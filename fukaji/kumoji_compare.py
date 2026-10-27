"""Research-only G7 vs frozen official KUMOJI, with independent openings."""
import argparse
import json
from collections import Counter
from pathlib import Path
import shogi
from .arena import opening
from .direct_arena import digest

G7_SHA="417d18e7b64494c9febdf6d217afbe4e4979e68d04528f16db8b45c0bcde2a0e"
KUMOJI_REF="807db500ee2739c1eb6a527b614eeea40a5993d6"
KNOWLEDGE_SHA="6da204cd1e65600ccdbb9f3dbe463627687d6cf7eb8347da6a103239129dac4c"
SEED,PAIRS,SHARDS,MAX_PLIES=20274027,32,8,240
METRICS=("nodes","turns","depth_sum","elapsed_ms","knowledge_probes","knowledge_hits","ponder_hits","long_think_reports")

def expected_request():
    return dict(research_only=True,training_allowed=False,model_adoption_allowed=False,
      official_engine_changes_allowed=False,g7_source_run=40434466661,
      g7_sha256=G7_SHA,kumoji_ref=KUMOJI_REF,knowledge_sha256=KNOWLEDGE_SHA,
      seed=SEED,pairs=PAIRS,shards=SHARDS,opening_plies=6,max_plies=MAX_PLIES,
      g7_movetime_ms=200,g7_depth=2,g7_mix_percent=25,g7_fast_relative=True,
      g7_root_alpha_beta=True,kumoji_go="go movetime 200",
      kumoji_adaptive_long_think=True,kumoji_position_knowledge=True,
      kumoji_ponder=True,kumoji_search_threads=2)

def validate_request(cfg):
    if cfg!=expected_request(): raise ValueError("unapproved frozen research request")
    return cfg

def empty_stats():
    return {side+"_"+key:0 for side in ("g7","kumoji") for key in METRICS}

def verify(row):
    board=shogi.Board()
    if len(row["opening"])!=6 or len(row["opening"])+len(row["moves"])>MAX_PLIES:
        raise ValueError("incorrect opening/move count")
    for word in row["opening"]+row["moves"]:
        move=shogi.Move.from_usi(word)
        if move not in board.legal_moves: raise ValueError("illegal replay "+word)
        board.push(move)
    status,winner=row["status"],row["winner"]
    if status=="checkmate":
        if not board.is_checkmate() or winner!=int(not board.turn): raise ValueError("false mate")
    elif status=="move_limit_unresolved":
        if len(row["opening"])+len(row["moves"])!=MAX_PLIES or winner is not None or board.is_checkmate(): raise ValueError("false move limit")
    elif status=="repetition_unresolved":
        if winner is not None or not board.is_fourfold_repetition(): raise ValueError("false repetition")
    elif status=="perpetual_check":
        if winner not in (0,1) or not board.is_fourfold_repetition(): raise ValueError("false perpetual check")
    elif status=="resignation":
        if row.get("terminal_token")!="resign" or winner!=int(not board.turn): raise ValueError("false resignation")
    elif status=="declare_win":
        from benchmarks.arena import can_declare_win
        if row.get("terminal_token")!="win" or winner!=int(board.turn) or not can_declare_win(board): raise ValueError("false declaration")
    elif status=="no_legal_moves":
        if any(board.legal_moves) or winner!=int(not board.turn): raise ValueError("false no-legal result")
    else: raise ValueError("unknown termination "+str(status))
    if set(row["stats"])!=set(empty_stats()): raise ValueError("missing telemetry")
    return True

def play_one(head,g7_black,g7,kumoji,seed):
    from benchmarks.arena import Adjudicator,can_declare_win
    board=shogi.Board()
    for word in head:
        move=shogi.Move.from_usi(word)
        if move not in board.legal_moves: raise ValueError("illegal opening")
        board.push(move)
    g7.configure_game(seed)
    kumoji.configure_game(seed+1)
    adjudicator=Adjudicator(board)
    history=list(head);moves=[];stats=empty_stats()
    status,winner,terminal_token="move_limit_unresolved",None,None
    for _ in range(MAX_PLIES-len(head)):
        if board.is_checkmate():
            status,winner="checkmate",int(not board.turn);break
        if board.is_fourfold_repetition():
            status="repetition_unresolved";break
        is_g7=(board.turn==shogi.BLACK)==g7_black
        side="g7" if is_g7 else "kumoji"
        engine=g7 if is_g7 else kumoji
        token=engine.bestmove(history)
        info=engine.last_search
        stats[side+"_turns"]+=1
        for field,source in (("nodes","nodes"),("depth_sum","depth"),("elapsed_ms","elapsed_ms"),
                             ("knowledge_probes","knowledge_probes"),("knowledge_hits","knowledge_hits")):
            stats[side+"_"+field]+=info.get(source,0)
        stats[side+"_ponder_hits"]+=int(bool(info.get("ponder_hit")))
        stats[side+"_long_think_reports"]+=int(bool(info.get("long_think_used")))
        if token=="resign":
            status,winner,terminal_token="resignation",int(not board.turn),token;break
        if token=="win":
            if not can_declare_win(board): raise ValueError("illegal declaration")
            status,winner,terminal_token="declare_win",int(board.turn),token;break
        move=shogi.Move.from_usi(token)
        if move not in board.legal_moves: raise ValueError("illegal USI response "+token)
        board.push(move);moves.append(token);history.append(token)
        terminal=adjudicator.after_move(board)
        if terminal is not None:
            winner,status=terminal
            if status=="fourfold_repetition": status="repetition_unresolved"
            break
        if len(history)<MAX_PLIES: engine.begin_ponder(history,board)
    else:
        if board.is_checkmate(): status,winner="checkmate",int(not board.turn)
    row=dict(status=status,winner=winner,terminal_token=terminal_token,
             opening=head,moves=moves,stats=stats)
    verify(row)
    return row

def play_shard(request,g7_path,kumoji_path,knowledge_path,shard,out):
    validate_request(json.loads(Path(request).read_text()))
    if type(shard)!=int or not 0<=shard<SHARDS: raise ValueError("invalid shard")
    if digest(g7_path)!=G7_SHA or digest(knowledge_path)!=KNOWLEDGE_SHA: raise ValueError("model/knowledge hash mismatch")
    from benchmarks.arena import Engine
    g7=Engine("build/fukaji-usi","g7",
      dict(ValueModelFile=str(Path(g7_path).resolve()),ValueModelType="relative",
           ValueModelMix=25,SearchDepth=2,FastRelativeEval=True,
           RootAlphaBeta=True,USI_Ponder=False),"go movetime 200")
    kumoji=Engine(kumoji_path,"kumoji",
      dict(AdaptiveLongThink=True,PositionKnowledge=True,
           PositionKnowledgeFile=str(Path(knowledge_path).resolve()),
           SearchThreads=2,USI_Ponder=True),"go movetime 200")
    rows=[]
    try:
        for pair in range(shard,PAIRS,SHARDS):
            head=opening(SEED+pair,6)
            for black in (True,False):
                row=play_one(head,black,g7,kumoji,SEED+4*pair+int(not black))
                row.update(pair=pair,g7_black=black)
                rows.append(row)
    finally:
        g7.close();kumoji.close()
    if len(rows)!=8: raise ValueError("wrong shard game count")
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    games=root/"games.jsonl"
    games.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    report=dict(shard=shard,games=8,request=expected_request(),g7_sha256=G7_SHA,
                kumoji_binary_sha256=digest(kumoji_path),knowledge_sha256=KNOWLEDGE_SHA,
                games_sha256=digest(games),legal_replay_verified=True)
    (root/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

def aggregate(request,parts,out):
    validate_request(json.loads(Path(request).read_text()))
    paths=list(Path(parts).glob("*/report.json"))
    if len(paths)!=SHARDS: raise ValueError("wrong shard count")
    rows=[];seen=set();binary_hashes=set()
    for path in paths:
        rep=json.loads(path.read_text());shard=rep.get("shard")
        if (type(shard)!=int or shard not in range(SHARDS) or shard in seen or
            rep.get("games")!=8 or rep.get("request")!=expected_request() or
            rep.get("g7_sha256")!=G7_SHA or rep.get("knowledge_sha256")!=KNOWLEDGE_SHA or
            rep.get("legal_replay_verified") is not True): raise ValueError("invalid shard")
        seen.add(shard);binary_hashes.add(rep["kumoji_binary_sha256"])
        gp=path.parent/"games.jsonl"
        if digest(gp)!=rep["games_sha256"]: raise ValueError("shard hash mismatch")
        group=[json.loads(x) for x in gp.read_text().splitlines()]
        if len(group)!=8: raise ValueError("wrong rows")
        for pair in range(shard,PAIRS,SHARDS):
            p=[r for r in group if r.get("pair")==pair]
            if len(p)!=2 or {r.get("g7_black") for r in p}!={True,False}: raise ValueError("unpaired")
            for row in p:
                if row["opening"]!=opening(SEED+pair,6): raise ValueError("opening mismatch")
                verify(row);rows.append(row)
    if len(binary_hashes)!=1 or len(rows)!=64: raise ValueError("wrong binaries/games")
    rows.sort(key=lambda r:(r["pair"],not r["g7_black"]))
    if len({tuple(r["opening"]+r["moves"]) for r in rows})!=64: raise ValueError("duplicate games")
    wins=Counter();pairs=Counter()
    for row in rows:
        if row["winner"] is None: wins["unresolved"]+=1
        elif (row["winner"]==0)==row["g7_black"]: wins["g7"]+=1
        else: wins["kumoji"]+=1
    for pair in range(PAIRS):
        subset=[r for r in rows if r["pair"]==pair]
        w=sum(r["winner"] is not None and (r["winner"]==0)==r["g7_black"] for r in subset)
        l=sum(r["winner"] is not None and (r["winner"]==0)!=r["g7_black"] for r in subset)
        pairs["g7" if w>l else "kumoji" if l>w else "equal"]+=1
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    games=root/"all_games.jsonl"
    games.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    summary=dict(games=64,pairs=32,g7_wins=wins["g7"],kumoji_wins=wins["kumoji"],
      unresolved=wins["unresolved"],pair_comparison=dict(pairs),
      termination=dict(Counter(r["status"] for r in rows)),
      request=expected_request(),kumoji_binary_sha256=next(iter(binary_hashes)),
      games_sha256=digest(games),legal_replay_verified=True,
      stats={key:sum(r["stats"][key] for r in rows) for key in rows[0]["stats"]},
      caution="Unequal compute by design: G7 200ms/depth2 vs KUMOJI 200ms+adaptive long think+ponder+adopted knowledge; research only, no training/adoption.")
    (root/"aggregate_report.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("play","aggregate"),required=True)
    p.add_argument("--request",required=True)
    p.add_argument("--g7-model");p.add_argument("--kumoji-engine");p.add_argument("--knowledge")
    p.add_argument("--shard",type=int);p.add_argument("--parts");p.add_argument("--out",required=True)
    a=p.parse_args()
    if a.mode=="play":
        if a.shard is None or not all((a.g7_model,a.kumoji_engine,a.knowledge)): p.error("missing play args")
        play_shard(a.request,a.g7_model,a.kumoji_engine,a.knowledge,a.shard,a.out)
    else:
        if not a.parts:p.error("missing parts")
        aggregate(a.request,a.parts,a.out)

if __name__=="__main__": main()
