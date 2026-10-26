"""Paired, auditable FUKAJI search-versus-search arena.

Each opening is played twice with candidate alternating color. Only
verified checkmates are wins; move limits/repetitions are unresolved.
Do not equate "unresolved" with "draw" or use it as a training target.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shogi
from .engine import select_move
from .value_model import LinearValue

def opening(seed, plies):
    rng=random.Random(seed)
    b=shogi.Board()
    history=[]
    for _ in range(plies):
        options=list(b.legal_moves)
        if not options or b.is_checkmate() or b.is_fourfold_repetition():
            break
        m=options[rng.randrange(len(options))]
        b.push(m)
        history.append(m.usi())
    return history

def play(history, candidate_black, candidate, movetime, depth, max_plies, search_mode="timed"):
    b=shogi.Board()
    for word in history:
        move=shogi.Move.from_usi(word)
        if move not in b.legal_moves:
            raise ValueError("illegal arena opening")
        b.push(move)
    moves=[]
    stats={"candidate_nodes":0,"baseline_nodes":0,"candidate_turns":0,"baseline_turns":0,
           "candidate_depth_zero":0,"baseline_depth_zero":0,
           "candidate_depth_sum":0,"baseline_depth_sum":0}
    for _ in range(max_plies):
        if b.is_checkmate():
            return {"status":"checkmate","winner":int(not b.turn),"moves":moves,"stats":stats}
        if b.is_fourfold_repetition():
            return {"status":"repetition_unresolved","winner":None,"moves":moves,"stats":stats}
        is_candidate=(b.turn==shogi.BLACK)==candidate_black
        model=candidate if is_candidate else None
        word,finished,nodes=select_move(b,movetime,depth,model=model,
                                        complete_depth=(search_mode=="complete_depth"))
        side="candidate" if is_candidate else "baseline"
        stats[side+"_nodes"]+=nodes
        stats[side+"_turns"]+=1
        stats[side+"_depth_sum"]+=finished
        if finished==0:
            stats[side+"_depth_zero"]+=1
        if word=="resign":
            # No end reason is invented from the engine's lack of moves.
            return {"status":"no_move_unresolved","winner":None,"moves":moves,"stats":stats}
        move=shogi.Move.from_usi(word)
        if move not in b.legal_moves:
            raise ValueError("engine produced illegal move")
        moves.append(word)
        b.push(move)
    status="checkmate" if b.is_checkmate() else "move_limit_unresolved"
    return {"status":status,"winner":int(not b.turn) if b.is_checkmate() else None,"moves":moves,"stats":stats}

def run(model_path, out, pairs=2, movetime=2, depth=1, max_plies=30, opening_plies=4, mix=50, seed=20261026, search_mode="timed", model_type="legacy"):
    if not (1<=pairs<=500 and 1<=movetime<=10000 and 1<=depth<=6 and 1<=max_plies<=800 and 0<=opening_plies<=20):
        raise ValueError("unsafe arena limits")
    if search_mode not in ("timed","complete_depth"):
        raise ValueError("invalid search mode")
    if search_mode=="complete_depth" and depth>2:
        raise ValueError("complete-depth diagnostic limited to depth <=2")
    if model_type not in ("legacy", "mover_relative_v1"):
        raise ValueError("unknown model schema")
    if model_type == "mover_relative_v1":
        from .relative_value import RelativeValue
        candidate = RelativeValue(model_path, mix_percent=mix)
    else:
        candidate = LinearValue(model_path, mix_percent=mix)
    rows=[]
    for i in range(pairs):
        hist=opening(seed+i,opening_plies)
        for black in (True,False):
            result=play(hist,black,candidate,movetime,depth,max_plies,search_mode=search_mode)
            result.update({"pair":i,"candidate_black":black,"opening":hist})
            rows.append(result)
    wins=losses=unresolved=0
    for r in rows:
        if r["winner"] is None: unresolved+=1
        elif (r["winner"]==int(shogi.BLACK))==r["candidate_black"]:wins+=1
        else:losses+=1
    target=Path(out);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w",encoding="utf-8") as f:
        for r in rows:f.write(json.dumps(r)+"\n")
    summary={"pairs":pairs,"games":len(rows),"verified_candidate_wins":wins,"verified_baseline_wins":losses,"unresolved":unresolved,
        "candidate_score_on_resolved":None if wins+losses==0 else wins/(wins+losses),
        "search_diagnostics":{key:sum(r["stats"][key] for r in rows) for key in rows[0]["stats"]},
        "max_plies":max_plies,"opening_plies":opening_plies,
        "model_sha256":candidate.sha256,"mix":mix,"movetime_ms":movetime if search_mode=="timed" else None,
        "search_mode":search_mode,"depth":depth,"seed":seed,"model_type":model_type,
        "arena_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
        "caution":"No Elo claim from unresolved games or tiny arena."}
    (target.parent/(target.stem+".report.json")).write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--out",default="build/arena.jsonl")
    p.add_argument("--pairs",type=int,default=2)
    p.add_argument("--movetime",type=int,default=2)
    p.add_argument("--depth",type=int,default=1)
    p.add_argument("--max-plies",type=int,default=30)
    p.add_argument("--opening-plies",type=int,default=4)
    p.add_argument("--mix",type=int,default=50)
    a=p.parse_args()
    run(a.model,a.out,a.pairs,a.movetime,a.depth,a.max_plies,a.opening_plies,a.mix)
if __name__=="__main__":main()
