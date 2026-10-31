"""Exactly 50 color-swapped opening pairs: G8 research FUKAJI vs full-archive KUMOJI.

Separate research candidate; no automatic promotion, training or production writes.
Imports the authoritative KUMOJI USI runner rather than reimplementing adjudication.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import shogi
from .arena import opening
from .kumoji_match import audit_game
from .kumoji_paired import play_paired_game, summarize_pairs
from .relative_value import RelativeValue

G8_SHA = "18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492"
KUMOJI_COMMIT = "76ee74d04a800285e24dd81ef01711e990cfa181"
OLD_KNOWLEDGE_SHA = "6da204cd1e65600ccdbb9f3dbe463627687d6cf7eb8347da6a103239129dac4c"
PAIRS, SHARDS, MAX_PLIES = 50, 20, 300
SEED = 202610315

def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def config():
    return dict(research_only=True,formal_adoption=False,
       fukaji_model_kind="G8_frozen_studied_depth3",fukaji_g8_sha256=G8_SHA,
       kumoji_commit=KUMOJI_COMMIT,prior_adopted_knowledge_sha256=OLD_KNOWLEDGE_SHA,
       games=100,pairs=PAIRS,shards=SHARDS,opening_plies=6,max_plies=MAX_PLIES,
       seed=SEED,movetime_ms=200,depth_fukaji=3,
       g8_mix_percent=25,fast_relative=True,fast_move_list=True,root_alpha_beta=True,
       kumoji_adaptive_long_think=True,kumoji_search_threads=2,kumoji_ponder=True)

def validate_request(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg != config():
        raise ValueError("100-game protocol does not match preregistered frozen request")
    return cfg

def input_files(folder):
    folder=Path(folder)
    cfg=json.loads((folder/"receipt.json").read_text(encoding="utf-8"))
    for name in ("kumoji","g8.npz","full-knowledge.tsv","full-report.json"):
        if not (folder/name).is_file():
            raise ValueError("missing pinned source binary/model/knowledge/audit: "+name)
    from benchmarks.arena import position_knowledge_metadata
    learned = RelativeValue(folder/"g8.npz", mix_percent=25)
    if digest(folder/"g8.npz") != G8_SHA:
        raise ValueError("the independent latest FUKAJI G8 model differs from expected SHA")
    report=json.loads((folder/"full-report.json").read_text(encoding="utf-8"))
    if (report.get("source_observations")!=1079 or report.get("prior_positions")!=390
        or report.get("newly_validated_hints",0)<1 or report.get("formal_adoption") is not False
        or report.get("candidate_sha256")!=digest(folder/"full-knowledge.tsv")):
        raise ValueError("the entire archived research was not actually analyzed/packaged")
    meta=position_knowledge_metadata(dict(PositionKnowledge=True,PositionKnowledgeFile=str(folder/"full-knowledge.tsv")))
    if not meta.get("available") or meta["positions"] != report["playable_positions"]:
        raise ValueError("new full-research knowledge data not loaded correctly")
    if (cfg.get("model_sha256")!=G8_SHA or cfg.get("knowledge_sha256")!=meta["sha256"]
        or cfg.get("binary_sha256")!=digest(folder/"kumoji")):
        raise ValueError("playing-engine input hashes differ from prepared receipt")
    return cfg,report

def indices(shard):
    if type(shard) is not int or not 0<=shard<SHARDS:
        raise ValueError("invalid shard")
    return list(range(shard,PAIRS,SHARDS))

def play_shard(req,folder,shard,out):
    validate_request(req)
    receipt,knowledge_report=input_files(folder)
    from benchmarks import arena
    folder=Path(folder).resolve()
    f_path=folder/"g8.npz"
    f_usi=folder/"fukaji-usi"
    f_usi.parent.mkdir(exist_ok=True,parents=True)
    f_usi.write_text("#!/bin/sh\nexec python3 -m fukaji.engine\n")
    f_usi.chmod(0o755)
    a=arena.Engine(f_usi,"A",{
        "ValueModelType":"relative","ValueModelFile":str(f_path),
        "ValueModelMix":25,"SearchDepth":3,"FastRelativeEval":True,
        "RootAlphaBeta":True,"FastMoveList":True,"USI_Ponder":False},
        "go movetime 200")
    b=arena.Engine(folder/"kumoji","B",{
        "PositionKnowledge":True,"PositionKnowledgeFile":str(folder/"full-knowledge.tsv"),
        "AdaptiveLongThink":True,"SearchThreads":2,"RiskVerify":True,
        "USI_Ponder":True}, "go movetime 200")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p=out/"games.jsonl"
    rows=[]
    try:
        with p.open("w",encoding="utf-8") as dst:
            for pair in indices(shard):
                head=opening(SEED+pair,6)
                if len(head)!=6: raise ValueError("opening must contain six plies")
                for seat,black in enumerate((True,False)):
                    row=play_paired_game(arena,a,b,head,black,MAX_PLIES,SEED+pair*4+seat*2)
                    row.update(pair=pair,game_id=2*pair+seat,kind="G8_vs_whole_archive_KUMOJI")
                    audit_game(row)
                    dst.write(json.dumps(row,ensure_ascii=False)+"\n")
                    dst.flush()
                    rows.append(row)
                    print("paired",row["game_id"],row["winner"],row["reason"],flush=True)
    finally:
        a.close();b.close()
    if len(rows)!=2*len(indices(shard)):
        raise ValueError("shard incomplete")
    report=dict(request=config(),shard=shard,games=len(rows),
        game_ids=[r["game_id"] for r in rows],games_sha256=digest(p),
        input_receipt_sha256=digest(folder/"receipt.json"),
        knowledge_sha256=receipt["knowledge_sha256"],
        fukaji_model_sha256=G8_SHA,
        full_research_observations=knowledge_report["source_observations"])
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    return report

def aggregate(req, folder, parts, out):
    validate_request(req)
    receipt, full_report=input_files(folder)
    receipt_sha=digest(Path(folder)/"receipt.json")
    files=list(Path(parts).glob("*/report.json"))
    if len(files)!=SHARDS:
        raise ValueError("20/20 shards required, got "+str(len(files)))
    observed_shards=set();observed_games=set();rows=[]
    for path in files:
        report=json.loads(path.read_text(encoding="utf-8"))
        shard=report.get("shard")
        if (type(shard) is not int or shard in observed_shards or not 0<=shard<SHARDS
            or report.get("request")!=config()
            or report.get("input_receipt_sha256")!=receipt_sha
            or report.get("knowledge_sha256")!=receipt["knowledge_sha256"]
            or report.get("fukaji_model_sha256")!=G8_SHA):
            raise ValueError("invalid shard identity or hash")
        observed_shards.add(shard)
        game_file=path.parent/"games.jsonl"
        if digest(game_file)!=report.get("games_sha256"):raise ValueError("games SHA changed")
        games=[json.loads(x) for x in game_file.read_text().splitlines()]
        expected={2*p+s for p in indices(shard) for s in (0,1)}
        if len(games)!=len(expected) or {x["game_id"] for x in games}!=expected:
            raise ValueError("shard missing games or wrong seats")
        for row in games:
            game_id=row["game_id"]
            pair,seat=divmod(game_id,2)
            if (game_id in observed_games or row.get("pair")!=pair
                or row.get("a_black") is not (seat==0)
                or row.get("opening")!=opening(SEED+pair,6)
                or row.get("kind")!="G8_vs_whole_archive_KUMOJI"
                or len(row.get("moves",[]))>MAX_PLIES):
                raise ValueError("game/seed/seat mismatch")
            if audit_game(row)!=len(row["moves"]):
                raise ValueError("illegal game replay")
            if len(row["move_records"])!=len(row["played_moves"]):
                raise ValueError("incomplete per-move telemetry")
            observed_games.add(game_id)
            rows.append(row)
    rows.sort(key=lambda x:x["game_id"])
    if len(rows)!=100 or observed_games!=set(range(100)):
        raise ValueError("not exactly 100 verified games")
    winners=Counter(r["winner"] or "unresolved" for r in rows)
    reasons=Counter(r["reason"] for r in rows)
    all_plays=[m for row in rows for m in row["move_records"]]
    kmoves=[m for m in all_plays if m["engine"]=="B"]
    fmoves=[m for m in all_plays if m["engine"]=="A"]
    direct=[m for m in kmoves if m["search"].get("decision_source")=="research"]
    def sum_search(moves,term):
        return sum(m["search"].get(term,0) for m in moves)
    report=dict(protocol=config(),status="finished_100_games_subject_to_owner_adoption",
      games=100,opening_pairs=50,shards=SHARDS,
      fukaji_wins=winners["A"],kumoji_wins=winners["B"],
      unresolved=winners["unresolved"],reasons=dict(reasons),
      paired_comparison=summarize_pairs(rows,PAIRS),
      fukaji_plays=len(fmoves),kumoji_plays=len(kmoves),
      kumoji_research_direct_moves=len(direct),
      kumoji_research_direct_games=sum(any(m["engine"]=="B" and m["search"].get("decision_source")=="research" for m in row["move_records"]) for row in rows),
      kumoji_research_ids=dict(Counter(m["search"].get("research_id") for m in direct)),
      kumoji_knowledge_probes=sum_search(kmoves,"knowledge_probes"),
      kumoji_knowledge_hits=sum_search(kmoves,"knowledge_hits"),
      kumoji_knowledge_promotions=sum_search(kmoves,"knowledge_promotions"),
      kumoji_ponder_hits=sum(1 for m in kmoves if m["search"].get("ponder_hit")),
      known_positions=full_report["playable_positions"],
      full_corpus_scanned=1079,full_corpus_category_counts=full_report["category_counts"],
      input=receipt,
      limitations=["Cross-engine compute and knowledge conditions are asymmetric.",
        "FUKAJI G8/depth3 research configuration is not a formally adopted FUKAJI release.",
        "Unresolved games are neither wins nor draws.",
        "Only vetted 227 long studies are direct research decisions; older valid studies are move-order hints.",
        "A 100-game result does not prove broad playing strength or formal adoption."])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=out/"all_games.jsonl"
    path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),encoding="utf-8")
    report["all_games_sha256"]=digest(path)
    (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="input"},indent=2,ensure_ascii=False))
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("check","play","aggregate"),required=True)
    parser.add_argument("--request",required=True)
    parser.add_argument("--input")
    parser.add_argument("--shard",type=int)
    parser.add_argument("--parts")
    parser.add_argument("--out")
    args=parser.parse_args()
    validate_request(args.request)
    if args.mode=="check": print("full100 protocol valid");return
    if not args.input or not args.out:parser.error("input and out required")
    if args.mode=="play":
        if args.shard is None:parser.error("shard required")
        play_shard(args.request,args.input,args.shard,args.out)
    else:
        if not args.parts:parser.error("parts required")
        aggregate(args.request,args.input,args.parts,args.out)

if __name__=="__main__": main()
