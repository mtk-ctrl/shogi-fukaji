"""Frozen G8 search-speed study on early and late roots from past KUMOJI losses.

One-factor A/B: short-circuit impossible checkmate tests at nonchecked nodes.
The current engine already exposes guard_checkmate opt-in; this script NEVER
enables it in official USI defaults. Historical positions are development only.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time
import shogi

from .engine import select_move
from .kumoji_loss_study import GAMES_SHA, PROBE_PAIRS, snapshot
from .relative_value import RelativeValue

G8_SHA="18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492"

def positions(games_path):
    raw=Path(games_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=GAMES_SHA:raise ValueError("wrong frozen 100 games")
    rows=[json.loads(s) for s in raw.splitlines()]
    if len(rows)!=100 or {r["game_id"] for r in rows}!=set(range(100)):
        raise ValueError("wrong game collection")
    positions=[]
    for row in rows:
        if row["pair"] not in PROBE_PAIRS:continue
        words=row["moves"]
        if words[:6]!=row["opening"] or len(words)<24:raise ValueError("game missing early root")
        cutoff=18 if row["a_black"] else 19
        for phase,count in (("opening_early",cutoff),("pre_last_defense",len(words)-2)):
            board=shogi.Board()
            for word in words[:count]:
                m=shogi.Move.from_usi(word)
                if m not in board.legal_moves:raise ValueError("illegal source history")
                board.push(m)
            if (board.turn==shogi.BLACK)!=row["a_black"]:
                raise ValueError("unexpected research side to move")
            if not any(board.legal_moves):raise ValueError("no legal sample moves")
            positions.append((row["game_id"],phase,board))
    positions.sort(key=lambda z:(z[0],z[1]))
    if len(positions)!=40:raise ValueError("not 20 early plus 20 late samples")
    return positions


def run(games,model_path,output):
    if hashlib.sha256(Path(model_path).read_bytes()).hexdigest()!=G8_SHA:
        raise ValueError("wrong pinned G8")
    model=RelativeValue(model_path,mix_percent=25)
    roots=positions(games)
    records=[]
    for game_id,phase,board in roots:
        initial=snapshot(board)
        outcomes={}
        for trial,guard in enumerate((False,True,True,False)):
            started=time.perf_counter()
            telemetry={}
            move,depth,nodes=select_move(
                board,movetime_ms=200,max_depth=3,model=model,
                avoid_repetition=True,fast_relative=True,root_alpha_beta=True,
                fast_move_list=True,deadline_from_call=True,
                guard_checkmate=guard,telemetry=telemetry)
            elapsed=round((time.perf_counter()-started)*1000,3)
            if snapshot(board)!=initial:raise ValueError("search altered board/history")
            if move=="resign" or shogi.Move.from_usi(move) not in board.legal_moves:
                raise ValueError("illegal research move")
            record={"game_id":game_id,"phase":phase,"trial":trial,
                    "variant":"skip_no_check_mate_test" if guard else "baseline",
                    "move":move,"depth":depth,"nodes":nodes,
                    "elapsed_ms":elapsed,"root_preparation_ms":telemetry.get("root_preparation_ms")}
            records.append(record)
        print("verified",game_id,phase,"4 search responses",flush=True)
    if len(records)!=160:raise ValueError("not 40 root x 4 completed calls")
    result={}
    for phase in ("opening_early","pre_last_defense"):
        for name in ("baseline","skip_no_check_mate_test"):
            subset=[x for x in records if x["phase"]==phase and x["variant"]==name]
            result[phase+"/"+name]={
                "calls":len(subset),
                "median_ms":round(statistics.median(x["elapsed_ms"] for x in subset),3),
                "mean_ms":round(statistics.mean(x["elapsed_ms"] for x in subset),3),
                "median_nodes":statistics.median(x["nodes"] for x in subset),
                "completed_depth":dict(Counter(x["depth"] for x in subset)),
                "depth_three_completed":sum(x["depth"]>=3 for x in subset),
                "over_200_ms":sum(x["elapsed_ms"]>200 for x in subset)}
    report={"status":"historical_G8_noncheck_guard_ABBA_throughput_not_strength",
            "source_sha256":GAMES_SHA,"g8_model_sha256":G8_SHA,
            "historical_roots":40,"search_calls":160,
            "order":"A_B_B_A", "one_factor":"guard_checkmate_false_versus_true",
            "groups":result,"calls":records,
            "limitations":["Old opponent games are development, never test split.",
              "No same-Cpu match wins or Elo measured.",
              "Whole-call clock applied to both arms; original 100 games had internal deadline.",
              "Research engine and official options not modified."]}
    target=Path(output);target.mkdir(parents=True,exist_ok=True)
    (target/"noncheck_guard_probe.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="calls"},ensure_ascii=False,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--games",required=True)
    parser.add_argument("--g8-model",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    run(args.games,args.g8_model,args.out)


if __name__=="__main__":main()
