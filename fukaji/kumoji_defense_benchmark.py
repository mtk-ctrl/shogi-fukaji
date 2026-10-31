"""Opt-in, replay-audited 20-game G8 defense-guard development benchmark.

Uses only old KUMOJI games to locate a handful of FUKAJI roots.
B-A-A-B fixed replay on a single runner controls run order; these historic
positions are NOT independent new game strength evidence. Models/USI are frozen.
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
from .kumoji_defense_guard import select_guarded, _unsafe_root_move
from .kumoji_loss_study import GAMES_SHA, PROBE_PAIRS, snapshot
from .relative_value import RelativeValue

MODEL_SHA="18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492"

def fetch_roots(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=GAMES_SHA:
        raise ValueError("historical game hash changed")
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows)!=100 or {r["game_id"] for r in rows}!=set(range(100)):
        raise ValueError("unexpected original games")
    roots=[]
    for row in rows:
        if row["pair"] not in PROBE_PAIRS: continue
        words=row["moves"]
        if words[:6]!=row["opening"] or len(words)<8:
            raise ValueError("opening prefix mismatch")
        board=shogi.Board()
        for word in words[:-2]:
            m=shogi.Move.from_usi(word)
            if m not in board.legal_moves: raise ValueError("illegal source")
            board.push(m)
        if ((board.turn == shogi.BLACK)!=row["a_black"] or
            words[-2] not in [m.usi() for m in board.legal_moves]):
            raise ValueError("wrong last FUKAJI root")
        roots.append((row["game_id"],words[-2],board))
    roots.sort(key=lambda x:x[0])
    if len(roots)!=20:raise ValueError("20 sampled roots required")
    return roots


def benchmark(games,model_path,output):
    if hashlib.sha256(Path(model_path).read_bytes()).hexdigest()!=MODEL_SHA:
        raise ValueError("frozen G8 model changed")
    model=RelativeValue(model_path,mix_percent=25)
    roots=fetch_roots(games)
    runs=[]
    for game_id,original,board in roots:
        before=snapshot(board)
        for trial,kind in enumerate(("baseline","guard","guard","baseline")):
            started=time.perf_counter()
            if kind=="baseline":
                stat={}
                move,depth,nodes=select_move(
                    board,movetime_ms=200,max_depth=3,model=model,
                    avoid_repetition=True,fast_relative=True,
                    root_alpha_beta=True,fast_move_list=True,
                    deadline_from_call=True,telemetry=stat)
                status="baseline"
            else:
                got=select_guarded(board,model,movetime_ms=200,reserve_ms=40,
                                   depth=3,fast_relative=True,
                                   root_alpha_beta=True,fast_move_list=True)
                move,depth,nodes=got["move"],got["depth"],got["nodes"]
                status=got["guard_status"]
            elapsed=round((time.perf_counter()-started)*1000,3)
            if snapshot(board)!=before:raise ValueError("search or guard modified history")
            if move=="resign" or shogi.Move.from_usi(move) not in board.legal_moves:
                raise ValueError("illegal candidate result")
            # Measure actual immediate-mate vulnerability separately from the
            # capped playing time; this diagnostic is NOT charged to play.
            proven_unsafe=_unsafe_root_move(
                board,shogi.Move.from_usi(move),time.monotonic()+20)
            runs.append({"game_id":game_id,"trial":trial,"arm":kind,
                         "move":move,"source_historical_move":original,
                         "depth":depth,"nodes":nodes,"elapsed_ms":elapsed,
                         "allows_immediate_mate":proven_unsafe,
                         "guard_status":status})
        print("root",game_id,"verified four ABBA runs",flush=True)
    if len(runs)!=80:raise ValueError("not 20 x 4 complete replay")
    def aggregate(kind):
        rs=[x for x in runs if x["arm"]==kind]
        return {"calls":len(rs),"moves_allow_immediate_mate":sum(x["allows_immediate_mate"] for x in rs),
                "median_wall_ms":round(statistics.median(x["elapsed_ms"] for x in rs),3),
                "mean_wall_ms":round(statistics.mean(x["elapsed_ms"] for x in rs),3),
                "over_200_ms":sum(x["elapsed_ms"]>200 for x in rs),
                "depth_three_completed":sum(x["depth"]>=3 for x in rs)}
    bygame={}
    for game_id,_,_ in roots:
        b=[x for x in runs if x["game_id"]==game_id and x["arm"]=="baseline"]
        g=[x for x in runs if x["game_id"]==game_id and x["arm"]=="guard"]
        bygame[game_id]={"baseline_unsafe":[x["allows_immediate_mate"] for x in b],
                          "guard_unsafe":[x["allows_immediate_mate"] for x in g],
                          "guard_status":[x["guard_status"] for x in g]}
    summary={
        "status":"historical_G8_defense_candidate_dev_benchmark_only",
        "source_games_sha256":GAMES_SHA,"model_sha256":MODEL_SHA,
        "historical_sampled_games":20,"same_runner_total_calls":80,
        "order":"baseline_guard_guard_baseline",
        "baseline":aggregate("baseline"),"guard":aggregate("guard"),
        "by_game":bygame,"per_call":runs,
        "limitations":["The 20 roots are retrospective development data, not independent playing strength.",
          "Baseline uses whole-call clock; original historical USI used internal clock; do not equate their moves.",
          "Defense guard only prevents one-ply mate when proof completes before deadline.",
          "Equal requested 200ms does not imply equal measured wall-clock; regressions must be reported.",
          "No model trained, engine defaults changed, or candidate formally adopted."]
    }
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/"defense_benchmark.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("by_game","per_call")},ensure_ascii=False,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--games",required=True)
    p.add_argument("--g8-model",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    benchmark(args.games,args.g8_model,args.out)


if __name__=="__main__":
    main()
