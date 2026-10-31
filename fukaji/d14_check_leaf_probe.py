"""D14 historical FUKAJI under-check roots, baseline vs check-leaf at 200 ms.

Development data only (100 prior KUMOJI losses). All source plies replayed and
source SHA and G8 model pinned. 12 distinct game roots fixed by deterministic
game-ID order *before* looking at selected moves. Baseline/experimental ABBA
on one runner; no formal adoption, no opponent-policy labels.
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
from .kumoji_loss_study import GAMES_SHA, snapshot
from .kumoji_defense_guard import _unsafe_root_move
from .relative_value import RelativeValue

MODEL_SHA = "18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492"
SELECT_COUNT = 12


def roots(games_path):
    data = Path(games_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != GAMES_SHA:
        raise ValueError("D14 source 100 games drift")
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if len(rows) != 100 or {r["game_id"] for r in rows} != set(range(100)):
        raise ValueError("D14 requires exact 100 games")
    found = []
    seen_positions = set()
    for row in sorted(rows, key=lambda g: g["game_id"]):
        if row["winner"] != "B" or row["reason"] != "checkmate":
            raise ValueError("D14 old failure source outcome changed")
        board = shogi.Board()
        for ply, word in enumerate(row["moves"]):
            if (ply >= 6 and (board.turn == shogi.BLACK) == row["a_black"]
                and board.is_check() and any(board.legal_moves)):
                key=" ".join(board.sfen().split()[:3])
                if key not in seen_positions:
                    seen_positions.add(key)
                    found.append({"game_id":row["game_id"],"ply":ply,
                                  "before":list(row["moves"][:ply]),"sfen":board.sfen()})
                break
            move = shogi.Move.from_usi(word)
            if move not in board.legal_moves:
                raise ValueError("D14 old source illegal game move")
            board.push(move)
        if len(found) >= SELECT_COUNT:
            break
    if len(found) != SELECT_COUNT:
        raise ValueError(f"D14 has only {len(found)} actual checked roots")
    if len(seen_positions) != SELECT_COUNT:
        raise ValueError("D14 root positions are not distinct")
    return found


def replay(prefix, expected):
    board=shogi.Board()
    for word in prefix:
        move=shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("D14 replay illegal move")
        board.push(move)
    if board.sfen()!=expected or not board.is_check():
        raise ValueError("D14 check root drift")
    return board


def run(games, model_path, output):
    if hashlib.sha256(Path(model_path).read_bytes()).hexdigest()!=MODEL_SHA:
        raise ValueError("D14 G8 weight mismatch")
    model=RelativeValue(model_path, mix_percent=25)
    samples=roots(games)
    calls=[]
    for row in samples:
        board=replay(row["before"],row["sfen"])
        before=snapshot(board)
        for trial,on in enumerate((False,True,True,False)):
            started=time.perf_counter()
            telemetry={}
            move,depth,nodes=select_move(
                board,movetime_ms=200,max_depth=3,model=model,
                avoid_repetition=True,fast_relative=True,
                root_alpha_beta=True,fast_move_list=True,
                fast_legal_guard=True, leaf_check_evasion=on,
                deadline_from_call=True,telemetry=telemetry)
            elapsed=round((time.perf_counter()-started)*1000,3)
            if snapshot(board)!=before:
                raise ValueError("D14 search changed root history")
            if move=="resign" or shogi.Move.from_usi(move) not in board.legal_moves:
                raise ValueError("D14 invalid root move")
            # Independent legal-move mate-in-one proof; not charged to player time.
            unsafe=_unsafe_root_move(board,shogi.Move.from_usi(move),time.monotonic()+30)
            if snapshot(board)!=before:
                raise ValueError("D14 independent proof changed board")
            calls.append({
                "game_id":row["game_id"],"ply":row["ply"],
                "research_candidate":on,"trial":trial,"move":move,
                "completed_depth":depth,"nodes":nodes,"elapsed_ms":elapsed,
                "allows_opponent_mate_in_one":unsafe})
        print("D14 replayed",row["game_id"],row["ply"],"4 calls",flush=True)
    if len(calls)!=SELECT_COUNT*4:
        raise ValueError("D14 ABBA incomplete")
    totals={}
    for flag,label in ((False,"baseline"),(True,"checked_horizon_one_reply")):
        subset=[x for x in calls if x["research_candidate"]==flag]
        totals[label]={
            "calls":len(subset),
            "allows_immediate_mate":sum(x["allows_opponent_mate_in_one"] for x in subset),
            "median_elapsed_ms":round(statistics.median(x["elapsed_ms"] for x in subset),3),
            "mean_elapsed_ms":round(statistics.mean(x["elapsed_ms"] for x in subset),3),
            "over_200ms":sum(x["elapsed_ms"]>200 for x in subset),
            "completed_depth3":sum(x["completed_depth"]>=3 for x in subset),
            "depth_histogram":dict(Counter(x["completed_depth"] for x in subset)),
            "nodes_sum":sum(x["nodes"] for x in subset)}
    report={
        "status":"D14_development_check_leaf_ABBA_completed_not_formal_strength",
        "source_100_games_sha256":GAMES_SHA,"G8_model_sha256":MODEL_SHA,
        "historical_checked_roots":SELECT_COUNT,"source_game_ply":[[x["game_id"],x["ply"]] for x in samples],
        "total_abba_calls":len(calls),"order":"OFF-ON-ON-OFF",
        "one_factor":"leaf_check_evasion",
        "same_frozen_G8_and_200ms":True,
        "totals":totals,"calls":calls,
        "limitations":[
            "Prior KUMOJI games are development only and not independent strength testing.",
            "Allows-mate metric checks an exact next-move mate only, not longer tactics.",
            "200ms is requested, actual outer wall time is measured and can exceed deadline.",
            "The one-evasion extension is not formally adopted, not in official USI options.",
            "No model training or external teacher move copy."]
    }
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/"diagnosis.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="calls"},ensure_ascii=False,indent=2))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--games",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    run(a.games,a.model,a.out)


if __name__=="__main__":
    main()
