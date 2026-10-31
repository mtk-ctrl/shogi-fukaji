"""Exact legal move equivalence and frozen G8 fast-legal-guard research.

Probes all 100 previously played game histories every seventh ply, including
a tactical subset; any different legal-move collection fails BEFORE A/B.
Then compares 200ms G8 search FastLegalGuard OFF and ON with A-B-B-A order
on 40 older early/late roots. Historical opponent games are not training.
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
from .engine import select_move,fast_legal_moves
from .kumoji_search_speed_probe import G8_SHA,positions
from .kumoji_loss_study import GAMES_SHA,snapshot
from .relative_value import RelativeValue


def audit_legal_sequences(games_path,stride=7):
    if not 1 <= stride <= 50:raise ValueError("invalid sample stride")
    raw=Path(games_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=GAMES_SHA:raise ValueError("source changed")
    rows=[json.loads(s) for s in raw.splitlines()]
    if len(rows)!=100:raise ValueError("not 100 games")
    audited=0
    for game in rows:
        board=shogi.Board()
        for ply,word in enumerate(game["moves"]):
            if ply%stride==0 or ply>=len(game["moves"])-3:
                before=snapshot(board)
                slow=[m.usi() for m in board.legal_moves]
                fast=[m.usi() for m in fast_legal_moves(board)]
                if (len(fast)!=len(set(fast)) or
                    sorted(fast)!=sorted(slow) or snapshot(board)!=before):
                    raise ValueError(f"fast legal guard move-set mismatch game={game['game_id']} ply={ply}")
                audited+=1
            move=shogi.Move.from_usi(word)
            if move not in board.legal_moves:raise ValueError("illegal saved game")
            board.push(move)
        if not board.is_checkmate():raise ValueError("wrong game outcome")
    return audited


def compare(games_path,model_path,output):
    if hashlib.sha256(Path(model_path).read_bytes()).hexdigest()!=G8_SHA:
        raise ValueError("frozen G8 mismatch")
    validated_positions=audit_legal_sequences(games_path)
    roots=positions(games_path)
    model=RelativeValue(model_path,mix_percent=25)
    records=[]
    for game_id,phase,board in roots:
        before=snapshot(board)
        for index,fast in enumerate((False,True,True,False)):
            started=time.perf_counter()
            telemetry={}
            word,depth,nodes=select_move(
                board,movetime_ms=200,max_depth=3,model=model,
                avoid_repetition=True,fast_relative=True,
                root_alpha_beta=True,fast_move_list=True,
                fast_legal_guard=fast,deadline_from_call=True,
                telemetry=telemetry)
            elapsed=round((time.perf_counter()-started)*1000,3)
            if snapshot(board)!=before:raise ValueError("search mutated saved board")
            if word=="resign" or shogi.Move.from_usi(word) not in board.legal_moves:
                raise ValueError("invalid selected move")
            records.append({"game_id":game_id,"phase":phase,"trial":index,
                "arm":"fast_legal_guard" if fast else "standard_legal",
                "move":word,"depth":depth,"nodes":nodes,"elapsed_ms":elapsed,
                "root_preparation_ms":telemetry.get("root_preparation_ms")})
        print("verified pair",game_id,phase,flush=True)
    if len(records)!=160:raise ValueError("incomplete balanced legal guard trial")
    grouped={}
    for phase in ("opening_early","pre_last_defense"):
        for arm in ("standard_legal","fast_legal_guard"):
            subset=[z for z in records if z["phase"]==phase and z["arm"]==arm]
            grouped[phase+"/"+arm]={
                "calls":len(subset),
                "median_ms":round(statistics.median(z["elapsed_ms"] for z in subset),3),
                "mean_ms":round(statistics.mean(z["elapsed_ms"] for z in subset),3),
                "median_nodes":statistics.median(z["nodes"] for z in subset),
                "depth_histogram":dict(Counter(z["depth"] for z in subset)),
                "depth_three_completed":sum(z["depth"]>=3 for z in subset),
                "over_200_ms":sum(z["elapsed_ms"]>200 for z in subset)}
    out={"status":"past_KUMOJI_frozen_G8_legal_generation_throughput_only",
         "source_sha256":GAMES_SHA,"g8_model_sha256":G8_SHA,
         "all_100_game_legal_replay":True,
         "board_move_sets_independently_compared":validated_positions,
         "roots":40,"search_calls":160,"order":"A-B-B-A",
         "single_factor":"fast_legal_guard_false_true",
         "groups":grouped,"calls":records,
         "limitations":["This is retrospective development and not new match strength.",
            "All 100 game histories were legally replayed and all sampled legal moves compared.",
            "The legal set equivalence is exhaustive only at sampled historical positions.",
            "Do not promote research-only fast guard before separate regression and independent games.",
            "Both 200ms budgets are requested; actual elapsed times are separately measured."]}
    target=Path(output);target.mkdir(parents=True,exist_ok=True)
    (target/"legal_fast_probe.json").write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k!="calls"},indent=2,ensure_ascii=False))
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--games",required=True)
    p.add_argument("--g8-model",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    compare(args.games,args.g8_model,args.out)


if __name__=="__main__":main()
