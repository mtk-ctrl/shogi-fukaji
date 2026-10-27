"""Frozen G7 move-list A/B: exact search semantics and equal 200ms throughput."""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
import shogi
from .engine import select_move
from .g7_depth_diagnosis import (G7_SHA, GAMES_SHA, MODEL_RUN, SOURCE_RUN,
                                 sample_positions, digest)
from .relative_value import RelativeValue

def expected_request():
    return dict(research_only=True, training_allowed=False, adoption_allowed=False,
                source_run=SOURCE_RUN, model_run=MODEL_RUN,
                games_sha256=GAMES_SHA, model_sha256=G7_SHA,
                samples=32, movetime_ms=200, max_depth=4,
                fixed_equivalence_depth=2, mix_percent=25,
                fast_relative=True, root_alpha_beta=True,
                variants=["legacy", "fast_list"])

def validate_request(request):
    if request != expected_request():
        raise ValueError("unapproved frozen move-list research request")
    return request

def execute(request, games, model, out):
    validate_request(json.loads(Path(request).read_text(encoding="utf-8")))
    if digest(games) != GAMES_SHA or digest(model) != G7_SHA:
        raise ValueError("source games or frozen G7 model SHA mismatch")
    rows = [json.loads(s) for s in Path(games).read_text(encoding="utf-8").splitlines() if s.strip()]
    positions = sample_positions(rows)
    frozen = RelativeValue(model, mix_percent=25)
    records = []
    for i, pos in enumerate(positions):
        baseline = shogi.Board(pos["sfen"])
        before = baseline.sfen()
        fixed = {}
        for fast_list in (False, True):
            move, depth, nodes = select_move(
                baseline, movetime_ms=200, max_depth=2, complete_depth=True,
                model=frozen, fast_relative=True, root_alpha_beta=True,
                fast_move_list=fast_list)
            if baseline.sfen() != before or shogi.Move.from_usi(move) not in baseline.legal_moves:
                raise ValueError("fixed search modified position or returned illegal move")
            fixed[fast_list] = (move, depth, nodes)
        if fixed[False] != fixed[True]:
            raise ValueError("fast_list changed fixed-depth move, completed depth or node count")
        for fast_list in ((False, True) if i % 2 == 0 else (True, False)):
            board = shogi.Board(pos["sfen"])
            t0 = time.perf_counter()
            move, depth, nodes = select_move(
                board, movetime_ms=200, max_depth=4, model=frozen,
                fast_relative=True, root_alpha_beta=True,
                fast_move_list=fast_list)
            elapsed = round((time.perf_counter() - t0) * 1000, 3)
            if board.sfen() != before or move == "resign" or shogi.Move.from_usi(move) not in board.legal_moves:
                raise ValueError("timed search modified position or returned illegal move")
            records.append(dict(sample=i, pair=pos["pair"], g7_black=pos["g7_black"],
                                variant="fast_list" if fast_list else "legacy",
                                move=move, complete_depth=depth, nodes=nodes,
                                elapsed_ms=elapsed, fixed_depth2_nodes=fixed[False][2]))
    summary = {}
    for variant in ("legacy", "fast_list"):
        subset = [r for r in records if r["variant"] == variant]
        summary[variant] = dict(median_elapsed_ms=statistics.median(r["elapsed_ms"] for r in subset),
                                median_nodes=statistics.median(r["nodes"] for r in subset),
                                total_nodes=sum(r["nodes"] for r in subset),
                                depth_distribution=dict(Counter(r["complete_depth"] for r in subset)),
                                median_nodes_per_second=statistics.median(
                                    r["nodes"] * 1000 / max(0.001, r["elapsed_ms"]) for r in subset))
    changes = sum(next(r for r in records if r["sample"] == i and r["variant"] == "legacy")["move"]
                  != next(r for r in records if r["sample"] == i and r["variant"] == "fast_list")["move"]
                  for i in range(32))
    result = dict(status="research_diagnosis_only", source_run=SOURCE_RUN,
                  model_run=MODEL_RUN, games_sha256=GAMES_SHA,
                  model_sha256=G7_SHA, samples=32, movetime_ms=200,
                  fixed_depth2_equivalent=32, timed_move_changes=changes,
                  summary=summary, records=records,
                  caution="Equal 200ms throughput diagnostic; not playing strength or formal adoption.")
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (output / "positions.json").write_text(json.dumps(positions, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "records"}, indent=2))
    return result

def main():
    p = argparse.ArgumentParser()
    for name in ("request", "games", "model", "out"):
        p.add_argument("--"+name, required=True)
    args = p.parse_args()
    execute(args.request, args.games, args.model, args.out)

if __name__ == "__main__":
    main()
