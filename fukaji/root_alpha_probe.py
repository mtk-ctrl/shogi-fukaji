"""Read-only, frozen-position test of opt-in root alpha-beta reuse.

The original full-window root search remains the default. The candidate reuses
the best root score as the alpha bound when examining later legal root moves.
Timed move differences are NOT correctness labels or evidence of playing strength.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
import shogi
from .depth_probe import sample_positions
from .direct_arena import replay_check
from .engine import select_move
from .relative_value import RelativeValue

SOURCE_RUN = 40378086136
MODEL_RUN = 40359123574
MODEL_SHA = "881af10b5bc0cde591e37a9aee883213fa7dd602345b786b9b93d69c7e6ed239"
GAMES_SHA = "0af0c939049bca6998ae56f0ba623768845014c98af3bdfe05750b7cf52601fb"
REPORT_SHA = "b23e2ec9f1c6b33d465c00f63733b1b0ecb80a00ddec762baf6b3b45016fcd8a"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_request(c):
    required = {
        "research_only": True, "source_run_id": SOURCE_RUN,
        "model_run_id": MODEL_RUN, "model_sha256": MODEL_SHA,
        "games_sha256": GAMES_SHA, "report_sha256": REPORT_SHA,
        "source_seed": 20271227, "source_pairs": 16,
        "max_samples": 12, "time_ms": 200, "max_depth": 2,
        "mix_percent": 25, "fast_relative": True,
        "root_pruning_default": False, "training_allowed": False,
    }
    if c != required:
        raise ValueError("root-alpha research request must exactly match frozen protocol")
    return c


def evaluate_one(board, model, enabled, time_ms, depth):
    before = board.sfen()
    t0 = time.perf_counter()
    move, completed, nodes = select_move(
        board, movetime_ms=time_ms, max_depth=depth,
        model=model, fast_relative=True, root_alpha_beta=enabled)
    elapsed = round((time.perf_counter() - t0) * 1000, 3)
    if board.sfen() != before or move == "resign" or shogi.Move.from_usi(move) not in board.legal_moves:
        raise ValueError("root search mutated board or returned illegal move")
    return {"root_alpha_beta": enabled, "move": move, "complete_depth": completed,
            "nodes": nodes, "elapsed_ms": elapsed}


def execute(request_path, games_path, report_path, model_path, out_dir):
    c = validate_request(json.loads(Path(request_path).read_text()))
    if (sha256(games_path) != GAMES_SHA or sha256(report_path) != REPORT_SHA
            or sha256(model_path) != MODEL_SHA):
        raise ValueError("frozen source or model SHA mismatch")
    report = json.loads(Path(report_path).read_text())
    if (report.get("arena_seed") != c["source_seed"]
            or report.get("games_sha256") != GAMES_SHA
            or report.get("model_sha256") != MODEL_SHA
            or report.get("games") != 32 or report.get("pairs") != 16):
        raise ValueError("source provenance mismatch")
    rows = [json.loads(x) for x in Path(games_path).read_text().splitlines() if x.strip()]
    if len(rows) != 32:
        raise ValueError("frozen source row count mismatch")
    for row in rows:
        replay_check(row["opening"], row["moves"], row)
    samples = sample_positions(rows, c["max_samples"])
    if len(samples) != 12 or len({p["pair"] for p in samples}) != 12:
        raise ValueError("independent frozen pairs missing")
    model = RelativeValue(model_path, mix_percent=25)
    records = []
    for i, sample in enumerate(samples):
        board = shogi.Board(sample["sfen"])
        order = (False, True) if i % 2 == 0 else (True, False)
        by_mode = {}
        for enabled in order:
            by_mode[enabled] = evaluate_one(board, model, enabled, 200, 2)
        records.append({"sample": i, "pair": sample["pair"],
                        "phase_ply": sample["phase_ply"],
                        "baseline": by_mode[False], "candidate": by_mode[True]})
    # Full-depth, no-clock checks independently verify that the pruning bound
    # preserves the chosen move and never visits more nodes on frozen endgames.
    fixed_checks = []
    for i, sample in enumerate(samples[:2]):
        board = shogi.Board(sample["sfen"])
        before = board.sfen()
        a = select_move(board, max_depth=2, model=model, fast_relative=True,
                        complete_depth=True)
        b = select_move(board, max_depth=2, model=model, fast_relative=True,
                        complete_depth=True, root_alpha_beta=True)
        if board.sfen() != before or a[:2] != b[:2] or a[1] != 2 or b[2] > a[2]:
            raise ValueError("fixed-depth root pruning failed equivalence")
        fixed_checks.append({"pair": sample["pair"], "baseline_nodes": a[2],
                             "candidate_nodes": b[2], "same_move": True})
    a = [r["baseline"] for r in records]
    b = [r["candidate"] for r in records]
    result = {
        "status": "read_only_root_alpha_probe_no_promotion",
        "source_run_id": SOURCE_RUN, "model_sha256": MODEL_SHA,
        "games_sha256": GAMES_SHA, "report_sha256": REPORT_SHA,
        "samples": len(samples),
        "move_changed": sum(x["move"] != y["move"] for x,y in zip(a,b)),
        "baseline_depth2_complete": sum(x["complete_depth"] == 2 for x in a),
        "candidate_depth2_complete": sum(x["complete_depth"] == 2 for x in b),
        "baseline_nodes_total": sum(x["nodes"] for x in a),
        "candidate_nodes_total": sum(x["nodes"] for x in b),
        "median_baseline_ms": statistics.median(x["elapsed_ms"] for x in a),
        "median_candidate_ms": statistics.median(x["elapsed_ms"] for x in b),
        "fixed_depth_checks": fixed_checks,
        "records": records,
        "note": "Timed move changes are not labels. No learning, adoption, or KUMOJI modification."
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "root_alpha_probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "records"}, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    for key in ("request","games","report","model","out"):
        p.add_argument("--"+key, required=True)
    a = p.parse_args()
    execute(a.request, a.games, a.report, a.model, a.out)


if __name__ == "__main__":
    main()
