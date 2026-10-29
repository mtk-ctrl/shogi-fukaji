"""D2: G8 whole-call 200ms FastMoveList OFF/ON ABBA research diagnostic.

The D1 48 parents are development data, not an unseen strength benchmark.
Only FastMoveList changes between timed variants; both use call-start deadlines.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import shogi

from .depth_limit_probe import (
    ORDER, QUOTAS, TARGET_PLIES, repetition_snapshot, select_positions,
)
from .engine import select_move
from .g8_paired_arena import G8_SHA
from .relative_value import RelativeValue
from .search_method_probe import load_source, _board_from_prefix
from .tactical_search_probe import SOURCE_SHA

REQUEST_FILE = ".github/fukaji-d2-fastmove-request.json"
EXPECTED_REQUEST = {
    "enabled": False, "research_only": True, "training_allowed": False,
    "adoption_allowed": False, "source_run": 40637486947,
    "g8_model_run": 40636602223, "max_positions": 48,
    "movetime_ms": 200, "depth_cap": 6,
    "mix_percent": 25, "fast_relative": True, "root_alpha_beta": True,
    "deadline_from_call": True, "runner_jobs": 1,
}
D1_MANIFEST_SHA = "823d97b85df4a8a6c8638e9fb6cdbcafc1a5d08edff1c03f34256df9c1a56d73"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_request(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg not in (EXPECTED_REQUEST, dict(EXPECTED_REQUEST, enabled=True)):
        raise ValueError("D2 request differs from frozen research protocol")
    enabled = cfg["enabled"]
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            print("enabled=" + str(enabled).lower(), file=f)
    return enabled


def snapshot(board):
    return (board.sfen(), tuple(board.move_stack), repetition_snapshot(board))


def reconstruct(item):
    board = _board_from_prefix(item["prefix"])
    if board.sfen() != item["sfen"]:
        raise ValueError("D2 position/history reconstruction mismatch")
    if board.is_checkmate() or board.is_fourfold_repetition():
        raise ValueError("D2 terminal position")
    return board


def fixed_depth_parity(positions, model, per_category=3):
    """Check identical move/score-proxy node count before timed measurement.

    Equal node count and move at fixed depth are necessary, not sufficient,
    to prove all search scores equal. This probe does not expose root score.
    """
    counts = Counter()
    results = []
    for item in positions:
        if counts[item["category"]] >= per_category:
            continue
        board = reconstruct(item)
        before = snapshot(board)
        variants = []
        for fast in (False, True):
            move, depth, nodes = select_move(
                board, movetime_ms=200, max_depth=2, complete_depth=True,
                model=model, fast_relative=True, root_alpha_beta=True,
                fast_move_list=fast, deadline_from_call=True)
            if snapshot(board) != before:
                raise AssertionError("D2 fixed-depth board/history changed")
            if move == "resign" or shogi.Move.from_usi(move) not in board.legal_moves:
                raise AssertionError("D2 fixed-depth illegal move")
            if depth != 2:
                raise AssertionError("D2 fixed-depth incomplete")
            variants.append((move, depth, nodes))
        if variants[0] != variants[1]:
            raise AssertionError("D2 fixed-depth move/depth/nodes mismatch")
        results.append({"pair": item["pair"], "category": item["category"],
                        "sfen": item["sfen"], "move": variants[0][0],
                        "depth": variants[0][1], "nodes": variants[0][2]})
        counts[item["category"]] += 1
    if any(counts[c] < min(per_category, sum(p["category"] == c for p in positions))
           for c in QUOTAS):
        raise AssertionError("D2 parity coverage incomplete")
    return results


def measure_position(item, model):
    board = reconstruct(item)
    before = snapshot(board)
    observations = []
    for variant in ORDER:
        timing = {}
        outer_start = time.perf_counter()
        move, depth, nodes = select_move(
            board, movetime_ms=200, max_depth=6,
            model=model, fast_relative=True, root_alpha_beta=True,
            fast_move_list=(variant == "B"), deadline_from_call=True,
            telemetry=timing)
        outer_ms = (time.perf_counter() - outer_start) * 1000
        if snapshot(board) != before:
            raise AssertionError("D2 timed board/history changed")
        if move == "resign" or shogi.Move.from_usi(move) not in board.legal_moves:
            raise AssertionError("D2 timed illegal move")
        if timing["completed_depth"] != depth or timing["nodes"] != nodes:
            raise AssertionError("D2 telemetry mismatch")
        if not (0 <= timing["root_preparation_ms"] <= timing["total_call_ms"]
                and 0 <= timing["internal_search_ms"] <= timing["total_call_ms"]
                and timing["total_call_ms"] <= outer_ms + 1):
            raise AssertionError("D2 invalid timing")
        observations.append({
            "variant": variant, "fast_move_list": variant == "B",
            "deadline_from_call": True, "depth_cap": 6, "move": move,
            "completed_depth": depth, "nodes": nodes,
            "outer_call_ms": round(outer_ms, 3),
            "root_preparation_ms": round(timing["root_preparation_ms"], 3),
            "internal_search_ms": round(timing["internal_search_ms"], 3),
            "engine_call_ms": round(timing["total_call_ms"], 3),
            "over_200ms": outer_ms > 200,
        })
    return observations


def probe(games_path, report_path, model_path, out_path):
    if sha(Path(model_path).read_bytes()) != G8_SHA:
        raise ValueError("D2 frozen G8 model SHA mismatch")
    rows, source_sha = load_source(games_path, report_path)
    if source_sha != SOURCE_SHA:
        raise ValueError("D2 frozen 240-game source SHA mismatch")
    model = RelativeValue(model_path, mix_percent=25)
    if model.sha256 != G8_SHA:
        raise ValueError("D2 model loader SHA mismatch")
    positions, shortages = select_positions(rows)
    if len(positions) != 48 or any(shortages.values()):
        raise ValueError("D2 expected exactly D1 48 independent parents")
    d1_manifest = {
        "status": "frozen_before_measurement",
        "selection": "first eligible row by category",
        "categories": QUOTAS, "target_plies": TARGET_PLIES,
        "source_sha256": source_sha, "model_sha256": G8_SHA,
        "positions": positions, "shortages": shortages,
        "independent_parent_pairs": len(positions),
        "development_data_not_unseen": True,
    }
    frozen = (json.dumps(d1_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    if sha(frozen) != D1_MANIFEST_SHA:
        raise ValueError("D2 D1 selection manifest SHA mismatch")
    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection_manifest.json").write_bytes(frozen)
    parity = fixed_depth_parity(positions, model)
    parity_raw = (json.dumps(parity, sort_keys=True, indent=2) + "\n").encode()
    (out / "fixed_depth_parity.json").write_bytes(parity_raw)
    records = [{"pair": p["pair"], "category": p["category"], "ply": p["ply"],
                "sfen": p["sfen"], "observations": measure_position(p, model)}
               for p in positions]
    raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records).encode()
    (out / "d2_observations.jsonl").write_bytes(raw)
    a = [v for row in records for v in row["observations"] if v["variant"] == "A"]
    b = [v for row in records for v in row["observations"] if v["variant"] == "B"]
    better = sum(
        sum(v["completed_depth"] for v in row["observations"] if v["variant"] == "B") >
        sum(v["completed_depth"] for v in row["observations"] if v["variant"] == "A")
        for row in records)
    summary = {
        "status": "research_diagnostic_not_strength_or_adoption",
        "experiment": "D2_fast_move_list_whole_call_200ms",
        "model_sha256": G8_SHA, "source_sha256": source_sha,
        "source_games": 240, "positions": 48, "observations": len(a) + len(b),
        "order": list(ORDER), "movetime_ms": 200, "depth_cap": 6,
        "mix_percent": 25, "fast_relative": True, "root_alpha_beta": True,
        "shared_deadline_from_call": True, "fixed_depth_parity_positions": len(parity),
        "a_over_200ms": sum(x["over_200ms"] for x in a),
        "b_over_200ms": sum(x["over_200ms"] for x in b),
        "a_median_outer_ms": round(statistics.median(x["outer_call_ms"] for x in a), 3),
        "b_median_outer_ms": round(statistics.median(x["outer_call_ms"] for x in b), 3),
        "a_median_depth": statistics.median(x["completed_depth"] for x in a),
        "b_median_depth": statistics.median(x["completed_depth"] for x in b),
        "b_deeper_than_a_positions": better,
        "selection_manifest_sha256": sha(frozen),
        "fixed_depth_parity_sha256": sha(parity_raw),
        "observations_sha256": sha(raw),
        "full_240_game_legal_replay": True,
        "new_games": 0, "teacher_labels": 0, "formal_adoption": False,
        "strength_improvement_proven": False,
        "limitations": [
            "D1 development parents are not an unseen test.",
            "D1 used an internal-start deadline; D2 shares a call-start deadline.",
            "Fixed-depth parity checks move/depth/nodes, not root scores.",
            "A/B/B/A timing on one runner is environment dependent.",
            "Search checks the deadline periodically; this is not hard 200ms OS enforcement.",
            "Depth and move changes alone do not prove tactical or playing strength.",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-request")
    parser.add_argument("--games")
    parser.add_argument("--source-report")
    parser.add_argument("--model")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.check_request:
        print(json.dumps({"enabled": validate_request(args.check_request)}))
        return
    if not all((args.games, args.source_report, args.model, args.out)):
        parser.error("D2 requires games, source-report, model, out")
    if not validate_request(REQUEST_FILE):
        raise ValueError("D2 request disabled")
    print(json.dumps(probe(args.games, args.source_report, args.model, args.out),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
