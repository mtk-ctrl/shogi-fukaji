"""Research-only phase timing for two UCT positions that exceeded 200ms.

No engine adoption, no training, no selfplay. Trace overhead is nonzero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import shogi

from .g8_paired_arena import G8_SHA
from .mcts_research import select_mcts
from .relative_value import RelativeValue
from .search_method_probe import build_positions, load_source, _board_from_prefix

REQUEST = ".github/fukaji-uct-phase-request.json"
EXPECTED = {
    "enabled": False, "research_only": True,
    "model_adoption_allowed": False, "source_run": 40637486947,
    "g8_model_run": 40636602223, "parent_indices": [6, 11],
    "repeats": 2, "movetime_ms": 200, "runner_jobs": 1,
}
SOURCE_SHA = "60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf"


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_request(path):
    request = json.loads(Path(path).read_text(encoding="utf-8"))
    if request not in (EXPECTED, dict(EXPECTED, enabled=True)):
        raise ValueError("unapproved UCT phase request")
    enabled = request["enabled"]
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            print("enabled=" + str(enabled).lower(), file=output)
    return enabled


def percentile(values, pct):
    if not values or not 0 <= pct <= 100:
        raise ValueError("invalid percentile")
    ordered = sorted(values)
    x = (len(ordered) - 1) * pct / 100
    lo = int(x)
    hi = min(len(ordered) - 1, lo + 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (x - lo)


def summarize_trace(trace):
    if not trace:
        raise ValueError("no iterations observed")
    phases = ("selection_ms", "expansion_ms", "leaf_ms",
              "backup_ms", "undo_ms")
    result = {
        "iterations": len(trace),
        "iteration_p50_ms": percentile([r["total_ms"] for r in trace], 50),
        "iteration_p90_ms": percentile([r["total_ms"] for r in trace], 90),
        "iteration_p99_ms": percentile([r["total_ms"] for r in trace], 99),
        "iteration_max_ms": max(r["total_ms"] for r in trace),
        "first_iteration_ms": trace[0]["total_ms"],
        "phase_sum_ms": {p: sum(r[p] for r in trace) for p in phases},
        "unexpected_long_iterations": sum(
            r["total_ms"] > trace[i - 1]["reserve_ms"]
            for i, r in enumerate(trace) if i > 0
        ),
    }
    return result


def run(games, report, model_path, out):
    if sha256(Path(model_path).read_bytes()) != G8_SHA:
        raise ValueError("G8 model SHA mismatch")
    rows, source_sha = load_source(games, report)
    if source_sha != SOURCE_SHA:
        raise ValueError("240-game JSONL SHA mismatch")
    positions = build_positions(rows)
    if len(positions) != 12:
        raise ValueError("wrong number of frozen positions")
    model = RelativeValue(model_path, mix_percent=25)
    if model.sha256 != G8_SHA:
        raise ValueError("model loader mismatch")
    result = []
    for index in EXPECTED["parent_indices"]:
        position = positions[index - 1]
        board = _board_from_prefix(position["prefix"])
        if board.sfen() != position["sfen"]:
            raise ValueError("position replay mismatch")
        original = (board.sfen(), tuple(board.move_stack))
        for repeat in range(EXPECTED["repeats"]):
            trace = []
            move, iterations, elapsed = select_mcts(
                board, movetime_ms=200, model=model, fast_relative=True,
                max_tree_depth=12, seed=20279028 + index - 1,
                iteration_trace=trace)
            if (board.sfen(), tuple(board.move_stack)) != original:
                raise ValueError("UCT changed board or history")
            if move == "resign" or shogi.Move.from_usi(move) not in board.legal_moves:
                raise ValueError("UCT selected illegal move")
            if iterations != len(trace) or iterations < 1:
                raise ValueError("iteration trace mismatch")
            result.append({
                "parent_index": index, "repeat": repeat + 1,
                "pair": position["pair"], "target_ply": position["target_ply"],
                "sfen": position["sfen"], "in_check": position["in_check"],
                "hand_pieces": position["hand_pieces"], "move": move,
                "elapsed_ms": elapsed, "trace": trace,
                "trace_summary": summarize_trace(trace),
            })
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in result).encode("utf-8")
    (root / "phase_iterations.jsonl").write_bytes(raw)
    summary = {
        "status": "research_only_no_adoption",
        "source_run": 40637486947, "source_games_sha256": source_sha,
        "g8_model_run": 40636602223, "g8_model_sha256": G8_SHA,
        "parents": [6, 11], "repeats_per_parent": 2,
        "rows": len(result), "movetime_ms": 200,
        "trace_jsonl_sha256": sha256(raw),
        "over_200ms_traced": sum(r["elapsed_ms"] > 200 for r in result),
        "unresolved_source_games_used_as_teacher": False,
        "model_promoted": False, "official_kumoji_changed": False,
        "limitations": [
            "Instrumentation changes execution cost; traced elapsed cannot be compared directly with untraced ABBA timings.",
            "Two parent positions and two repetitions do not establish strength or real-time guarantees.",
            "Phase durations are observational, not a proof of causal bottlenecks.",
        ],
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check-request")
    p.add_argument("--games")
    p.add_argument("--report")
    p.add_argument("--model")
    p.add_argument("--out")
    args = p.parse_args()
    if args.check_request:
        print(json.dumps({"enabled": validate_request(args.check_request)}))
        return
    if not all((args.games, args.report, args.model, args.out)):
        p.error("missing input")
    if not validate_request(REQUEST):
        raise ValueError("research request disabled")
    print(json.dumps(run(args.games, args.report, args.model, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
