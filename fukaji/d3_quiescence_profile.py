"""Research-only cProfile diagnosis for D3 quiescence; no G8, no 200ms claim.

Exclusive CPU time is grouped by function name for hypothesis generation.
It is *not* a wall-clock speedup estimate and categories are approximate:
a board accessor can be used by both ordering and evaluation.
"""
from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time

import shogi

from .d3_tactical_probe import CASES, RECAPTURE_VARIANTS, snapshot
from .engine import select_move


def category(filename, function):
    """Disjoint buckets of self CPU time, not inclusive phase timing."""
    if function in ("ordered_moves", "order", "sort", "tactical_moves"):
        return "ordering_and_tactical_filter"
    if function in ("generate_pseudo_legal_moves", "generate_legal_moves",
                    "is_suicide_or_check_by_dropping_pawn", "is_legal"):
        return "legal_move_generation"
    if function in ("evaluate", "evaluate_relative_fused", "combine"):
        return "evaluation"
    if function in ("is_check", "is_checkmate", "is_attacked_by",
                    "is_fourfold_repetition"):
        return "check_and_repetition"
    if function in ("push", "pop", "zobrist_hash"):
        return "board_state_and_history"
    if function in ("piece_at", "pieces", "piece_map"):
        return "board_access_shared"
    return "other"


def profile_one(name, sfen, enabled, fast_tactical=False):
    board = shogi.Board(sfen)
    before = snapshot(board)
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    try:
        move, depth, nodes = select_move(
            board, movetime_ms=200, max_depth=1, model=None,
            complete_depth=True, avoid_repetition=False,
            fast_relative=False, root_alpha_beta=True,
            fast_move_list=False, deadline_from_call=True,
            quiescence=enabled, quiescence_fast_tactical=fast_tactical,
        )
    finally:
        profiler.disable()
    wall_ms = (time.perf_counter() - started) * 1000
    if snapshot(board) != before:
        raise AssertionError(name + ": board/history changed")
    if move == "resign" or shogi.Move.from_usi(move) not in board.legal_moves:
        raise AssertionError(name + ": illegal search move")
    stats = pstats.Stats(profiler).stats
    grouped = {}
    functions = []
    for (filename, line, function), (primitive, total_calls, self_s, cumulative_s, _) in stats.items():
        group = category(filename, function)
        grouped[group] = grouped.get(group, 0.0) + self_s * 1000
        functions.append({
            "function": function, "module": filename.rsplit("/", 1)[-1],
            "calls": total_calls, "self_ms": round(self_s * 1000, 4),
            "cumulative_ms": round(cumulative_s * 1000, 4),
        })
    return {
        "case": name, "quiescence": enabled,
        "fast_tactical": fast_tactical, "move": move,
        "depth": depth, "nodes": nodes,
        "wall_ms_instrumented": round(wall_ms, 4),
        "self_cpu_ms_by_category": {k: round(v, 4) for k, v in sorted(grouped.items())},
        "top_self_cpu_functions": sorted(functions, key=lambda x: -x["self_ms"])[:12],
    }


def run_profile(compare_fast=False):
    scenarios = [(name, sfen) for name, sfen, *_ in RECAPTURE_VARIANTS]
    for name in ("check_evasion", "quiet_promotion"):
        scenarios.append((name, dict(CASES)[name]))
    rows = []
    for name, sfen in scenarios:
        if compare_fast:
            rows.append(profile_one(name, sfen, True, False))
            rows.append(profile_one(name, sfen, True, True))
        else:
            for enabled in (False, True):
                rows.append(profile_one(name, sfen, enabled))
    aggregate = {}
    for row in rows:
        side = (("Q_ON_FAST" if row["fast_tactical"] else "Q_ON_BASE")
                if compare_fast else ("Q_ON" if row["quiescence"] else "Q_OFF"))
        target = aggregate.setdefault(side, {})
        for group, ms in row["self_cpu_ms_by_category"].items():
            target[group] = round(target.get(group, 0.0) + ms, 4)
    return {
        "experiment": ("D3_Q_early_filter_cProfile" if compare_fast
                       else "D3_Q_cost_cProfile_development_only"),
        "positions": len(scenarios), "searches": len(rows),
        "clock": "fixed_depth_1_unbounded_classical_no_G8",
        "interpretation": "instrumented exclusive CPU time; not wall-time speedup or playing strength",
        "aggregate_self_cpu_ms": aggregate, "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--compare-fast", action="store_true")
    args = parser.parse_args()
    result = run_profile(compare_fast=args.compare_fast)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps({
            k: v for k, v in result.items() if k != "rows"
        }, indent=2, sort_keys=True))
        for row in result["rows"]:
            print(json.dumps({
                "case": row["case"], "Q": row["quiescence"],
                "fast_tactical": row["fast_tactical"],
                "move": row["move"], "nodes": row["nodes"],
                "top_self_cpu_functions": row["top_self_cpu_functions"][:6],
            }, sort_keys=True))


if __name__ == "__main__":
    main()
