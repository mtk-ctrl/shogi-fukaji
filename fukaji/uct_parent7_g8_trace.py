"""G8 parent-7 traced timing probe; research only, no games or adoption."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import shogi

from .g8_paired_arena import G8_SHA
from .mcts_research import select_mcts
from .relative_value import RelativeValue
from .search_method_probe import _board_from_prefix, build_positions, load_source
from .uct_cleanup_reserve_abba import MODEL_SHA, SOURCE_SHA

REQUEST_PATH = ".github/fukaji-uct-parent7-g8-trace-request.json"
REQUEST = {
    "enabled": False, "research_only": True, "model_adoption_allowed": False,
    "source_run": 40637486947, "g8_model_run": 40636602223,
    "parent_index": 7, "target_ply": 50, "movetime_ms": 200,
    "runner_jobs": 1, "calls": 8, "cleanup_aware_reserve": True,
    "single_pass_legal_moves": True, "total_budget_from_call": True,
    "linear_root_choice": True,
}
PARENT7_SFEN = (
    "3g5/1rskg3l/2p2s1pn/4p1p2/ln+B3n1p/+b1P1PP1P1/"
    "3PS1P1P/3SGK1R1/5G1NL b 5Plp 51"
)
ORDER = ("traced_a", "plain_a", "plain_b", "traced_b",
         "plain_c", "traced_c", "traced_d", "plain_d")
SEED = 20279028 + 7


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_request(path):
    request = json.loads(Path(path).read_text(encoding="utf-8"))
    if request not in (REQUEST, dict(REQUEST, enabled=True)):
        raise ValueError("unapproved parent-7 G8 diagnostic request")
    enabled = request["enabled"]
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
            print("enabled=" + str(enabled).lower(), file=handle)
    return enabled


def summarize(records):
    if len(records) != len(ORDER):
        raise ValueError("expected eight fixed parent-7 observations")
    for expected, row in zip(ORDER, records):
        traced = expected.startswith("traced")
        if (row.get("pass") != expected or row.get("traced") is not traced
                or row.get("sfen") != PARENT7_SFEN
                or row.get("seed") != SEED or row.get("target_ply") != 50
                or row.get("iterations", -1) < 0
                or row.get("move") in (None, "", "resign")
                or row.get("external_ms", 0) <= 0
                or row.get("internal_ms", -1) < 0
                or row.get("cpu_ms", -1) < 0
                or (traced and not isinstance(row.get("phase"), dict))
                or (not traced and row.get("phase") is not None)):
            raise ValueError("G8 trace row or frozen identity mismatch")
    out = {
        "status": "research_only_parent7_g8_instrumentation_not_strength",
        "source_games_sha256": SOURCE_SHA, "g8_model_sha256": MODEL_SHA,
        "parent_index": 7, "sfen": PARENT7_SFEN, "movetime_ms": 200,
        "trace_calls": 4, "untraced_calls": 4,
        "same_parent_and_seed": True, "cleanup_aware_reserve_fixed": True,
        "unresolved_games_as_teacher": False, "model_promoted": False,
        "official_kumoji_changed": False,
    }
    for name, group in (("traced", [r for r in records if r["traced"]]),
                        ("plain", [r for r in records if not r["traced"]])):
        out[name + "_external_median_ms"] = round(
            statistics.median(r["external_ms"] for r in group), 3)
        out[name + "_over_200ms"] = sum(r["external_ms"] > 200 for r in group)
        out[name + "_iterations"] = sum(r["iterations"] for r in group)
    out["limitations"] = [
        "Instrumented and uninstrumented observations are not interchangeable.",
        "Eight calls from one selected parent do not prove a hard 200ms bound.",
        "Process CPU vs wall time is suggestive, not proof of OS interruption.",
        "G8 model is frozen; no games, training, teacher labels or adoption.",
        "A change in chosen move or iteration count does not prove strength.",
    ]
    return out


def probe(games_path, report_path, model_path, output):
    if sha256(Path(model_path).read_bytes()) != MODEL_SHA or G8_SHA != MODEL_SHA:
        raise ValueError("frozen G8 model SHA mismatch")
    rows, source_sha = load_source(games_path, report_path)
    if source_sha != SOURCE_SHA:
        raise ValueError("frozen 240-game source SHA mismatch")
    positions = build_positions(rows)
    if len(positions) != 12:
        raise ValueError("expected twelve independent parent positions")
    pos = positions[7]
    if (pos["target_ply"] != 50 or pos["pair"] != 7
            or pos["sfen"] != PARENT7_SFEN):
        raise ValueError("parent-7 source reconstruction mismatch")
    model = RelativeValue(model_path, mix_percent=25)
    if model.sha256 != MODEL_SHA:
        raise ValueError("G8 loader SHA mismatch")
    records = []
    for pass_name in ORDER:
        traced = pass_name.startswith("traced")
        board = _board_from_prefix(pos["prefix"])
        before = (board.sfen(), tuple(board.move_stack))
        phase = [] if traced else None
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        move, iterations, internal = select_mcts(
            board, movetime_ms=200, model=model, fast_relative=True,
            max_tree_depth=12, seed=SEED,
            single_pass_legal_moves=True, total_budget_from_call=True,
            linear_root_choice=True, cleanup_aware_reserve=True,
            clock_phase_trace=phase)
        external = (time.perf_counter() - wall_start) * 1000.0
        cpu_ms = (time.process_time() - cpu_start) * 1000.0
        if (board.sfen(), tuple(board.move_stack)) != before:
            raise AssertionError("parent-7 board or history was changed")
        if move == "resign" or shogi.Move.from_usi(move) not in board.legal_moves:
            raise ValueError("illegal G8 UCT move")
        if traced and (len(phase) != 1 or phase[0]["terminal_no_legal_moves"]
                       or phase[0]["iterations"] != iterations):
            raise ValueError("invalid phase trace")
        records.append({
            "pass": pass_name, "traced": traced, "sfen": PARENT7_SFEN,
            "seed": SEED, "target_ply": 50, "move": move,
            "iterations": iterations, "external_ms": round(external, 3),
            "internal_ms": round(internal, 3), "cpu_ms": round(cpu_ms, 3),
            "wall_minus_cpu_ms": round(external - cpu_ms, 3),
            "phase": phase[0] if traced else None,
        })
    summary = summarize(records)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                  for r in records).encode("utf-8")
    (root / "parent7_g8_observations.jsonl").write_bytes(raw)
    summary["observations_sha256"] = sha256(raw)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-request")
    parser.add_argument("--games")
    parser.add_argument("--report")
    parser.add_argument("--model")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.check_request:
        print(json.dumps({"enabled": validate_request(args.check_request)}))
        return
    if not all((args.games, args.report, args.model, args.out)):
        parser.error("missing source, model or output")
    if not validate_request(REQUEST_PATH):
        raise ValueError("research request is disabled")
    print(json.dumps(probe(args.games, args.report, args.model, args.out),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
