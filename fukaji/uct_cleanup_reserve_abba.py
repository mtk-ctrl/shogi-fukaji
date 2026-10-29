"""Research-only ABBA timing of cleanup-aware reserve (single-pass, total clock and linear root fixed ON).

No games, labels, model updates or formal adoption. All 240 source games are
legally replayed before the 12 fixed, independent parent positions are probed.
"""
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

SOURCE_SHA = "60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf"
MODEL_SHA = "18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492"
REQUEST_PATH = ".github/fukaji-uct-cleanup-reserve-abba-request.json"
REQUEST = {
    "enabled": False, "research_only": True, "model_adoption_allowed": False,
    "source_run": 40637486947, "g8_model_run": 40636602223,
    "positions": 12, "movetime_ms": 200, "repeats_per_variant": 2,
    "runner_jobs": 1, "single_pass_legal_moves": True,
    "total_budget_from_call": True, "linear_root_choice": True,
    "variable": "cleanup_aware_reserve",
}
ORDER = (("off_a", False), ("on_a", True), ("on_b", True), ("off_b", False))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_request(path):
    request = json.loads(Path(path).read_text(encoding="utf-8"))
    if request not in (REQUEST, dict(REQUEST, enabled=True)):
        raise ValueError("unapproved cleanup-aware UCT ABBA request")
    enabled = request["enabled"]
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
            print("enabled=" + str(enabled).lower(), file=handle)
    return enabled


def summarize(records):
    if len(records) != 48:
        raise ValueError("expected 48 UCT observations")
    keys = [(r["pair"], r["target_ply"], r["sfen"]) for r in records[:12]]
    if (len(set(keys)) != 12 or len({r["pair"] for r in records[:12]}) != 12
            or sorted(k[1] for k in keys) != [12] * 4 + [50] * 4 + [110] * 4):
        raise ValueError("not twelve independent source positions")
    by_variant = {"off": [], "on": []}
    for pass_index, (name, flag) in enumerate(ORDER):
        section = records[pass_index * 12:(pass_index + 1) * 12]
        for i, r in enumerate(section):
            if (r["pass"] != name or r["cleanup_reserve"] is not flag or r["single_pass"] is not True
                    or r.get("total_budget") is not True or r.get("linear_root_choice") is not True
                    or (r["pair"], r["target_ply"], r["sfen"]) != keys[i]
                    or r["iterations"] < 0 or r["elapsed_ms"] <= 0
                    or r.get("internal_elapsed_ms", -1) < 0
                    or not r["move"] or r["move"] == "resign"):
                raise ValueError("ABBA row, identity or legal move mismatch")
            by_variant["on" if flag else "off"].append(r)
    out = {"status": "research_only_cleanup_reserve_abba_no_strength_claim",
           "run_order": [name for name, _ in ORDER], "positions": 12,
           "single_pass_legal_moves_fixed": True,
           "total_budget_from_call_fixed": True, "linear_root_choice_fixed": True,
           "cleanup_aware_reserve_isolated": True,
           "observations_per_variant": 24, "movetime_ms": 200,
           "source_games_sha256": SOURCE_SHA, "g8_model_sha256": MODEL_SHA,
           "unresolved_source_games_used_as_teacher": False,
           "model_promoted": False, "official_kumoji_changed": False}
    for key, rows in by_variant.items():
        times = [r["elapsed_ms"] for r in rows]
        out[key + "_median_ms"] = round(statistics.median(times), 3)
        out[key + "_mean_ms"] = round(statistics.mean(times), 3)
        out[key + "_max_ms"] = max(times)
        out[key + "_over_200ms"] = sum(t > 200 for t in times)
        out[key + "_iterations"] = sum(r["iterations"] for r in rows)
        out[key + "_zero_iteration_rows"] = sum(r["iterations"] == 0 for r in rows)
        out[key + "_internal_median_ms"] = round(statistics.median(r["internal_elapsed_ms"] for r in rows), 3)
    faster = 0
    for i in range(12):
        off = [records[i]["elapsed_ms"], records[36 + i]["elapsed_ms"]]
        on = [records[12 + i]["elapsed_ms"], records[24 + i]["elapsed_ms"]]
        faster += statistics.median(on) < statistics.median(off)
    out["on_faster_positions"] = faster
    out["limitations"] = [
        "Same-run ABBA reduces but does not eliminate scheduler noise.",
        "Two observations per variant are descriptive, not a significance test.",
        "The 12 frozen positions are selected, not representative random samples.",
        "Cleanup-aware reservation is best-effort, not a 200ms hard limit.",
        "Zero iterations is valid if a legal root move is returned.",
        "Move changes and iteration counts do not prove playing strength.",
        "No new games, teacher labels, training, or formal adoption.",
    ]
    return out


def probe(games_path, report_path, model_path, output):
    if digest(Path(model_path).read_bytes()) != MODEL_SHA or G8_SHA != MODEL_SHA:
        raise ValueError("G8 model SHA mismatch")
    rows, source_sha = load_source(games_path, report_path)
    if source_sha != SOURCE_SHA:
        raise ValueError("240-game source SHA mismatch")
    positions = build_positions(rows)
    model = RelativeValue(model_path, mix_percent=25)
    if model.sha256 != MODEL_SHA:
        raise ValueError("loaded model SHA mismatch")
    records = []
    for pass_name, flag in ORDER:
        for index, pos in enumerate(positions):
            board = _board_from_prefix(pos["prefix"])
            if board.sfen() != pos["sfen"]:
                raise ValueError("position reconstruction mismatch")
            before = (board.sfen(), tuple(board.move_stack))
            start = time.perf_counter()
            move, iterations, internal_ms = select_mcts(
                board, movetime_ms=200, model=model, fast_relative=True,
                max_tree_depth=12, seed=20279028 + index,
                single_pass_legal_moves=True, total_budget_from_call=True,
                linear_root_choice=True, cleanup_aware_reserve=flag)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
            if (board.sfen(), tuple(board.move_stack)) != before:
                raise AssertionError("UCT mutated source board/history")
            if move == "resign" or shogi.Move.from_usi(move) not in board.legal_moves:
                raise ValueError("illegal UCT output")
            records.append({
                "pass": pass_name, "single_pass": True, "total_budget": True,
                "linear_root_choice": True, "cleanup_reserve": flag,
                "pair": pos["pair"], "target_ply": pos["target_ply"],
                "sfen": pos["sfen"], "hand_pieces": pos["hand_pieces"],
                "move": move, "iterations": iterations,
                "elapsed_ms": elapsed_ms, "internal_elapsed_ms": round(internal_ms, 3),
            })
    summary = summarize(records)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records).encode("utf-8")
    (root / "abba_observations.jsonl").write_bytes(raw)
    summary["observations_sha256"] = digest(raw)
    (root / "abba_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
