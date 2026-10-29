"""D1: fixed-G8 depth-cap 2 vs 6 diagnostic; research-only, default OFF.

The 240-game source is development data, never an unseen strength test.
Root preparation is outside the current engine's internal deadline; record both
whole-call and internal timings. No new games, labels, or model promotion.
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

from .engine import select_move
from .g8_paired_arena import G8_SHA
from .relative_value import RelativeValue
from .search_method_probe import load_source, _board_from_prefix
from .tactical_search_probe import SOURCE_SHA

REQUEST_FILE = ".github/fukaji-d1-depth-request.json"
EXPECTED_REQUEST = {
    "enabled": False, "research_only": True, "training_allowed": False,
    "adoption_allowed": False, "source_run": 40637486947,
    "g8_model_run": 40636602223, "max_positions": 48,
    "movetime_ms": 200, "depth_a": 2, "depth_b": 6,
    "mix_percent": 25, "fast_relative": True, "root_alpha_beta": True,
    "runner_jobs": 1,
}
QUOTAS = {"opening": 12, "middlegame": 12, "endgame": 12, "tactical": 12}
TARGET_PLIES = {"opening": 12, "middlegame": 50, "endgame": 110}
ORDER = ("A", "B", "B", "A")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_request(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg not in (EXPECTED_REQUEST, dict(EXPECTED_REQUEST, enabled=True)):
        raise ValueError("D1 protocol differs from frozen depth-only comparison")
    enabled = cfg["enabled"]
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            print("enabled=" + str(enabled).lower(), file=f)
    return enabled


def select_positions(rows, quotas=QUOTAS):
    """Deterministic first-eligible order, one independent parent per position.

    Stage targets are fixed before measurement. Tactical means a nonterminal
    position in check or with pieces in hand, after ply 20. No synthetic fills.
    """
    used_pairs, used_sfen = set(), set()
    selected, shortages = [], {}
    for category in ("opening", "middlegame", "endgame", "tactical"):
        needed = quotas[category]
        if type(needed) is not int or needed < 0 or needed > 12:
            raise ValueError("invalid D1 category quota")
        if needed == 0:
            shortages[category] = 0
            continue
        found = 0
        for row in rows:
            if row["pair"] in used_pairs:
                continue
            seq = row["opening"] + row["moves"]
            if category != "tactical":
                ply = TARGET_PLIES[category]
                if len(seq) <= ply:
                    continue
                board = _board_from_prefix(seq[:ply])
            else:
                board = shogi.Board()
                ply = None
                for i, word in enumerate(seq, 1):
                    move = shogi.Move.from_usi(word)
                    if move not in board.legal_moves:
                        raise ValueError("illegal tactical prefix")
                    board.push(move)
                    if i >= 20 and not board.is_checkmate() and not board.is_fourfold_repetition():
                        hands = sum(int(n) for h in board.pieces_in_hand for n in h.values())
                        if board.is_check() or hands:
                            ply = i
                            break
                if ply is None:
                    continue
            if (board.is_checkmate() or board.is_fourfold_repetition()
                    or board.sfen() in used_sfen):
                continue
            prefix = seq[:ply]
            selected.append({
                "category": category, "pair": row["pair"], "g8_black": row["g8_black"],
                "ply": ply, "prefix": prefix, "sfen": board.sfen(),
                "in_check": bool(board.is_check()),
                "hand_pieces": sum(int(n) for h in board.pieces_in_hand for n in h.values()),
                "source_status": row["status"],
            })
            used_pairs.add(row["pair"])
            used_sfen.add(board.sfen())
            found += 1
            if found == needed:
                break
        shortages[category] = needed - found
    if len(selected) > 48 or len({p["pair"] for p in selected}) != len(selected):
        raise AssertionError("D1 independent-parent bound violated")
    return selected, shortages


def repetition_snapshot(board):
    """Ignore zero-count keys left by temporary push/pop during legal enumeration."""
    return {key: count for key, count in board.transpositions.items() if count > 0}


def measure_position(item, model):
    board = _board_from_prefix(item["prefix"])
    if board.sfen() != item["sfen"]:
        raise ValueError("D1 board/history reconstruction mismatch")
    if board.is_checkmate() or board.is_fourfold_repetition():
        raise ValueError("D1 selected terminal position")
    before = (board.sfen(), tuple(board.move_stack), repetition_snapshot(board))
    observations = []
    for variant in ORDER:
        telemetry = {}
        start = time.perf_counter()
        word, depth, nodes = select_move(
            board, movetime_ms=200, max_depth=2 if variant == "A" else 6,
            model=model, fast_relative=True, root_alpha_beta=True,
            telemetry=telemetry)
        outer_ms = (time.perf_counter() - start) * 1000
        if (board.sfen(), tuple(board.move_stack), repetition_snapshot(board)) != before:
            raise AssertionError("D1 board or repetition history mutated")
        if word == "resign" or shogi.Move.from_usi(word) not in board.legal_moves:
            raise ValueError("D1 illegal nonterminal move")
        if telemetry["completed_depth"] != depth or telemetry["nodes"] != nodes:
            raise AssertionError("D1 timing metadata does not match search")
        if not (0 <= telemetry["root_preparation_ms"] <= telemetry["total_call_ms"]
                and 0 <= telemetry["internal_search_ms"] <= telemetry["total_call_ms"]
                and telemetry["total_call_ms"] <= outer_ms + 1):
            raise AssertionError("D1 invalid monotonic timing")
        observations.append({
            "variant": variant, "depth_cap": 2 if variant == "A" else 6,
            "move": word, "completed_depth": depth, "nodes": nodes,
            "outer_call_ms": round(outer_ms, 3),
            "root_preparation_ms": round(telemetry["root_preparation_ms"], 3),
            "internal_search_ms": round(telemetry["internal_search_ms"], 3),
            "engine_call_ms": round(telemetry["total_call_ms"], 3),
            "over_200ms": outer_ms > 200,
        })
    return observations


def probe(games_path, report_path, model_path, out_path):
    raw_model = Path(model_path).read_bytes()
    if digest(raw_model) != G8_SHA:
        raise ValueError("D1 frozen G8 model SHA mismatch")
    rows, source_sha = load_source(games_path, report_path)
    if source_sha != SOURCE_SHA:
        raise ValueError("D1 frozen 240-game source SHA mismatch")
    model = RelativeValue(model_path, mix_percent=25)
    if model.sha256 != G8_SHA:
        raise ValueError("D1 model loader SHA mismatch")
    positions, shortages = select_positions(rows)
    root = Path(out_path)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "frozen_before_measurement", "selection": "first eligible row by category",
        "categories": QUOTAS, "target_plies": TARGET_PLIES,
        "source_sha256": source_sha, "model_sha256": G8_SHA,
        "positions": positions, "shortages": shortages,
        "independent_parent_pairs": len(positions),
        "development_data_not_unseen": True,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    (root / "selection_manifest.json").write_bytes(manifest_bytes)
    observations = []
    for item in positions:
        observations.append({
            "pair": item["pair"], "category": item["category"], "ply": item["ply"],
            "sfen": item["sfen"], "observations": measure_position(item, model),
        })
    raw = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in observations)
    (root / "d1_observations.jsonl").write_text(raw, encoding="utf-8")
    flat = [v for row in observations for v in row["observations"]]
    a = [v for v in flat if v["variant"] == "A"]
    b = [v for v in flat if v["variant"] == "B"]
    depth_advantage = sum(
        sum(v["completed_depth"] for v in row["observations"] if v["variant"] == "B") >
        sum(v["completed_depth"] for v in row["observations"] if v["variant"] == "A")
        for row in observations)
    summary = {
        "status": "research_diagnostic_not_strength_or_adoption",
        "experiment": "D1_depth_cap_2_vs_6", "model_sha256": G8_SHA,
        "source_sha256": source_sha, "source_games": 240,
        "positions": len(positions), "independent_parent_pairs": len(positions),
        "selected_by_category": dict(Counter(p["category"] for p in positions)),
        "shortages": shortages, "order": list(ORDER), "movetime_ms": 200,
        "mix_percent": 25, "fast_relative": True, "root_alpha_beta": True,
        "root_preparation_outside_engine_deadline": True,
        "b_deeper_than_a_positions": depth_advantage,
        "a_over_200ms": sum(v["over_200ms"] for v in a),
        "b_over_200ms": sum(v["over_200ms"] for v in b),
        "a_median_outer_ms": round(statistics.median(v["outer_call_ms"] for v in a), 3) if a else None,
        "b_median_outer_ms": round(statistics.median(v["outer_call_ms"] for v in b), 3) if b else None,
        "a_median_depth": statistics.median(v["completed_depth"] for v in a) if a else None,
        "b_median_depth": statistics.median(v["completed_depth"] for v in b) if b else None,
        "selection_manifest_sha256": digest(manifest_bytes),
        "observations_sha256": digest(raw.encode()),
        "full_240_game_legal_replay": True,
        "new_games": 0, "teacher_labels": 0, "formal_adoption": False,
        "strength_improvement_proven": False,
        "limitations": [
            "Existing 240 games are development data, not an unseen test.",
            "Depth changes and move disagreements are not proven tactical improvements.",
            "Root preparation is excluded from the engine's internal 200ms deadline.",
            "Whole-call overruns invalidate equal-total-time strength claims.",
            "A/B/B/A is one runner sequence; OS timing is environment-dependent.",
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
    p.add_argument("--source-report")
    p.add_argument("--model")
    p.add_argument("--out")
    args = p.parse_args()
    if args.check_request:
        print(json.dumps({"enabled": validate_request(args.check_request)}))
        return
    if not all((args.games, args.source_report, args.model, args.out)):
        p.error("D1 requires games, source-report, model and out")
    if not validate_request(REQUEST_FILE):
        raise ValueError("D1 request disabled")
    print(json.dumps(probe(args.games, args.source_report, args.model, args.out),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
