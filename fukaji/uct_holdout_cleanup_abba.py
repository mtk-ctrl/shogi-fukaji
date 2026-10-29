"""Frozen holdout-only UCT cleanup-reserve ABBA; no training or adoption.

This reuses the existing ABBA order/summary and the full 240-game legal-replay
audit, but selects only the disjoint, frozen pair12..23 holdout positions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import shogi

from .uct_cleanup_reserve_abba import ORDER, summarize, MODEL_SHA, SOURCE_SHA
from .uct_generalization_audit import validate_holdout_abba
from .uct_holdout_freeze import verify as verify_holdout, MANIFEST
from .search_method_probe import load_source, _board_from_prefix
from .relative_value import RelativeValue
from .mcts_research import select_mcts

REQUEST_PATH = ".github/fukaji-uct-holdout-reserve-abba-request.json"
REQUEST = {
    "enabled": False, "research_only": True, "model_adoption_allowed": False,
    "source_run": 40637486947, "g8_model_run": 40636602223,
    "holdout_manifest": str(MANIFEST), "split": "holdout_only",
    "positions": 12, "movetime_ms": 200, "repeats_per_variant": 2,
    "runner_jobs": 1, "single_pass_legal_moves": True,
    "total_budget_from_call": True, "linear_root_choice": True,
    "variable": "cleanup_aware_reserve",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_request(path):
    request = json.loads(Path(path).read_text(encoding="utf-8"))
    if request not in (REQUEST, dict(REQUEST, enabled=True)):
        raise ValueError("unapproved holdout-only UCT request")
    enabled = request["enabled"]
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
            print("enabled=" + str(enabled).lower(), file=handle)
    return enabled


def prepare_holdout_positions(rows, manifest):
    """Require exact pair/row/ply/seed and reconstruct legal history."""
    parents = manifest["holdout"]
    if len(parents) != 12 or [p["pair"] for p in parents] != list(range(12, 24)):
        raise ValueError("frozen holdout identities changed")
    positions = []
    for i, p in enumerate(parents):
        pair, ply = 12 + i, (12, 50, 110)[i // 4]
        if (p["pair"] != pair or p["game_row_index"] != 2 * pair
                or p["ply"] != ply or p["seed"] != 20279128 + i
                or p["g8_black"] is not True or
                p["source_sha256"] != SOURCE_SHA):
            raise ValueError("holdout identity/seed/source mismatch")
        row = rows[p["game_row_index"]]
        if row["pair"] != pair or row["g8_black"] is not True:
            raise ValueError("wrong source game for holdout")
        sequence = row["opening"] + row["moves"]
        if len(sequence) <= ply:
            raise ValueError("holdout history too short")
        board = _board_from_prefix(sequence[:ply])
        if (board.sfen() != p["sfen"] or board.is_checkmate()
                or board.is_fourfold_repetition() or not list(board.legal_moves)):
            raise ValueError("holdout SFEN, legality or terminal mismatch")
        positions.append((p, sequence[:ply]))
    return positions


def attest_replayed_legality(metadata_validation, replay_receipt):
    """Promote a metadata-only audit flag only after actual legal replay."""
    if (replay_receipt.get("legality_replayed") is not True
            or replay_receipt.get("source_games_legally_replayed") != 240
            or replay_receipt.get("holdout_prefixes_legally_replayed") != 12):
        raise ValueError("cannot attest legality without full replay")
    return {**metadata_validation, "legality_replayed": True,
            "source_games_legally_replayed": 240,
            "holdout_prefixes_legally_replayed": 12}


def probe(games_path, report_path, model_path, out_path):
    if digest(Path(model_path).read_bytes()) != MODEL_SHA:
        raise ValueError("G8 model bytes mismatch")
    receipt = verify_holdout(games_path, report_path)
    if (receipt["legality_replayed"] is not True
            or receipt["source_games_legally_replayed"] != 240
            or receipt["holdout_prefixes_legally_replayed"] != 12):
        raise ValueError("source/holdout legal replay missing")
    rows, source_sha = load_source(games_path, report_path)
    if source_sha != SOURCE_SHA:
        raise ValueError("frozen source bytes mismatch")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    positions = prepare_holdout_positions(rows, manifest)
    model = RelativeValue(model_path, mix_percent=25)
    if model.sha256 != MODEL_SHA:
        raise ValueError("loaded G8 model SHA mismatch")
    observations = []
    for name, enabled in ORDER:
        for parent, prefix in positions:
            board = _board_from_prefix(prefix)
            before = (board.sfen(), tuple(board.move_stack))
            started = time.perf_counter()
            move, iterations, internal_ms = select_mcts(
                board, movetime_ms=200, model=model, fast_relative=True,
                max_tree_depth=12, seed=parent["seed"],
                single_pass_legal_moves=True, total_budget_from_call=True,
                linear_root_choice=True, cleanup_aware_reserve=enabled)
            external_ms = round((time.perf_counter() - started) * 1000, 3)
            restored = (board.sfen(), tuple(board.move_stack)) == before
            legal = move != "resign" and shogi.Move.from_usi(move) in board.legal_moves
            if not restored or not legal:
                raise ValueError("illegal move or board/history mutation")
            observations.append({
                "pass": name, "cleanup_reserve": enabled,
                "single_pass": True, "total_budget": True,
                "linear_root_choice": True, "pair": parent["pair"],
                "target_ply": parent["ply"], "parent_id": parent["parent_id"],
                "game_id": parent["game_id"], "seed": parent["seed"],
                "sfen": parent["sfen"], "source_sha256": SOURCE_SHA,
                "model_sha256": MODEL_SHA, "movetime_ms": 200,
                "move": move, "legal": legal, "board_restored": restored,
                "iterations": iterations, "external_ms": external_ms,
                "elapsed_ms": external_ms,
                "internal_elapsed_ms": round(internal_ms, 3),
            })
    validation = attest_replayed_legality(
        validate_holdout_abba(
            observations, manifest["holdout"],
            model_sha256=MODEL_SHA, source_sha256=SOURCE_SHA),
        receipt)
    summary = summarize(observations)
    summary.update({
        "split": "frozen_holdout_only", "holdout_manifest": str(MANIFEST),
        "development_parents_used": 0, "holdout_parents_used": 12,
        "holdout_validated": True, "source_legally_replayed": 240,
        "holdout_prefixes_legally_replayed": 12,
        "holdout_validation": validation,
        "holdout_not_used_for_coefficient_tuning": True,
        "measurement_is_not_strength_or_hard_deadline_proof": True,
    })
    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                  for r in observations).encode("utf-8")
    (out / "holdout_observations.jsonl").write_bytes(raw)
    summary["observations_sha256"] = digest(raw)
    (out / "holdout_summary.json").write_text(
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
        parser.error("missing source/model/output")
    if not validate_request(REQUEST_PATH):
        raise ValueError("holdout research request disabled")
    print(json.dumps(probe(args.games, args.report, args.model, args.out),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
