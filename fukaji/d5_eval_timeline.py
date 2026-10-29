"""D5-3: development-only chronological evaluation and tactical-event audit.

Read-only diagnostic, not a teacher or proof of a best move. Never score
the sealed D5 parent group. Historical game outcome is a weak label.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile

import shogi

from .d4_realgame_probe import (
    SOURCE_ID, SOURCE_NAME, SOURCE_RUN, SOURCE_ZIP_SHA, SOURCE_GAMES_SHA,
    source_rows,
)
from .d5_eval_diagnosis import frozen_split, feature_snapshot, sha
from .engine import evaluate
from .g8_artifact_selection import artifact_bytes
from .g8_paired_arena import G8_ID, G8_NAME, G8_RUN, G8_SHA, G8_ZIP_SHA
from .relative_value import RelativeValue

TARGET_PAIRS = (49, 52, 109)
CHECKPOINTS = (20, 40, 60, 80, 100)


def development_cases(games):
    """Inspect ONLY the three preselected development parents."""
    development, _sealed = frozen_split()
    if not set(TARGET_PAIRS).issubset(development):
        raise ValueError("target pair is outside frozen development set")
    if len(games) != 240:
        raise ValueError("original 240-game manifest incomplete")
    cases = [
        row for row in games
        if row["pair"] in TARGET_PAIRS and row["status"] == "checkmate"
    ]
    if set(row["pair"] for row in cases) != set(TARGET_PAIRS):
        raise ValueError("one of the three development parents lacks a decisive game")
    return cases


def trace_game(row, model, require_mate=True):
    """Replay and score actual moves; preserve one fixed black-side perspective."""
    board = shogi.Board()
    initial = (board.sfen(), board.zobrist_hash(), len(board.move_stack))
    moves = row["opening"] + row["moves"]
    trace = []
    previous = None
    for ply, word in enumerate(moves, 1):
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal source move at ply %d" % ply)
        captured = board.piece_at(move.to_square)
        board.push(move)
        before = (board.sfen(), board.zobrist_hash(), len(board.move_stack))
        classical = evaluate(board)
        g8 = evaluate(board, model, fast_relative=True)
        features = feature_snapshot(board)
        if (board.sfen(), board.zobrist_hash(), len(board.move_stack)) != before:
            raise AssertionError("evaluation changed board or history")
        black_factor = 1 if int(board.turn) == int(shogi.BLACK) else -1
        material = black_factor * (
            features["board_material_cp"] + features["hand_material_cp"]
        )
        black_king_attacks = (features["own_king_zone_attacks"]
                              if board.turn == shogi.BLACK else
                              features["opponent_king_zone_attacks"])
        white_king_attacks = (features["opponent_king_zone_attacks"]
                              if board.turn == shogi.BLACK else
                              features["own_king_zone_attacks"])
        point = {
            "ply": ply, "move": word,
            "g8_black_cp": int(g8 * black_factor),
            "classical_black_cp": int(classical * black_factor),
            "material_black_cp": int(material),
            "black_king_zone_attacks": black_king_attacks,
            "white_king_zone_attacks": white_king_attacks,
            "gave_check": bool(board.is_check()),
            "captured_piece_type": (int(captured.piece_type)
                                    if captured is not None else None),
            "promoted": bool(move.promotion),
            "drop_piece_type": (int(move.drop_piece_type)
                                if move.drop_piece_type else None),
        }
        point["g8_swing_cp"] = (point["g8_black_cp"] - previous["g8_black_cp"]
                                if previous is not None else None)
        point["material_swing_cp"] = (point["material_black_cp"] -
                                      previous["material_black_cp"]
                                      if previous is not None else None)
        trace.append(point)
        previous = point

    if require_mate and (
        not board.is_checkmate() or int(not board.turn) != row["winner"]
    ):
        raise ValueError("decisive source does not end in verified checkmate")

    while board.move_stack:
        board.pop()
    if (board.sfen(), board.zobrist_hash(), len(board.move_stack)) != initial:
        raise AssertionError("source board/history not restored")

    winner_sign = 1 if row["winner"] == int(shogi.BLACK) else -1
    checkpoints = [
        {
            "ply": p["ply"], "g8_black_cp": p["g8_black_cp"],
            "classical_black_cp": p["classical_black_cp"],
            "material_black_cp": p["material_black_cp"],
            "black_king_zone_attacks": p["black_king_zone_attacks"],
            "white_king_zone_attacks": p["white_king_zone_attacks"],
            "gave_check": p["gave_check"],
            "g8_matches_final_winner": p["g8_black_cp"] * winner_sign > 0,
        }
        for p in trace
        if p["ply"] in CHECKPOINTS and p["ply"] < len(moves)
    ]
    top_swings = sorted(
        (p for p in trace if p["g8_swing_cp"] is not None),
        key=lambda p: (-abs(p["g8_swing_cp"]), p["ply"]),
    )[:8]
    return {
        "pair": row["pair"], "g8_black": row["g8_black"],
        "winner": row["winner"], "total_plies": len(moves),
        "checkpoints": checkpoints, "top_evaluation_swings": top_swings,
        "check_moves": sum(p["gave_check"] for p in trace),
        "capture_moves": sum(p["captured_piece_type"] is not None for p in trace),
        "promotion_moves": sum(p["promoted"] for p in trace),
        "full_move_trace": trace,
        "board_history_restored": True,
    }


def diagnose(games, model):
    cases = development_cases(games)
    results = [trace_game(row, model) for row in cases]
    return {
        "scope": "D5_three_development_parents_only_not_unseen_not_strength",
        "source_run": SOURCE_RUN, "source_zip_sha256": SOURCE_ZIP_SHA,
        "source_games_sha256": SOURCE_GAMES_SHA,
        "g8_model_sha256": G8_SHA,
        "target_pairs": list(TARGET_PAIRS),
        "selected_decisive_games": len(cases),
        "sealed_pair_scores_computed": False,
        "unresolved_games_used_as_labels": False,
        "new_games_played": 0, "model_updated": False,
        "causal_attribution_proven": False,
        "interpretation": (
            "Black-perspective evaluations and material/king-zone/check/capture "
            "events are descriptive. Final checkmate winner is a weak outcome "
            "label, not proof of an earlier best move or evaluation error."
        ),
        "games": results,
    }


def run(output):
    if os.environ.get("GITHUB_REPOSITORY") != "mtk-ctrl/shogi-fukaji":
        raise ValueError("wrong research repository")
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise ValueError("read-only artifact token missing")
    source = artifact_bytes(SOURCE_ID, SOURCE_NAME, SOURCE_RUN, SOURCE_ZIP_SHA, token)
    model_zip = artifact_bytes(G8_ID, G8_NAME, G8_RUN, G8_ZIP_SHA, token)
    with ZipFile(io.BytesIO(model_zip)) as z:
        raw_model = z.read("mover-relative-generation8-research.npz")
    if sha(raw_model) != G8_SHA:
        raise ValueError("G8 model SHA mismatch")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "g8-frozen.npz"
    model_path.write_bytes(raw_model)
    model = RelativeValue(model_path, mix_percent=25)
    report = diagnose(source_rows(source), model)
    (root / "d5-three-parent-timeline.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    compact = {k: v for k, v in report.items() if k != "games"}
    compact["games"] = [
        {k: v for k, v in game.items() if k != "full_move_trace"}
        for game in report["games"]
    ]
    (root / "d5-three-parent-summary.json").write_text(
        json.dumps(compact, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True))
    return compact


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    run(parser.parse_args().out)
