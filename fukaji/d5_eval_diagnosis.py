"""D5 read-only G8 evaluation diagnosis on previously unused parent pairs.

No search, no new games, no training, no automatic model adoption.
Only development parents are scored; the sealed parent split is not inspected.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from .d4_realgame_probe import (
    EXCLUDED_DEVELOPMENT_PAIRS, FROZEN_PAIRS, SOURCE_ID, SOURCE_NAME,
    SOURCE_RUN, SOURCE_ZIP_SHA, SOURCE_GAMES_SHA, source_rows,
)
from .direct_arena import replay_check
from .engine import evaluate
from .g8_artifact_selection import artifact_bytes
from .g8_paired_arena import G8_ID, G8_NAME, G8_RUN, G8_SHA, G8_ZIP_SHA
from .relative_value import RelativeValue

SEED = "D5-evaluation-parent-split-20261030-v1"
PLIES = (20, 60, 100)
EXCLUDED = frozenset(range(48)) | EXCLUDED_DEVELOPMENT_PAIRS | frozenset(FROZEN_PAIRS)
REMAINING = frozenset(range(120)) - EXCLUDED


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def frozen_split():
    if len(EXCLUDED) != 96 or len(REMAINING) != 24:
        raise ValueError("D5 historical parent exclusion has changed")
    ordered = sorted(REMAINING, key=lambda p: sha((SEED + "|" + str(p)).encode()))
    return tuple(ordered[:12]), tuple(ordered[12:])


def auc(rows, key):
    positive = [r[key] for r in rows if r["outcome"] == 1]
    negative = [r[key] for r in rows if r["outcome"] == -1]
    if not positive or not negative:
        return None
    favorable = sum((a > b) + 0.5 * (a == b) for a in positive for b in negative)
    return favorable / (len(positive) * len(negative))


def score_summary(rows, key):
    if not rows:
        return {"n": 0, "sign_accuracy": None, "auc": None}
    correct = sum((r[key] > 0 and r["outcome"] == 1) or
                  (r[key] < 0 and r["outcome"] == -1) for r in rows)
    ties = sum(r[key] == 0 for r in rows)
    return {"n": len(rows), "sign_accuracy": round(correct / len(rows), 6),
            "zero_scores": ties, "auc": None if auc(rows, key) is None
            else round(auc(rows, key), 6)}


def pair_bootstrap(rows, key, repetitions=300):
    """Pair-cluster percentile interval; never resample individual plies."""
    groups = {}
    for row in rows:
        groups.setdefault(row["pair"], []).append(row)
    parents = sorted(groups)
    if len(parents) < 2:
        return None
    rng = random.Random(20261030)
    values = []
    for _ in range(repetitions):
        sampled = [r for _ in parents for r in groups[rng.choice(parents)]]
        if sampled:
            correct = sum((r[key] > 0 and r["outcome"] == 1) or
                          (r[key] < 0 and r["outcome"] == -1) for r in sampled)
            values.append(correct / len(sampled))
    values.sort()
    return [round(values[int((len(values) - 1) * q)], 6) for q in (0.025, 0.975)]


def verify_source(games):
    """Check ALL 240 original games before looking at any evaluation score."""
    if len(games) != 240:
        raise ValueError("not all source games were obtained")
    statuses = Counter()
    for row in games:
        replay_check(row["opening"], row["moves"], row)
        statuses[row["status"]] += 1
    if statuses["checkmate"] != 147 or sum(statuses.values()) - 147 != 93:
        raise ValueError("unexpected decisive/unresolved source totals")
    return dict(statuses)



def feature_snapshot(board):
    """Interpretable, side-to-move material and king-zone risk (not labels)."""
    from .engine import VALUES
    own, other = int(board.turn), int(not board.turn)
    board_balance = 0
    for square in range(81):
        piece = board.piece_at(square)
        if piece is not None and piece.piece_type != 8:
            board_balance += VALUES[piece.piece_type] * (1 if int(piece.color) == own else -1)
    hand_balance = 0
    for color in (own, other):
        factor = 1 if color == own else -1
        for kind, count in board.pieces_in_hand[color].items():
            hand_balance += factor * VALUES[kind] * int(count)

    def king_zone_attacks(color):
        king = board.king_squares[color]
        if king is None:
            return None
        file, rank = divmod(king, 9)
        enemy = int(not color)
        return sum(board.is_attacked_by(enemy, nf * 9 + nr)
                   for nf in range(max(0, file - 1), min(8, file + 1) + 1)
                   for nr in range(max(0, rank - 1), min(8, rank + 1) + 1))
    return {"board_material_cp": board_balance, "hand_material_cp": hand_balance,
            "own_king_zone_attacks": king_zone_attacks(own),
            "opponent_king_zone_attacks": king_zone_attacks(other),
            "in_check": bool(board.is_check())}


def development_error_cases(observations):
    """Only inspected development observations; sealed group is untouched."""
    return [
        {"pair": row["pair"], "ply": row["ply"], "g8_black": row["g8_black"],
         "outcome": row["outcome"], "classical": row["classical"], "g8": row["g8"],
         "classical_correct": (row["classical"] * row["outcome"] > 0),
         "g8_correct": (row["g8"] * row["outcome"] > 0),
         "features": row["features"]}
        for row in observations
        if row["g8"] * row["outcome"] <= 0 or
           row["classical"] * row["outcome"] <= 0
    ]

def development_scores(games, model):
    """Do not inspect or compute scores for sealed parent games."""
    development, sealed = frozen_split()
    dev = set(development)
    observations = []
    for row in games:
        if row["pair"] not in dev or row["status"] != "checkmate":
            continue
        moves = row["opening"] + row["moves"]
        # The game must continue beyond the sampled ply.
        import shogi
        board = shogi.Board()
        targets = set(PLIES)
        for ply, word in enumerate(moves, 1):
            move = shogi.Move.from_usi(word)
            if move not in board.legal_moves:
                raise ValueError("illegal source game")
            board.push(move)
            if ply not in targets or ply >= len(moves):
                continue
            before = (board.sfen(), len(board.move_stack), board.zobrist_hash())
            start = time.perf_counter()
            classical = evaluate(board)
            classical_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            learned = evaluate(board, model, fast_relative=True)
            learned_ms = (time.perf_counter() - start) * 1000
            if (board.sfen(), len(board.move_stack), board.zobrist_hash()) != before:
                raise AssertionError("evaluation mutated source board")
            observations.append({
                "pair": row["pair"], "g8_black": row["g8_black"], "ply": ply,
                "outcome": 1 if int(board.turn) == row["winner"] else -1,
                "classical": classical, "g8": learned,
                "classical_ms": round(classical_ms, 6),
                "g8_ms": round(learned_ms, 6),
                "sfen_sha256": sha(board.sfen().encode()),
                "features": feature_snapshot(board),
            })
        if not board.is_checkmate() or row["winner"] != int(not board.turn):
            raise ValueError("decisive game does not end in proven mate")
    return observations


def report(games, model):
    statuses = verify_source(games)
    development, sealed = frozen_split()
    observations = development_scores(games, model)
    stages = {}
    for ply in PLIES:
        subset = [r for r in observations if r["ply"] == ply]
        stages[str(ply)] = {
            "classical": score_summary(subset, "classical"),
            "g8": score_summary(subset, "g8"),
            "g8_sign_accuracy_pair_bootstrap_95pct": pair_bootstrap(subset, "g8"),
            "g8_eval_ms_median": (round(statistics.median(r["g8_ms"] for r in subset), 6)
                                  if subset else None),
        }
    return {
        "scope": "D5_development_only_outcome_correlation_not_strength",
        "source_zip_sha256": SOURCE_ZIP_SHA, "source_games_sha256": SOURCE_GAMES_SHA,
        "g8_model_sha256": G8_SHA, "seed": SEED, "sample_plies": list(PLIES),
        "source_games": len(games), "source_statuses": statuses,
        "development_pairs": list(development),
        "sealed_pairs_sha256": sha(json.dumps(sealed).encode()),
        "sealed_pair_count": len(sealed), "sealed_scores_computed": False,
        "development_decisive_games": sum(r["pair"] in set(development) and
                                          r["status"] == "checkmate" for r in games),
        "observations": observations, "by_ply": stages,
        "development_error_cases": development_error_cases(observations),
        "training_executed": False, "games_played": 0, "model_adopted": False,
    }


def run(output):
    if os.environ.get("GITHUB_REPOSITORY") != "mtk-ctrl/shogi-fukaji":
        raise ValueError("not running in the original research repository")
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
    result = report(source_rows(source), model)
    # Do not include model bytes or unsealed original game histories in artifact.
    (root / "d5-development-diagnosis.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if k != "observations"}
    summary["observation_count"] = len(result["observations"])
    summary["development_error_count"] = len(result["development_error_cases"])
    (root / "d5-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.out)
