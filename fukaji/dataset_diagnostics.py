"""Replay-verified diagnostic of outcome labels, split bias and value features.

This does not train or promote a model.  It inspects one SHA-frozen selfplay
dataset and the exact model trained from it, including independent parent
groups and the known side-to-move representation limitation.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import shogi

from .learning import FEATURES, features, load_games, partition_rows


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _split_stats(games, weights):
    samples = positives = 0
    brier_sum = loss_sum = confidence_sum = 0.0
    per_game_lengths = []
    for x, y in games:
        z = np.clip(x @ weights, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-z))
        samples += len(y)
        positives += int(np.sum(y))
        brier_sum += float(np.sum((p - y) ** 2))
        loss_sum += float(np.sum(np.logaddexp(0, z) - y * z))
        confidence_sum += float(np.sum(np.abs(p - 0.5)))
        per_game_lengths.append(len(y))
    return {
        "games": len(games), "positions": samples,
        "positive_labels": positives, "negative_labels": samples - positives,
        "positive_fraction": positives / samples if samples else None,
        "mean_game_plies": sum(per_game_lengths) / len(per_game_lengths) if games else None,
        "log_loss": loss_sum / samples if samples else None,
        "brier_score": brier_sum / samples if samples else None,
        "mean_distance_from_half": confidence_sum / samples if samples else None,
    }


def turn_feature_probe():
    """Same piece layout, opposite turn: all non-turn features stay unchanged.

    This means absolute piece coefficients cannot change sign with the side
    to move.  It is a structural limitation, not proof of any Elo difference.
    """
    b = shogi.Board()
    words = b.sfen().split()
    words[1] = "w" if words[1] == "b" else "b"
    other = shogi.Board(" ".join(words))
    a, z = features(b), features(other)
    return {
        "same_board_non_turn_features_equal": bool(np.array_equal(a[:-1], z[:-1])),
        "turn_feature_sign_reverses": bool(a[-1] == -z[-1]),
        "relative_piece_feature_representation": False,
        "limitation": "Only the global turn coefficient flips for an unchanged board; "
                      "piece and hand contributions do not reverse for the player to move.",
    }


def execute(games_path, model_path, request_path, out):
    cfg = json.loads(Path(request_path).read_text(encoding="utf-8"))
    expected = {"source_run_id", "dataset_sha256", "model_sha256"}
    if not expected.issubset(cfg) or type(cfg["source_run_id"]) is not int or cfg["source_run_id"] <= 0:
        raise ValueError("invalid source run or missing expected SHA256")
    for key, path in (("dataset_sha256", games_path), ("model_sha256", model_path)):
        sha = cfg[key]
        if (not isinstance(sha, str) or len(sha) != 64 or
            any(c not in "0123456789abcdef" for c in sha) or _sha(path) != sha):
            raise ValueError(key + " mismatch or invalid SHA256")
    with np.load(model_path, allow_pickle=False) as data:
        if set(data.files) != {"weight"}:
            raise ValueError("unexpected model payload")
        weights = data["weight"]
    if weights.shape != (FEATURES,) or not np.isfinite(weights).all():
        raise ValueError("bad model weights")
    # load_games checks every move, terminal mate and winner; rejects duplicates.
    rows = load_games(games_path)
    games = [json.loads(line) for line in Path(games_path).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if len(games) != len(rows):
        raise ValueError("duplicate or nondecisive game in frozen dataset")
    splits, parent_counts = partition_rows(rows)
    winner_counts = Counter(str(g["winner"]) for g in games)
    parent_sizes = Counter(key for key, _, _ in rows)
    starts = Counter(g["start_sfen"] for g in games)
    branch_games = sum("parent_id" in g for g in games)
    report = {
        "status": "verified_diagnostic_only",
        "source_run_id": cfg["source_run_id"],
        "dataset_sha256": _sha(games_path),
        "model_sha256": _sha(model_path),
        "verified_checkmates": len(rows),
        "winner_counts": dict(winner_counts),
        "black_win_fraction": winner_counts.get("0", 0) / len(games) if games else None,
        "branch_games": branch_games,
        "branch_fraction": branch_games / len(games) if games else None,
        "distinct_start_sfen": len(starts),
        "largest_start_sfen_frequency": max(starts.values(), default=0),
        "independent_parent_groups": len(parent_sizes),
        "largest_parent_group": max(parent_sizes.values(), default=0),
        "parent_split_counts": parent_counts,
        "splits": {k: _split_stats(v, weights) for k, v in splits.items()},
        "turn_feature_probe": turn_feature_probe(),
        "limitations": [
            "The training labels are final game outcomes copied to every pre-move position, not independent positional evaluations.",
            "Absolute piece and hand features are not side-to-move normalized; a single turn coefficient cannot reverse piece value contributions.",
            "Related endgame branches share a parent split, but long games and branch-rich parents can dominate position-weighted fitting.",
            "Held-out outcome log loss is not an independent measure of playing strength.",
        ],
        "recommendation": "Test a separately versioned, mover-relative feature representation on frozen parent-disjoint splits before any arena adoption.",
        "promote_model": False,
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "dataset_diagnostic.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--out", default="build/dataset_audit")
    a = p.parse_args()
    execute(a.games, a.model, a.request, a.out)


if __name__ == "__main__":
    main()
