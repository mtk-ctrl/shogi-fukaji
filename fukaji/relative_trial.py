"""Bounded v1 mover-relative feature experiment on one SHA-frozen dataset.

Retrains ONLY a separate candidate, never changes the existing legacy model,
and uses parent-disjoint splits and equal-completed-depth arena controls.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .arena import run as arena
from .learning import FEATURES, fit, load_games, partition_rows
from .relative_value import relative_features


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(games_path, legacy_model_path, request_path, out):
    cfg = json.loads(Path(request_path).read_text(encoding="utf-8"))
    for key, path in (("dataset_sha256", games_path), ("model_sha256", legacy_model_path)):
        if sha(path) != cfg[key]:
            raise ValueError("source " + key + " mismatch")
    pairs = cfg["arena_pairs"]
    seed = cfg["arena_seed"]
    if (type(cfg["source_run_id"]) is not int or cfg["source_run_id"] <= 0 or
        type(pairs) is not int or not 1 <= pairs <= 20 or
        type(seed) is not int or not 0 <= seed <= 2**31 - 1):
        raise ValueError("invalid bounded trial request")
    legacy_rows = load_games(games_path)
    relative_rows = load_games(games_path, feature_fn=relative_features)
    if len(legacy_rows) != len(relative_rows) or any(
        a[0] != b[0] or not np.array_equal(a[2], b[2])
        for a, b in zip(legacy_rows, relative_rows)
    ):
        raise ValueError("source, grouping or labels changed between schemas")
    _, groups = partition_rows(relative_rows)
    if min(groups.values()) < 2:
        raise ValueError("insufficient independent parent groups")
    legacy_weights, legacy_metrics = fit(legacy_rows, epochs=4)
    relative_weights, relative_metrics = fit(relative_rows, epochs=4)
    with np.load(legacy_model_path, allow_pickle=False) as data:
        if set(data.files) != {"weight"}:
            raise ValueError("unexpected legacy model schema")
        frozen_weights = data["weight"]
    if frozen_weights.shape != (FEATURES,):
        raise ValueError("invalid frozen model")
    weight_delta = float(np.max(np.abs(frozen_weights - legacy_weights)))
    if weight_delta > 1e-4:
        raise ValueError("legacy reproduction mismatch: training configuration drift")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    relative_path = root / "mover-relative-v1.npz"
    np.savez_compressed(relative_path, weight=relative_weights)
    common = dict(pairs=pairs, movetime=20, depth=1, max_plies=160,
                  opening_plies=6, mix=25, seed=seed, search_mode="complete_depth")
    baseline = arena(legacy_model_path, root / "legacy_mix25.jsonl",
                     model_type="legacy", **common)
    candidate = arena(relative_path, root / "relative_mix25.jsonl",
                      model_type="mover_relative_v1", **common)
    report = {
        "status": "research_candidate_measured_no_promotion",
        "source_run_id": cfg["source_run_id"],
        "dataset_sha256": sha(games_path),
        "legacy_model_sha256": sha(legacy_model_path),
        "relative_model_sha256": sha(relative_path),
        "verified_checkmates": len(relative_rows),
        "parent_groups": groups,
        "same_parent_splits_and_labels": True,
        "legacy_reproduced_max_abs_weight_delta": weight_delta,
        "legacy_split_metrics": legacy_metrics,
        "relative_split_metrics": relative_metrics,
        "arena_control": {"pairs_per_model": pairs, "search_mode": "complete_depth",
                          "depth": 1, "seed": seed, "mix_percent": 25},
        "legacy_arena": baseline,
        "relative_arena": candidate,
        "model_adopted": False,
        "limitations": [
            "The candidate changes only feature perspective, not labels or training objective.",
            "Arena compares each model against the same classical baseline, not a direct candidate-versus-legacy match.",
            "Outcome labels are terminal-game labels, not independent position-quality targets.",
            "Do not tune on the held-out test split; this experiment is exploratory and not a strength claim.",
        ],
    }
    (root / "relative_trial.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", required=True)
    p.add_argument("--legacy-model", required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--out", default="build/relative_trial")
    a = p.parse_args()
    execute(a.games, a.legacy_model, a.request, a.out)


if __name__ == "__main__":
    main()
