"""Isolated FUKAJI generation-2 research update from verified selfplay mates.

Start from the frozen parent model; train on parent-disjoint groups only.
No adoption, no changes to KUMOJI, no claim of playing-strength improvement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .learning import FEATURES, load_games, partition_rows
from .relative_value import relative_features

DATA_RUN = 40404298089
DATA_SHA = "cdcf8985ec81e2011f3b82d188e13148bd2d0e6cd36ddaf6638098e7bdaacda0"
PARENT_RUN = 40359123574
PARENT_SHA = "881af10b5bc0cde591e37a9aee883213fa7dd602345b786b9b93d69c7e6ed239"
EPOCHS, LR, SEED = 8, 0.02, 42


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def incremental_fit(train, parent, epochs=EPOCHS, lr=LR, seed=SEED):
    if not train:
        raise ValueError("empty training groups")
    w = parent.astype(np.float32).copy()
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for index in rng.permutation(len(train)):
            x, y = train[index]
            logits = np.clip(x @ w, -20, 20)
            p = 1.0 / (1.0 + np.exp(-logits))
            gradient = (x.T @ (p - y)) / len(y) + 0.001 * (w - parent)
            w -= lr * gradient.astype(np.float32)
    if not np.isfinite(w).all():
        raise ValueError("non-finite candidate")
    return w


def score(groups, w):
    if not groups:
        return {"games": 0, "positions": 0, "log_loss": None, "brier": None}
    loss, brier, n = 0.0, 0.0, 0
    for x, y in groups:
        logits = np.clip(x @ w, -20, 20)
        p = 1.0 / (1.0 + np.exp(-logits))
        loss += float(np.sum(np.logaddexp(0, logits) - y * logits))
        brier += float(np.sum((p - y) ** 2))
        n += len(y)
    return {"games": len(groups), "positions": n, "log_loss": loss / n, "brier": brier / n}


def execute(data_path, manifest_path, parent_path, out):
    if sha(data_path) != DATA_SHA or sha(parent_path) != PARENT_SHA:
        raise ValueError("frozen dataset or parent model SHA mismatch")
    manifest = json.loads(Path(manifest_path).read_text())
    if (manifest.get("status") != "replay_verified_generation_input_only_no_training_no_adoption" or
            manifest.get("dataset_sha256") != DATA_SHA or
            manifest.get("frozen_model_sha256") != PARENT_SHA or
            manifest.get("accepted_checkmate_games") != 31 or
            manifest.get("unresolved_excluded_from_labels") != 32 or
            manifest.get("independent_parent_groups") != 25):
        raise ValueError("input manifest mismatch")
    with np.load(parent_path, allow_pickle=False) as d:
        if set(d.files) != {"weight"}:
            raise ValueError("unexpected parent model schema")
        parent = d["weight"]
    if parent.shape != (FEATURES,) or not np.isfinite(parent).all():
        raise ValueError("bad parent weights")
    rows = load_games(data_path, feature_fn=relative_features)
    if len(rows) != 31:
        raise ValueError("dataset replay count mismatch")
    parts, groups = partition_rows(rows)
    if groups != {"train": 21, "validation": 2, "test": 2}:
        raise ValueError("split parent counts changed")
    candidate = incremental_fit(parts["train"], parent)
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "mover-relative-generation2-research.npz"
    np.savez_compressed(model_path, weight=candidate)
    report = {
        "status": "research_candidate_not_promoted",
        "dataset_run": DATA_RUN, "dataset_sha256": DATA_SHA,
        "parent_run": PARENT_RUN, "parent_model_sha256": PARENT_SHA,
        "candidate_model_sha256": sha(model_path),
        "training": {"epochs": EPOCHS, "lr": LR, "seed": SEED, "l2_toward_parent": 0.001},
        "parent_groups": groups,
        "metrics": {part: {"parent": score(games, parent), "candidate": score(games, candidate)}
                    for part, games in parts.items()},
        "delta_l2": float(np.linalg.norm(candidate - parent)),
        "delta_max_abs": float(np.max(np.abs(candidate - parent))),
        "updated_coefficients": int(np.count_nonzero(candidate != parent)),
        "candidate_adopted": False,
        "limitations": ["The held-out test set is diagnostic, not used for tuning.",
                        "Game outcome is a weak label for individual move quality.",
                        "Validation loss is not evidence of playing strength.",
                        "Independent same-time paired games are required before any promotion."],
    }
    (target / "generation2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    for key in ("data", "manifest", "parent", "out"):
        ap.add_argument("--" + key, required=True)
    args = ap.parse_args()
    execute(args.data, args.manifest, args.parent, args.out)


if __name__ == "__main__":
    main()
