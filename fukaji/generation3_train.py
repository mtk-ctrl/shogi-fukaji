"""Third research-generation update from newly completed frozen selfplay mates.

Preserve every original parent-group split, especially the never-tuned test
group. Add only replay-verified checkmates, no unfinished outcome labels.
All hyperparameters are frozen from generation 2; no automatic adoption.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .generation_train import incremental_fit, score, sha
from .learning import FEATURES, load_games
from .relative_value import relative_features

OLD_DATA_SHA = "cdcf8985ec81e2011f3b82d188e13148bd2d0e6cd36ddaf6638098e7bdaacda0"
NEW_DATA_SHA = "6a4f0efceb38f72b61aa25833cb535adfc4093091a6d10e01af43d62401b6604"
NEW_REPORT_SHA = "1ece44c096902f7556c96dca9f6c67e0a9f30a68a4313fed09e31115e8aa8e71"
G1_SHA = "881af10b5bc0cde591e37a9aee883213fa7dd602345b786b9b93d69c7e6ed239"
G2_SHA = "00a6186dd710c992cb5df2fe96dd42066eef37a6e0f212eecd5931691a20e6b4"


def validate_request(cfg):
    expected = {
        "research_only": True, "source_runs": [40404298089, 40414249980],
        "parent_model_runs": [40359123574, 40404769521],
        "training_allowed": True, "model_adoption_allowed": False,
        "preserve_existing_test": True,
    }
    if cfg != expected:
        raise ValueError("unexpected research generation-3 request")
    return cfg


def group_sort(key):
    return hashlib.sha256(("fukaji-group-v2|" + key).encode()).hexdigest()


def frozen_assignments(old_groups, new_groups):
    """The original 25 parents retain their exact generation-2 split.

    Six previously unseen parents get a separate fixed 4/1/1 assignment.
    Never rebalance or train on any previously held-out test parent.
    """
    old = sorted(set(old_groups), key=group_sort)
    new = sorted(set(new_groups) - set(old_groups), key=group_sort)
    if len(old) != 25 or len(new) != 6:
        raise ValueError("unexpected independent parent group counts")
    mapping = {}
    for i, key in enumerate(old):
        mapping[key] = "train" if i < 21 else "validation" if i < 23 else "test"
    for i, key in enumerate(new):
        mapping[key] = "train" if i < 4 else "validation" if i == 4 else "test"
    if len(mapping) != 31:
        raise ValueError("parent split overlap")
    return mapping


def read_weights(path, expected):
    if sha(path) != expected:
        raise ValueError("frozen model SHA mismatch")
    with np.load(path, allow_pickle=False) as d:
        if set(d.files) != {"weight"}:
            raise ValueError("unknown model schema")
        w = d["weight"].astype(np.float32)
    if w.shape != (FEATURES,) or not np.isfinite(w).all():
        raise ValueError("invalid model weights")
    return w


def execute(request, old_path, new_path, old_manifest, new_report,
            g1_path, g2_path, out):
    validate_request(json.loads(Path(request).read_text(encoding="utf-8")))
    if sha(old_path) != OLD_DATA_SHA or sha(new_path) != NEW_DATA_SHA:
        raise ValueError("frozen training dataset SHA mismatch")
    original_manifest = json.loads(Path(old_manifest).read_text(encoding="utf-8"))
    extension_manifest = json.loads(Path(new_report).read_text(encoding="utf-8"))
    if (original_manifest.get("dataset_sha256") != OLD_DATA_SHA or
            original_manifest.get("accepted_checkmate_games") != 31 or
            original_manifest.get("independent_parent_groups") != 25 or
            extension_manifest.get("games_sha256") != NEW_REPORT_SHA or
            extension_manifest.get("mates_sha256") != NEW_DATA_SHA or
            extension_manifest.get("games") != 32 or
            extension_manifest.get("new_verified_mates") != 15 or
            extension_manifest.get("legal_replay_verified") is not True):
        raise ValueError("frozen dataset provenance mismatch")
    g1 = read_weights(g1_path, G1_SHA)
    g2 = read_weights(g2_path, G2_SHA)
    old = [json.loads(x) for x in Path(old_path).read_text(
        encoding="utf-8").splitlines() if x.strip()]
    new = [json.loads(x) for x in Path(new_path).read_text(
        encoding="utf-8").splitlines() if x.strip()]
    if len(old) != 31 or len(new) != 15:
        raise ValueError("frozen game count mismatch")
    old_groups = {g["parent_id"] for g in old}
    new_groups = {g["parent_id"] for g in new}
    mapping = frozen_assignments(old_groups, new_groups)
    all_games = old + new
    keys = {(g["start_sfen"], tuple(g["moves"])) for g in all_games}
    if len(keys) != len(all_games):
        raise ValueError("duplicate mate across frozen datasets")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "merged_verified_mates.jsonl"
    dataset.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n"
                               for g in all_games), encoding="utf-8")
    # load_games rejects illegal moves, non-checkmate labels and wrong winners.
    rows = load_games(dataset, feature_fn=relative_features)
    if len(rows) != 46 or {r[0] for r in rows} != set(mapping):
        raise ValueError("replay-verified learner row mismatch")
    splits = {k: [] for k in ("train", "validation", "test")}
    for parent_id, x, y in rows:
        splits[mapping[parent_id]].append((x, y))
    if (len(splits["train"]) < 25 or
            len(splits["validation"]) < 3 or len(splits["test"]) < 3):
        raise ValueError("insufficient frozen parent-disjoint split")
    # No hyperparameter search; preserve generation-2 optimizer exactly.
    g3 = incremental_fit(splits["train"], g2, epochs=8, lr=0.02, seed=42)
    model_path = root / "mover-relative-generation3-research.npz"
    np.savez_compressed(model_path, weight=g3)
    counts = {k: sum(label == k for label in mapping.values())
              for k in splits}
    if counts != {"train": 25, "validation": 3, "test": 3}:
        raise ValueError("frozen parent assignment changed")
    report = {
        "status": "generation3_research_candidate_not_promoted",
        "old_dataset_sha256": OLD_DATA_SHA,
        "new_dataset_sha256": NEW_DATA_SHA,
        "merged_dataset_sha256": sha(dataset),
        "g1_model_sha256": G1_SHA, "g2_model_sha256": G2_SHA,
        "g3_model_sha256": sha(model_path),
        "verified_mate_games": len(rows),
        "independent_parent_groups": len(mapping),
        "split_parent_groups": counts,
        "split_games": {k: len(v) for k, v in splits.items()},
        "old_heldout_groups_preserved": True,
        "training": {"epochs": 8, "lr": 0.02, "seed": 42,
                     "l2_toward_g2": 0.001, "initialized_from": "generation2_research"},
        "metrics": {k: {"generation1": score(v, g1),
                        "generation2": score(v, g2),
                        "generation3": score(v, g3)}
                    for k, v in splits.items()},
        "updated_coefficients": int(np.count_nonzero(g3 != g2)),
        "candidate_adopted": False,
        "limitations": [
            "Held-out test data is diagnostic only and not used for hyperparameter tuning.",
            "Game outcome does not prove individual move quality.",
            "Independent paired head-to-head games are needed to establish strength.",
            "Research generation 2 and 3 are not formally adopted.",
        ],
    }
    (root / "generation3_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "frozen_parent_splits.json").write_text(
        json.dumps(mapping, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    for arg in ("request", "old", "new", "old-manifest", "new-report",
                "generation1", "generation2", "out"):
        p.add_argument("--" + arg, required=True)
    a = p.parse_args()
    execute(a.request, a.old, a.new, a.old_manifest, a.new_report,
            a.generation1, a.generation2, a.out)


if __name__ == "__main__":
    main()
