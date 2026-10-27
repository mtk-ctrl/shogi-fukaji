"""G7 research-only training from independently replayed G6 selfplay.

Never train on any prior validation/test parent; new parents have frozen,
disjoint splits. Fixed optimizer, no test-driven tuning, no model promotion.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .generation3_train import read_weights, group_sort
from .generation_train import incremental_fit, score, sha
from .learning import load_games
from .relative_value import relative_features

G6_RUN, NEW_RUN = 40429892376, 40431771135
G6_SHA = "99984388447a26cef9ca770bd3efe0cd38af784aba056395e95f90f5a4d4ff08"
OLD_SHA = "e5389f8ae2b74f0fe8fde97d93229fefc2102cd5141c947cfe341ccd5c86c9a3"
OLD_REPORT_SHA = "b472751d10753208a61a8980b4f0dafd6f45fb046a4aa7e0821d2443e40af853"
OLD_SPLITS_SHA = "a6783832ee90d43f53d21a17b22220db34a1618dc04aed48073179d17a1d87ae"
NEW_SHA = "e1c686805ae373be2117e9e03a83d2c8821b187a33a5970528923a3dfb473f26"
NEW_REPORT_SHA = "26f2b950df2d30bd82e2e06e4429dd7618bc701f44c2963ca1a39dd6b5a57e30"
ALL_GAMES_SHA = "3a0f762df9ee4c57d353ef0b8bf2f9b3a6a258134176ae2fa4ead59fababc13e"

def expected_request():
    return {"research_only": True, "training_allowed": True,
            "model_adoption_allowed": False, "source_runs": [G6_RUN, NEW_RUN],
            "preserve_prior_parent_splits": True, "old_test_for_tuning": False,
            "reuse_frozen_optimizer": True, "parent_model_sha256": G6_SHA,
            "new_mates_sha256": NEW_SHA,
            "fixed_update_shrinkage": 0.5}

def validate_request(cfg):
    if cfg != expected_request():
        raise ValueError("unapproved generation-7 research protocol")
    return cfg

def frozen_assignments(old_mapping, old_groups, new_groups):
    if (len(old_mapping) != 367 or set(old_mapping) != set(old_groups)
            or {k: list(old_mapping.values()).count(k)
                for k in ("train", "validation", "test")}
            != {"train": 295, "validation": 36, "test": 36}
            or set(old_mapping.values()) != {"train", "validation", "test"}):
        raise ValueError("prior heldout parent assignments changed")
    if len(new_groups) != 190 or set(old_groups) & set(new_groups):
        raise ValueError("new independent parent groups invalid or overlapping")
    mapping = dict(old_mapping)
    for i, key in enumerate(sorted(new_groups, key=group_sort)):
        mapping[key] = "train" if i < 152 else "validation" if i < 171 else "test"
    if {k: list(mapping.values()).count(k) for k in ("train", "validation", "test")} != {
            "train": 447, "validation": 55, "test": 55}:
        raise ValueError("invalid frozen parent counts")
    return mapping

def execute(request, old_data, old_report, old_splits, new_data, new_report, generation6, out):
    validate_request(json.loads(Path(request).read_text(encoding="utf-8")))
    for path, digest in ((old_data, OLD_SHA), (old_report, OLD_REPORT_SHA),
                         (old_splits, OLD_SPLITS_SHA), (new_data, NEW_SHA),
                         (new_report, NEW_REPORT_SHA), (generation6, G6_SHA)):
        if sha(path) != digest:
            raise ValueError("frozen source SHA mismatch: " + str(path))
    prior = json.loads(Path(old_report).read_text(encoding="utf-8"))
    recent = json.loads(Path(new_report).read_text(encoding="utf-8"))
    if (prior.get("merged_dataset_sha256") != OLD_SHA or
        prior.get("g6_model_sha256") != G6_SHA or
        prior.get("verified_mate_games") != 382 or
        prior.get("independent_parent_groups") != 367 or
        prior.get("old_parent_assignments_preserved") is not True or
        recent.get("source_run") != G6_RUN or
        recent.get("model_sha256") != G6_SHA or
        recent.get("verified_mates_sha256") != NEW_SHA or
        recent.get("all_games_sha256") != ALL_GAMES_SHA or
        recent.get("games") != 256 or
        recent.get("accepted_unique_mates") != 190 or
        recent.get("independent_parent_groups") != 190 or
        recent.get("terminal_counts") != {"checkmate": 190, "move_limit_unresolved": 66} or
        recent.get("legal_replay_verified") is not True):
        raise ValueError("frozen dataset provenance mismatch")
    old = [json.loads(s) for s in Path(old_data).read_text(encoding="utf-8").splitlines() if s.strip()]
    new = [json.loads(s) for s in Path(new_data).read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(old) != 382 or len(new) != 190:
        raise ValueError("wrong mate count")
    mapping = frozen_assignments(
        json.loads(Path(old_splits).read_text(encoding="utf-8")),
        {g["parent_id"] for g in old}, {g["parent_id"] for g in new})
    games = old + new
    if len({(g["start_sfen"], tuple(g["moves"])) for g in games}) != 572:
        raise ValueError("duplicate or overlapping games")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    merged = root / "merged_verified_mates.jsonl"
    merged.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in games),
                      encoding="utf-8")
    # The strict learner replays every move, checks actual mate and winner.
    rows = load_games(merged, feature_fn=relative_features)
    if len(rows) != 572 or {r[0] for r in rows} != set(mapping):
        raise ValueError("strict learner rejected a source game")
    splits = {k: [] for k in ("train", "validation", "test")}
    old_parts = {k: [] for k in splits}
    new_parts = {k: [] for k in splits}
    for i, (parent_id, x, y) in enumerate(rows):
        part = mapping[parent_id]
        splits[part].append((x, y))
        (old_parts if i < 382 else new_parts)[part].append((x, y))
    if {k: len(v) for k, v in splits.items()} != {
            "train": 459, "validation": 55, "test": 58}:
        raise ValueError("unexpected split game counts")
    g6 = read_weights(generation6, G6_SHA)
    # No hyperparameter search and no use of validation/test to select epochs.
    # Predeclared conservative update; no validation/test-based selection.
    raw = incremental_fit(splits["train"], g6, epochs=8, lr=0.02, seed=42)
    g7 = (g6 + np.float32(0.5) * (raw - g6)).astype(np.float32)
    model = root / "mover-relative-generation7-research.npz"
    np.savez_compressed(model, weight=g7)
    (root / "frozen_parent_splits.json").write_text(
        json.dumps(mapping, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "generation7_research_candidate_not_promoted",
        "source_runs": [G6_RUN, NEW_RUN],
        "old_dataset_sha256": OLD_SHA, "new_dataset_sha256": NEW_SHA,
        "merged_dataset_sha256": sha(merged),
        "g6_model_sha256": G6_SHA, "g7_model_sha256": sha(model),
        "verified_mate_games": len(rows), "independent_parent_groups": len(mapping),
        "old_parent_assignments_preserved": True,
        "new_parent_groups": {"train": 152, "validation": 19, "test": 19},
        "split_parent_groups": {k: list(mapping.values()).count(k) for k in splits},
        "split_games": {k: len(v) for k, v in splits.items()},
        "training": {"epochs": 8, "lr": 0.02, "seed": 42, "l2_toward_g6": 0.001,
                     "fixed_update_shrinkage": 0.5,
                     "no_holdout_selection": True},
        "metrics": {k: {"generation6": score(v, g6), "generation7": score(v, g7)}
                    for k, v in splits.items()},
        "old_holdout_diagnostics": {k: {"generation6": score(old_parts[k], g6),
                                       "generation7": score(old_parts[k], g7)}
                                    for k in ("validation", "test")},
        "new_holdout_diagnostics": {k: {"generation6": score(new_parts[k], g6),
                                       "generation7": score(new_parts[k], g7)}
                                    for k in ("validation", "test")},
        "updated_coefficients": int(np.count_nonzero(g7 != g6)),
        "candidate_adopted": False,
        "limitations": [
            "Every old validation/test parent retains its frozen assignment.",
            "All new validation/test parents are excluded from fitting.",
            "Held-out test metrics are diagnostic, never optimizer selection.",
            "A fixed 0.5 update shrinkage is predeclared; it is not a tuned or proven remedy for overfitting.",
            "G7 strength requires independent paired G7 vs G1 and G6 games.",
            "No formal promotion and no changes to KUMOJI.",
        ]}
    (root / "generation7_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report

def main():
    p = argparse.ArgumentParser()
    for name in ("request", "old-data", "old-report", "old-splits",
                 "new-data", "new-report", "generation6", "out"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    execute(a.request, a.old_data, a.old_report, a.old_splits,
            a.new_data, a.new_report, a.generation6, a.out)

if __name__ == "__main__":
    main()
