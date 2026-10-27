"""G6 research-only training from independently replayed G5 selfplay.

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

G5_RUN, NEW_RUN = 40424947260, 40427032590
G5_SHA = "ff1128d9b139fdf71beae1b49a48506bfdcf3e9cd31e08053341783985d92e03"
OLD_SHA = "663f0a31dadcc689d82a55678bd4a7ecd730806e8ed8a79f9338b492701dc475"
OLD_REPORT_SHA = "0ad3173f21a090cfc4564b23338f4c611f8493d932a9ec159f57fa7e7d8d9d90"
OLD_SPLITS_SHA = "ebda1afdfd1bf557c8e7bbd8ef3688bfaee063192a1f05119e340060c9fabdcf"
NEW_SHA = "fdba16b653b8bea2739645d188c8951f8d41cfe49bb97b733a2c74fbb7f78b08"
NEW_REPORT_SHA = "b229b04c1404a941a2347bfdfb66261015fe7c0ac2ae9a8d1ea86dbd37544924"
ALL_GAMES_SHA = "b51eccc67a6f1ea41b15aa61ef548f2586c8572f8aa5b4d2b76598f222b217bc"

def expected_request():
    return {"research_only": True, "training_allowed": True,
            "model_adoption_allowed": False, "source_runs": [G5_RUN, NEW_RUN],
            "preserve_prior_parent_splits": True, "old_test_for_tuning": False,
            "reuse_frozen_optimizer": True, "parent_model_sha256": G5_SHA,
            "new_mates_sha256": NEW_SHA}

def validate_request(cfg):
    if cfg != expected_request():
        raise ValueError("unapproved generation-5 research protocol")
    return cfg

def frozen_assignments(old_mapping, old_groups, new_groups):
    if (len(old_mapping) != 207 or set(old_mapping) != set(old_groups)
            or {k: list(old_mapping.values()).count(k)
                for k in ("train", "validation", "test")}
            != {"train": 295, "validation": 36, "test": 36}
            or set(old_mapping.values()) != {"train", "validation", "test"}):
        raise ValueError("prior heldout parent assignments changed")
    if len(new_groups) != 160 or set(old_groups) & set(new_groups):
        raise ValueError("new independent parent groups invalid or overlapping")
    mapping = dict(old_mapping)
    for i, key in enumerate(sorted(new_groups, key=group_sort)):
        mapping[key] = "train" if i < 128 else "validation" if i < 144 else "test"
    if {k: list(mapping.values()).count(k) for k in ("train", "validation", "test")} != {
            "train": 295, "validation": 36, "test": 36}:
        raise ValueError("invalid frozen parent counts")
    return mapping

def execute(request, old_data, old_report, old_splits, new_data, new_report, generation5, out):
    validate_request(json.loads(Path(request).read_text(encoding="utf-8")))
    for path, digest in ((old_data, OLD_SHA), (old_report, OLD_REPORT_SHA),
                         (old_splits, OLD_SPLITS_SHA), (new_data, NEW_SHA),
                         (new_report, NEW_REPORT_SHA), (generation5, G5_SHA)):
        if sha(path) != digest:
            raise ValueError("frozen source SHA mismatch: " + str(path))
    prior = json.loads(Path(old_report).read_text(encoding="utf-8"))
    recent = json.loads(Path(new_report).read_text(encoding="utf-8"))
    if (prior.get("merged_dataset_sha256") != OLD_SHA or
        prior.get("g5_model_sha256") != G5_SHA or
        prior.get("verified_mate_games") != 222 or
        prior.get("independent_parent_groups") != 207 or
        prior.get("old_parent_assignments_preserved") is not True or
        recent.get("source_run") != G5_RUN or
        recent.get("model_sha256") != G5_SHA or
        recent.get("verified_mates_sha256") != NEW_SHA or
        recent.get("all_games_sha256") != ALL_GAMES_SHA or
        recent.get("games") != 256 or
        recent.get("accepted_unique_mates") != 160 or
        recent.get("independent_parent_groups") != 160 or
        recent.get("terminal_counts") != {"checkmate": 160, "move_limit_unresolved": 96} or
        recent.get("legal_replay_verified") is not True):
        raise ValueError("frozen dataset provenance mismatch")
    old = [json.loads(s) for s in Path(old_data).read_text(encoding="utf-8").splitlines() if s.strip()]
    new = [json.loads(s) for s in Path(new_data).read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(old) != 222 or len(new) != 160:
        raise ValueError("wrong mate count")
    mapping = frozen_assignments(
        json.loads(Path(old_splits).read_text(encoding="utf-8")),
        {g["parent_id"] for g in old}, {g["parent_id"] for g in new})
    games = old + new
    if len({(g["start_sfen"], tuple(g["moves"])) for g in games}) != 382:
        raise ValueError("duplicate or overlapping games")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    merged = root / "merged_verified_mates.jsonl"
    merged.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in games),
                      encoding="utf-8")
    # The strict learner replays every move, checks actual mate and winner.
    rows = load_games(merged, feature_fn=relative_features)
    if len(rows) != 222 or {r[0] for r in rows} != set(mapping):
        raise ValueError("strict learner rejected a source game")
    splits = {k: [] for k in ("train", "validation", "test")}
    old_parts = {k: [] for k in splits}
    new_parts = {k: [] for k in splits}
    for i, (parent_id, x, y) in enumerate(rows):
        part = mapping[parent_id]
        splits[part].append((x, y))
        (old_parts if i < 222 else new_parts)[part].append((x, y))
    if {k: len(v) for k, v in splits.items()} != {
            "train": 307, "validation": 36, "test": 39}:
        raise ValueError("unexpected split game counts")
    g5 = read_weights(generation5, G5_SHA)
    # No hyperparameter search and no use of validation/test to select epochs.
    g6 = incremental_fit(splits["train"], g5, epochs=8, lr=0.02, seed=42)
    model = root / "mover-relative-generation6-research.npz"
    np.savez_compressed(model, weight=g6)
    (root / "frozen_parent_splits.json").write_text(
        json.dumps(mapping, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "generation6_research_candidate_not_promoted",
        "source_runs": [G5_RUN, NEW_RUN],
        "old_dataset_sha256": OLD_SHA, "new_dataset_sha256": NEW_SHA,
        "merged_dataset_sha256": sha(merged),
        "g5_model_sha256": G5_SHA, "g6_model_sha256": sha(model),
        "verified_mate_games": len(rows), "independent_parent_groups": len(mapping),
        "old_parent_assignments_preserved": True,
        "new_parent_groups": {"train": 128, "validation": 16, "test": 16},
        "split_parent_groups": {k: list(mapping.values()).count(k) for k in splits},
        "split_games": {k: len(v) for k, v in splits.items()},
        "training": {"epochs": 8, "lr": 0.02, "seed": 42, "l2_toward_g5": 0.001},
        "metrics": {k: {"generation5": score(v, g5), "generation6": score(v, g6)}
                    for k, v in splits.items()},
        "old_holdout_diagnostics": {k: {"generation5": score(old_parts[k], g5),
                                       "generation6": score(old_parts[k], g6)}
                                    for k in ("validation", "test")},
        "new_holdout_diagnostics": {k: {"generation5": score(new_parts[k], g5),
                                       "generation6": score(new_parts[k], g6)}
                                    for k in ("validation", "test")},
        "updated_coefficients": int(np.count_nonzero(g6 != g5)),
        "candidate_adopted": False,
        "limitations": [
            "Every old validation/test parent retains its frozen assignment.",
            "All new validation/test parents are excluded from fitting.",
            "Held-out test metrics are diagnostic, never optimizer selection.",
            "G6 strength requires independent paired G6 vs G1 and G5 games.",
            "No formal promotion and no changes to KUMOJI.",
        ]}
    (root / "generation6_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report

def main():
    p = argparse.ArgumentParser()
    for name in ("request", "old-data", "old-report", "old-splits",
                 "new-data", "new-report", "generation5", "out"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    execute(a.request, a.old_data, a.old_report, a.old_splits,
            a.new_data, a.new_report, a.generation5, a.out)

if __name__ == "__main__":
    main()
