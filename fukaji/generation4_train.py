"""G4 research learning; never reassign any G3 held-out parent or promote."""
import argparse
import json
from pathlib import Path
import numpy as np
from .generation3_train import read_weights, group_sort
from .generation_train import incremental_fit, score, sha
from .learning import load_games
from .relative_value import relative_features

G3_RUN, NEW_RUN = 40414745515, 40416658637
G3_SHA = "7740cd133fa05b21998109ca35abcccfa1b365f005c7a7407468f397957e74bd"
OLD_SHA = "f9c4a0fbc7eb26e7a25b614089852bd91b7c0f4c344bb46c604b128964ed8011"
OLD_REPORT_SHA = "5524b03bf0162ddbc74db65bf686da7b8d1632b222615ce735b5414298c078da"
OLD_SPLITS_SHA = "7c921029c0494a4a61b0c6c20ad9f49642737ef6ed51496d1817082701747553"
NEW_SHA = "ede7915c3dd4e05f8cabee02da6177cd18c95850e381f771d8db53a40084be0f"
NEW_REPORT_SHA = "853cfff94e473326244d3c12dfcab3544282adc0c622a989925d0246c070bc78"

def expected_request():
    return {"research_only": True, "training_allowed": True,
            "model_adoption_allowed": False, "source_runs": [G3_RUN, NEW_RUN],
            "preserve_prior_parent_splits": True, "old_test_for_tuning": False,
            "reuse_frozen_optimizer": True, "parent_model_sha256": G3_SHA,
            "new_mates_sha256": NEW_SHA}

def validate_request(cfg):
    if cfg != expected_request():
        raise ValueError("unapproved G4 research protocol")
    return cfg

def frozen_assignments(old_mapping, old_groups, new_groups):
    if (len(old_mapping) != 31 or set(old_mapping) != set(old_groups)
        or {k: list(old_mapping.values()).count(k)
            for k in ("train", "validation", "test")}
        != {"train": 25, "validation": 3, "test": 3}
        or set(old_mapping.values()) != {"train", "validation", "test"}):
        raise ValueError("prior heldout parent assignments changed")
    if len(new_groups) != 72 or set(old_groups) & set(new_groups):
        raise ValueError("new independent parent groups invalid or overlapping")
    mapping = dict(old_mapping)
    for i, key in enumerate(sorted(new_groups, key=group_sort)):
        mapping[key] = "train" if i < 58 else "validation" if i < 65 else "test"
    if {k: list(mapping.values()).count(k) for k in ("train", "validation", "test")} != {
            "train": 83, "validation": 10, "test": 10}:
        raise ValueError("invalid frozen parent counts")
    return mapping

def execute(request, old_data, old_report, old_splits, new_data, new_report, generation3, out):
    validate_request(json.loads(Path(request).read_text(encoding="utf-8")))
    for path, digest in ((old_data, OLD_SHA), (old_report, OLD_REPORT_SHA),
                         (old_splits, OLD_SPLITS_SHA), (new_data, NEW_SHA),
                         (new_report, NEW_REPORT_SHA), (generation3, G3_SHA)):
        if sha(path) != digest:
            raise ValueError("frozen source SHA mismatch: " + str(path))
    prior = json.loads(Path(old_report).read_text(encoding="utf-8"))
    recent = json.loads(Path(new_report).read_text(encoding="utf-8"))
    if (prior.get("merged_dataset_sha256") != OLD_SHA or
        prior.get("g3_model_sha256") != G3_SHA or
        prior.get("verified_mate_games") != 46 or
        prior.get("independent_parent_groups") != 31 or
        prior.get("old_heldout_groups_preserved") is not True or
        recent.get("source_run") != G3_RUN or
        recent.get("model_sha256") != G3_SHA or
        recent.get("verified_mates_sha256") != NEW_SHA or
        recent.get("games") != 128 or
        recent.get("accepted_unique_mates") != 72 or
        recent.get("independent_parent_groups") != 72 or
        recent.get("terminal_counts") != {"checkmate": 72, "move_limit_unresolved": 56} or
        recent.get("legal_replay_verified") is not True):
        raise ValueError("frozen dataset provenance mismatch")
    old = [json.loads(s) for s in Path(old_data).read_text(encoding="utf-8").splitlines() if s.strip()]
    new = [json.loads(s) for s in Path(new_data).read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(old) != 46 or len(new) != 72:
        raise ValueError("wrong mate count")
    mapping = frozen_assignments(
        json.loads(Path(old_splits).read_text(encoding="utf-8")),
        {g["parent_id"] for g in old}, {g["parent_id"] for g in new})
    games = old + new
    if len({(g["start_sfen"], tuple(g["moves"])) for g in games}) != 118:
        raise ValueError("duplicate or overlapping games")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    merged = root / "merged_verified_mates.jsonl"
    merged.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in games),
                      encoding="utf-8")
    # The learner replays every legal move and checks actual mate/winner.
    rows = load_games(merged, feature_fn=relative_features)
    if len(rows) != 118 or {r[0] for r in rows} != set(mapping):
        raise ValueError("strict learner rejected a source game")
    splits = {k: [] for k in ("train", "validation", "test")}
    old_parts = {k: [] for k in splits}
    new_parts = {k: [] for k in splits}
    for i, (parent_id, x, y) in enumerate(rows):
        part = mapping[parent_id]
        splits[part].append((x, y))
        (old_parts if i < 46 else new_parts)[part].append((x, y))
    if {k: len(v) for k, v in splits.items()} != {
            "train": 95, "validation": 10, "test": 13}:
        raise ValueError("unexpected split game counts")
    g3 = read_weights(generation3, G3_SHA)
    g4 = incremental_fit(splits["train"], g3, epochs=8, lr=0.02, seed=42)
    model = root / "mover-relative-generation4-research.npz"
    np.savez_compressed(model, weight=g4)
    (root / "frozen_parent_splits.json").write_text(
        json.dumps(mapping, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "generation4_research_candidate_not_promoted",
        "source_runs": [G3_RUN, NEW_RUN],
        "old_dataset_sha256": OLD_SHA, "new_dataset_sha256": NEW_SHA,
        "merged_dataset_sha256": sha(merged),
        "g3_model_sha256": G3_SHA, "g4_model_sha256": sha(model),
        "verified_mate_games": len(rows), "independent_parent_groups": len(mapping),
        "old_parent_assignments_preserved": True,
        "new_parent_groups": {"train": 58, "validation": 7, "test": 7},
        "split_parent_groups": {k: list(mapping.values()).count(k) for k in splits},
        "split_games": {k: len(v) for k, v in splits.items()},
        "training": {"epochs": 8, "lr": 0.02, "seed": 42, "l2_toward_g3": 0.001},
        "metrics": {k: {"generation3": score(v, g3), "generation4": score(v, g4)}
                    for k, v in splits.items()},
        "old_holdout_diagnostics": {k: {"generation3": score(old_parts[k], g3),
                                       "generation4": score(old_parts[k], g4)}
                                    for k in ("validation", "test")},
        "new_holdout_diagnostics": {k: {"generation3": score(new_parts[k], g3),
                                       "generation4": score(new_parts[k], g4)}
                                    for k in ("validation", "test")},
        "updated_coefficients": int(np.count_nonzero(g4 != g3)),
        "candidate_adopted": False,
        "limitations": [
            "Old and new validation/test parents are excluded from training.",
            "Held-out test metrics are diagnostics, never optimizer selection.",
            "G4 playing strength requires independent paired G4 vs G1/G3 games.",
            "No formal promotion and no changes to KUMOJI.",
        ]}
    (root / "generation4_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report

def main():
    p = argparse.ArgumentParser()
    for name in ("request", "old-data", "old-report", "old-splits",
                 "new-data", "new-report", "generation3", "out"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    execute(a.request, a.old_data, a.old_report, a.old_splits,
            a.new_data, a.new_report, a.generation3, a.out)

if __name__ == "__main__":
    main()
