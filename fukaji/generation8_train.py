"""SHA-frozen, parent-disjoint G8 research training. No engine or model promotion.

Only real checkmate results are labeled. The original 557 parent assignments
and all 400 new parent assignments are immutable. Long-think observations are
diagnostics, never teacher moves. No validation/test-driven optimizer tuning.
"""
import argparse
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .g8_artifact_selection import artifact_bytes
from .generation8_selection import (
    G7_MODEL_SHA, OLD_G7_ZIP_SHA, SOURCE_AGGREGATE_SHA,
    frozen_splits, verify_all_games,
)
from .generation3_train import read_weights
from .generation_train import incremental_fit, score
from .learning import load_games
from .relative_value import relative_features

OLD_RUN, OLD_ID = 40434466661, 12702814791
OLD_NAME = "fukaji-generation7-research-40434466661"
NEW_RUN, NEW_ID = 40610843978, 12783855111
NEW_NAME = "fukaji-g7-400-aggregate-40610843978"
SELECTION_RUN, SELECTION_ID = 40624316138, 12788612272
SELECTION_NAME = "fukaji-g8-verified-selection-40624316138"
SELECTION_ZIP_SHA = "5206f1ffbf9cd0e8a71ac65f8d6d2f9828853b408eaf5c3189ed285b8e47db52"
NEW_MATES_SHA = "d8dc9204428ed474180e5caab1c0eeb7ed56f629d221b7a92ed29cc682045014"
ALL_GAMES_SHA = "ab03df8ec3e64d89f32bec0ace74ac77da4e3a95a7085ee2ae13908337031834"
EPOCHS, LR, SEED, SHRINK = 8, 0.02, 42, 0.5
LABEL = "generation8_research_candidate_not_promoted"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def expected_request(enabled=False):
    return {
        "enabled": enabled, "research_only": True,
        "training_allowed": bool(enabled), "model_adoption_allowed": False,
        "old_run": OLD_RUN, "old_artifact_id": OLD_ID,
        "old_artifact_zip_sha256": OLD_G7_ZIP_SHA,
        "selfplay_run": NEW_RUN, "selfplay_artifact_id": NEW_ID,
        "selfplay_artifact_zip_sha256": SOURCE_AGGREGATE_SHA,
        "selection_run": SELECTION_RUN, "selection_artifact_id": SELECTION_ID,
        "selection_artifact_zip_sha256": SELECTION_ZIP_SHA,
        "parent_model_sha256": G7_MODEL_SHA,
        "new_verified_mates_sha256": NEW_MATES_SHA,
        "epochs": EPOCHS, "learning_rate": LR, "seed": SEED,
        "fixed_update_shrinkage": SHRINK,
        "holdout_used_for_training": False,
        "longthink_used_as_teacher": False,
        "unresolved_used_as_teacher": False,
    }


def validate_request(cfg):
    if cfg == expected_request(False):
        return False
    if cfg == expected_request(True):
        return True
    raise ValueError("unapproved G8 training request or hyperparameters")


def safe_read(z, name, limit=16 * 1024 * 1024):
    info = z.getinfo(name)
    if info.is_dir() or info.file_size > limit:
        raise ValueError("unapproved archive member size")
    data = z.read(info)
    if len(data) != info.file_size:
        raise ValueError("archive member truncated")
    return data


def lines(raw):
    return [json.loads(s) for s in raw.decode("utf-8").splitlines() if s.strip()]


def validate_splits(old_map, new_map, old_mates, new_mates, all_games):
    if (len(old_map) != 557 or Counter(old_map.values()) !=
            {"train": 447, "validation": 55, "test": 55}):
        raise ValueError("old 557 parent assignments changed")
    if (len(new_map) != 400 or Counter(new_map.values()) !=
            {"train": 320, "validation": 40, "test": 40}):
        raise ValueError("new 400 parent assignments changed")
    if set(old_map) & set(new_map):
        raise ValueError("prior and new parent groups overlap")
    if (len(old_mates) != 572 or len(new_mates) != 258 or
            len(all_games) != 400 or
            len({g["parent_id"] for g in all_games}) != 400):
        raise ValueError("mate or source game count mismatch")
    if {g["parent_id"] for g in old_mates} != set(old_map):
        raise ValueError("old parent assignments do not cover old mates")
    mate_parents = {g["parent_id"] for g in new_mates}
    if len(mate_parents) != 258 or mate_parents - set(new_map):
        raise ValueError("new mate parents invalid")
    if (Counter(new_map[g["parent_id"]] for g in new_mates) !=
            {"train": 206, "validation": 26, "test": 26}):
        raise ValueError("new checkmate split changed")
    if (Counter(new_map[g["parent_id"]] for g in all_games
                if g["status"] == "move_limit_unresolved") !=
            {"train": 114, "validation": 14, "test": 14}):
        raise ValueError("unresolved split changed")
    if len({(g["start_sfen"], tuple(g["moves"])) for g in old_mates + new_mates}) != 830:
        raise ValueError("exact duplicate or cross-generation replay")
    return dict(old_map, **new_map)


def train(request, out):
    if not validate_request(json.loads(Path(request).read_text(encoding="utf-8"))):
        raise ValueError("disabled G8 research training request")
    if os.environ.get("GITHUB_REPOSITORY") != "mtk-ctrl/shogi-fukaji":
        raise ValueError("wrong repository")
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise ValueError("missing GitHub Actions read token")

    original = artifact_bytes(OLD_ID, OLD_NAME, OLD_RUN, OLD_G7_ZIP_SHA, token)
    recent = artifact_bytes(NEW_ID, NEW_NAME, NEW_RUN, SOURCE_AGGREGATE_SHA, token)
    selected = artifact_bytes(SELECTION_ID, SELECTION_NAME, SELECTION_RUN,
                              SELECTION_ZIP_SHA, token)
    with ZipFile(io.BytesIO(original)) as z:
        old_raw = safe_read(z, "merged_verified_mates.jsonl")
        old_map = json.loads(safe_read(z, "frozen_parent_splits.json"))
        old_report = json.loads(safe_read(z, "generation7_report.json"))
        model_raw = safe_read(z, "mover-relative-generation7-research.npz")
    with ZipFile(io.BytesIO(recent)) as z:
        new_raw = safe_read(z, "verified_mates.jsonl")
        all_raw = safe_read(z, "all_selfplay_games.jsonl")
        new_report = json.loads(safe_read(z, "selfplay_aggregate_report.json"))
    with ZipFile(io.BytesIO(selected)) as z:
        new_map_raw = safe_read(z, "selected/new_parent_splits_400.json")
        new_map = json.loads(new_map_raw)
        selection = json.loads(safe_read(z, "selected/selection_manifest.json"))
        receipt = json.loads(safe_read(z, "verification_receipt.json"))
    if (digest(model_raw) != G7_MODEL_SHA or
            digest(old_raw) != old_report.get("merged_dataset_sha256") or
            old_report.get("g7_model_sha256") != G7_MODEL_SHA or
            old_report.get("verified_mate_games") != 572 or
            old_report.get("independent_parent_groups") != 557 or
            old_report.get("split_games") !=
            {"train": 459, "validation": 55, "test": 58} or
            digest(new_raw) != NEW_MATES_SHA or
            digest(all_raw) != ALL_GAMES_SHA or
            new_report.get("verified_mates_sha256") != NEW_MATES_SHA or
            new_report.get("all_games_sha256") != ALL_GAMES_SHA or
            new_report.get("accepted_unique_mates") != 258 or
            new_report.get("terminal_counts") !=
            {"checkmate": 258, "move_limit_unresolved": 142} or
            new_report.get("legal_replay_verified") is not True or
            selection.get("new_splits_sha256") != digest(new_map_raw) or
            selection.get("all_400_games_full_legal_replay_verified") is not True or
            receipt.get("all_artifact_metadata_and_bytes_verified") is not True or
            receipt.get("training_executed") is not False):
        raise ValueError("frozen G7/G8 dataset, split or provenance mismatch")

    old_mates, new_mates, all_games = lines(old_raw), lines(new_raw), lines(all_raw)
    if (len({g["game_id"] for g in all_games}) != 400 or
            len({g["parent_id"] for g in all_games}) != 400):
        raise ValueError("invalid selfplay IDs or parents")
    if new_map != frozen_splits(all_games, old_map):
        raise ValueError("G8 parent mapping was not reproduced")
    if verify_all_games(all_games) != 72326:
        raise ValueError("source 400 full legal replay differs")
    by_parent = {g["parent_id"]: g for g in all_games if g["status"] == "checkmate"}
    for mate in new_mates:
        src = by_parent.get(mate["parent_id"])
        if (src is None or mate["moves"] != src["opening"] + src["moves"] or
                mate["winner"] != src["winner"] or
                mate["source"] != "selfplay"):
            raise ValueError("mate is not an exact checkmate game from source")
    mapping = validate_splits(old_map, new_map, old_mates, new_mates, all_games)

    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    parent_file = root / "frozen-g7-parent.npz"
    parent_file.write_bytes(model_raw)
    parent = read_weights(parent_file, G7_MODEL_SHA)
    parent_file.unlink()  # No duplicate parent model in output artifact.
    merged = root / "merged_verified_mates.jsonl"
    merged.write_bytes(old_raw + new_raw)
    rows = load_games(merged, feature_fn=relative_features)
    if len(rows) != 830:
        raise ValueError("strict learner rejected or deduplicated games")
    parts = {key: [] for key in ("train", "validation", "test")}
    for group, x, y in rows:
        parts[mapping[group]].append((x, y))
    if {k: len(v) for k, v in parts.items()} != {
            "train": 665, "validation": 81, "test": 84}:
        raise ValueError("training partition leakage or game count mismatch")

    # Frozen G7 optimizer, no epoch/LR search and no holdout-driven selection.
    raw = incremental_fit(parts["train"], parent, epochs=EPOCHS, lr=LR, seed=SEED)
    candidate = (parent + np.float32(SHRINK) * (raw - parent)).astype(np.float32)
    if not np.isfinite(candidate).all():
        raise ValueError("nonfinite G8 candidate")
    model = root / "mover-relative-generation8-research.npz"
    np.savez_compressed(model, weight=candidate)
    (root / "frozen_parent_splits.json").write_text(
        json.dumps(mapping, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": LABEL, "parent_model_sha256": G7_MODEL_SHA,
        "candidate_model_sha256": digest(model.read_bytes()),
        "old_zip_sha256": OLD_G7_ZIP_SHA,
        "new_zip_sha256": SOURCE_AGGREGATE_SHA,
        "selection_zip_sha256": SELECTION_ZIP_SHA,
        "new_verified_mates_sha256": NEW_MATES_SHA,
        "merged_verified_mates_sha256": digest(merged.read_bytes()),
        "verified_mate_games": 830, "new_verified_mates": 258,
        "new_train_mates": 206, "unresolved_excluded": 142,
        "parent_groups": {"old": 557, "new_all": 400, "new_mates": 258},
        "split_parent_groups": dict(Counter(mapping.values())),
        "split_mate_games": {k: len(v) for k, v in parts.items()},
        "training": {"epochs": EPOCHS, "lr": LR, "seed": SEED,
                     "fixed_update_shrinkage": SHRINK,
                     "training_only_split": "train", "no_holdout_tuning": True,
                     "no_longthink_teacher": True},
        "metrics": {k: {"G7": score(v, parent), "G8": score(v, candidate)}
                    for k, v in parts.items()},
        "updated_coefficients": int(np.count_nonzero(candidate != parent)),
        "model_promoted": False, "kumoji_modified": False,
        "strength_proven": False,
        "limitations": [
            "Outcome labels from verified selfplay checkmates are weak move-quality targets.",
            "The 40 long-think changed moves are NOT teacher labels.",
            "Validation/test scores are post-hoc diagnostics, never optimizer selection.",
            "No new validation/test parent or old holdout was fitted.",
            "Independent 240 paired games are needed to evaluate playing strength.",
        ],
    }
    (root / "generation8_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--out")
    a = p.parse_args()
    cfg = json.loads(Path(a.request).read_text(encoding="utf-8"))
    enabled = validate_request(cfg)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write("enabled=" + ("true" if enabled else "false") + "\n")
    if a.out:
        train(a.request, a.out)
    else:
        print("G8 research request valid; enabled:", enabled)


if __name__ == "__main__":
    main()
