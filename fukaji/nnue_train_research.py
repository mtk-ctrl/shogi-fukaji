"""Frozen-parent, self-play-only small NNUE research trainer (never auto-adopt).

No trained weights are used by the current USI engine. The training entry point
is explicitly opt-in; it reads a SHA-verified G8 artifact and uses only the
original 665 train checkmates. Validation/test are diagnostics, not tuning.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from .g8_artifact_selection import artifact_bytes
from .generation8_train import safe_read
from .learning import load_games
from .relative_value import relative_features
from .nnue_research import HIDDEN, INPUTS, SmallNNUE

SOURCE_RUN = 40636602223
SOURCE_ID = 12793956048
SOURCE_NAME = "fukaji-generation8-research-40636602223"
SOURCE_ZIP_SHA = "9b5c004446f4b93a529d52fa96243ccbd5ac434373fe1ff69191bee9c8e6306e"
MERGED_SHA = "115079770f7677ba22d7fc8603a08eaa52bfd21af0374cf3e7862a553f8dddd2"
EPOCHS = 4
LEARNING_RATE = 0.005
SEED = 20261028
MAX_POSITIONS_PER_GAME = 16
WEIGHT_DECAY = 0.0001


def sha(data):
    return hashlib.sha256(data).hexdigest()


def expected_request(enabled=False):
    return {
        "enabled": bool(enabled), "research_only": True,
        "model_adoption_allowed": False, "engine_integration_allowed": False,
        "source_run": SOURCE_RUN, "source_artifact_id": SOURCE_ID,
        "source_zip_sha256": SOURCE_ZIP_SHA,
        "merged_games_sha256": MERGED_SHA,
        "train_games": 665, "validation_games": 81, "test_games": 84,
        "unresolved_used_as_teacher": False,
        "holdout_used_for_training": False,
        "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
        "seed": SEED, "max_positions_per_game": MAX_POSITIONS_PER_GAME,
        "weight_decay": WEIGHT_DECAY,
    }


def validate_request(cfg):
    if cfg == expected_request(False):
        return False
    if cfg == expected_request(True):
        return True
    raise ValueError("unapproved NNUE research training request")


def frozen_parts(rows, mapping):
    if (len(mapping) != 957 or Counter(mapping.values()) !=
            {"train": 767, "validation": 95, "test": 95}):
        raise ValueError("G8 frozen parent group split changed")
    parts = {key: [] for key in ("train", "validation", "test")}
    for group, x, y in rows:
        if group not in mapping or x.shape[1] != INPUTS + 1 or len(x) != len(y):
            raise ValueError("invalid parent, feature layout or labels")
        parts[mapping[group]].append((x, y))
    if {key: len(value) for key, value in parts.items()} != {
            "train": 665, "validation": 81, "test": 84}:
        raise ValueError("frozen mate split counts changed")
    return parts


def fit_train_only(train, *, epochs=EPOCHS, learning_rate=LEARNING_RATE,
                   seed=SEED, max_positions=MAX_POSITIONS_PER_GAME,
                   weight_decay=WEIGHT_DECAY):
    """Train only from the supplied train rows; no holdout is accepted."""
    if not train or not (1 <= epochs <= 20 and 0 < learning_rate <= 0.05 and
                     1 <= max_positions <= 64 and 0 <= weight_decay <= 0.01):
        raise ValueError("invalid training data or parameters")
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, (INPUTS, HIDDEN)).astype(np.float32)
    b = np.zeros(HIDDEN, dtype=np.float32)
    v = rng.normal(0, 0.01, HIDDEN).astype(np.float32)
    out = np.float32(0.0)
    steps = 0
    positions = 0
    for _ in range(epochs):
        for index in rng.permutation(len(train)):
            x, y = train[index]
            if (x.ndim != 2 or x.shape[1] != INPUTS + 1 or
                    y.ndim != 1 or len(x) != len(y) or len(y) == 0 or
                    not np.isfinite(x).all() or not np.isfinite(y).all() or
                    not np.isin(y, [0.0, 1.0]).all()):
                raise ValueError("invalid train-only feature/label matrix")
            selected = np.linspace(0, len(y) - 1,
                                   min(max_positions, len(y)), dtype=np.int64)
            batch = x[selected, :INPUTS].astype(np.float32, copy=False)
            labels = y[selected].astype(np.float32, copy=False)
            pre = batch @ w + b
            hidden = np.clip(pre, 0.0, 1.0)
            logits = hidden @ v + out
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
            error = probabilities - labels
            n = len(labels)
            grad_v = hidden.T @ error / n + weight_decay * v
            grad_out = np.float32(error.mean())
            grad_hidden = (error[:, None] * v[None, :] *
                           ((pre > 0) & (pre < 1)))
            grad_w = batch.T @ grad_hidden / n + weight_decay * w
            grad_b = grad_hidden.mean(axis=0)
            w -= np.float32(learning_rate) * grad_w
            b -= np.float32(learning_rate) * grad_b
            v -= np.float32(learning_rate) * grad_v
            out -= np.float32(learning_rate) * grad_out
            steps += 1
            positions += n
    model = SmallNNUE(w, b, v, float(out))
    return model, {"epochs": epochs, "optimizer_steps": steps,
                   "sampled_positions_total": positions,
                   "max_positions_per_game_per_epoch": max_positions}


def loss(part, model):
    total = 0.0
    samples = 0
    for x, y in part:
        if x.shape[1] != INPUTS + 1 or len(x) != len(y):
            raise ValueError("invalid evaluation rows")
        for start in range(0, len(y), 128):
            batch = x[start:start + 128, :INPUTS]
            labels = y[start:start + 128]
            hidden = np.clip(batch @ model.hidden_weights + model.hidden_bias,
                             0.0, 1.0)
            logits = hidden @ model.output_weights + model.output_bias
            total += float((np.logaddexp(0, logits) - labels * logits).sum())
            samples += len(labels)
    return {"games": len(part), "positions": samples,
            "log_loss": total / samples if samples else None}


def run(request_path, out_dir=None):
    enabled = validate_request(json.loads(Path(request_path).read_text(encoding="utf-8")))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write("enabled=" + str(enabled).lower() + "\n")
    if out_dir is None:
        return {"enabled": enabled, "training_executed": False}
    if not enabled:
        raise ValueError("NNUE research training is disabled")
    if os.environ.get("GITHUB_REPOSITORY") != "mtk-ctrl/shogi-fukaji":
        raise ValueError("wrong repository")
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise ValueError("missing read token")
    raw = artifact_bytes(SOURCE_ID, SOURCE_NAME, SOURCE_RUN, SOURCE_ZIP_SHA, token)
    with ZipFile(io.BytesIO(raw)) as z:
        games_raw = safe_read(z, "merged_verified_mates.jsonl")
        mapping = json.loads(safe_read(z, "frozen_parent_splits.json"))
        source_report = json.loads(safe_read(z, "generation8_report.json"))
    if (sha(games_raw) != MERGED_SHA or
            source_report.get("merged_verified_mates_sha256") != MERGED_SHA or
            source_report.get("split_mate_games") !=
            {"train": 665, "validation": 81, "test": 84} or
            source_report.get("unresolved_excluded") != 142 or
            source_report.get("model_promoted") is not False):
        raise ValueError("G8 frozen provenance/teacher exclusion mismatch")
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "mates.jsonl"
        path.write_bytes(games_raw)
        rows = load_games(path, feature_fn=relative_features)
    if len(rows) != 830:
        raise ValueError("unexpected verified mate count")
    parts = frozen_parts(rows, mapping)
    model, stats = fit_train_only(parts["train"])
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "small-nnue-research-unadopted.npz"
    np.savez_compressed(model_path, hidden_weights=model.hidden_weights,
                        hidden_bias=model.hidden_bias,
                        output_weights=model.output_weights,
                        output_bias=np.array(model.output_bias, dtype=np.float32))
    report = {
        "status": "research_candidate_not_adopted",
        "source_run": SOURCE_RUN, "source_artifact_id": SOURCE_ID,
        "source_zip_sha256": SOURCE_ZIP_SHA,
        "source_merged_sha256": MERGED_SHA,
        "model_sha256": sha(model_path.read_bytes()),
        "architecture": [INPUTS, HIDDEN, 1],
        "training": {**stats, "lr": LEARNING_RATE, "seed": SEED,
                     "weight_decay": WEIGHT_DECAY, "train_games_only": 665,
                     "validation_test_used_for_tuning": False},
        "metrics": {key: loss(value, model) for key, value in parts.items()},
        "unresolved_teacher_games": 0, "longthink_teacher_games": 0,
        "model_promoted": False, "engine_modified": False,
        "strength_proven": False,
    }
    (root / "nnue_research_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--out")
    args = p.parse_args()
    print(json.dumps(run(args.request, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
