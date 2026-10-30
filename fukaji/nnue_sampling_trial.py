"""Frozen sampling-only trial with future G8 selfplay reserved for evaluation.

No old holdout evaluation, external teacher, search integration or adoption.
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
import shogi

from .arena import opening
from .direct_arena import replay_check
from .g8_artifact_selection import artifact_bytes
from .generation_arena import play
from .generation8_train import safe_read
from .learning import iter_games
from .nnue_research import SmallNNUE
from .nnue_sampling import paired_sample_indices
from .nnue_train_research import (
    SOURCE_RUN, SOURCE_ID, SOURCE_NAME, SOURCE_ZIP_SHA, MERGED_SHA,
    EPOCHS, LEARNING_RATE, SEED, WEIGHT_DECAY, INPUTS, fit_train_only, loss,
)
from .relative_value import RelativeValue, relative_features

G8_SHA = "18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492"
FRESH_SEED = 202610300
GAMES, SHARDS, MAX_PLIES = 400, 20, 240
MODEL_FILES = {"original": "original-nnue.npz", "adjacent_pairs": "paired-nnue.npz"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def expected_request(enabled=False):
    return {"enabled": enabled, "research_only": True,
            "model_adoption_allowed": False, "engine_integration_allowed": False,
            "old_holdout_evaluation_allowed": False, "fresh_games_training_allowed": False,
            "source_run": SOURCE_RUN, "source_zip_sha256": SOURCE_ZIP_SHA,
            "merged_games_sha256": MERGED_SHA, "g8_model_sha256": G8_SHA,
            "train_games": 665, "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
            "training_seed": SEED, "weight_decay": WEIGHT_DECAY,
            "sample_positions_per_game": 16,
            "samplers": ["original", "adjacent_pairs"],
            "fresh_seed": FRESH_SEED, "evaluation_games": GAMES, "shards": SHARDS,
            "opening_plies": 6, "max_plies": MAX_PLIES, "movetime_ms": 200,
            "depth": 2, "mix_percent": 25, "fast_relative": True,
            "root_alpha_beta_both": True, "fresh_teacher": "verified_checkmates_only"}


def validate_request(cfg):
    # JSON comparison also distinguishes booleans from numeric 0/1.
    actual = json.dumps(cfg, sort_keys=True)
    for enabled in (False, True):
        if actual == json.dumps(expected_request(enabled), sort_keys=True):
            return enabled
    raise ValueError("sampling trial differs from preregistered protocol")


def train_only_bytes(raw, mapping):
    selected = []
    for line in raw.splitlines():
        game = json.loads(line)
        if mapping.get(game["parent_id"]) == "train":
            selected.append(line)
    return b"\n".join(selected) + b"\n"


def save_model(path, model):
    np.savez_compressed(path, hidden_weights=model.hidden_weights,
                        hidden_bias=model.hidden_bias,
                        output_weights=model.output_weights,
                        output_bias=np.array(model.output_bias, dtype=np.float32))


def load_model(path):
    with np.load(path, allow_pickle=False) as a:
        return SmallNNUE(a["hidden_weights"], a["hidden_bias"],
                         a["output_weights"], a["output_bias"])


def read_receipt(root):
    root = Path(root)
    receipt = json.loads((root / "training_receipt.json").read_text())
    if receipt.get("protocol") != expected_request(True):
        raise ValueError("input protocol mismatch")
    expected_names = set(MODEL_FILES.values()) | {"g8.npz"}
    if set(receipt.get("model_sha256", {})) != expected_names:
        raise ValueError("input model inventory mismatch")
    for name, digest in receipt["model_sha256"].items():
        if sha((root / name).read_bytes()) != digest:
            raise ValueError("input model SHA mismatch")
    if receipt["model_sha256"]["g8.npz"] != G8_SHA:
        raise ValueError("G8 model changed")
    return receipt


def prepare(out):
    raw = artifact_bytes(SOURCE_ID, SOURCE_NAME, SOURCE_RUN, SOURCE_ZIP_SHA,
                         os.environ["GH_TOKEN"])
    with ZipFile(io.BytesIO(raw)) as z:
        games = safe_read(z, "merged_verified_mates.jsonl")
        mapping = json.loads(safe_read(z, "frozen_parent_splits.json"))
        g8 = safe_read(z, "mover-relative-generation8-research.npz")
    if sha(games) != MERGED_SHA or sha(g8) != G8_SHA:
        raise ValueError("frozen source bytes changed")
    if Counter(mapping.values()) != {"train": 767, "validation": 95, "test": 95}:
        raise ValueError("frozen parent mapping changed")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "g8.npz").write_bytes(g8)
    metrics, training = {}, {}
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "train_only.jsonl"
        path.write_bytes(train_only_bytes(games, mapping))
        rows = {key: [] for key in MODEL_FILES}
        for group, x, y in iter_games(path, feature_fn=relative_features):
            if mapping[group] != "train":
                raise ValueError("holdout entered training")
            for key in rows:
                ix = (np.linspace(0, len(y) - 1, min(16, len(y)), dtype=np.int64)
                      if key == "original" else paired_sample_indices(len(y)))
                rows[key].append((x[ix], y[ix]))
        if any(len(part) != 665 or sum(len(y) for _, y in part) != 10640
               for part in rows.values()):
            raise ValueError("train sample counts changed")
        for key, part in rows.items():
            model, stats = fit_train_only(part)
            if stats["optimizer_steps"] != 2660 or stats["sampled_positions_total"] != 42560:
                raise ValueError("training counters changed")
            training[key] = stats
            metrics[key] = {"sampled": loss(part, model), "all_train": loss(
                ((x, y) for _, x, y in iter_games(path, feature_fn=relative_features)), model)}
            save_model(root / MODEL_FILES[key], model)
    old_rows = [json.loads(x) for x in games.splitlines()]
    # Old records are used only to reject reused openings/full games. Old
    # validation/test outcomes and features are never evaluated or tuned on.
    excluded_openings = sorted({sha(" ".join(g["moves"][:6]).encode()) for g in old_rows})
    fresh_openings = set()
    for i in range(GAMES):
        head = opening(FRESH_SEED + i, 6)
        opening_sha = sha(" ".join(head).encode())
        if (len(head) != 6 or opening_sha in excluded_openings or
                opening_sha in fresh_openings):
            raise ValueError("new evaluation opening overlaps old source")
        fresh_openings.add(opening_sha)
    receipt = {"protocol": expected_request(True), "training": training,
               "train_metrics": metrics,
               "model_sha256": {name: sha((root / name).read_bytes())
                                 for name in [*MODEL_FILES.values(), "g8.npz"]},
               "excluded_opening_sha256": excluded_openings,
               "old_game_sha256": sorted({sha(" ".join(g["moves"]).encode()) for g in old_rows}),
               "old_holdout_evaluated": False, "model_promoted": False,
               "fresh_games_used_for_training": False}
    (root / "training_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"train_metrics": metrics, "training": training,
                      "old_holdout_evaluated": False}), flush=True)
    return receipt


def play_shard(input_dir, out, shard):
    receipt = read_receipt(input_dir)
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("invalid shard")
    model = RelativeValue(str(Path(input_dir) / "g8.npz"), mix_percent=25)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "evaluation_games.jsonl"
    with path.open("w") as f:
        for i in range(shard * 20, (shard + 1) * 20):
            head = opening(FRESH_SEED + i, 6)
            if sha(" ".join(head).encode()) in receipt["excluded_opening_sha256"]:
                raise ValueError("old opening reused")
            row = play(head, True, model, model, max_plies=MAX_PLIES)
            row.update({"game_id": i, "opening": head,
                        "parent_id": sha(("nnue-sampling-eval-v1|" + str(FRESH_SEED + i) +
                                          "|" + " ".join(head)).encode())})
            replay_check(head, row["moves"], row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print("completed", i + 1 - shard * 20, "/20", row["status"], flush=True)
    report = {"protocol": expected_request(True), "shard": shard, "games": 20,
              "games_sha256": sha(path.read_bytes()),
              "input_receipt_sha256": sha((Path(input_dir) / "training_receipt.json").read_bytes())}
    (root / "evaluation_shard_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def aggregate(input_dir, parts, out):
    receipt = read_receipt(input_dir)
    reports = sorted(Path(parts).glob("*/evaluation_shard_report.json"))
    if len(reports) != SHARDS:
        raise ValueError("expected exactly twenty evaluation shards")
    rows, seen = [], set()
    input_sha = sha((Path(input_dir) / "training_receipt.json").read_bytes())
    for path in reports:
        r = json.loads(path.read_text())
        shard = r.get("shard")
        if (type(shard) is not int or not 0 <= shard < SHARDS or shard in seen or
                r.get("protocol") != expected_request(True) or r.get("games") != 20 or
                r.get("input_receipt_sha256") != input_sha):
            raise ValueError("evaluation shard identity mismatch")
        seen.add(shard)
        game_path = path.parent / "evaluation_games.jsonl"
        if sha(game_path.read_bytes()) != r.get("games_sha256"):
            raise ValueError("evaluation shard bytes changed")
        shard_rows = [json.loads(x) for x in game_path.read_text().splitlines()]
        if [x.get("game_id") for x in shard_rows] != list(range(shard * 20, (shard + 1) * 20)):
            raise ValueError("missing or duplicate evaluation games")
        for row in shard_rows:
            head = opening(FRESH_SEED + row["game_id"], 6)
            parent = sha(("nnue-sampling-eval-v1|" + str(FRESH_SEED + row["game_id"]) +
                          "|" + " ".join(head)).encode())
            if row.get("opening") != head or row.get("parent_id") != parent:
                raise ValueError("evaluation opening changed")
            if sha(" ".join(head).encode()) in receipt["excluded_opening_sha256"]:
                raise ValueError("evaluation opening reused")
            replay_check(head, row["moves"], row)
            if len(head) + len(row["moves"]) > MAX_PLIES:
                raise ValueError("evaluation move cap exceeded")
            if sha(" ".join(head + row["moves"]).encode()) in receipt["old_game_sha256"]:
                raise ValueError("old game reused for independent evaluation")
            rows.append(row)
    rows.sort(key=lambda r: r["game_id"])
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    all_path = root / "all_evaluation_games.jsonl"
    all_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    mate_path = root / "verified_evaluation_mates.jsonl"
    mates = [{"start_sfen": shogi.Board().sfen(), "moves": r["opening"] + r["moves"],
              "winner": r["winner"], "source": "selfplay", "parent_id": r["parent_id"]}
             for r in rows if r["status"] == "checkmate"]
    mate_path.write_text("".join(json.dumps(r) + "\n" for r in mates))
    models = {key: load_model(Path(input_dir) / name) for key, name in MODEL_FILES.items()}
    with np.load(Path(input_dir) / "g8.npz", allow_pickle=False) as a:
        g8_weight = a["weight"]
    totals = {key: 0.0 for key in [*models, "g8"]}
    games = positions = 0
    for _, x, y in iter_games(mate_path, feature_fn=relative_features):
        games += 1
        positions += len(y)
        for key, model in models.items():
            totals[key] += loss([(x, y)], model)["log_loss"] * len(y)
        logits = x @ g8_weight
        totals["g8"] += float((np.logaddexp(0, logits) - y * logits).sum())
    if games != len(mates):
        raise ValueError("strict learner rejected evaluation mates")
    report = {"protocol": expected_request(True), "games": len(rows),
              "verified_checkmates": games, "evaluated_positions": positions,
              "terminal_counts": dict(Counter(r["status"] for r in rows)),
              "log_loss": {key: total / positions if positions else None for key, total in totals.items()},
              "all_games_sha256": sha(all_path.read_bytes()),
              "verified_mates_sha256": sha(mate_path.read_bytes()),
              "model_sha256": receipt["model_sha256"], "old_holdout_evaluated": False,
              "fresh_games_used_for_training": False, "model_promoted": False,
              "strength_proven": False,
              "limitations": ["G8 selfplay outcomes are a prediction proxy, not position truth.",
                              "Unresolved games are retained but never given outcome labels.",
                              "No NNUE search integration or strength matches were performed.",
                              "These 400 parent seeds are evaluation-only in all later research."]}
    (root / "evaluation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("check", "prepare", "play", "aggregate"), required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--input")
    p.add_argument("--parts")
    p.add_argument("--shard", type=int)
    p.add_argument("--out")
    a = p.parse_args()
    enabled = validate_request(json.loads(Path(a.request).read_text()))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("enabled=" + str(enabled).lower() + "\n")
    if a.mode == "check":
        print("sampling trial enabled:", enabled)
        return
    if not enabled or os.environ.get("GITHUB_REPOSITORY") != "mtk-ctrl/shogi-fukaji":
        raise ValueError("disabled trial or wrong repository")
    if a.mode == "prepare":
        prepare(a.out)
    elif a.mode == "play":
        play_shard(a.input, a.out, a.shard)
    else:
        aggregate(a.input, a.parts, a.out)


if __name__ == "__main__":
    main()
