"""Reproduce a metadata-only, training-parent-only G8 sampling audit.

Input legality is inherited from the SHA-fixed previously replay-verified
artifact. This script does not independently replay chess, train any model,
or inspect validation/test outcomes or moves after parent filtering.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from fukaji.nnue_sampling import paired_sample_indices

ZIP_SHA = "9b5c004446f4b93a529d52fa96243ccbd5ac434373fe1ff69191bee9c8e6306e"
GAMES_SHA = "115079770f7677ba22d7fc8603a08eaa52bfd21af0374cf3e7862a553f8dddd2"


def audit(path):
    raw_zip = Path(path).read_bytes()
    if hashlib.sha256(raw_zip).hexdigest() != ZIP_SHA:
        raise ValueError("source ZIP SHA mismatch")
    with ZipFile(path) as z:
        mapping = json.loads(z.read("frozen_parent_splits.json"))
        raw = z.read("merged_verified_mates.jsonl")
    if hashlib.sha256(raw).hexdigest() != GAMES_SHA:
        raise ValueError("source games SHA mismatch")
    games = positions = wins = 0
    summaries = {k: Counter() for k in ("original", "adjacent_pairs")}
    histograms = {k: Counter() for k in summaries}
    for line in raw.splitlines():
        g = json.loads(line)
        if mapping.get(g["parent_id"]) != "train":
            continue
        if g["source"] != "selfplay" or type(g["winner"]) is not int or g["winner"] not in (0, 1):
            raise ValueError("invalid verified training source")
        n = len(g["moves"])
        turn = {"b": 0, "w": 1}[g["start_sfen"].split()[1]]
        labels = ((np.arange(n) + turn) % 2 == g["winner"]).astype(int)
        if n < 2 or labels[-1] != 1:
            raise ValueError("mate winner parity mismatch")
        games += 1
        positions += n
        wins += int(labels.sum())
        choices = {"original": np.linspace(0, n - 1, min(16, n), dtype=np.int64),
                   "adjacent_pairs": paired_sample_indices(n)}
        for name, ix in choices.items():
            c = summaries[name]
            winning = int(labels[ix].sum())
            c["sampled_positions"] += len(ix)
            c["winner_to_move_labels"] += winning
            c["loser_to_move_labels"] += len(ix) - winning
            c["games_with_last_winner_turn"] += int(n - 1 in ix)
            c["games_with_last_loser_turn"] += int(n - 2 in ix)
            c["games_with_single_label_only"] += int(winning in (0, len(ix)))
            histograms[name][winning] += 1
    if (games, positions, wins) != (665, 97137, 48731):
        raise ValueError("frozen train count mismatch")
    return {"source_zip_sha256": ZIP_SHA, "source_games_sha256": GAMES_SHA,
            "train_games": games, "train_positions": positions,
            "all_positions_winner_to_move_labels": wins,
            "all_positions_winner_label_rate": wins / positions,
            "samplers": {k: {**dict(v),
                             "winner_label_rate": v["winner_to_move_labels"] / v["sampled_positions"],
                             "winner_labels_per_game_histogram": dict(sorted(histograms[k].items()))}
                         for k, v in summaries.items()},
            "model_trained": False, "strength_proven": False,
            "holdout_outcomes_analyzed": False,
            "limitations": ["Metadata parity audit, not independent legal replay.",
                            "Observed sampling imbalance is not a causal explanation of NNUE loss.",
                            "Adjacent pairs are an unadopted candidate; frozen trainer is unchanged."]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    report = audit(a.source)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
