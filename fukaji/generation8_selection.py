"""Reproducible G8 analysis candidates from frozen G7 400-game research data.

Research-only. Candidate is NOT a proven blunder. No training, promotion or
engine change. Exact legal replay and board-state dedup require python-shogi.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import shogi
from .direct_arena import replay_check

SEED = "fukaji-g8-selection-v1|20261028|40610843978"
SOURCE_RUN = 40610843978
SOURCE_AGGREGATE_SHA = "4b906197065ee8f50f50dad53b65185d57e04eb78706b39005cf5866402e7508"
G7_MODEL_SHA = "417d18e7b64494c9febdf6d217afbe4e4979e68d04528f16db8b45c0bcde2a0e"
QUOTAS = (("mate_tail", 40), ("mate_middle", 20),
          ("unresolved_late", 20), ("early_mixed", 20))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def group_sort(parent):
    return digest(("fukaji-group-v2|" + parent).encode())


def stable(*parts):
    return digest((SEED + "|" + "|".join(map(str, parts))).encode())


def frozen_splits(games, prior):
    if len(games) != 400 or len({g["game_id"] for g in games}) != 400:
        raise ValueError("invalid 400-game source")
    if len(prior) != 557 or set(prior.values()) != {"train", "validation", "test"}:
        raise ValueError("invalid frozen G7 parent split")
    groups = {g["parent_id"] for g in games}
    if len(groups) != 400 or groups & set(prior):
        raise ValueError("overlapping old/new parent groups")
    mapping = {}
    for status, expected, train_n, val_n in (
        ("checkmate", 258, 206, 26),
        ("move_limit_unresolved", 142, 114, 14),
    ):
        rows = sorted((g for g in games if g["status"] == status),
                      key=lambda g: group_sort(g["parent_id"]))
        if len(rows) != expected:
            raise ValueError("unexpected terminal status counts")
        for i, g in enumerate(rows):
            mapping[g["parent_id"]] = (
                "train" if i < train_n else
                "validation" if i < train_n + val_n else "test")
    if Counter(mapping.values()) != {"train": 320, "validation": 40, "test": 40}:
        raise ValueError("unexpected split counts")
    return mapping


def target_index(g, kind):
    moves = g["opening"] + g["moves"]
    n = len(moves)
    if kind == "mate_tail":
        lo, hi = max(6, n - 20), max(6, n - 2)
    elif kind == "mate_middle":
        lo, hi = 30, min(n - 12, 120)
    elif kind == "unresolved_late":
        lo, hi = 130, min(n - 1, 225)
    else:
        lo, hi = 12, min(n - 10, 70)
    if hi < lo:
        return None
    def rank(i):
        future = moves[i:min(n, i + 4)]
        triggers = sum(2 if "+" in m else 1 if "*" in m else 0 for m in future)
        return (-triggers, stable(g["parent_id"], kind, i))
    return min(range(lo, hi + 1), key=rank)


def verified_position(g, i):
    # Verify the complete game before taking a prefix, not only the prefix.
    replay_check(g["opening"], g["moves"], g)
    full = g["opening"] + g["moves"]
    b = shogi.Board()
    for move in full[:i]:
        m = shogi.Move.from_usi(move)
        if m not in b.legal_moves:
            raise ValueError("illegal candidate history")
        b.push(m)
    # SFEN's final integer is a move counter, not part of the board state.
    state = " ".join(b.sfen().split()[:3])
    return state


def select(games, splits):
    used_parents, used_boards, selected = set(), set(), []
    for kind, quota in QUOTAS:
        eligible = [g for g in games if splits[g["parent_id"]] == "train"
                    and g["parent_id"] not in used_parents
                    and (kind == "early_mixed" or
                         (g["status"] == "checkmate" if kind.startswith("mate_")
                          else g["status"] == "move_limit_unresolved"))]
        eligible.sort(key=lambda g: stable(kind, g["parent_id"]))
        taken = 0
        for g in eligible:
            i = target_index(g, kind)
            if i is None:
                continue
            state = verified_position(g, i)
            if state in used_boards:
                continue
            full = g["opening"] + g["moves"]
            selected.append({
                "candidate_id": len(selected), "game_id": g["game_id"],
                "parent_id": g["parent_id"], "split": "train",
                "reason_category": kind, "terminal_status": g["status"],
                "winner_for_proven_mates_only": g["winner"] if g["status"] == "checkmate" else None,
                "position_before_ply": i + 1, "history_usi": full[:i],
                "position_sfen_state": state,
                "next_played_move_usi": full[i],
                "full_game_sha256": digest(" ".join(full).encode()),
                "analysis_label": "candidate_only_not_proven_error",
            })
            used_parents.add(g["parent_id"])
            used_boards.add(state)
            taken += 1
            if taken == quota:
                break
        if taken != quota:
            raise ValueError("insufficient unique legal train positions for " + kind)
    if len(selected) != 100 or len(used_parents) != 100 or len(used_boards) != 100:
        raise ValueError("candidate dedup failed")
    return selected


def execute(aggregate_zip, old_model_zip, output):
    if digest(Path(aggregate_zip).read_bytes()) != SOURCE_AGGREGATE_SHA:
        raise ValueError("aggregate GitHub artifact SHA mismatch")
    with ZipFile(aggregate_zip) as z:
        report = json.loads(z.read("selfplay_aggregate_report.json"))
        raw = z.read("all_selfplay_games.jsonl")
    if (report["source_run"] != 40434466661 or
        report["model_sha256"] != G7_MODEL_SHA or
        report["games"] != 400 or
        report["all_games_sha256"] != digest(raw) or
        report["terminal_counts"] != {"checkmate": 258, "move_limit_unresolved": 142} or
        report["legal_replay_verified"] is not True):
        raise ValueError("invalid frozen aggregate provenance")
    games = [json.loads(s) for s in raw.decode().splitlines()]
    with ZipFile(old_model_zip) as z:
        old = json.loads(z.read("frozen_parent_splits.json"))
        old_report = json.loads(z.read("generation7_report.json"))
    if old_report["g7_model_sha256"] != G7_MODEL_SHA:
        raise ValueError("wrong G7 frozen parent")
    splits = frozen_splits(games, old)
    selected = select(games, splits)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "analysis_candidates_100.jsonl"
    path.write_text("".join(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n"
                            for x in selected), encoding="utf-8")
    split_path = root / "new_parent_splits_400.json"
    split_path.write_text(json.dumps(splits, sort_keys=True, indent=2) + "\n",
                          encoding="utf-8")
    manifest = {
        "status": "research_only_legal_replayed_candidate_selection_not_trained",
        "source_run": SOURCE_RUN, "source_aggregate_sha256": SOURCE_AGGREGATE_SHA,
        "g7_model_sha256": G7_MODEL_SHA, "selection_seed": SEED,
        "candidate_count": len(selected),
        "candidate_groups": dict(Counter(x["reason_category"] for x in selected)),
        "new_parent_splits": dict(Counter(splits.values())),
        "prior_parent_splits_preserved": len(old),
        "all_candidates_in_train": True,
        "all_selected_board_states_unique": True,
        "candidate_sha256": digest(path.read_bytes()),
        "new_splits_sha256": digest(split_path.read_bytes()),
        "no_training_or_adoption": True,
        "limitations": ["Selection triggers do not establish actual bad moves.",
                        "No deep analysis, G8 training, or playing-strength comparison."],
    }
    (root / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aggregate-zip", required=True)
    p.add_argument("--g7-model-zip", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(json.dumps(execute(a.aggregate_zip, a.g7_model_zip, a.out),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
