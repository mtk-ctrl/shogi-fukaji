"""Research-only independent generation-5 selfplay dataset, no training/adoption.

256 independently seeded openings, one game per parent group. Uses frozen
generation-5 model on BOTH sides. Only replay-verified mates are eligible
for a future learner; move-limit and repetition remain unresolved.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import shogi

from .arena import opening
from .direct_arena import digest, replay_check
from .generation_arena import play
from .relative_value import RelativeValue

MODEL_SHA = "ff1128d9b139fdf71beae1b49a48506bfdcf3e9cd31e08053341783985d92e03"
SOURCE_RUN = 40424947260
SEED = 20272827
GAMES = 256
SHARDS = 8
MAX_PLIES = 240
FIELDS = ("nodes", "turns", "depth_sum", "incomplete", "elapsed_ms")


def expected_request():
    return {
        "research_only": True, "model_adoption_allowed": False,
        "training_allowed": False, "source_run": SOURCE_RUN,
        "model_sha256": MODEL_SHA, "seed": SEED,
        "games": GAMES, "shards": SHARDS, "opening_plies": 6,
        "max_plies": MAX_PLIES, "movetime_ms": 200, "depth": 2,
        "mix_percent": 25, "fast_relative": True,
        "root_alpha_beta_both": True,
    }


def validate_request(cfg):
    if cfg != expected_request():
        raise ValueError("unapproved frozen generation-5 selfplay protocol")
    return cfg


def parent_id(seed, head):
    return hashlib.sha256(
        ("fukaji-g5-selfplay-v1|" + str(seed) + "|" + " ".join(head)).encode()
    ).hexdigest()


def play_shard(request, model_path, out, shard):
    validate_request(json.loads(Path(request).read_text(encoding="utf-8")))
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("invalid selfplay shard")
    if digest(model_path) != MODEL_SHA:
        raise ValueError("frozen generation-5 model SHA mismatch")
    model = RelativeValue(model_path, mix_percent=25)
    rows = []
    for i in range(shard * (GAMES // SHARDS), (shard + 1) * (GAMES // SHARDS)):
        head = opening(SEED + i, 6)
        if len(head) != 6:
            raise ValueError("incomplete seeded opening")
        # Both seats are the exact same frozen model and search configuration.
        result = play(head, True, model, model, max_plies=MAX_PLIES)
        stats = {field: result["stats"]["g3_" + field] +
                         result["stats"]["g2_" + field] for field in FIELDS}
        row = {"game_id": i, "parent_id": parent_id(SEED + i, head),
               "opening": head, "moves": result["moves"],
               "status": result["status"], "winner": result["winner"],
               "stats": stats}
        replay_check(head, row["moves"], row)
        rows.append(row)
    if len(rows) != GAMES // SHARDS:
        raise ValueError("wrong shard game count")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    games_path = root / "selfplay_games.jsonl"
    games_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                  for r in rows), encoding="utf-8")
    report = {"status": "research_only_no_training_no_adoption",
              "shard": shard, "games": len(rows), "seed": SEED,
              "model_sha256": MODEL_SHA, "max_plies": MAX_PLIES,
              "movetime_ms": 200, "depth": 2, "mix_percent": 25,
              "fast_relative": True, "root_alpha_beta_both": True,
              "legal_replay_verified": True,
              "games_sha256": digest(games_path)}
    (root / "selfplay_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return report


def aggregate(parts, out):
    reports = sorted(Path(parts).glob("*/selfplay_report.json"))
    if len(reports) != SHARDS:
        raise ValueError("expected eight independent selfplay shards")
    seen_shards, rows = set(), []
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        shard = report.get("shard")
        if (type(shard) is not int or shard not in range(SHARDS) or
                shard in seen_shards or
                report.get("status") != "research_only_no_training_no_adoption" or
                report.get("games") != GAMES // SHARDS or
                report.get("seed") != SEED or
                report.get("model_sha256") != MODEL_SHA or
                report.get("max_plies") != MAX_PLIES or
                report.get("movetime_ms") != 200 or
                report.get("depth") != 2 or report.get("mix_percent") != 25 or
                report.get("fast_relative") is not True or
                report.get("root_alpha_beta_both") is not True or
                report.get("legal_replay_verified") is not True):
            raise ValueError("invalid frozen selfplay shard")
        seen_shards.add(shard)
        games_path = path.parent / "selfplay_games.jsonl"
        if digest(games_path) != report.get("games_sha256"):
            raise ValueError("selfplay shard SHA mismatch")
        part = [json.loads(x) for x in games_path.read_text(
            encoding="utf-8").splitlines() if x.strip()]
        if len(part) != GAMES // SHARDS:
            raise ValueError("wrong shard rows")
        for row in part:
            i = row.get("game_id")
            if (type(i) is not int or
                    not shard * (GAMES // SHARDS) <= i < (shard + 1) * (GAMES // SHARDS) or
                    row.get("opening") != opening(SEED + i, 6) or
                    row.get("parent_id") != parent_id(SEED + i, row["opening"]) or
                    set(row.get("stats", {})) != set(FIELDS)):
                raise ValueError("selfplay source identity or telemetry mismatch")
            replay_check(row["opening"], row["moves"], row)
            if len(row["opening"]) + len(row["moves"]) > MAX_PLIES:
                raise ValueError("move cap exceeded")
            rows.append(row)
    rows.sort(key=lambda x: x["game_id"])
    if len(rows) != GAMES or [r["game_id"] for r in rows] != list(range(GAMES)):
        raise ValueError("missing or duplicate selfplay games")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    all_path = root / "all_selfplay_games.jsonl"
    all_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                for r in rows), encoding="utf-8")
    # Training input is a separate file with strictly verified mate labels.
    # Keep all unresolved games in the raw dataset; never invent their winner.
    mates, seen_games = [], set()
    for row in rows:
        if row["status"] != "checkmate":
            continue
        moves = row["opening"] + row["moves"]
        key = hashlib.sha256(" ".join(moves).encode()).hexdigest()
        if key in seen_games:
            continue
        seen_games.add(key)
        mates.append({"start_sfen": shogi.Board().sfen(), "moves": moves,
                      "winner": row["winner"], "source": "selfplay",
                      "parent_id": row["parent_id"]})
    mate_path = root / "verified_mates.jsonl"
    mate_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                 for r in mates), encoding="utf-8")
    # Independently replay every selected training row with the strict learner.
    from .learning import load_games, partition_rows
    verified = load_games(mate_path)
    if len(verified) != len(mates):
        raise ValueError("learner rejected mate dataset")
    _, groups = partition_rows(verified)
    counts = dict(Counter(r["status"] for r in rows))
    report = {"status": "research_only_dataset_not_trained_or_adopted",
              "source_run": SOURCE_RUN, "model_sha256": MODEL_SHA,
              "games": GAMES, "independent_opening_seeds": GAMES,
              "independent_parent_groups": len({r["parent_id"] for r in mates}),
              "terminal_counts": counts,
              "accepted_unique_mates": len(mates),
              "duplicate_mates_removed": counts.get("checkmate", 0) - len(mates),
              "split_parent_groups": groups, "seed": SEED,
              "movetime_ms": 200, "depth": 2, "max_plies": MAX_PLIES,
              "mix_percent": 25, "fast_relative": True,
              "root_alpha_beta_both": True,
              "all_games_sha256": digest(all_path),
              "verified_mates_sha256": digest(mate_path),
              "legal_replay_verified": True,
              "limitations": [
                  "G5-vs-G5 selfplay is training-source research, not strength comparison.",
                  "No unfinished result is labeled a win or a draw.",
                  "Existing held-out parents must remain held out in later training.",
                  "No model is trained or formally adopted by this workflow.",
              ]}
    (root / "selfplay_aggregate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("play", "aggregate"), required=True)
    p.add_argument("--request")
    p.add_argument("--model")
    p.add_argument("--shard", type=int)
    p.add_argument("--parts")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.mode == "play":
        if not a.request or not a.model or a.shard is None:
            p.error("missing play arguments")
        play_shard(a.request, a.model, a.out, a.shard)
    else:
        if not a.parts:
            p.error("missing aggregate parts")
        aggregate(a.parts, a.out)


if __name__ == "__main__":
    main()
