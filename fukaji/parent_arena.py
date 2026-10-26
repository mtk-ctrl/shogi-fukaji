"""Frozen research comparisons of generations 2 and 3 against generation 1.

Two independent, color-swapped 16-pair arenas, same 200ms/depth2 settings.
Only model weights differ. Never promote models or train on this test set.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from .arena import opening
from .direct_arena import digest, replay_check
from .generation_arena import play as frozen_play
from .relative_value import RelativeValue

MODEL_SHA = {
    "g1": "881af10b5bc0cde591e37a9aee883213fa7dd602345b786b9b93d69c7e6ed239",
    "g2": "00a6186dd710c992cb5df2fe96dd42066eef37a6e0f212eecd5931691a20e6b4",
    "g3": "7740cd133fa05b21998109ca35abcccfa1b365f005c7a7407468f397957e74bd",
}
SEEDS = {"g2": 20271827, "g3": 20271927}
PAIRS, SHARDS, MAX_PLIES = 16, 8, 160
FIELDS = ("nodes", "turns", "depth_sum", "incomplete", "elapsed_ms")


def validate_request(cfg):
    expected = {
        "research_only": True, "training_allowed": False,
        "model_adoption_allowed": False, "model_sha256": MODEL_SHA,
        "model_runs": {"g1": 40359123574, "g2": 40404769521, "g3": 40414745515},
        "seeds": SEEDS, "pairs_per_comparison": PAIRS, "shards": SHARDS,
        "opening_plies": 6, "max_plies": MAX_PLIES, "movetime_ms": 200,
        "depth": 2, "mix_percent": 25, "fast_relative": True,
        "root_alpha_beta_both": True,
    }
    if cfg != expected:
        raise ValueError("unexpected independent frozen parent comparison protocol")
    return cfg


def execute(request, model_paths, out, shard):
    validate_request(json.loads(Path(request).read_text(encoding="utf-8")))
    if type(shard) is not int or shard not in range(SHARDS):
        raise ValueError("invalid parent comparison shard")
    for generation, path in model_paths.items():
        if generation not in MODEL_SHA or digest(path) != MODEL_SHA[generation]:
            raise ValueError("frozen model SHA mismatch: " + generation)
    if set(model_paths) != set(MODEL_SHA):
        raise ValueError("missing frozen generation model")
    models = {k: RelativeValue(path, mix_percent=25) for k, path in model_paths.items()}
    rows = []
    for candidate in ("g2", "g3"):
        for pair in (shard, shard + SHARDS):
            head = opening(SEEDS[candidate] + pair, 6)
            if len(head) != 6:
                raise ValueError("incomplete independent opening")
            for black in (True, False):
                # Existing, regression-tested play() uses g3/g2 labels internally;
                # rename both to candidate/parent before storing the result.
                row = frozen_play(head, black, models[candidate], models["g1"],
                                  max_plies=MAX_PLIES)
                row["stats"] = {
                    ("candidate_" if key.startswith("g3_") else "parent_") +
                    key.split("_", 1)[1]: value
                    for key, value in row["stats"].items()
                }
                row.update({"candidate": candidate, "pair": pair,
                            "candidate_black": black, "opening": head})
                rows.append(row)
    if len(rows) != 8:
        raise ValueError("incorrect number of games per shard")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    game_file = root / "parent_games.jsonl"
    game_file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                         encoding="utf-8")
    report = {
        "status": "research_only_not_adopted", "shard": shard, "games": 8,
        "pairs_per_comparison": 2, "seeds": SEEDS, "model_sha256": MODEL_SHA,
        "movetime_ms": 200, "depth": 2, "mix_percent": 25,
        "max_plies": MAX_PLIES, "root_alpha_beta_both": True,
        "fast_relative": True, "legal_replay_verified": True,
        "games_sha256": digest(game_file),
    }
    (root / "parent_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def aggregate(parts, out):
    reports = sorted(Path(parts).glob("*/parent_report.json"))
    if len(reports) != SHARDS:
        raise ValueError("expected exactly eight parent comparison shards")
    seen_shards, rows = set(), []
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        shard = report.get("shard")
        if (type(shard) is not int or shard not in range(SHARDS) or
                shard in seen_shards or report.get("status") != "research_only_not_adopted" or
                report.get("games") != 8 or report.get("pairs_per_comparison") != 2 or
                report.get("seeds") != SEEDS or report.get("model_sha256") != MODEL_SHA or
                report.get("movetime_ms") != 200 or report.get("depth") != 2 or
                report.get("mix_percent") != 25 or report.get("max_plies") != MAX_PLIES or
                report.get("root_alpha_beta_both") is not True or
                report.get("fast_relative") is not True or
                report.get("legal_replay_verified") is not True):
            raise ValueError("invalid parent comparison shard report")
        seen_shards.add(shard)
        game_path = path.parent / "parent_games.jsonl"
        if digest(game_path) != report.get("games_sha256"):
            raise ValueError("parent comparison shard SHA mismatch")
        part_rows = [json.loads(line) for line in game_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if len(part_rows) != 8:
            raise ValueError("wrong shard row count")
        for candidate in ("g2", "g3"):
            for pair in (shard, shard + SHARDS):
                games = [r for r in part_rows if r.get("candidate") == candidate
                         and r.get("pair") == pair]
                if (len(games) != 2 or
                        [g.get("candidate_black") for g in games] != [True, False] or
                        games[0].get("opening") != games[1].get("opening") or
                        games[0].get("opening") != opening(SEEDS[candidate] + pair, 6)):
                    raise ValueError("missing or unbalanced paired openings")
                for game in games:
                    replay_check(game["opening"], game["moves"], game)
                    if len(game["opening"]) + len(game["moves"]) > MAX_PLIES:
                        raise ValueError("move cap exceeded")
                    expected_keys = {side + "_" + field for side in ("candidate", "parent")
                                     for field in FIELDS}
                    if set(game.get("stats", {})) != expected_keys:
                        raise ValueError("missing or extra search telemetry")
                    rows.append(game)
    if len(rows) != 2 * len(SEEDS) * PAIRS:
        raise ValueError("wrong aggregate game count")
    rows.sort(key=lambda r: (r["candidate"], r["pair"], not r["candidate_black"]))
    results = {}
    for candidate in ("g2", "g3"):
        group = [r for r in rows if r["candidate"] == candidate]
        wins = Counter()
        pair_results = Counter()
        for r in group:
            if r["status"] != "checkmate":
                wins["unresolved"] += 1
            elif (r["winner"] == 0) == r["candidate_black"]:
                wins["candidate"] += 1
            else:
                wins["parent"] += 1
        for pair in range(PAIRS):
            games = [r for r in group if r["pair"] == pair]
            cw = sum(r["status"] == "checkmate" and
                     (r["winner"] == 0) == r["candidate_black"] for r in games)
            pw = sum(r["status"] == "checkmate" and
                     (r["winner"] == 0) != r["candidate_black"] for r in games)
            pair_results["candidate_advantage" if cw > pw else
                         "parent_advantage" if pw > cw else "equal"] += 1
        results[candidate] = {
            "games": len(group), "pairs": PAIRS,
            "candidate_wins": wins["candidate"], "parent_wins": wins["parent"],
            "unresolved": wins["unresolved"], "pair_comparison": dict(pair_results),
            "seed": SEEDS[candidate],
            "stats": {key: sum(r["stats"][key] for r in group)
                      for key in group[0]["stats"]},
        }
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    game_file = root / "all_parent_games.jsonl"
    game_file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                         encoding="utf-8")
    result = {
        "status": "research_only_no_adoption", "games": len(rows),
        "models_sha256": MODEL_SHA, "comparisons": results,
        "movetime_ms": 200, "depth": 2, "mix_percent": 25,
        "max_plies": MAX_PLIES, "fast_relative": True,
        "root_alpha_beta_both": True, "legal_replay_verified": True,
        "all_pairs_color_swapped": True, "games_sha256": digest(game_file),
        "caution": "Independent test seeds; no training or formal promotion. Unresolved is not a win.",
    }
    (root / "aggregate_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("play", "aggregate"), required=True)
    p.add_argument("--request")
    for generation in MODEL_SHA:
        p.add_argument("--" + generation)
    p.add_argument("--shard", type=int)
    p.add_argument("--parts")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.mode == "play":
        if args.shard is None or not args.request or any(
                not getattr(args, g) for g in MODEL_SHA):
            p.error("missing frozen play inputs")
        execute(args.request, {g: getattr(args, g) for g in MODEL_SHA},
                args.out, args.shard)
    else:
        if not args.parts:
            p.error("missing aggregate input")
        aggregate(args.parts, args.out)


if __name__ == "__main__":
    main()
