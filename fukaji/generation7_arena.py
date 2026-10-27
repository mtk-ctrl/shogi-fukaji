"""Research-only independent G7-vs-G1 and G7-vs-G6 equal-time arenas.

Both opponents share the same new openings and swapped colors. Never use
these comparison games as training labels or promote any research model.
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
    "g6": "99984388447a26cef9ca770bd3efe0cd38af784aba056395e95f90f5a4d4ff08",
    "g7": "417d18e7b64494c9febdf6d217afbe4e4979e68d04528f16db8b45c0bcde2a0e",
}
MODEL_RUNS = {"g1": 40359123574, "g6": 40429892376, "g7": 40434466661}
OPPONENTS = ("g1", "g6")
SEED, PAIRS, SHARDS, MAX_PLIES = 20273727, 32, 8, 200
FIELDS = ("nodes", "turns", "depth_sum", "incomplete", "elapsed_ms")

def expected_request():
    return {"research_only": True, "training_allowed": False,
            "model_adoption_allowed": False, "model_sha256": MODEL_SHA,
            "model_runs": MODEL_RUNS, "seed": SEED, "pairs_per_comparison": PAIRS,
            "shards": SHARDS, "opening_plies": 6, "max_plies": MAX_PLIES,
            "movetime_ms": 200, "depth": 2, "mix_percent": 25,
            "fast_relative": True, "root_alpha_beta_both": True}

def validate_request(cfg):
    if cfg != expected_request():
        raise ValueError("unapproved frozen G7 paired comparison protocol")
    return cfg

def execute(request, model_paths, out, shard):
    validate_request(json.loads(Path(request).read_text(encoding="utf-8")))
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("invalid shard")
    if set(model_paths) != set(MODEL_SHA):
        raise ValueError("missing frozen generation model")
    for generation, path in model_paths.items():
        if digest(path) != MODEL_SHA[generation]:
            raise ValueError("frozen model SHA mismatch: " + generation)
    models = {k: RelativeValue(path, mix_percent=25) for k, path in model_paths.items()}
    rows = []
    for opponent in OPPONENTS:
        for pair in range(shard, PAIRS, SHARDS):
            head = opening(SEED + pair, 6)
            if len(head) != 6:
                raise ValueError("incomplete independent opening")
            for black in (True, False):
                row = frozen_play(head, black, models["g7"], models[opponent],
                                  max_plies=MAX_PLIES)
                row["stats"] = {
                    ("candidate_" if key.startswith("g3_") else "opponent_") +
                    key.split("_", 1)[1]: value for key, value in row["stats"].items()
                }
                row.update({"opponent": opponent, "pair": pair,
                            "candidate_black": black, "opening": head})
                rows.append(row)
    if len(rows) != 16:
        raise ValueError("incorrect number of games per shard")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    games = root / "paired_games.jsonl"
    games.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                     encoding="utf-8")
    report = {"status": "research_only_not_adopted", "shard": shard,
              "games": 16, "pairs_per_comparison": 4, "seed": SEED,
              "model_sha256": MODEL_SHA, "movetime_ms": 200, "depth": 2,
              "mix_percent": 25, "max_plies": MAX_PLIES,
              "root_alpha_beta_both": True, "fast_relative": True,
              "legal_replay_verified": True, "games_sha256": digest(games)}
    (root / "paired_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report

def aggregate(parts, out):
    reports = sorted(Path(parts).glob("*/paired_report.json"))
    if len(reports) != SHARDS:
        raise ValueError("expected eight independent paired shards")
    seen_shards, seen_pairs, rows = set(), set(), []
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        shard = report.get("shard")
        if (type(shard) is not int or shard not in range(SHARDS) or
                shard in seen_shards or report.get("status") != "research_only_not_adopted" or
                report.get("games") != 16 or report.get("pairs_per_comparison") != 4 or
                report.get("seed") != SEED or report.get("model_sha256") != MODEL_SHA or
                report.get("movetime_ms") != 200 or report.get("depth") != 2 or
                report.get("mix_percent") != 25 or report.get("max_plies") != MAX_PLIES or
                report.get("root_alpha_beta_both") is not True or
                report.get("fast_relative") is not True or
                report.get("legal_replay_verified") is not True):
            raise ValueError("invalid frozen paired shard report")
        seen_shards.add(shard)
        games_path = path.parent / "paired_games.jsonl"
        if digest(games_path) != report.get("games_sha256"):
            raise ValueError("paired shard SHA mismatch")
        part = [json.loads(s) for s in games_path.read_text(
            encoding="utf-8").splitlines() if s.strip()]
        if len(part) != 16:
            raise ValueError("wrong paired shard rows")
        for opponent in OPPONENTS:
            for pair in range(shard, PAIRS, SHARDS):
                key = (opponent, pair)
                games = [r for r in part if r.get("opponent") == opponent
                         and r.get("pair") == pair]
                if (key in seen_pairs or len(games) != 2 or
                        [g.get("candidate_black") for g in games] != [True, False] or
                        games[0].get("opening") != games[1].get("opening") or
                        games[0].get("opening") != opening(SEED + pair, 6)):
                    raise ValueError("missing or unbalanced color-swapped pair")
                seen_pairs.add(key)
                for game in games:
                    replay_check(game["opening"], game["moves"], game)
                    if len(game["opening"]) + len(game["moves"]) > MAX_PLIES:
                        raise ValueError("move cap exceeded")
                    expected = {side + "_" + field
                                for side in ("candidate", "opponent") for field in FIELDS}
                    if set(game.get("stats", {})) != expected:
                        raise ValueError("missing search telemetry")
                    rows.append(game)
    if (seen_pairs != {(opp, i) for opp in OPPONENTS for i in range(PAIRS)} or
            len(rows) != 128):
        raise ValueError("wrong total game count or missing paired opening")
    rows.sort(key=lambda r: (r["opponent"], r["pair"], not r["candidate_black"]))
    results = {}
    for opponent in OPPONENTS:
        group = [r for r in rows if r["opponent"] == opponent]
        wins, pairs = Counter(), Counter()
        for r in group:
            if r["status"] != "checkmate":
                wins["unresolved"] += 1
            elif (r["winner"] == 0) == r["candidate_black"]:
                wins["g7"] += 1
            else:
                wins["opponent"] += 1
        for pair in range(PAIRS):
            games = [r for r in group if r["pair"] == pair]
            cw = sum(r["status"] == "checkmate" and
                     (r["winner"] == 0) == r["candidate_black"] for r in games)
            ow = sum(r["status"] == "checkmate" and
                     (r["winner"] == 0) != r["candidate_black"] for r in games)
            pairs["g7_advantage" if cw > ow else
                  "opponent_advantage" if ow > cw else "equal"] += 1
        results[opponent] = {
            "games": len(group), "pairs": PAIRS,
            "g7_wins": wins["g7"], "opponent_wins": wins["opponent"],
            "unresolved": wins["unresolved"], "pair_comparison": dict(pairs),
            "seed": SEED,
            "stats": {key: sum(r["stats"][key] for r in group)
                      for key in group[0]["stats"]}}
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    games = root / "all_paired_games.jsonl"
    games.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                     encoding="utf-8")
    report = {"status": "research_only_no_adoption", "games": len(rows),
              "models_sha256": MODEL_SHA, "comparisons": results,
              "movetime_ms": 200, "depth": 2, "mix_percent": 25,
              "max_plies": MAX_PLIES, "fast_relative": True,
              "root_alpha_beta_both": True, "legal_replay_verified": True,
              "all_pairs_color_swapped": True, "games_sha256": digest(games),
              "caution": "Independent new seeds; unresolved is not a win; no training or promotion."}
    (root / "aggregate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report

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
