"""Research-only G7 FastMoveList ON/OFF 256-game independent replication.

The frozen G7 model is identical on both sides. Only the optional legal-move
materialization changes. Every opening is played with swapped colors.
Only replay-verified checkmates count as wins. No training or promotion.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import shogi

from .arena import opening
from .direct_arena import digest, replay_check
from .engine import select_move
from .relative_value import RelativeValue

G7_SHA = "417d18e7b64494c9febdf6d217afbe4e4979e68d04528f16db8b45c0bcde2a0e"
G7_RUN = 40434466661
SEED, PAIRS, SHARDS, MAX_PLIES = 20274627, 128, 20, 240
FIELDS = ("nodes", "turns", "depth_sum", "incomplete", "elapsed_ms")


def expected_request():
    return {
        "research_only": True, "training_allowed": False,
        "model_adoption_allowed": False, "model_run": G7_RUN,
        "model_sha256": G7_SHA, "seed": SEED, "pairs": PAIRS,
        "shards": SHARDS, "opening_plies": 6,
        "max_plies": MAX_PLIES, "movetime_ms": 200,
        "depth": 4, "mix_percent": 25, "fast_relative": True,
        "root_alpha_beta_both": True,
        "fast_move_list_candidate": True,
        "fast_move_list_baseline": False,
    }


def validate_request(cfg):
    if cfg != expected_request():
        raise ValueError("unapproved G7 FastMoveList paired protocol")
    return cfg


def play(head, fast_black, model, max_plies=MAX_PLIES):
    board = shogi.Board()
    for word in head:
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal opening")
        board.push(move)
    moves = []
    stats = {side + "_" + field: 0 for side in ("fast", "baseline")
             for field in FIELDS}
    status, winner = "move_limit_unresolved", None
    for _ in range(max_plies - len(head)):
        if board.is_checkmate():
            status, winner = "checkmate", int(not board.turn)
            break
        if board.is_fourfold_repetition():
            status = "repetition_unresolved"
            break
        is_fast = (board.turn == shogi.BLACK) == fast_black
        side = "fast" if is_fast else "baseline"
        start = time.perf_counter()
        word, depth, nodes = select_move(
            board, movetime_ms=200, max_depth=4, model=model,
            fast_relative=True, root_alpha_beta=True,
            fast_move_list=is_fast)
        elapsed = round((time.perf_counter() - start) * 1000, 3)
        stats[side + "_nodes"] += nodes
        stats[side + "_turns"] += 1
        stats[side + "_depth_sum"] += depth
        stats[side + "_incomplete"] += int(depth < 4)
        stats[side + "_elapsed_ms"] += elapsed
        if word == "resign":
            status = "no_move_unresolved"
            break
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal selected move")
        moves.append(word)
        board.push(move)
    else:
        if board.is_checkmate():
            status, winner = "checkmate", int(not board.turn)
        elif board.is_fourfold_repetition():
            status = "repetition_unresolved"
    row = {"status": status, "winner": winner, "moves": moves, "stats": stats}
    replay_check(head, moves, row)
    return row


def execute(request, model_path, out, shard):
    validate_request(json.loads(Path(request).read_text(encoding="utf-8")))
    if type(shard) is not int or shard not in range(SHARDS):
        raise ValueError("invalid shard")
    if digest(model_path) != G7_SHA:
        raise ValueError("frozen G7 model SHA mismatch")
    model = RelativeValue(model_path, mix_percent=25)
    rows = []
    for pair in range(shard, PAIRS, SHARDS):
        head = opening(SEED + pair, 6)
        if len(head) != 6:
            raise ValueError("incomplete independent opening")
        for fast_black in (True, False):
            row = play(head, fast_black, model)
            row.update({"pair": pair, "fast_black": fast_black, "opening": head})
            rows.append(row)
    if len(rows) != 2 * len(range(shard, PAIRS, SHARDS)):
        raise ValueError("incorrect shard game count")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    games = root / "paired_games.jsonl"
    games.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                             for row in rows), encoding="utf-8")
    report = {
        "status": "research_only_not_adopted", "shard": shard,
        "games": len(rows), "pairs": len(rows) // 2, "seed": SEED, "model_sha256": G7_SHA,
        "movetime_ms": 200, "depth": 4, "mix_percent": 25,
        "max_plies": MAX_PLIES, "fast_relative": True,
        "root_alpha_beta_both": True,
        "fast_move_list_candidate": True,
        "fast_move_list_baseline": False,
        "legal_replay_verified": True, "games_sha256": digest(games),
    }
    (root / "paired_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def aggregate(parts, out):
    reports = sorted(Path(parts).glob("*/paired_report.json"))
    if len(reports) != SHARDS:
        raise ValueError("expected twenty independent paired shards")
    seen_shards, seen_pairs, seen_openings, rows = set(), set(), set(), []
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        shard = report.get("shard")
        expected_pairs = len(range(shard, PAIRS, SHARDS)) if type(shard) is int and shard in range(SHARDS) else -1
        if (type(shard) is not int or shard not in range(SHARDS)
                or shard in seen_shards
                or report.get("status") != "research_only_not_adopted"
                or report.get("games") != 2 * expected_pairs or report.get("pairs") != expected_pairs
                or report.get("seed") != SEED
                or report.get("model_sha256") != G7_SHA
                or report.get("movetime_ms") != 200
                or report.get("depth") != 4
                or report.get("mix_percent") != 25
                or report.get("max_plies") != MAX_PLIES
                or report.get("fast_relative") is not True
                or report.get("root_alpha_beta_both") is not True
                or report.get("fast_move_list_candidate") is not True
                or report.get("fast_move_list_baseline") is not False
                or report.get("legal_replay_verified") is not True):
            raise ValueError("invalid frozen paired shard report")
        seen_shards.add(shard)
        games_path = path.parent / "paired_games.jsonl"
        if digest(games_path) != report.get("games_sha256"):
            raise ValueError("paired shard SHA mismatch")
        part = [json.loads(line) for line in games_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if len(part) != 2 * expected_pairs:
            raise ValueError("wrong shard game count")
        for pair in range(shard, PAIRS, SHARDS):
            games = [r for r in part if r.get("pair") == pair]
            head = opening(SEED + pair, 6)
            key = tuple(head)
            if (pair in seen_pairs or len(games) != 2
                    or [g.get("fast_black") for g in games] != [True, False]
                    or len(head) != 6 or key in seen_openings
                    or any(g.get("opening") != head for g in games)):
                raise ValueError("invalid color-swapped independent opening")
            seen_pairs.add(pair)
            seen_openings.add(key)
            for game in games:
                replay_check(head, game["moves"], game)
                if len(head) + len(game["moves"]) > MAX_PLIES:
                    raise ValueError("move cap exceeded")
                expected_stats = {side + "_" + field
                                  for side in ("fast", "baseline")
                                  for field in FIELDS}
                stats = game.get("stats")
                if not isinstance(stats, dict) or set(stats) != expected_stats:
                    raise ValueError("missing search telemetry")
                if any(type(v) not in (int, float) or v < 0
                       for v in stats.values()):
                    raise ValueError("invalid search telemetry")
                turns = stats["fast_turns"] + stats["baseline_turns"]
                if turns not in (len(game["moves"]), len(game["moves"]) + 1):
                    raise ValueError("search turn count inconsistent")
                rows.append(game)
    if seen_pairs != set(range(PAIRS)) or len(rows) != 2 * PAIRS:
        raise ValueError("wrong total game count")
    rows.sort(key=lambda r: (r["pair"], not r["fast_black"]))
    wins, pair_scores = Counter(), Counter()
    for row in rows:
        if row["status"] != "checkmate":
            wins["unresolved"] += 1
        elif (row["winner"] == int(shogi.BLACK)) == row["fast_black"]:
            wins["fast"] += 1
        else:
            wins["baseline"] += 1
    for pair in range(PAIRS):
        group = rows[2 * pair:2 * pair + 2]
        fast = sum(r["status"] == "checkmate" and
                   (r["winner"] == int(shogi.BLACK)) == r["fast_black"]
                   for r in group)
        baseline = sum(r["status"] == "checkmate" and
                       (r["winner"] == int(shogi.BLACK)) != r["fast_black"]
                       for r in group)
        pair_scores["fast_advantage" if fast > baseline else
                    "baseline_advantage" if baseline > fast else "equal"] += 1
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    games = root / "all_paired_games.jsonl"
    games.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                             for row in rows), encoding="utf-8")
    report = {
        "status": "research_only_no_adoption",
        "games": len(rows), "pairs": PAIRS, "seed": SEED,
        "fast_wins": wins["fast"], "baseline_wins": wins["baseline"],
        "unresolved": wins["unresolved"], "pair_comparison": dict(pair_scores),
        "model_sha256": G7_SHA, "model_run": G7_RUN,
        "movetime_ms": 200, "depth": 4, "mix_percent": 25,
        "max_plies": MAX_PLIES, "fast_relative": True,
        "root_alpha_beta_both": True,
        "fast_move_list_candidate": True,
        "fast_move_list_baseline": False,
        "all_pairs_color_swapped": True, "legal_replay_verified": True,
        "stats": {key: sum(row["stats"][key] for row in rows)
                  for key in rows[0]["stats"]},
        "games_sha256": digest(games),
        "caution": "Unresolved games are not wins or draws. No automatic adoption.",
    }
    (root / "aggregate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("play", "aggregate"), required=True)
    parser.add_argument("--request")
    parser.add_argument("--model")
    parser.add_argument("--shard", type=int)
    parser.add_argument("--parts")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "play":
        if args.request is None or args.model is None or args.shard is None:
            parser.error("missing frozen play inputs")
        execute(args.request, args.model, args.out, args.shard)
    else:
        if not args.parts:
            parser.error("missing aggregate input")
        aggregate(args.parts, args.out)


if __name__ == "__main__":
    main()
