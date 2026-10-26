"""Frozen research generation-4 vs generation-3 paired selfplay arena.

Two independently frozen mover-relative value models, equal search/time,
eight parallel lanes, color-swapped openings, legal replay and SHA aggregation.
No automatic generation adoption.
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

G3_SHA = "7740cd133fa05b21998109ca35abcccfa1b365f005c7a7407468f397957e74bd"
G4_SHA = "db79c7d32cd6e03e6f417a461789fa1736ac8eb3fe2159030531cb5303afb2fd"
SEED, PAIRS, SHARDS, MAX_PLIES = 20272327, 32, 8, 160


def validate_request(cfg):
    expected = {
        "research_only": True, "generation3_run": 40414745515,
        "generation4_run": 40418598213,
        "generation3_sha256": G3_SHA, "generation4_sha256": G4_SHA,
        "seed": SEED, "pairs": PAIRS, "shards": SHARDS,
        "opening_plies": 6, "max_plies": MAX_PLIES,
        "movetime_ms": 200, "depth": 2, "mix_percent": 25,
        "fast_relative": True, "root_alpha_beta_both": True,
        "training_allowed": False, "model_adoption_allowed": False,
    }
    if cfg != expected:
        raise ValueError("unapproved generational paired research request")
    return cfg


def play(head, g4_black, g4, g3, max_plies=MAX_PLIES):
    board = shogi.Board()
    for word in head:
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal paired opening")
        board.push(move)
    moves = []
    stats = {side + "_" + metric: 0 for side in ("g4", "g3")
             for metric in ("nodes", "turns", "depth_sum", "incomplete", "elapsed_ms")}
    status, winner = "move_limit_unresolved", None
    for _ in range(max_plies - len(head)):
        if board.is_checkmate():
            status, winner = "checkmate", int(not board.turn)
            break
        if board.is_fourfold_repetition():
            status = "repetition_unresolved"
            break
        is_g4 = (board.turn == shogi.BLACK) == g4_black
        side = "g4" if is_g4 else "g3"
        start = time.perf_counter()
        word, depth, nodes = select_move(
            board, movetime_ms=200, max_depth=2,
            model=g4 if is_g4 else g3,
            fast_relative=True, root_alpha_beta=True)
        elapsed = (time.perf_counter() - start) * 1000
        stats[side + "_nodes"] += nodes
        stats[side + "_turns"] += 1
        stats[side + "_depth_sum"] += depth
        stats[side + "_incomplete"] += int(depth < 2)
        stats[side + "_elapsed_ms"] += round(elapsed, 3)
        if word == "resign":
            status = "no_move_unresolved"
            break
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal generational search move")
        moves.append(word)
        board.push(move)
    else:
        if board.is_checkmate():
            status, winner = "checkmate", int(not board.turn)
    row = {"status": status, "winner": winner, "moves": moves, "stats": stats}
    replay_check(head, moves, row)
    return row


def execute(request, g3_path, g4_path, out, shard):
    validate_request(json.loads(Path(request).read_text(encoding="utf-8")))
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("invalid shard")
    if digest(g3_path) != G3_SHA or digest(g4_path) != G4_SHA:
        raise ValueError("frozen generation model SHA mismatch")
    g3 = RelativeValue(g3_path, mix_percent=25)
    g4 = RelativeValue(g4_path, mix_percent=25)
    rows = []
    for pair in range(PAIRS):
        if pair % SHARDS != shard:
            continue
        head = opening(SEED + pair, 6)
        if len(head) != 6:
            raise ValueError("incomplete paired opening")
        for black in (True, False):
            row = play(head, black, g4, g3)
            row.update({"pair": pair, "g4_black": black, "opening": head})
            rows.append(row)
    if len(rows) != 8:
        raise ValueError("wrong shard game count")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    games_path = root / "generation_games.jsonl"
    games_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                  for row in rows), encoding="utf-8")
    report = {
        "status": "research_only_not_promoted", "shard": shard,
        "pairs": 4, "games": 8, "seed": SEED,
        "generation3_sha256": G3_SHA, "generation4_sha256": G4_SHA,
        "max_plies": MAX_PLIES, "movetime_ms": 200, "depth": 2,
        "mix_percent": 25, "fast_relative": True, "root_alpha_beta_both": True,
        "games_sha256": digest(games_path),
        "legal_replay_verified": True,
    }
    (root / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def aggregate(parts, out):
    reports = sorted(Path(parts).glob("*/generation_report.json"))
    if len(reports) != SHARDS:
        raise ValueError("expected exactly eight generational shards")
    rows, seen_shards, seen_pairs = [], set(), set()
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        shard = report.get("shard")
        if (type(shard) is not int or shard not in range(SHARDS) or
                shard in seen_shards or report.get("status") != "research_only_not_promoted" or
                report.get("generation3_sha256") != G3_SHA or
                report.get("generation4_sha256") != G4_SHA or
                report.get("seed") != SEED or report.get("games") != 8 or
                report.get("pairs") != 4 or report.get("max_plies") != MAX_PLIES or
                report.get("movetime_ms") != 200 or report.get("depth") != 2 or
                report.get("mix_percent") != 25 or report.get("fast_relative") is not True or
                report.get("root_alpha_beta_both") is not True or
                report.get("legal_replay_verified") is not True):
            raise ValueError("invalid generational shard")
        seen_shards.add(shard)
        game_path = path.parent / "generation_games.jsonl"
        if digest(game_path) != report.get("games_sha256"):
            raise ValueError("shard game SHA mismatch")
        shard_rows = [json.loads(x) for x in game_path.read_text(
            encoding="utf-8").splitlines() if x.strip()]
        if len(shard_rows) != 8:
            raise ValueError("shard row count mismatch")
        for pair in (shard + i * SHARDS for i in range(PAIRS // SHARDS)):
            games = [r for r in shard_rows if r.get("pair") == pair]
            if (pair in seen_pairs or len(games) != 2 or
                    games[0].get("g4_black") is not True or
                    games[1].get("g4_black") is not False or
                    games[0].get("opening") != games[1].get("opening") or
                    games[0].get("opening") != opening(SEED + pair, 6)):
                raise ValueError("invalid color-swapped opening pair")
            seen_pairs.add(pair)
            for game in games:
                replay_check(game["opening"], game["moves"], game)
                if len(game["opening"]) + len(game["moves"]) > MAX_PLIES:
                    raise ValueError("move cap exceeded")
                rows.append(game)
    if seen_pairs != set(range(PAIRS)) or len(rows) != 64:
        raise ValueError("missing generational paired games")
    rows.sort(key=lambda x: (x["pair"], not x["g4_black"]))
    wins = Counter()
    for row in rows:
        if row["status"] != "checkmate":
            wins["unresolved"] += 1
        elif (row["winner"] == int(shogi.BLACK)) == row["g4_black"]:
            wins["generation4"] += 1
        else:
            wins["generation3"] += 1
    pairs = Counter()
    for i in range(PAIRS):
        pair_rows = rows[2*i:2*i+2]
        g4_wins = sum(r["status"] == "checkmate" and
                      (r["winner"] == int(shogi.BLACK)) == r["g4_black"]
                      for r in pair_rows)
        g3_wins = sum(r["status"] == "checkmate" and
                      (r["winner"] == int(shogi.BLACK)) != r["g4_black"]
                      for r in pair_rows)
        pairs["g4_advantage" if g4_wins > g3_wins else
              "g3_advantage" if g3_wins > g4_wins else "equal"] += 1
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    games_path = root / "all_generation_games.jsonl"
    games_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                  for row in rows), encoding="utf-8")
    report = {
        "status": "research_only_no_adoption", "games": len(rows), "pairs": PAIRS,
        "generation4_wins": wins["generation4"],
        "generation3_wins": wins["generation3"],
        "unresolved": wins["unresolved"], "pair_comparison": dict(pairs),
        "model_generation4_sha256": G4_SHA, "model_generation3_sha256": G3_SHA,
        "seed": SEED, "movetime_ms": 200, "depth": 2,
        "mix_percent": 25, "max_plies": MAX_PLIES,
        "fast_relative": True, "root_alpha_beta_both": True,
        "all_pairs_color_swapped": True, "legal_replay_verified": True,
        "stats": {key: sum(row["stats"][key] for row in rows)
                  for key in rows[0]["stats"]},
        "games_sha256": digest(games_path),
        "caution": "Unresolved is not a win or a draw; exploratory comparison only.",
    }
    (root / "aggregate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("play", "aggregate"), required=True)
    p.add_argument("--request")
    p.add_argument("--generation3")
    p.add_argument("--generation4")
    p.add_argument("--shard", type=int)
    p.add_argument("--parts")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.mode == "play":
        if not all((a.request, a.generation3, a.generation4)) or a.shard is None:
            p.error("missing play inputs")
        execute(a.request, a.generation3, a.generation4, a.out, a.shard)
    else:
        if not a.parts:
            p.error("missing aggregate inputs")
        aggregate(a.parts, a.out)


if __name__ == "__main__":
    main()
