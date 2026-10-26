"""Frozen-model head-to-head arena, mover-relative versus legacy value.

This is a bounded diagnostic, never an automatic promotion. Both models use
the same mix, complete the same depth, and alternate colors per opening.
Only replay-verified checkmates are counted as wins.
"""
import argparse
import hashlib
import json
from pathlib import Path

import shogi

from .arena import opening
from .engine import select_move
from .relative_value import RelativeValue
from .value_model import LinearValue


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_check(opening_moves, moves, result):
    board = shogi.Board()
    for word in opening_moves + moves:
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal replay move: " + word)
        board.push(move)
    if result["status"] == "checkmate":
        if not board.is_checkmate() or result["winner"] != int(not board.turn):
            raise ValueError("invalid checkmate winner")
    elif result["winner"] is not None:
        raise ValueError("nondecisive game has winner")
    elif result["status"] == "repetition_unresolved":
        if not board.is_fourfold_repetition():
            raise ValueError("false repetition")
    elif result["status"] == "move_limit_unresolved":
        if board.is_checkmate():
            raise ValueError("missed checkmate")
    elif result["status"] == "no_move_unresolved":
        pass
    else:
        raise ValueError("unknown termination")


def play(head, relative_black, relative_model, legacy_model, depth, max_plies,\n         movetime_ms=20, search_mode="complete_depth"):
    board = shogi.Board()
    for word in head:
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal opening")
        board.push(move)
    moves = []
    stats = {"relative_nodes": 0, "legacy_nodes": 0,
             "relative_turns": 0, "legacy_turns": 0,
             "relative_depth_zero": 0, "legacy_depth_zero": 0,\n             "relative_depth_sum": 0, "legacy_depth_sum": 0,\n             "relative_incomplete": 0, "legacy_incomplete": 0}
    status, winner = "move_limit_unresolved", None
    for _ in range(max_plies):
        if board.is_checkmate():
            status, winner = "checkmate", int(not board.turn)
            break
        if board.is_fourfold_repetition():
            status = "repetition_unresolved"
            break
        is_relative = (board.turn == shogi.BLACK) == relative_black
        label = "relative" if is_relative else "legacy"
        model = relative_model if is_relative else legacy_model
        word, finished, nodes = select_move(
            board, movetime_ms, depth, model=model,\n            complete_depth=(search_mode == "complete_depth"))
        stats[label + "_nodes"] += nodes
        stats[label + "_turns"] += 1
        stats[label + "_depth_sum"] += finished\n        if finished == 0:\n            stats[label + "_depth_zero"] += 1\n        if finished != depth:\n            stats[label + "_incomplete"] += 1\n            if search_mode == "complete_depth":\n                raise ValueError("incomplete equal-depth search")
        if word == "resign":
            status = "no_move_unresolved"
            break
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal engine move")
        moves.append(word)
        board.push(move)
    else:
        if board.is_checkmate():
            status, winner = "checkmate", int(not board.turn)
    row = {"status": status, "winner": winner, "moves": moves, "stats": stats}
    replay_check(head, moves, row)
    return row


def execute(relative_path, legacy_path, request_path, out):
    cfg = json.loads(Path(request_path).read_text(encoding="utf-8"))
    for key, path in (("relative_model_sha256", relative_path),
                      ("legacy_model_sha256", legacy_path)):
        expected = cfg.get(key)
        if not isinstance(expected, str) or len(expected) != 64 or digest(path) != expected:
            raise ValueError(key + " mismatch")
    for key, lo, hi in (("relative_source_run_id", 1, 2**63-1),
                        ("legacy_source_run_id", 1, 2**63-1),
                        ("arena_seed", 0, 2**31-1),
                        ("arena_pairs", 1, 40),
                        ("max_plies", 1, 240),
                        ("opening_plies", 0, 20)):
        v = cfg.get(key)
        if type(v) is not int or not lo <= v <= hi:
            raise ValueError("invalid " + key)
    mode = cfg.get("search_mode", "complete_depth")\n    depth = cfg.get("depth")\n    movetime_ms = cfg.get("movetime_ms", 20)\n    if mode not in ("complete_depth", "timed"):\n        raise ValueError("invalid search mode")\n    if type(movetime_ms) is not int or not 1 <= movetime_ms <= 1000:\n        raise ValueError("invalid movetime_ms")\n    if (type(depth) is not int or\n            (mode == "complete_depth" and depth != 1) or\n            (mode == "timed" and depth not in (1, 2)) or\n            cfg.get("mix_percent") != 25):\n        raise ValueError("invalid fixed-model comparison settings")
    relative = RelativeValue(relative_path, mix_percent=25)
    legacy = LinearValue(legacy_path, mix_percent=25)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in range(cfg["arena_pairs"]):
        head = opening(cfg["arena_seed"] + pair, cfg["opening_plies"])
        for relative_black in (True, False):
            row = play(head, relative_black, relative, legacy, depth,\n                       cfg["max_plies"], movetime_ms=movetime_ms, search_mode=mode)
            row.update({"pair": pair, "relative_black": relative_black, "opening": head})
            rows.append(row)
    wins = sum(r["status"] == "checkmate" and
               (r["winner"] == int(shogi.BLACK)) == r["relative_black"] for r in rows)
    losses = sum(r["status"] == "checkmate" and
                 (r["winner"] == int(shogi.BLACK)) != r["relative_black"] for r in rows)
    unresolved = len(rows) - wins - losses
    path = root / "direct_games.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {"status": "research_only_no_promotion", "games": len(rows),
              "pairs": cfg["arena_pairs"], "relative_wins": wins,
              "legacy_wins": losses, "unresolved": unresolved,
              "relative_model_sha256": digest(relative_path),
              "legacy_model_sha256": digest(legacy_path),
              "arena_seed": cfg["arena_seed"], "mix_percent": 25,
              "depth": depth, "search_mode": mode,\n              "movetime_ms": movetime_ms if mode == "timed" else None,\n              "max_plies": cfg["max_plies"],
              "legal_replay_verified": True,
              "all_pairs_alternate_color": True,
              "stats": {key: sum(row["stats"][key] for row in rows)
                        for key in rows[0]["stats"]},
              "games_sha256": digest(path),
              "caution": "Unresolved games are not wins. No formal strength claim."}
    (root / "direct_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--relative-model", required=True)
    p.add_argument("--legacy-model", required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--out", default="build/direct_arena")
    a = p.parse_args()
    execute(a.relative_model, a.legacy_model, a.request, a.out)


if __name__ == "__main__":
    main()
