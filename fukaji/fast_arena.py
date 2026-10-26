"""Research-only paired 200ms FUKAJI fast-leaf vs unchanged baseline.

Both sides load the exact same frozen mover-relative model and use the same
search; the only difference is the opt-in single-pass leaf evaluation.
Unresolved games are never counted as wins. No model promotion.
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


def validate_request(c):
    if (c.get("research_only") is not True
            or c.get("relative_source_run_id") != 40359123574
            or type(c.get("arena_seed")) is not int
            or not 0 <= c["arena_seed"] <= 2147483647 - 100
            or type(c.get("arena_pairs")) is not int
            or not 1 <= c["arena_pairs"] <= 16
            or c.get("max_plies") not in (80, 120, 160)
            or c.get("opening_plies") != 6
            or c.get("movetime_ms") != 200
            or c.get("depth") != 2
            or c.get("mix_percent") != 25
            or c.get("fast_eval_default") is not False):
        raise ValueError("invalid bounded fast evaluation research protocol")
    expected = c.get("relative_model_sha256")
    if (not isinstance(expected, str) or len(expected) != 64
            or any(ch not in "0123456789abcdef" for ch in expected)):
        raise ValueError("invalid frozen model SHA")
    return c


def play(head, fast_black, model, max_plies):
    board = shogi.Board()
    for word in head:
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal opening")
        board.push(move)
    moves = []
    stats = {key + "_" + field: 0
             for key in ("fast", "baseline")
             for field in ("nodes", "turns", "depth_sum", "depth_zero", "incomplete", "elapsed_ms")}
    status, winner = "move_limit_unresolved", None
    for _ in range(max_plies - len(head)):
        if board.is_checkmate():
            status, winner = "checkmate", int(not board.turn)
            break
        if board.is_fourfold_repetition():
            status = "repetition_unresolved"
            break
        is_fast = (board.turn == shogi.BLACK) == fast_black
        label = "fast" if is_fast else "baseline"
        start = time.perf_counter()
        word, depth, nodes = select_move(
            board, movetime_ms=200, max_depth=2, model=model,
            fast_relative=is_fast)
        elapsed = (time.perf_counter() - start) * 1000
        stats[label + "_nodes"] += nodes
        stats[label + "_turns"] += 1
        stats[label + "_depth_sum"] += depth
        stats[label + "_elapsed_ms"] += round(elapsed, 3)
        if depth == 0:
            stats[label + "_depth_zero"] += 1
        if depth < 2:
            stats[label + "_incomplete"] += 1
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
    row = {"status": status, "winner": winner, "moves": moves, "stats": stats}
    replay_check(head, moves, row)
    return row


def summarize_pairs(rows, pairs):
    if len(rows) != pairs * 2:
        raise ValueError("wrong game count")
    out = Counter()
    for i in range(pairs):
        a, b = rows[2 * i:2 * i + 2]
        if (a["opening"] != b["opening"] or a["fast_black"] is not True
                or b["fast_black"] is not False):
            raise ValueError("invalid color-swapped opening pair")
        f = sum(x["status"] == "checkmate" and
                (x["winner"] == int(shogi.BLACK)) == x["fast_black"]
                for x in (a, b))
        old = sum(x["status"] == "checkmate" and
                  (x["winner"] == int(shogi.BLACK)) != x["fast_black"]
                  for x in (a, b))
        out["fast_advantage" if f > old else
            "baseline_advantage" if old > f else "equal"] += 1
    return dict(out)


def execute(request_path, model_path, out_dir):
    cfg = validate_request(json.loads(Path(request_path).read_text(encoding="utf-8")))
    if digest(model_path) != cfg["relative_model_sha256"]:
        raise ValueError("frozen relative model SHA mismatch")
    model = RelativeValue(model_path, mix_percent=25)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in range(cfg["arena_pairs"]):
        head = opening(cfg["arena_seed"] + pair, cfg["opening_plies"])
        if len(head) != 6:
            raise ValueError("incomplete opening")
        for fast_black in (True, False):
            row = play(head, fast_black, model, cfg["max_plies"])
            row.update({"pair": pair, "fast_black": fast_black, "opening": head})
            rows.append(row)
    fast_wins = sum(x["status"] == "checkmate" and
                    (x["winner"] == int(shogi.BLACK)) == x["fast_black"]
                    for x in rows)
    baseline_wins = sum(x["status"] == "checkmate" and
                        (x["winner"] == int(shogi.BLACK)) != x["fast_black"]
                        for x in rows)
    game_file = root / "fast_games.jsonl"
    game_file.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                         encoding="utf-8")
    result = {
        "status": "research_only_no_promotion",
        "games": len(rows), "pairs": cfg["arena_pairs"],
        "fast_wins": fast_wins, "baseline_wins": baseline_wins,
        "unresolved": len(rows) - fast_wins - baseline_wins,
        "pair_comparison": summarize_pairs(rows, cfg["arena_pairs"]),
        "model_sha256": digest(model_path),
        "arena_seed": cfg["arena_seed"],
        "movetime_ms": 200, "depth": 2, "mix_percent": 25,
        "max_plies": cfg["max_plies"], "all_pairs_alternate_color": True,
        "legal_replay_verified": True,
        "stats": {key: sum(row["stats"][key] for row in rows)
                  for key in rows[0]["stats"]},
        "games_sha256": digest(game_file),
        "caution": "Only leaf evaluation implementation differs. Time-limited outcomes vary; no strength claim or adoption.",
    }
    (root / "fast_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    execute(a.request, a.model, a.out)


if __name__ == "__main__":
    main()
