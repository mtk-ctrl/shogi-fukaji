"""Research-only paired root alpha-beta bound reuse vs full-window root.

Identical frozen model, evaluation, time, depth, and openings. The only change
is root_alpha_beta=True for the candidate. No model or KUMOJI promotion.
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

MODEL_SHA = "881af10b5bc0cde591e37a9aee883213fa7dd602345b786b9b93d69c7e6ed239"
MODEL_RUN = 40359123574


def validate_request(c):
    # Two explicitly frozen, independently seeded protocols; no arbitrary
    # request edits can silently change the experimental conditions.
    allowed = {20271327: (8, 120), 20271427: (16, 160)}
    if c.get("arena_seed") not in allowed:
        raise ValueError("unknown independent arena protocol")
    pairs, max_plies = allowed[c["arena_seed"]]
    expected = {
        "research_only": True, "relative_source_run_id": MODEL_RUN,
        "relative_model_sha256": MODEL_SHA, "arena_seed": c["arena_seed"],
        "arena_pairs": pairs, "max_plies": max_plies, "opening_plies": 6,
        "movetime_ms": 200, "depth": 2, "mix_percent": 25,
        "fast_relative": True, "root_alpha_beta_default": False,
        "root_alpha_beta_candidate": True, "training_allowed": False,
    }
    if c != expected:
        raise ValueError("root-alpha arena request must match frozen protocol")
    return c


def play(head, candidate_black, model, max_plies):
    board = shogi.Board()
    for word in head:
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal opening")
        board.push(move)
    moves = []
    stats = {key + "_" + field: 0
             for key in ("candidate", "baseline")
             for field in ("nodes", "turns", "depth_sum", "depth_zero", "incomplete", "elapsed_ms")}
    status, winner = "move_limit_unresolved", None
    for _ in range(max_plies - len(head)):
        if board.is_checkmate():
            status, winner = "checkmate", int(not board.turn)
            break
        if board.is_fourfold_repetition():
            status = "repetition_unresolved"
            break
        is_candidate = (board.turn == shogi.BLACK) == candidate_black
        label = "candidate" if is_candidate else "baseline"
        start = time.perf_counter()
        word, depth, nodes = select_move(
            board, movetime_ms=200, max_depth=2, model=model,
            fast_relative=True, root_alpha_beta=is_candidate)
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
    if len(rows) != 2 * pairs:
        raise ValueError("wrong game count")
    out = Counter()
    for i in range(pairs):
        a, b = rows[2*i:2*i+2]
        if (a["opening"] != b["opening"] or
                a["candidate_black"] is not True or b["candidate_black"] is not False):
            raise ValueError("invalid paired openings or colors")
        candidate = sum(x["status"] == "checkmate" and
                        (x["winner"] == int(shogi.BLACK)) == x["candidate_black"]
                        for x in (a,b))
        baseline = sum(x["status"] == "checkmate" and
                       (x["winner"] == int(shogi.BLACK)) != x["candidate_black"]
                       for x in (a,b))
        out["candidate_advantage" if candidate > baseline else
            "baseline_advantage" if baseline > candidate else "equal"] += 1
    return dict(out)


def execute(request_path, model_path, out_dir):
    cfg = validate_request(json.loads(Path(request_path).read_text()))
    if digest(model_path) != MODEL_SHA:
        raise ValueError("frozen model SHA mismatch")
    model = RelativeValue(model_path, mix_percent=25)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in range(cfg["arena_pairs"]):
        head = opening(cfg["arena_seed"] + pair, cfg["opening_plies"])
        if len(head) != 6:
            raise ValueError("incomplete opening")
        for candidate_black in (True, False):
            row = play(head, candidate_black, model, cfg["max_plies"])
            row.update({"pair": pair, "candidate_black": candidate_black, "opening": head})
            rows.append(row)
    candidate_wins = sum(x["status"] == "checkmate" and
                         (x["winner"] == int(shogi.BLACK)) == x["candidate_black"]
                         for x in rows)
    baseline_wins = sum(x["status"] == "checkmate" and
                        (x["winner"] == int(shogi.BLACK)) != x["candidate_black"]
                        for x in rows)
    game_file = root / "root_alpha_games.jsonl"
    game_file.write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in rows))
    result = {
        "status": "research_only_no_promotion",
        "games": len(rows), "pairs": cfg["arena_pairs"],
        "candidate_wins": candidate_wins, "baseline_wins": baseline_wins,
        "unresolved": len(rows) - candidate_wins - baseline_wins,
        "pair_comparison": summarize_pairs(rows, cfg["arena_pairs"]),
        "model_sha256": digest(model_path), "arena_seed": cfg["arena_seed"],
        "movetime_ms": 200, "depth": 2, "mix_percent": 25,
        "max_plies": cfg["max_plies"], "all_pairs_alternate_color": True,
        "legal_replay_verified": True, "root_alpha_beta_default": False,
        "stats": {key: sum(row["stats"][key] for row in rows)
                  for key in rows[0]["stats"]},
        "games_sha256": digest(game_file),
        "caution": "Time-limited games are exploratory; unresolved is not a win; no strength claim or adoption.",
    }
    (root/"root_alpha_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    p=argparse.ArgumentParser()
    for key in ("request","model","out"):
        p.add_argument("--"+key, required=True)
    a=p.parse_args()
    execute(a.request,a.model,a.out)


if __name__ == "__main__":
    main()
