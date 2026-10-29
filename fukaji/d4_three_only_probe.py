"""D4 research-only discovery of complete three-ply mates with NO mate in one.

Deterministic development search, NOT a held-out puzzle set or strength claim.
No training labels, USI options, external engine moves, or formal adoption.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import shogi

from .engine import select_move
from .mate_oracle import ProofIncomplete, prove_mate_up_to_three

SEED = 20261030
BASE = {"1a": "k", "2c": "R", "2d": "L", "9i": "K"}
PIECES = ("R", "B", "G", "S", "L", "N")
DEFENDERS = ("p", "s", "g", "r")
FILES = "12345"
RANKS = "abcde"


def to_sfen(pieces, hand="-"):
    """Convert a small, explicit position map to an SFEN board."""
    rows = []
    for rank in "abcdefghi":
        row = ""
        empty = 0
        for file in "987654321":
            piece = pieces.get(file + rank)
            if piece is None:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += piece
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows) + " b " + hand + " 1"


def positions(limit=400):
    """Two explicitly distinguished construction sources; never claim unseen."""
    # First source: single-defender mutations of the known D4 development board.
    for kind in DEFENDERS:
        for rank in "abcd":
            for file in "1234":
                square = file + rank
                if square in BASE:
                    continue
                board = dict(BASE)
                board[square] = kind
                yield "guarded_mutation", to_sfen(board)
    # Second source: seeded independent layouts with varied attacking pieces.
    rng = random.Random(SEED)
    for _ in range(limit):
        king = rng.choice(("1a", "2a", "3a"))
        board = {king: "k", "9i": "K"}
        available = [f + r for f in FILES for r in RANKS if f + r != king]
        rng.shuffle(available)
        for i in range(rng.choice((2, 3, 3, 4))):
            board[available.pop()] = rng.choice(PIECES)
        for i in range(rng.choice((0, 1, 1, 2))):
            board[available.pop()] = rng.choice(DEFENDERS)
        yield "seeded_layout", to_sfen(board)


def snapshot(board):
    return (board.sfen(), tuple(m.usi() for m in board.move_stack),
            board.zobrist_hash(),
            {k: v for k, v in board.transpositions.items() if v > 0})


def discover(max_positions=500, max_legal_moves=4000, max_examples=2):
    if max_positions < 1 or max_examples < 1:
        raise ValueError("limits must be positive")
    result = {"seed": SEED, "positions_attempted": 0, "fully_proved": 0,
              "incomplete": 0, "invalid": 0, "three_only": [],
              "sources": {"guarded_mutation": 0, "seeded_layout": 0}}
    seen = set()
    for source, sfen in positions():
        if result["positions_attempted"] >= max_positions:
            break
        if sfen in seen:
            continue
        seen.add(sfen)
        result["positions_attempted"] += 1
        result["sources"][source] += 1
        board = shogi.Board(sfen)
        # White king cannot already be under attack on Black's turn.
        white_king = board.king_squares[shogi.WHITE]
        if (white_king is None or board.king_squares[shogi.BLACK] is None
                or board.is_attacked_by(shogi.BLACK, white_king)
                or board.is_check()):
            result["invalid"] += 1
            continue
        before = snapshot(board)
        try:
            proof = prove_mate_up_to_three(board, max_legal_moves=max_legal_moves)
        except ProofIncomplete:
            result["incomplete"] += 1
            if snapshot(board) != before:
                raise AssertionError("oracle failed to restore board after budget")
            continue
        if snapshot(board) != before:
            raise AssertionError("oracle modified board")
        result["fully_proved"] += 1
        if proof["mate_in_one"] or not proof["mate_in_three"]:
            continue
        # This is a complete, bounded positive, not a guess from one line.
        example = {"source": source, "sfen": sfen,
                   "first_moves": [p["first"] for p in proof["mate_in_three"]],
                   "proof": proof, "search": []}
        for depth in (1, 2):
            telemetry = {}
            started = time.perf_counter()
            move, finished, nodes = select_move(
                board, max_depth=depth, complete_depth=True,
                avoid_repetition=False, telemetry=telemetry)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            if snapshot(board) != before:
                raise AssertionError("search modified board")
            if shogi.Move.from_usi(move) not in board.legal_moves:
                raise AssertionError("search selected an illegal move")
            example["search"].append({
                "depth": depth, "move": move, "nodes": nodes,
                "completed_depth": finished, "elapsed_ms": elapsed_ms,
                "is_proven_three_ply": move in example["first_moves"]})
        result["three_only"].append(example)
        if len(result["three_only"]) >= max_examples:
            break
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-positions", type=int, default=500)
    p.add_argument("--max-legal-moves", type=int, default=4000)
    p.add_argument("--max-examples", type=int, default=2)
    a = p.parse_args()
    print(json.dumps(discover(a.max_positions, a.max_legal_moves,
                              a.max_examples), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
