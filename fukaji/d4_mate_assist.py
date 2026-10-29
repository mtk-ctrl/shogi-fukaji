"""D4 optional, research-only mate-first selection (NOT wired into USI).

Prove all legal defender replies within a small fraction of the move budget.
A timed-out proof is UNKNOWN and falls back to the unchanged search. This
is a development probe, not a guaranteed 200ms operating-system deadline.
"""
from __future__ import annotations

import argparse
import json
import time
import shogi

from .engine import select_move
from .mate_oracle import ProofIncomplete, prove_mate_up_to_three

DEVELOPMENT_POSITIONS = (
    ("A", "8k/9/9/8N/4N1R2/9/9/9/K8 b - 1"),
    ("B", "5g2k/7L1/8s/4N4/4B2L1/9/9/9/K8 b - 1"),
    ("A_without_rook", "8k/9/9/8N/4N4/9/9/9/K8 b - 1"),
)


def select_with_mate_assist(board, *, movetime_ms=200, max_depth=2,
                            model=None, proof_ms=25, max_legal_moves=1500):
    """Research-only candidate. Return proof status and total measured time.

    If a complete mate-in-1 or mate-in-3 is proved, return that legal first
    move. If no mate is proved or the proof is incomplete, run the ordinary
    engine with the remaining clock. A complete negative is not a no-mate
    result beyond three ply.
    """
    if movetime_ms < 1 or proof_ms < 1 or max_legal_moves < 1:
        raise ValueError("positive budgets required")
    started = time.monotonic()
    before = (board.sfen(), tuple(m.usi() for m in board.move_stack),
              board.zobrist_hash(),
              {k: v for k, v in board.transpositions.items() if v > 0})
    limit_ms = min(proof_ms, movetime_ms * 0.25)
    status = "complete_no_short_mate"
    try:
        proof = prove_mate_up_to_three(
            board, max_legal_moves=max_legal_moves,
            deadline=started + limit_ms / 1000)
    except ProofIncomplete:
        proof = None
        status = "incomplete"
    if proof is not None:
        if proof["mate_in_one"]:
            move = proof["mate_in_one"][0]
            status = "proven_one"
        elif proof["mate_in_three"]:
            move = proof["mate_in_three"][0]["first"]
            status = "proven_three"
        else:
            move = None
    else:
        move = None
    proof_elapsed = (time.monotonic() - started) * 1000
    if move is None:
        remaining = max(1, int(movetime_ms - proof_elapsed))
        move, depth, nodes = select_move(
            board, movetime_ms=remaining, max_depth=max_depth,
            model=model, deadline_from_call=True)
    else:
        depth, nodes = 0, 0
    total = (time.monotonic() - started) * 1000
    after = (board.sfen(), tuple(m.usi() for m in board.move_stack),
             board.zobrist_hash(),
             {k: v for k, v in board.transpositions.items() if v > 0})
    if after != before:
        raise AssertionError("mate assist changed board/history")
    if move != "resign" and shogi.Move.from_usi(move) not in board.legal_moves:
        raise AssertionError("mate assist chose an illegal move")
    return {"move": move, "status": status, "proof_ms": round(proof_elapsed, 3),
            "total_ms": round(total, 3), "search_depth": depth, "nodes": nodes,
            "over_200ms": total > 200}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--movetime-ms", type=int, default=200)
    args = p.parse_args()
    rows = []
    for name, sfen in DEVELOPMENT_POSITIONS:
        board = shogi.Board(sfen)
        row = select_with_mate_assist(board, movetime_ms=args.movetime_ms)
        rows.append({"name": name, "sfen": sfen, **row})
    print(json.dumps({"scope": "development_only_not_G8_or_strength",
                      "results": rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
