"""D4 research-only, exhaustive legal mate-in-1 / mate-in-3 proof.

This is an independent tactical *oracle*, not a USI engine feature or a
trained label.  The attacker's first move must give check. Every legal reply
must permit a checking mate on the next attacking move. A move limit raises
INCOMPLETE rather than silently treating an unsearched line as no mate.
All board mutations are undone, including on budget exhaustion.
"""
from __future__ import annotations

import argparse
import json
import shogi


class ProofIncomplete(RuntimeError):
    """The legal-move budget expired; neither mate nor no-mate is proved."""


class _Budget:
    def __init__(self, maximum):
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
            raise ValueError("maximum must be a positive integer")
        self.maximum = maximum
        self.examined = 0

    def step(self):
        if self.examined >= self.maximum:
            raise ProofIncomplete("legal-move budget exhausted")
        self.examined += 1


def _checking_mates_in_one(board, budget):
    """Return all legal moves that checkmate immediately; restore board."""
    result = []
    for move in tuple(board.legal_moves):
        budget.step()
        board.push(move)
        try:
            if board.is_check() and not any(board.legal_moves):
                result.append(move.usi())
        finally:
            board.pop()
    return tuple(result)


def prove_mate_up_to_three(board, *, max_legal_moves=50000):
    """Exhaustive 1/3-ply proof using python-shogi's full legal-move rules.

    A three-ply proof contains the COMPLETE set of defender replies, each
    paired with at least one legal immediate mate. The one-ply and three-ply
    classes are separate. An exhausted budget raises ProofIncomplete; it
    never returns an unproved negative. This routine does not decide best
    positional moves, perpetual-check adjudication, or actual playing strength.
    """
    budget = _Budget(max_legal_moves)
    mate_one = []
    mate_three = []
    checked_first_moves = 0
    checked_defenses = 0
    for first in tuple(board.legal_moves):
        budget.step()
        checked_first_moves += 1
        board.push(first)
        try:
            if not board.is_check():
                continue
            replies = tuple(board.legal_moves)
            if not replies:
                mate_one.append(first.usi())
                continue
            defenses = []
            for reply in replies:
                budget.step()
                checked_defenses += 1
                board.push(reply)
                try:
                    finishes = _checking_mates_in_one(board, budget)
                finally:
                    board.pop()
                if not finishes:
                    break  # One legal defense refutes this candidate.
                defenses.append({"reply": reply.usi(), "mates": finishes})
            if len(defenses) == len(replies):
                mate_three.append({
                    "first": first.usi(),
                    "defenses": tuple(defenses),
                })
        finally:
            board.pop()
    return {
        "scope": "complete_legal_mate_in_1_or_3_not_best_move_or_Elo",
        "mate_in_one": tuple(mate_one),
        "mate_in_three": tuple(mate_three),
        "first_moves_checked": checked_first_moves,
        "defenses_checked": checked_defenses,
        "legal_moves_examined": budget.examined,
    }


def main():
    parser = argparse.ArgumentParser(description="D4 bounded mate proof, research only")
    parser.add_argument("--sfen", default="8k/9/7R1/7L1/9/9/9/9/8K b - 1")
    parser.add_argument("--max-legal-moves", type=int, default=50000)
    args = parser.parse_args()
    board = shogi.Board(args.sfen)
    before = (board.sfen(), tuple(m.usi() for m in board.move_stack))
    try:
        result = prove_mate_up_to_three(board, max_legal_moves=args.max_legal_moves)
    except ProofIncomplete as exc:
        print(json.dumps({"status": "INCOMPLETE", "reason": str(exc)}, sort_keys=True))
        return 2
    assert before == (board.sfen(), tuple(m.usi() for m in board.move_stack))
    print(json.dumps({"status": "PROVED", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
