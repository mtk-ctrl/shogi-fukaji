"""Research-only bounded quiescence search for FUKAJI (not enabled in USI).

The baseline evaluator is deliberately unchanged. A checked side cannot stand
pat: all legal evasions are searched, or the iteration is abandoned at a
finite check-chain bound. A quiet side searches only captures and promotions.
"""
from __future__ import annotations

import time

MATE = 1_000_000


def tactical_moves(board, ordered):
    """Select captures/promotions from the existing fully legal move list."""
    return [move for move in ordered
            if move.promotion or board.piece_at(move.to_square) is not None]


def quiesce(board, alpha, beta, deadline, nodes, evaluator, legal_order,
            *, quiet_left=4, check_left=8, ply=0, count_node=False):
    """Fail-soft negamax; TimeoutError aborts the entire unfinished iteration.

    This is a bounded tactical probe, not a proof of no mate. The caller must
    retain the last fully completed iterative-deepening result on timeout.
    """
    if count_node:
        nodes[0] += 1
    if time.monotonic() >= deadline:
        raise TimeoutError("quiescence deadline")
    if board.is_fourfold_repetition():
        return -25000
    if board.transpositions[board.zobrist_hash()] >= 2:
        return -1000

    in_check = board.is_check()
    if in_check:
        # No stand pat in check; never call a static score an evasion.
        legal = legal_order(board)
        if not legal:
            return -MATE + ply
        if check_left <= 0:
            # Incomplete check resolution must not be scored as a quiet leaf.
            raise TimeoutError("quiescence check-chain bound")
        choices = legal
    else:
        standing = evaluator(board)
        if standing >= beta:
            return standing
        if standing > alpha:
            alpha = standing
        if quiet_left <= 0:
            return standing
        choices = tactical_moves(board, legal_order(board))

    best = -10**9 if in_check else standing
    for move in choices:
        if time.monotonic() >= deadline:
            raise TimeoutError("quiescence deadline")
        board.push(move)
        try:
            score = -quiesce(
                board, -beta, -alpha, deadline, nodes, evaluator, legal_order,
                quiet_left=quiet_left - (0 if in_check else 1),
                check_left=check_left - 1,
                ply=ply + 1, count_node=True,
            )
        finally:
            board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best
