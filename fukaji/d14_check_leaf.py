"""D14 research-only: one exact legal evasion ply at a checking horizon.

This is not the old D3 quiescence search. It triggers only when the side to
move is in check at the exact alpha-beta depth=0 horizon. It examines every
legal evasion once, scores the resulting board, and never recursively expands
captures or check chains. A missed budget aborts the entire iteration rather
than returning a guessed safe value.
"""
from __future__ import annotations
import time

MATE = 1_000_000
INF = 10 ** 9


def score_checked_leaf(board, alpha, beta, deadline, nodes, evaluator, legal_order):
    """Fail-soft exact one-evasion-ply extension, only for a checked side.

    The caller already checks checkmate and repetition at the incoming root.
    Child repetitions and checkmate are evaluated as normal negamax nodes.
    """
    if not board.is_check():
        raise ValueError("check-only leaf extension called at quiet board")
    if time.monotonic() >= deadline:
        raise TimeoutError("D14 check-evasion horizon deadline")
    moves = legal_order(board)
    if not moves:
        return -MATE
    best = -INF
    for move in moves:
        if time.monotonic() >= deadline:
            raise TimeoutError("D14 check-evasion response deadline")
        board.push(move)
        nodes[0] += 1
        try:
            if board.is_fourfold_repetition():
                value = 25000
            elif board.transpositions[board.zobrist_hash()] >= 2:
                value = 1000
            elif board.is_checkmate():
                value = MATE
            else:
                value = -evaluator(board)
        finally:
            board.pop()
        if value > best:
            best = value
        if value > alpha:
            alpha = value
        if alpha >= beta:
            break
    return best
