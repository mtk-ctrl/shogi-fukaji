"""Experimental mover-relative linear value features, isolated from legacy v0.

All board squares and piece colors are represented from the side-to-move
perspective.  This is a new model schema, NOT compatible with legacy v0
weights.  No automatic model promotion.
"""
import numpy as np
import shogi

from .learning import FEATURES, PIECES
from .value_model import LinearValue


def relative_features(board):
    x = np.zeros(FEATURES, dtype=np.float32)
    turn = int(board.turn)
    for sq in range(81):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        relative_color = int(piece.color) ^ turn
        relative_sq = sq if turn == shogi.BLACK else 80 - sq
        x[(relative_color * PIECES + piece.piece_type - 1) * 81 + relative_sq] = 1.0
    base = 81 * PIECES * 2
    for color in (shogi.BLACK, shogi.WHITE):
        relative_color = int(color) ^ turn
        for kind in range(1, 8):
            count = int(board.pieces_in_hand[color].get(kind, 0))
            for j in range(min(20, count)):
                x[base + (relative_color * 7 + kind - 1) * 20 + j] = 1.0
    x[-1] = 1.0  # intercept; mover identity is already normalized.
    return x


class RelativeValue(LinearValue):
    """Sparse inference exactly matching relative_features @ trained weights."""

    def score(self, board):
        turn = int(board.turn)
        logit = self._turn_weight
        for sq in range(81):
            piece = board.piece_at(sq)
            if piece is None:
                continue
            owner = int(piece.color) ^ turn
            square = sq if turn == shogi.BLACK else 80 - sq
            logit += self._piece_weights[(owner * PIECES + piece.piece_type - 1) * 81 + square]
        for color in (shogi.BLACK, shogi.WHITE):
            owner = int(color) ^ turn
            for kind, count in board.pieces_in_hand[color].items():
                logit += self._hand_prefix[owner, kind][min(20, int(count))]
        return int(round(max(-10.0, min(10.0, logit)) * self.weight_cp))
