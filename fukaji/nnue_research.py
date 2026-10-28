"""Research-only small NNUE-style value accumulator, not a trained engine.

This module deliberately does not change FUKAJI's USI search, current G8
model, or evaluation defaults. A dual-perspective accumulator allows each
legal push/pop to update only changed squares and hand-count thresholds.
The random test weights are synthetic and are NEVER a playing-strength model.
"""
from __future__ import annotations

import numpy as np
import shogi

from .learning import FEATURES, PIECES

INPUTS = FEATURES - 1  # Exclude the existing constant intercept feature.
HIDDEN = 32
HAND_BASE = 81 * PIECES * 2
HAND_LIMIT = 20


def _piece_index(piece, square, perspective):
    owner = int(piece.color) ^ int(perspective)
    relative_square = square if perspective == shogi.BLACK else 80 - square
    return (owner * PIECES + piece.piece_type - 1) * 81 + relative_square


def _hand_index(color, kind, threshold, perspective):
    owner = int(color) ^ int(perspective)
    return HAND_BASE + (owner * 7 + kind - 1) * HAND_LIMIT + threshold


def active_indices(board, perspective):
    """Sparse 2548-input indices for either fixed player perspective."""
    if perspective not in (shogi.BLACK, shogi.WHITE):
        raise ValueError("invalid perspective")
    result = []
    for square in range(81):
        piece = board.piece_at(square)
        if piece is not None:
            result.append(_piece_index(piece, square, perspective))
    for color in (shogi.BLACK, shogi.WHITE):
        for kind in range(1, 8):
            count = min(HAND_LIMIT, int(board.pieces_in_hand[color].get(kind, 0)))
            result.extend(_hand_index(color, kind, j, perspective) for j in range(count))
    if len(result) != len(set(result)) or any(not 0 <= x < INPUTS for x in result):
        raise AssertionError("duplicate or out-of-range NNUE feature")
    return result


class SmallNNUE:
    """32-unit clipped-ReLU test architecture; no trained model is loaded."""

    def __init__(self, hidden_weights, hidden_bias, output_weights, output_bias=0.0):
        w = np.asarray(hidden_weights, dtype=np.float32)
        b = np.asarray(hidden_bias, dtype=np.float32)
        v = np.asarray(output_weights, dtype=np.float32)
        o = float(output_bias)
        if (w.shape != (INPUTS, HIDDEN) or b.shape != (HIDDEN,) or
                v.shape != (HIDDEN,) or not np.isfinite(w).all() or
                not np.isfinite(b).all() or not np.isfinite(v).all() or
                not np.isfinite(o)):
            raise ValueError("invalid NNUE research tensor shapes or values")
        self.hidden_weights = w.copy()
        self.hidden_bias = b.astype(np.float64).copy()
        self.output_weights = v.copy()
        self.output_bias = o

    @classmethod
    def synthetic_test_model(cls, seed=42):
        """Deterministic untrained weights for regression tests ONLY."""
        rng = np.random.default_rng(seed)
        return cls(rng.normal(0, 0.01, (INPUTS, HIDDEN)),
                   rng.normal(0, 0.01, HIDDEN),
                   rng.normal(0, 0.01, HIDDEN))

    def full_accumulator(self, board, perspective):
        indexes = active_indices(board, perspective)
        if not indexes:
            return self.hidden_bias.copy()
        return self.hidden_bias + self.hidden_weights[indexes].astype(np.float64).sum(axis=0)

    def logit(self, board, accumulator=None):
        if accumulator is None:
            accumulator = self.full_accumulator(board, board.turn)
        hidden = np.clip(np.asarray(accumulator, dtype=np.float64), 0.0, 1.0)
        result = float(np.dot(hidden, self.output_weights.astype(np.float64)) + self.output_bias)
        if not np.isfinite(result):
            raise ValueError("nonfinite NNUE output")
        return result


class DualAccumulator:
    """Owns a board's incremental evaluation state; restores it on pop.

    This is a correctness prototype, not a measured performance optimization.
    It never replaces the board's legality, repetition, or checkmate rules.
    """

    def __init__(self, board, model):
        if not isinstance(model, SmallNNUE):
            raise TypeError("expected SmallNNUE research model")
        self.board = board
        self.model = model
        self.base_depth = len(board.move_stack)
        self.acc = {p: model.full_accumulator(board, p)
                    for p in (shogi.BLACK, shogi.WHITE)}
        self.history = []
        # Catch external board replacement even when the move-stack depth agrees.
        self.position_hash = board.zobrist_hash()

    def _check_depth(self):
        if len(self.board.move_stack) != self.base_depth + len(self.history):
            raise ValueError("board stack modified outside incremental evaluator")
        if self.board.zobrist_hash() != self.position_hash:
            raise ValueError("board position modified outside incremental evaluator")

    def _snapshot(self, squares):
        pieces = {sq: self.board.piece_at(sq) for sq in squares}
        hands = {(color, kind): int(self.board.pieces_in_hand[color].get(kind, 0))
                 for color in (shogi.BLACK, shogi.WHITE) for kind in range(1, 8)}
        return pieces, hands

    def push(self, move):
        self._check_depth()
        if move not in self.board.legal_moves:
            raise ValueError("illegal move for incremental research")
        squares = sorted(set(([move.from_square] if move.from_square is not None else [])
                             + [move.to_square]))
        before_pieces, before_hands = self._snapshot(squares)
        self.board.push(move)
        try:
            after_pieces, after_hands = self._snapshot(squares)
            changes = {shogi.BLACK: [], shogi.WHITE: []}
            for perspective in (shogi.BLACK, shogi.WHITE):
                items = changes[perspective]
                for square in squares:
                    old = before_pieces[square]
                    new = after_pieces[square]
                    if old is not None:
                        items.append((-1, _piece_index(old, square, perspective)))
                    if new is not None:
                        items.append((+1, _piece_index(new, square, perspective)))
                for color in (shogi.BLACK, shogi.WHITE):
                    for kind in range(1, 8):
                        old = min(HAND_LIMIT, before_hands[color, kind])
                        new = min(HAND_LIMIT, after_hands[color, kind])
                        if new > old:
                            items.extend((+1, _hand_index(color, kind, j, perspective))
                                         for j in range(old, new))
                        elif old > new:
                            items.extend((-1, _hand_index(color, kind, j, perspective))
                                         for j in range(new, old))
            for perspective in (shogi.BLACK, shogi.WHITE):
                for sign, index in changes[perspective]:
                    self.acc[perspective] += sign * self.model.hidden_weights[index]
            self.history.append(changes)
            self.position_hash = self.board.zobrist_hash()
        except BaseException:
            # A failed research update must not leave the caller board pushed.
            self.board.pop()
            self.acc = {p: self.model.full_accumulator(self.board, p)
                        for p in (shogi.BLACK, shogi.WHITE)}
            self.position_hash = self.board.zobrist_hash()
            raise

    def pop(self):
        self._check_depth()
        if not self.history:
            raise IndexError("no incremental move to undo")
        changes = self.history[-1]
        move = self.board.pop()
        for perspective in (shogi.BLACK, shogi.WHITE):
            for sign, index in reversed(changes[perspective]):
                self.acc[perspective] -= sign * self.model.hidden_weights[index]
        self.history.pop()
        self.position_hash = self.board.zobrist_hash()
        return move

    def logit(self):
        self._check_depth()
        return self.model.logit(self.board, self.acc[self.board.turn])

    def verify(self, tolerance=1e-8):
        """Recompute both perspectives to audit every incremental step."""
        self._check_depth()
        for perspective in (shogi.BLACK, shogi.WHITE):
            expected = self.model.full_accumulator(self.board, perspective)
            delta = float(np.max(np.abs(self.acc[perspective] - expected)))
            if delta > tolerance:
                raise AssertionError("incremental NNUE accumulator drift: " + str(delta))
        return True
