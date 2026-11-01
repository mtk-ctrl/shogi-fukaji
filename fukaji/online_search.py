"""D17 research search: NNUE-only leaves; never changes the formal USI engine.

The search shares only FUKAJI's legal move ordering with engine.py. Outcomes
are rule facts, independent of the network. No score transposition cache is
used: repetition depends on history. A move-order cache is optional.
"""
from __future__ import annotations

import copy
import time
import numpy as np
import shogi

from .engine import ordered_moves
from .nnue_research import _piece_index, _hand_index, SmallNNUE, relative_piece_indices

MATE = 1000000.0


class Position:
    """Incremental two-perspective state; push receives a proven legal move.

    Public callers use push(); only the search's enumerated legal moves use
    push_legal(). Undo restores saved accumulators, avoiding floating drift
    across sibling branches. Only the one changed hand count is inspected.
    """
    def __init__(self, board, model, incremental=True, *, history=None):
        if not isinstance(model, SmallNNUE):
            raise TypeError("SmallNNUE required")
        self.board, self.model, self.incremental = board, model, incremental
        self.acc = np.stack([model.full_accumulator(board, p) for p in (0, 1)])
        self.undo = []
        if history is not None:
            # play() already maintains exact repetition/check history. Borrow
            # its facts, never its model-specific accumulators or undo stack.
            if (not isinstance(history, Position) or history.board is not board or
                len(history.keys) != len(board.move_stack) + 1 or
                len(history.checks) != len(board.move_stack) or
                history.keys[-1] != board.zobrist_hash()):
                raise ValueError("stale or unrelated repetition history")
            self.keys = list(history.keys)
            self.checks = list(history.checks)
        else:
            b = copy.deepcopy(board)
            reversed_history = []
            while b.move_stack:
                reversed_history.append((b.zobrist_hash(), int(not b.turn), b.is_check()))
                b.pop()
            self.keys = [b.zobrist_hash()]
            self.checks = []
            for key, mover, check in reversed(reversed_history):
                self.keys.append(key)
                self.checks.append((mover, check))
        self.base_depth = len(board.move_stack)

    def push(self, move):
        if move not in self.board.legal_moves:
            raise ValueError("illegal move")
        self.push_legal(move)

    def push_legal(self, move):
        b = self.board
        if len(b.move_stack) != self.base_depth + len(self.undo) or b.zobrist_hash() != self.keys[-1]:
            raise ValueError("board changed outside Position")
        color = b.turn
        squares = [move.to_square] + ([move.from_square] if move.from_square is not None else [])
        before = [(sq, b.piece_at(sq)) for sq in squares]
        king_move = (self.model.king_relative and move.from_square is not None and
                     b.piece_type_at(move.from_square) == shogi.KING)
        captured = b.piece_type_at(move.to_square)
        if captured == shogi.KING:
            raise ValueError("invalid position allows king capture")
        hand_kind = move.drop_piece_type or (shogi.PIECE_PROMOTED.index(captured)
                    if captured in shogi.PIECE_PROMOTED else captured)
        old_count = b.pieces_in_hand[color].get(hand_kind, 0) if hand_kind else 0
        saved = self.acc.copy()
        b.push(move)
        try:
            if self.incremental and king_move:
                # Moving either anchor changes both perspectives' relations.
                self.acc = np.stack([self.model.full_accumulator(b, p) for p in (0, 1)])
            elif self.incremental:
                for p in (0, 1):
                    for sq, piece in before:
                        if piece:
                            self.acc[p] -= self.model.hidden_weights[_piece_index(piece, sq, p)]
                            if self.model.king_relative:
                                for i in relative_piece_indices(piece, sq, p, b.king_squares):
                                    self.acc[p] -= self.model.hidden_weights[i]
                        piece = b.piece_at(sq)
                        if piece:
                            self.acc[p] += self.model.hidden_weights[_piece_index(piece, sq, p)]
                            if self.model.king_relative:
                                for i in relative_piece_indices(piece, sq, p, b.king_squares):
                                    self.acc[p] += self.model.hidden_weights[i]
                    if hand_kind:
                        new_count = b.pieces_in_hand[color].get(hand_kind, 0)
                        for j in range(min(old_count, new_count), max(old_count, new_count)):
                            self.acc[p] += (1 if new_count > old_count else -1) * self.model.hidden_weights[
                                _hand_index(color, hand_kind, j, p)]
            self.undo.append(saved)
            self.keys.append(b.zobrist_hash())
            self.checks.append((int(color), b.is_check()))
        except BaseException:
            b.pop()
            self.acc = saved
            raise

    def pop(self):
        if not self.undo:
            raise IndexError("empty search stack")
        self.board.pop()
        self.acc = self.undo.pop()
        self.keys.pop()
        self.checks.pop()

    def logit(self):
        return self.model.logit(self.board, self.acc[self.board.turn] if self.incremental else None)

    def repetition(self):
        if self.board.transpositions[self.keys[-1]] < 4:
            return None
        hits = [i for i, key in enumerate(self.keys) if key == self.keys[-1]]
        if len(hits) < 4:
            raise ValueError("incomplete repetition history")
        interval = self.checks[hits[-4]:]
        continuous = [color for color in (0, 1)
                      if any(m == color for m, _ in interval)
                      and all(check for m, check in interval if m == color)]
        if len(continuous) == 1:
            return {"status": "perpetual_check", "winner": 1 - continuous[0]}
        return {"status": "repetition_draw", "winner": None}

    def verify(self):
        if self.board.zobrist_hash() != self.keys[-1]:
            raise AssertionError("history drift")
        if self.incremental:
            for p in (0, 1):
                np.testing.assert_allclose(self.acc[p], self.model.full_accumulator(self.board, p),
                                           rtol=0, atol=1e-8)


def terminal(position, legal):
    if not legal:
        return {"status": "checkmate", "winner": 1 - position.board.turn} if position.board.is_check() else {
            "status": "no_legal_unresolved", "winner": None}
    return position.repetition()


def select(board, model, *, movetime_ms=50, max_depth=64, incremental=True,
           ordering=True, fixed_depth=False, fast_leaf=True, history=None):
    if not 1 <= movetime_ms <= 10000 or not 1 <= max_depth <= 64:
        raise ValueError("invalid search bounds")
    started = time.monotonic()
    deadline = float("inf") if fixed_depth else started + movetime_ms / 1000
    state = Position(board, model, incremental, history=history)
    original = board.sfen()
    original_counts = +board.transpositions
    nodes, evals = 0, 0
    hints = {}

    def moves_at(ply):
        moves = ordered_moves(board, fast_move_list=True, fast_legal_guard=True)
        hint = hints.get(board.zobrist_hash()) if ordering else None
        if hint is not None:
            moves.sort(key=lambda m: m.usi() != hint)
        return moves

    def search(depth, alpha, beta, ply):
        nonlocal nodes, evals
        nodes += 1
        if time.monotonic() >= deadline:
            raise TimeoutError
        # At a static leaf only existence of a legal move matters. Enumerating
        # and sorting every legal move here dominated the original 20ms pilot.
        # The generator still proves legality (including check evasions).
        if depth == 0 and fast_leaf:
            first = next(iter(board.legal_moves), None)
            moves = [] if first is None else [first]
        else:
            moves = moves_at(ply)
        end = terminal(state, moves)
        if end is not None:
            if end["winner"] is None:
                return 0.0
            return (MATE - ply) * (1 if end["winner"] == board.turn else -1)
        if depth == 0:
            evals += 1
            return float(np.clip(state.logit(), -20, 20)) * 1000
        best, word = -2*MATE, None
        for move in moves:
            state.push_legal(move)
            try:
                score = -search(depth-1, -beta, -alpha, ply+1)
            finally:
                state.pop()
            if score > best:
                best, word = score, move.usi()
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        if ordering and word:
            hints[board.zobrist_hash()] = word
        return best

    moves = moves_at(0)
    end = terminal(state, moves)
    best = moves[0] if moves and end is None else None
    completed, score = 0, None
    partial_root_moves = 0
    for depth in range(1, max_depth+1) if best is not None else ():
        current, value, alpha = best, -2*MATE, -2*MATE
        searched = 0
        try:
            for move in moves:
                if time.monotonic() >= deadline:
                    raise TimeoutError
                state.push_legal(move)
                try:
                    v = -search(depth-1, -2*MATE, -alpha, 1)
                finally:
                    state.pop()
                if v > value:
                    current, value = move, v
                alpha = max(alpha, value)
                searched += 1
        except TimeoutError:
            # Before the first completed iteration, use the best NNUE-evaluated
            # move we actually examined, not the arbitrary first ordered move.
            # This remains a partial root search, never a completed depth 1.
            if completed == 0 and searched:
                best, score, partial_root_moves = current, value, searched
            break
        best, score, completed = current, value, depth
        moves.sort(key=lambda m: m != best)
        if abs(score) > MATE - 1000:
            break
    state.verify()
    if board.sfen() != original or +board.transpositions != original_counts:
        raise AssertionError("search failed to restore board/history")
    return {"move": best.usi() if best else None, "depth": completed,
            "partial_root_moves": partial_root_moves, "root_moves": len(moves),
            "nodes": nodes, "evaluations": evals, "score": score,
            "elapsed_ms": (time.monotonic()-started)*1000, "terminal": end}
