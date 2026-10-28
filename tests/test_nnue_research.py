"""Safety tests for dual-perspective NNUE-style incremental research."""
import unittest

import numpy as np
import shogi

from fukaji.nnue_research import INPUTS, HIDDEN, SmallNNUE, DualAccumulator, active_indices
from fukaji.relative_value import relative_features


class NNUEFoundationTests(unittest.TestCase):
    def setUp(self):
        self.model = SmallNNUE.synthetic_test_model(seed=73)

    def test_feature_layout_matches_existing_mover_relative_vector(self):
        board = shogi.Board()
        for move in ("7g7f", "7c7d", "7f7e", "3c3d", "7e7d"):
            board.push(shogi.Move.from_usi(move))
        expected = set(np.flatnonzero(relative_features(board)[:-1]).tolist())
        self.assertEqual(set(active_indices(board, board.turn)), expected)
        self.assertEqual(INPUTS, 2548)
        self.assertEqual(HIDDEN, 32)

    def test_initial_two_perspectives_and_logit(self):
        board = shogi.Board()
        state = DualAccumulator(board, self.model)
        self.assertTrue(state.verify())
        self.assertAlmostEqual(state.logit(), self.model.logit(board), places=10)

    def test_capture_hands_and_reversible_history(self):
        board = shogi.Board()
        start = board.sfen()
        state = DualAccumulator(board, self.model)
        seq = ["7g7f", "7c7d", "7f7e", "3c3d", "7e7d"]
        for word in seq:
            state.push(shogi.Move.from_usi(word))
            self.assertTrue(state.verify())
            self.assertAlmostEqual(state.logit(), self.model.logit(board), places=9)
        self.assertEqual(board.pieces_in_hand[shogi.BLACK][shogi.PAWN], 1)
        for _ in seq:
            state.pop()
            self.assertTrue(state.verify())
        self.assertEqual(board.sfen(), start)
        self.assertEqual(len(board.move_stack), 0)

    def test_legal_drop_and_undo(self):
        board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b P 1")
        state = DualAccumulator(board, self.model)
        start = board.sfen()
        move = shogi.Move.from_usi("P*5e")
        self.assertIn(move, board.legal_moves)
        state.push(move)
        self.assertTrue(state.verify())
        self.assertEqual(board.pieces_in_hand[shogi.BLACK].get(shogi.PAWN, 0), 0)
        state.pop()
        self.assertTrue(state.verify())
        self.assertEqual(board.sfen(), start)

    def test_promotion_and_undo(self):
        board = shogi.Board("4k4/9/4P4/9/9/9/9/9/4K4 b - 1")
        state = DualAccumulator(board, self.model)
        start = board.sfen()
        move = shogi.Move.from_usi("5c5b+")
        self.assertIn(move, board.legal_moves)
        state.push(move)
        self.assertTrue(state.verify())
        state.pop()
        self.assertTrue(state.verify())
        self.assertEqual(board.sfen(), start)

    def test_reject_illegal_and_external_board_modification(self):
        board = shogi.Board()
        state = DualAccumulator(board, self.model)
        with self.assertRaises(ValueError):
            state.push(shogi.Move.from_usi("7g7g"))
        self.assertTrue(state.verify())
        board.push(shogi.Move.from_usi("7g7f"))
        with self.assertRaises(ValueError):
            state.verify()

    def test_reject_shape_and_nonfinite_weights(self):
        with self.assertRaises(ValueError):
            SmallNNUE(np.zeros((10, 32)), np.zeros(32), np.zeros(32))
        with self.assertRaises(ValueError):
            SmallNNUE(np.full((INPUTS, HIDDEN), np.nan), np.zeros(32), np.zeros(32))
        with self.assertRaises(ValueError):
            SmallNNUE(np.zeros((INPUTS, HIDDEN)), np.zeros(32), np.zeros(32), float("inf"))


if __name__ == "__main__":
    unittest.main()
