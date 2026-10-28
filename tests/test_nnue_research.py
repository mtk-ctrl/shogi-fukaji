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


    def test_checked_king_evasion_preserves_full_recalculation(self):
        board = shogi.Board("4k4/9/4R4/9/9/9/9/9/4K4 w - 1")
        self.assertTrue(board.is_check())
        self.assertFalse(board.is_checkmate())
        start = board.sfen()
        state = DualAccumulator(board, self.model)
        move = next(iter(board.legal_moves))
        state.push(move)
        self.assertTrue(state.verify())
        self.assertAlmostEqual(state.logit(), self.model.logit(board), places=9)
        state.pop()
        self.assertEqual(board.sfen(), start)
        self.assertTrue(board.is_check())
        self.assertTrue(state.verify())

    def test_checkmate_has_no_incremental_legal_move(self):
        board = shogi.Board("4k4/3GRG3/9/9/9/9/9/9/4K4 w - 1")
        self.assertTrue(board.is_checkmate())
        state = DualAccumulator(board, self.model)
        start = board.sfen()
        self.assertEqual(list(board.legal_moves), [])
        with self.assertRaises(ValueError):
            state.push(shogi.Move.from_usi("5a5b"))
        self.assertEqual(board.sfen(), start)
        self.assertTrue(state.verify())

    def test_fourfold_repetition_preserves_history_and_accumulators(self):
        board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        start = board.sfen()
        state = DualAccumulator(board, self.model)
        cycle = ("5i6i", "5a6a", "6i5i", "6a5a")
        self.assertFalse(board.is_fourfold_repetition())
        for word in cycle * 3:
            move = shogi.Move.from_usi(word)
            self.assertIn(move, board.legal_moves)
            state.push(move)
            self.assertTrue(state.verify())
        self.assertTrue(board.is_fourfold_repetition())
        for _ in range(12):
            state.pop()
            self.assertTrue(state.verify())
        self.assertEqual(board.sfen(), start)
        self.assertFalse(board.is_fourfold_repetition())

    def test_illegal_pawn_drop_mate_is_rejected(self):
        # The rook/gold net leaves the white king no escape after P*5b.
        board = shogi.Board("4k4/9/3RGR3/9/9/9/9/9/4K4 b P 1")
        state = DualAccumulator(board, self.model)
        start = board.sfen()
        move = shogi.Move.from_usi("P*5b")
        self.assertNotIn(move, board.legal_moves)
        with self.assertRaises(ValueError):
            state.push(move)
        self.assertEqual(board.sfen(), start)
        self.assertTrue(state.verify())

    def test_long_legal_history_reversible(self):
        import random
        rng = random.Random(127)
        board = shogi.Board()
        start = board.sfen()
        state = DualAccumulator(board, self.model)
        count = 0
        for _ in range(60):
            if board.is_game_over():
                break
            choices = list(board.legal_moves)
            move = choices[rng.randrange(len(choices))]
            state.push(move)
            count += 1
            self.assertTrue(state.verify())
            self.assertAlmostEqual(state.logit(), self.model.logit(board), places=9)
        self.assertGreater(count, 0)
        for _ in range(count):
            state.pop()
            self.assertTrue(state.verify())
        self.assertEqual(board.sfen(), start)
        self.assertEqual(len(board.move_stack), 0)

    def test_external_same_depth_position_replacement_is_rejected(self):
        board = shogi.Board()
        board.push(shogi.Move.from_usi("7g7f"))
        state = DualAccumulator(board, self.model)
        board.pop()
        board.push(shogi.Move.from_usi("2g2f"))
        self.assertEqual(len(board.move_stack), state.base_depth)
        with self.assertRaisesRegex(ValueError, "position modified"):
            state.verify()
        with self.assertRaisesRegex(ValueError, "position modified"):
            state.push(next(iter(board.legal_moves)))

if __name__ == "__main__":
    unittest.main()
