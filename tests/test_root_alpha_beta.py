"""Root-bound research candidate: fixed-depth equivalence before timed profiling."""
import unittest
import shogi
from fukaji.engine import select_move


class RootAlphaBetaTests(unittest.TestCase):
    def test_opt_in_preserves_fixed_depth_best_move_and_board(self):
        openings = [
            [],
            ["7g7f", "3c3d", "2g2f", "8c8d"],
            ["2g2f", "8c8d", "7g7f", "3c3d", "2f2e", "8d8e"],
        ]
        strict_savings = 0
        for opening in openings:
            board = shogi.Board()
            for word in opening:
                board.push_usi(word)
            before = board.sfen()
            original = select_move(board, max_depth=2, complete_depth=True)
            candidate = select_move(
                board, max_depth=2, complete_depth=True, root_alpha_beta=True
            )
            self.assertEqual(board.sfen(), before)
            self.assertEqual(original[:2], candidate[:2])
            self.assertEqual(original[1], 2)
            self.assertLessEqual(candidate[2], original[2])
            strict_savings += candidate[2] < original[2]
        self.assertGreater(strict_savings, 0)

    def test_default_is_explicitly_off(self):
        board = shogi.Board()
        self.assertEqual(
            select_move(board, max_depth=2, complete_depth=True),
            select_move(board, max_depth=2, complete_depth=True, root_alpha_beta=False),
        )

    def test_timed_candidate_returns_legal_move(self):
        board = shogi.Board()
        before = board.sfen()
        move, depth, nodes = select_move(
            board, movetime_ms=1, max_depth=3, root_alpha_beta=True
        )
        self.assertEqual(board.sfen(), before)
        self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
        self.assertGreaterEqual(depth, 0)
        self.assertGreater(nodes, 0)


if __name__ == "__main__":
    unittest.main()
