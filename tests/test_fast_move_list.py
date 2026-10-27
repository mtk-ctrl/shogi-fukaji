"""Opt-in legal-move enumeration semantic equivalence (research only)."""
import inspect
import subprocess
import sys
import unittest
import shogi
from fukaji.engine import ordered_moves, select_move, negamax

class FastMoveListTests(unittest.TestCase):
    def test_legal_order_is_identical(self):
        boards = [shogi.Board(), shogi.Board("4k4/9/4R4/9/9/9/9/9/4K4 w - 1")]
        for board in boards:
            self.assertEqual([m.usi() for m in ordered_moves(board)],
                             [m.usi() for m in ordered_moves(board, fast_move_list=True)])

    def test_fixed_depth_search_equivalence(self):
        boards = [shogi.Board(), shogi.Board("4k4/9/4R4/9/9/9/9/9/4K4 w - 1")]
        for board in boards:
            before = board.sfen()
            for depth in (1, 2):
                a = select_move(board, 1, max_depth=depth, complete_depth=True,
                                root_alpha_beta=True)
                b = select_move(board, 1, max_depth=depth, complete_depth=True,
                                root_alpha_beta=True, fast_move_list=True)
                self.assertEqual(a, b)
                self.assertEqual(board.sfen(), before)

    def test_default_is_unchanged(self):
        self.assertFalse(inspect.signature(ordered_moves).parameters["fast_move_list"].default)
        self.assertFalse(inspect.signature(select_move).parameters["fast_move_list"].default)
        self.assertFalse(inspect.signature(negamax).parameters["fast_move_list"].default)

    def test_usi_opt_in(self):
        p = subprocess.run([sys.executable, "-m", "fukaji.engine"],
            input="usi\nsetoption name FastMoveList value true\nisready\nposition startpos\ngo movetime 30\nquit\n",
            text=True, capture_output=True, timeout=15)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("option name FastMoveList type check default false", p.stdout)
        self.assertIn("bestmove ", p.stdout)

if __name__ == "__main__":
    unittest.main()
