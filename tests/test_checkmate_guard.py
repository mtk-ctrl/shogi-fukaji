"""Opt-in mate-guard semantic equivalence tests (research only)."""
import time
import unittest
from unittest.mock import patch
import shogi
from fukaji.engine import INF, MATE, negamax, select_move

class CheckmateGuardTests(unittest.TestCase):
    def test_non_check_short_circuits_mate_probe(self):
        board = shogi.Board()
        self.assertFalse(board.is_check())
        with patch.object(shogi.Board, "is_checkmate", side_effect=AssertionError("must not call")):
            score = negamax(board, 0, -INF, INF, float("inf"), [0], guard_checkmate=True)
        self.assertIsInstance(score, int)

    def test_in_check_non_mate_is_identical(self):
        board = shogi.Board("4k4/9/4R4/9/9/9/9/9/4K4 w - 1")
        self.assertTrue(board.is_check())
        self.assertFalse(board.is_checkmate())
        for depth in (0, 1):
            a = negamax(board, depth, -INF, INF, float("inf"), [0])
            b = negamax(board, depth, -INF, INF, float("inf"), [0], guard_checkmate=True)
            self.assertEqual(a, b)

    def test_checkmate_score_preserved(self):
        board = shogi.Board("4k4/3GRG3/9/9/9/9/9/9/4K4 w - 1")
        self.assertTrue(board.is_check())
        self.assertTrue(board.is_checkmate())
        for depth in (0, 1, 2):
            for flag in (False, True):
                self.assertEqual(negamax(board, depth, -INF, INF, float("inf"), [0], guard_checkmate=flag), -MATE-depth)

    def test_complete_depth_moves_and_nodes_match(self):
        boards = [
            shogi.Board(),
            shogi.Board("4k4/9/4R4/9/9/9/9/9/4K4 w - 1"),
        ]
        for board in boards:
            before = board.sfen()
            for depth in (1, 2):
                a = select_move(board, 1, max_depth=depth, complete_depth=True,
                                root_alpha_beta=True)
                b = select_move(board, 1, max_depth=depth, complete_depth=True,
                                root_alpha_beta=True, guard_checkmate=True)
                self.assertEqual(a, b)
                self.assertEqual(board.sfen(), before)

    def test_usi_option_is_exposed_and_opt_in(self):
        import subprocess
        import sys
        p = subprocess.run(
            [sys.executable, "-m", "fukaji.engine"],
            input="usi\nsetoption name GuardCheckmate value true\nisready\nposition startpos\ngo movetime 30\nquit\n",
            text=True, capture_output=True, timeout=15)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("option name GuardCheckmate type check default false", p.stdout)
        self.assertIn("bestmove ", p.stdout)

    def test_default_flag_is_false(self):
        import inspect
        self.assertFalse(inspect.signature(select_move).parameters["guard_checkmate"].default)
        self.assertFalse(inspect.signature(negamax).parameters["guard_checkmate"].default)

if __name__ == "__main__":
    unittest.main()
