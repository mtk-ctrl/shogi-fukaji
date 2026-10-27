"""Conservative legal-guard opt-in equivalence tests."""
import inspect
import random
import subprocess
import sys
import unittest
import shogi
from fukaji.engine import fast_legal_moves, ordered_moves, select_move, negamax

class FastLegalGuardTests(unittest.TestCase):
    def check_position(self, board):
        before = board.sfen()
        baseline = [m.usi() for m in board.legal_moves]
        guarded = [m.usi() for m in fast_legal_moves(board)]
        self.assertEqual(baseline, guarded, before)
        self.assertEqual(before, board.sfen())
        self.assertEqual(
            [m.usi() for m in ordered_moves(board, fast_move_list=True)],
            [m.usi() for m in ordered_moves(
                board, fast_move_list=True, fast_legal_guard=True)])

    def test_special_positions(self):
        for sfen in (
            shogi.Board().sfen(),
            "4r3k/9/9/9/9/9/9/4G4/4K4 b - 1",
            "4r3k/9/9/9/9/9/9/9/4K4 b - 1",
            "4k4/9/9/9/9/9/9/9/4K4 b PGLNS 1",
            "4k4/9/9/9/4P4/9/9/9/4K4 b - 1",
            "4k4/9/4R4/9/9/9/9/9/4K4 w - 1",
        ):
            with self.subTest(sfen=sfen):
                self.check_position(shogi.Board(sfen))

    def test_random_reachable_positions(self):
        rng = random.Random(20275027)
        for game in range(8):
            board = shogi.Board()
            for ply in range(18):
                with self.subTest(game=game, ply=ply):
                    self.check_position(board)
                moves = [m for m in board.legal_moves]
                if not moves:
                    break
                board.push(rng.choice(moves))

    def test_fixed_depth_and_default_off(self):
        for moves in ((), ("7g7f", "3c3d"), ("2g2f", "8c8d")):
            board = shogi.Board()
            for token in moves:
                m = shogi.Move.from_usi(token)
                self.assertIn(m, board.legal_moves)
                board.push(m)
            old = select_move(board, 1, max_depth=2, complete_depth=True,
                              root_alpha_beta=True, fast_move_list=True)
            new = select_move(board, 1, max_depth=2, complete_depth=True,
                              root_alpha_beta=True, fast_move_list=True,
                              fast_legal_guard=True)
            self.assertEqual(old, new)
        for fn in (ordered_moves, select_move, negamax):
            self.assertFalse(inspect.signature(fn).parameters["fast_legal_guard"].default)

    def test_usi_opt_in(self):
        p = subprocess.run([sys.executable, "-m", "fukaji.engine"],
            input="usi\nsetoption name FastLegalGuard value true\nisready\nposition startpos\ngo movetime 30\nquit\n",
            text=True, capture_output=True, timeout=15)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("option name FastLegalGuard type check default false", p.stdout)
        self.assertIn("bestmove ", p.stdout)

if __name__ == "__main__":
    unittest.main()
