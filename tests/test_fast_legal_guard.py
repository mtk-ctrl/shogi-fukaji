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

    def test_uchi_fuzume_pawn_drop_is_not_shortcut(self):
        # A pawn drop at 5b would mate the white king at 5a; prohibited.
        board = shogi.Board("3pkp3/3p1p3/4G4/9/9/9/9/9/4K4 b P 1")
        forbidden = shogi.Move.from_usi("P*5b")
        self.assertNotIn(forbidden, board.legal_moves)
        self.assertNotIn(forbidden, [m for m in fast_legal_moves(board)])
        self.check_position(board)

    def test_fourfold_repetition_state_preserved(self):
        board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        cycle = ("5i5h", "5a5b", "5h5i", "5b5a")
        for _ in range(3):
            for usi in cycle:
                move = shogi.Move.from_usi(usi)
                self.assertIn(move, board.legal_moves)
                board.push(move)
        self.assertTrue(board.is_fourfold_repetition())
        # python-shogi's push/pop may retain zero-count Counter keys; only
        # positive repetition counts are semantically meaningful.
        before = {k: v for k, v in board.transpositions.items() if v > 0}
        self.check_position(board)
        after = {k: v for k, v in board.transpositions.items() if v > 0}
        self.assertEqual(before, after)
        self.assertTrue(board.is_fourfold_repetition())

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

    def test_usi_formally_adopted_default_and_legacy_override(self):
        for choice in ("", "setoption name FastLegalGuard value false\n",
                       "setoption name FastLegalGuard value true\n"):
            with self.subTest(option=choice):
                p = subprocess.run([sys.executable, "-m", "fukaji.engine"],
                    input="usi\n" + choice + "isready\nposition startpos\ngo movetime 30\nquit\n",
                    text=True, capture_output=True, timeout=15)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertIn("id name FUKAJI v0.1.0", p.stdout)
                self.assertIn("option name FastLegalGuard type check default true", p.stdout)
                self.assertIn("bestmove ", p.stdout)

if __name__ == "__main__":
    unittest.main()
