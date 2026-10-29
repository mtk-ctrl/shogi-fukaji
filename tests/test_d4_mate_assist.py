"""D4 research-only mate-first selection safety and bounded proof checks."""
import time
import unittest

import shogi

from fukaji.d4_mate_assist import (
    DEVELOPMENT_POSITIONS, select_with_mate_assist,
)
from fukaji.d4_three_only_probe import snapshot
from fukaji.mate_oracle import ProofIncomplete, prove_mate_up_to_three


class D4MateAssistTests(unittest.TestCase):
    def test_two_proven_three_ply_moves_are_selected(self):
        expected = {"A": "3e3a+", "B": "2b2a+"}
        for name, sfen in DEVELOPMENT_POSITIONS[:2]:
            with self.subTest(name=name):
                board = shogi.Board(sfen)
                before = snapshot(board)
                row = select_with_mate_assist(
                    board, movetime_ms=200, max_depth=2,
                    proof_ms=50, max_legal_moves=1500)
                self.assertEqual(row["status"], "proven_three")
                self.assertEqual(row["move"], expected[name])
                self.assertEqual(snapshot(board), before)
                self.assertEqual(row["search_depth"], 0)

    def test_attacker_removed_falls_back_without_false_mate(self):
        name, sfen = DEVELOPMENT_POSITIONS[2]
        board = shogi.Board(sfen)
        before = snapshot(board)
        row = select_with_mate_assist(
            board, movetime_ms=200, max_depth=1,
            proof_ms=50, max_legal_moves=1500)
        self.assertEqual(row["status"], "complete_no_short_mate")
        self.assertIn(shogi.Move.from_usi(row["move"]), board.legal_moves)
        self.assertEqual(snapshot(board), before)

    def test_incomplete_proof_never_claims_mate(self):
        board = shogi.Board(DEVELOPMENT_POSITIONS[0][1])
        before = snapshot(board)
        row = select_with_mate_assist(
            board, movetime_ms=200, max_depth=1,
            proof_ms=50, max_legal_moves=1)
        self.assertEqual(row["status"], "incomplete")
        self.assertIn(shogi.Move.from_usi(row["move"]), board.legal_moves)
        self.assertEqual(snapshot(board), before)

    def test_expired_deadline_is_unknown_and_restores_board(self):
        board = shogi.Board(DEVELOPMENT_POSITIONS[0][1])
        before = snapshot(board)
        with self.assertRaises(ProofIncomplete):
            prove_mate_up_to_three(board, deadline=time.monotonic() - 1)
        self.assertEqual(snapshot(board), before)


if __name__ == "__main__":
    unittest.main()
