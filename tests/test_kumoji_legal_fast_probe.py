"""Regression for conservative fast legality on a real starting position."""
import tempfile
import unittest
from pathlib import Path
import shogi
from fukaji.engine import fast_legal_moves
from fukaji.kumoji_legal_fast_probe import audit_legal_sequences
from fukaji.kumoji_loss_study import snapshot


class FastLegalResearchTests(unittest.TestCase):
    def test_fast_source_has_exact_standard_moves_and_no_state_change(self):
        board=shogi.Board()
        before=snapshot(board)
        ordinary={move.usi() for move in board.legal_moves}
        fast=[move.usi() for move in fast_legal_moves(board)]
        self.assertEqual(set(fast),ordinary)
        self.assertEqual(len(fast),len(ordinary))
        self.assertEqual(snapshot(board),before)

    def test_source_hash_is_required(self):
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/"games.jsonl"
            file.write_text("{}\n")
            with self.assertRaisesRegex(ValueError,"source changed"):
                audit_legal_sequences(file)
        with self.assertRaises(ValueError):
            audit_legal_sequences(__file__,stride=0)


if __name__=="__main__":
    unittest.main()
