"""D3 independent legal-fact and explicit-request guard regression tests."""
import json
import tempfile
from pathlib import Path
import unittest

import shogi

from fukaji.d3_tactical_probe import (
    CASES, EXPECTED, ORDER, frozen_proofs, snapshot, validate_request, verify_case,
)


class D3TacticalProbeTests(unittest.TestCase):
    def test_five_frozen_cases_and_unique_sfen(self):
        proofs = frozen_proofs()
        self.assertEqual(len(proofs), 5)
        self.assertEqual(len({x["sfen"] for x in proofs}), 5)
        self.assertTrue(all("not a unique best move" in x["oracle_scope"]
                            for x in proofs))

    def test_oracles_restore_history(self):
        for name, sfen in CASES:
            with self.subTest(case=name):
                board = shogi.Board(sfen)
                before = snapshot(board)
                verify_case(name, board)
                self.assertEqual(snapshot(board), before)

    def test_poison_capture_has_forced_legal_recapture(self):
        board = shogi.Board(dict(CASES)["poison_recapture"])
        board.push(shogi.Move.from_usi("5e5d"))
        self.assertIn(shogi.Move.from_usi("5c5d"), board.legal_moves)

    def test_pawn_drop_mate_is_illegal(self):
        board = shogi.Board(dict(CASES)["illegal_pawn_drop_mate"])
        self.assertNotIn(shogi.Move.from_usi("P*5b"), board.legal_moves)

    def test_independent_mate_in_one_and_defensive_counterexample(self):
        """Enumerate every legal first move, then every legal evasion.

        The lance defends 2b after the rook vacates 2c. Removing it allows
        the defending king to capture the promoted rook; these are constructed
        development positions, not unseen strength-test examples.
        """
        mate_sfen = "8k/9/7R1/7L1/9/9/9/9/8K b - 1"
        board = shogi.Board(mate_sfen)
        before = snapshot(board)
        self.assertFalse(board.is_check())
        expected = shogi.Move.from_usi("2c2b+")
        self.assertIn(expected, board.legal_moves)
        mates = []
        for move in list(board.legal_moves):
            board.push(move)
            try:
                if board.is_checkmate():
                    self.assertTrue(board.is_check())
                    self.assertEqual(list(board.legal_moves), [])
                    mates.append(move.usi())
            finally:
                board.pop()
        self.assertIn(expected.usi(), mates)
        self.assertEqual(snapshot(board), before)

        # With no defending lance, the king can legally capture on 2b.
        unguarded = shogi.Board("8k/9/7R1/9/9/9/9/9/8K b - 1")
        unguarded_before = snapshot(unguarded)
        self.assertIn(expected, unguarded.legal_moves)
        unguarded.push(expected)
        try:
            self.assertFalse(unguarded.is_checkmate())
            self.assertIn(shogi.Move.from_usi("1a2b"), unguarded.legal_moves)
        finally:
            unguarded.pop()
        self.assertEqual(snapshot(unguarded), unguarded_before)

    def test_order_is_abba(self):
        self.assertEqual(ORDER, ("A", "B", "B", "A"))

    def test_request_is_disabled_by_default_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "request.json"
            path.write_text(json.dumps(EXPECTED))
            self.assertFalse(validate_request(path))
            path.write_text(json.dumps(dict(EXPECTED, enabled=True)))
            self.assertTrue(validate_request(path))
            path.write_text(json.dumps(dict(EXPECTED, movetime_ms=201)))
            with self.assertRaises(ValueError):
                validate_request(path)


if __name__ == "__main__":
    unittest.main()
