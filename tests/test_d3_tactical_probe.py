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
        board.push_usi("5e5d")
        self.assertIn(shogi.Move.from_usi("5c5d"), board.legal_moves)

    def test_pawn_drop_mate_is_illegal(self):
        board = shogi.Board(dict(CASES)["illegal_pawn_drop_mate"])
        self.assertNotIn(shogi.Move.from_usi("P*5b"), board.legal_moves)

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
