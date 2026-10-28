"""Fail-closed G8 60-position long-think protocol tests."""
import hashlib
import unittest
from collections import Counter
from unittest.mock import patch

import shogi

from fukaji.g8_longthink import (
    expected_request, validate_request, select_sixty, history_board,
    SHARDS, PER_SHARD, QUOTAS, BASE_MS, LONG_MS,
)


def fixtures():
    rows = []
    i = 0
    for kind, count in (("mate_tail", 40), ("mate_middle", 20),
                        ("unresolved_late", 20), ("early_mixed", 20)):
        for j in range(count):
            # Balanced in each category, with unique parent and board markers.
            parity = j % 2
            rows.append(dict(candidate_id=i, parent_id="parent-" + str(i),
                             position_sfen_state="state-" + str(i),
                             position_before_ply=100 + parity,
                             reason_category=kind, split="train"))
            i += 1
    return rows


class G8LongThinkTests(unittest.TestCase):
    def test_request_strict_and_disabled_default(self):
        self.assertFalse(validate_request(expected_request(False)))
        self.assertTrue(validate_request(expected_request(True)))
        for change in ({"positions": 61}, {"training_allowed": True},
                       {"long_ms": 30000}, {"source_artifact_id": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_request({**expected_request(True), **change})
        self.assertEqual((SHARDS, PER_SHARD, BASE_MS, LONG_MS),
                         (20, 3, 200, 20000))

    def test_deterministic_unique_balanced_sixty(self):
        rows = fixtures()
        a = select_sixty(rows)
        b = select_sixty(rows)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 60)
        self.assertEqual(Counter(r["reason_category"] for r in a), QUOTAS)
        self.assertEqual(Counter(r["position_before_ply"] % 2 for r in a),
                         {0: 30, 1: 30})
        self.assertEqual(len({r["parent_id"] for r in a}), 60)
        self.assertEqual([r["analysis_index"] for r in a], list(range(60)))

    def test_insufficient_side_and_duplicate_state_rejected(self):
        rows = fixtures()
        for r in rows:
            if r["reason_category"] == "early_mixed":
                r["position_before_ply"] = 101
        with self.assertRaisesRegex(ValueError, "side-balanced"):
            select_sixty(rows)
        rows = fixtures()
        for r in rows:
            r["position_sfen_state"] = "same"
        with self.assertRaisesRegex(ValueError, "independent"):
            select_sixty(rows)

    def test_legal_history_and_position_reconstruction(self):
        board = shogi.Board()
        move = shogi.Move.from_usi("7g7f")
        self.assertIn(move, board.legal_moves)
        board.push(move)
        row = {"history_usi": ["7g7f"],
               "position_before_ply": 2,
               "position_sfen_state": " ".join(board.sfen().split()[:3]),
               "next_played_move_usi": "3c3d"}
        restored = history_board(row)
        self.assertEqual(restored.sfen(), board.sfen())
        row["position_sfen_state"] = "invalid"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            history_board(row)

    def test_diagnostic_does_not_prove_training(self):
        self.assertFalse(expected_request(True)["training_allowed"])
        self.assertFalse(expected_request(True)["adoption_allowed"])


if __name__ == "__main__":
    unittest.main()
