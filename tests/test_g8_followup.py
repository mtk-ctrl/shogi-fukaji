"""Fail-closed tests for G8 40-position read-only follow-up."""
import unittest
from unittest.mock import patch

import shogi

from fukaji.g8_followup import (
    expected_request, validate_request, select_changed, opponent_reply,
    SHARDS, PER_SHARD, FOLLOWUP_SHA,
)


def fixtures():
    selected, results = [], []
    for i in range(60):
        played = "7g7f"
        deep = "2g2f" if i < 40 else played
        selected.append({
            "analysis_index": i, "candidate_id": i,
            "parent_id": "parent-" + str(i),
            "position_sfen_state": "state-" + str(i),
            "next_played_move_usi": played, "split": "train",
            "reason_category": "mate_tail",
        })
        results.append({
            "analysis_index": i, "candidate_id": i,
            "parent_id": "parent-" + str(i),
            "source_played_move": played,
            "baseline": {"move": played, "depth_cap": 2},
            "long_think": {"move": deep, "depth_cap": 5},
            "changed_from_baseline": i < 40,
            "changed_from_played": i < 40,
        })
    return selected, results


class FollowupTests(unittest.TestCase):
    def test_request_disabled_by_default_and_exact_protocol(self):
        self.assertFalse(validate_request(expected_request(False)))
        self.assertTrue(validate_request(expected_request(True)))
        self.assertEqual((SHARDS, PER_SHARD), (20, 2))
        self.assertEqual(len(FOLLOWUP_SHA), 64)
        for field, value in (
            ("training_allowed", True), ("adoption_allowed", True),
            ("positions", 41), ("independent_ms", 1000),
            ("aggregate_artifact_id", 1),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_request({**expected_request(True), field: value})

    def test_selects_exact_forty_with_frozen_identity(self):
        selected, results = fixtures()
        chosen = select_changed(selected, results)
        self.assertEqual(len(chosen), 40)
        self.assertEqual([r["followup_index"] for r in chosen], list(range(40)))
        self.assertEqual([r["analysis_index"] for r in chosen], list(range(40)))
        self.assertEqual(len({r["parent_id"] for r in chosen}), 40)
        self.assertEqual(chosen[0]["long_think"]["move"], "2g2f")

    def test_source_tampering_and_holdout_rejected(self):
        for tamper in ("duplicate", "holdout", "played", "change", "count"):
            with self.subTest(tamper=tamper), self.assertRaises(ValueError):
                selected, results = fixtures()
                if tamper == "duplicate":
                    selected[1]["parent_id"] = selected[0]["parent_id"]
                elif tamper == "holdout":
                    selected[0]["split"] = "test"
                elif tamper == "played":
                    results[0]["source_played_move"] = "3c3d"
                elif tamper == "change":
                    results[0]["changed_from_baseline"] = False
                else:
                    results[40]["changed_from_baseline"] = True
                    results[40]["changed_from_played"] = True
                    results[40]["long_think"]["move"] = "2g2f"
                select_changed(selected, results)

    def test_opponent_branch_undo_and_legality(self):
        board = shogi.Board()
        before = board.sfen()
        history = dict(board.transpositions)
        with patch("fukaji.g8_followup.search",
                   return_value={"move": "3c3d", "completed_depth": 2,
                                 "elapsed_ms": 1000.0, "nodes": 10}) as mocked:
            result = opponent_reply(board, "7g7f", object())
        self.assertEqual(result["terminal"], "nonterminal")
        self.assertEqual(result["reply"]["move"], "3c3d")
        self.assertEqual(board.sfen(), before)
        self.assertEqual(dict(board.transpositions), history)
        mocked.assert_called_once()
        with self.assertRaises(ValueError):
            opponent_reply(board, "invalid", object())
        self.assertEqual(board.sfen(), before)


if __name__ == "__main__":
    unittest.main()
