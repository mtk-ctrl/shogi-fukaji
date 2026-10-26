import unittest
from unittest.mock import patch
import shogi

from fukaji.continue_unresolved import advance, validate_request, MODEL_SHA, SOURCE_RUNS, execute
from fukaji.continuation_aggregate import aggregate


class ContinuationTest(unittest.TestCase):
    def valid(self):
        return {
            "research_only": True, "source_runs": list(SOURCE_RUNS),
            "original_plies": 160, "max_plies": 240,
            "relative_model_sha256": MODEL_SHA, "movetime_ms": 200,
            "depth": 2, "mix_percent": 25, "fast_relative": True,
            "root_alpha_beta_default": False, "root_alpha_beta_candidate": True,
            "training_allowed": False, "model_adoption_allowed": False,
        }

    def test_frozen_request(self):
        self.assertEqual(validate_request(self.valid()), self.valid())
        for field, bad in (("max_plies", 400), ("training_allowed", True),
                           ("movetime_ms", 1), ("source_runs", []),
                           ("root_alpha_beta_default", True)):
            altered = self.valid()
            altered[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_request(altered)

    def test_one_actual_legal_continuation_and_roles(self):
        board = shogi.Board()
        with patch("fukaji.continue_unresolved.select_move",
                   return_value=("7g7f", 1, 23)) as select:
            result = advance(board, True, None, 1)
        self.assertEqual(result["status"], "move_limit_unresolved")
        self.assertIsNone(result["winner"])
        self.assertEqual(result["continuation_moves"], ["7g7f"])
        self.assertEqual(result["continuation_stats"]["candidate_nodes"], 23)
        self.assertEqual(result["continuation_stats"]["baseline_nodes"], 0)
        self.assertTrue(select.call_args.kwargs["root_alpha_beta"])
        self.assertTrue(select.call_args.kwargs["fast_relative"])

    def test_baseline_turn_never_uses_candidate_bound(self):
        board = shogi.Board()
        with patch("fukaji.continue_unresolved.select_move",
                   return_value=("7g7f", 0, 10)) as select:
            result = advance(board, False, None, 1)
        self.assertFalse(select.call_args.kwargs["root_alpha_beta"])
        self.assertEqual(result["continuation_stats"]["baseline_incomplete"], 1)

    def test_no_move_does_not_fabricate_loss(self):
        with patch("fukaji.continue_unresolved.select_move",
                   return_value=("resign", 0, 1)):
            result = advance(shogi.Board(), True, None, 1)
        self.assertEqual(result["status"], "no_move_unresolved")
        self.assertIsNone(result["winner"])
        self.assertEqual(result["continuation_moves"], [])

    def test_illegal_search_move_fails_closed(self):
        with patch("fukaji.continue_unresolved.select_move",
                   return_value=("1a1a", 0, 1)):
            with self.assertRaises(ValueError):
                advance(shogi.Board(), True, None, 1)

    def test_aggregate_missing_shards_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'eight'):
                aggregate(d, d)

    def test_continuation_limit(self):
        with self.assertRaises(ValueError):
            advance(shogi.Board(), True, None, 81)


if __name__ == "__main__":
    unittest.main()
