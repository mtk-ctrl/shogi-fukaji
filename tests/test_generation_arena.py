import unittest
from unittest.mock import patch
import shogi

from fukaji.generation_arena import validate_request, play, aggregate, G2_SHA, G3_SHA


class GenerationArenaTest(unittest.TestCase):
    def request(self):
        return {"research_only": True, "generation2_run": 40404769521,
                "generation3_run": 40414745515, "generation2_sha256": G2_SHA,
                "generation3_sha256": G3_SHA, "seed": 20271727, "pairs": 16,
                "shards": 8, "opening_plies": 6, "max_plies": 160,
                "movetime_ms": 200, "depth": 2, "mix_percent": 25,
                "fast_relative": True, "root_alpha_beta_both": True,
                "training_allowed": False, "model_adoption_allowed": False}

    def test_research_request_frozen(self):
        self.assertEqual(validate_request(self.request()), self.request())
        for key, value in (("generation3_sha256", G2_SHA),
                           ("model_adoption_allowed", True),
                           ("movetime_ms", 1), ("root_alpha_beta_both", False)):
            cfg = self.request()
            cfg[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_request(cfg)

    def test_both_sides_same_root_bound_and_different_models(self):
        g2, g3 = object(), object()
        with patch("fukaji.generation_arena.select_move",
                   return_value=("7g7f", 1, 15)) as select:
            row = play([], True, g3, g2, max_plies=1)
        self.assertEqual(row["status"], "move_limit_unresolved")
        self.assertIsNone(row["winner"])
        self.assertEqual(row["moves"], ["7g7f"])
        self.assertIs(select.call_args.kwargs["model"], g3)
        self.assertTrue(select.call_args.kwargs["root_alpha_beta"])
        self.assertTrue(select.call_args.kwargs["fast_relative"])

    def test_no_move_is_unresolved(self):
        with patch("fukaji.generation_arena.select_move",
                   return_value=("resign", 0, 1)):
            row = play([], True, object(), object(), max_plies=1)
        self.assertEqual(row["status"], "no_move_unresolved")
        self.assertIsNone(row["winner"])

    def test_incomplete_aggregate_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(ValueError, "eight"):
                aggregate(path, path)


if __name__ == "__main__":
    unittest.main()
