import unittest
from unittest.mock import patch
import shogi

from fukaji.generation4_vs3_arena import validate_request, play, aggregate, G3_SHA, G4_SHA


class Generation4Vs3ArenaTest(unittest.TestCase):
    def request(self):
        return {"research_only": True, "generation3_run": 40414745515,
                "generation4_run": 40418598213, "generation3_sha256": G3_SHA,
                "generation4_sha256": G4_SHA, "seed": 20272327, "pairs": 32,
                "shards": 8, "opening_plies": 6, "max_plies": 160,
                "movetime_ms": 200, "depth": 2, "mix_percent": 25,
                "fast_relative": True, "root_alpha_beta_both": True,
                "training_allowed": False, "model_adoption_allowed": False}

    def test_research_request_frozen(self):
        self.assertEqual(validate_request(self.request()), self.request())
        for key, value in (("generation4_sha256", G3_SHA),
                           ("model_adoption_allowed", True),
                           ("movetime_ms", 1), ("root_alpha_beta_both", False)):
            cfg = self.request()
            cfg[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_request(cfg)

    def test_both_sides_same_root_bound_and_different_models(self):
        g3, g4 = object(), object()
        with patch("fukaji.generation4_vs3_arena.select_move",
                   return_value=("7g7f", 1, 15)) as select:
            row = play([], True, g4, g3, max_plies=1)
        self.assertEqual(row["status"], "move_limit_unresolved")
        self.assertIsNone(row["winner"])
        self.assertEqual(row["moves"], ["7g7f"])
        self.assertIs(select.call_args.kwargs["model"], g4)
        self.assertTrue(select.call_args.kwargs["root_alpha_beta"])
        self.assertTrue(select.call_args.kwargs["fast_relative"])

    def test_no_move_is_unresolved(self):
        with patch("fukaji.generation4_vs3_arena.select_move",
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
