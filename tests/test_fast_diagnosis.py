"""Frozen 32-game diagnostic input and replay tests."""
import unittest

import shogi

from fukaji.fast_diagnosis import MODEL_SHA, SOURCE_RUN, analyze_game, validate_request


class FastDiagnosisTests(unittest.TestCase):
    def config(self):
        return {"research_only": True, "source_run_id": SOURCE_RUN,
                "model_sha256": MODEL_SHA, "source_arena_seed": 20271227,
                "source_pairs": 16, "source_max_plies": 160,
                "training_allowed": False,
                "games_sha256": "a" * 64, "report_sha256": "b" * 64}

    def test_fail_closed_protocol(self):
        self.assertEqual(validate_request(self.config())["source_pairs"], 16)
        for key, value in (("source_run_id", 1), ("research_only", False),
                           ("model_sha256", "a" * 64), ("source_pairs", 8),
                           ("training_allowed", True), ("games_sha256", "bad")):
            with self.subTest(key=key):
                c = self.config()
                c[key] = value
                with self.assertRaises(ValueError):
                    validate_request(c)

    def test_unresolved_stays_unresolved(self):
        row = {"opening": ["7g7f", "3c3d"], "moves": ["2g2f"],
               "status": "move_limit_unresolved", "winner": None,
               "pair": 0, "fast_black": True}
        got = analyze_game(row, 3)
        self.assertEqual(got["status"], "move_limit_unresolved")
        self.assertEqual(got["plies"], 3)
        self.assertEqual(got["captures"], 0)
        with self.assertRaises(ValueError):
            analyze_game(row, 4)
        row["winner"] = shogi.BLACK
        with self.assertRaises(ValueError):
            analyze_game(row, 3)


if __name__ == "__main__":
    unittest.main()
