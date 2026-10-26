import unittest
from fukaji.root_alpha_probe import (
    validate_request, SOURCE_RUN, MODEL_RUN, MODEL_SHA, GAMES_SHA, REPORT_SHA,
)


class RootAlphaProbeTests(unittest.TestCase):
    def config(self):
        return {
            "research_only": True, "source_run_id": SOURCE_RUN,
            "model_run_id": MODEL_RUN, "model_sha256": MODEL_SHA,
            "games_sha256": GAMES_SHA, "report_sha256": REPORT_SHA,
            "source_seed": 20271227, "source_pairs": 16,
            "max_samples": 12, "time_ms": 200, "max_depth": 2,
            "mix_percent": 25, "fast_relative": True,
            "root_pruning_default": False, "training_allowed": False,
        }

    def test_frozen_protocol(self):
        self.assertEqual(validate_request(self.config()), self.config())
        for key, value in (
            ("source_run_id", 0), ("model_sha256", "bad"),
            ("max_samples", 24), ("time_ms", 1000),
            ("max_depth", 3), ("training_allowed", True),
            ("root_pruning_default", True), ("source_seed", 0),
        ):
            c = self.config()
            c[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_request(c)

    def test_unknown_field_rejected(self):
        c = self.config()
        c["dispatch_more_games"] = True
        with self.assertRaises(ValueError):
            validate_request(c)


if __name__ == "__main__":
    unittest.main()
