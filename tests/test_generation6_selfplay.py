"""Research-only generation-6 selfplay safeguards."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from fukaji.generation6_selfplay import (
    MODEL_SHA, expected_request, validate_request, parent_id,
    play_shard, aggregate,
)


class Generation6SelfplayTest(unittest.TestCase):
    def test_request_frozen_and_no_adoption(self):
        self.assertEqual(MODEL_SHA, "99984388447a26cef9ca770bd3efe0cd38af784aba056395e95f90f5a4d4ff08")
        cfg = expected_request()
        self.assertEqual(validate_request(cfg), cfg)
        for key, value in (
            ("training_allowed", True),
            ("model_adoption_allowed", True),
            ("model_sha256", "0" * 64),
            ("games", 128),
            ("max_plies", 160),
            ("movetime_ms", 1),
            ("root_alpha_beta_both", False),
        ):
            with self.subTest(key=key):
                changed = dict(cfg)
                changed[key] = value
                with self.assertRaises(ValueError):
                    validate_request(changed)

    def test_unique_parent_ids(self):
        self.assertNotEqual(parent_id(20273427, ["7g7f"]),
                            parent_id(20273428, ["7g7f"]))

    def test_invalid_shard_and_model_rejected_before_play(self):
        with tempfile.TemporaryDirectory() as td:
            request = Path(td) / "request.json"
            import json
            request.write_text(json.dumps(expected_request()))
            model = Path(td) / "model.npz"
            model.write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "shard"):
                play_shard(request, model, td, 8)
            with self.assertRaisesRegex(ValueError, "SHA"):
                play_shard(request, model, td, 0)

    def test_missing_aggregate_shards_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "eight"):
                aggregate(td, td)


if __name__ == "__main__":
    unittest.main()
