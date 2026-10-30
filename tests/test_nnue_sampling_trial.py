import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fukaji.nnue_sampling_trial import (
    expected_request, validate_request, train_only_bytes, main,
)


class SamplingTrialTests(unittest.TestCase):
    def test_exact_request_and_no_numeric_boolean_substitution(self):
        self.assertFalse(validate_request(expected_request(False)))
        self.assertTrue(validate_request(expected_request(True)))
        for key, value in (("enabled", 1), ("old_holdout_evaluation_allowed", True),
                           ("fresh_games_training_allowed", True),
                           ("evaluation_games", 401), ("training_seed", 42),
                           ("samplers", ["adjacent_pairs"])):
            with self.subTest(key=key):
                cfg = expected_request(True)
                cfg[key] = value
                with self.assertRaises(ValueError):
                    validate_request(cfg)

    def test_old_holdouts_filtered_before_replay_or_outcome_access(self):
        rows = [{"parent_id": "train", "moves": ["retained"], "winner": 1},
                {"parent_id": "validation", "moves": "POISON", "winner": "DO_NOT_READ"},
                {"parent_id": "test", "moves": None}]
        raw = b"\n".join(json.dumps(x).encode() for x in rows)
        result = train_only_bytes(raw, {"train": "train", "validation": "validation", "test": "test"})
        self.assertEqual([json.loads(x) for x in result.splitlines()], rows[:1])

    def test_disabled_entry_only_checks_protocol_without_preparing(self):
        with TemporaryDirectory() as tmp:
            req = Path(tmp) / "request.json"
            req.write_text(json.dumps(expected_request(False)))
            output = Path(tmp) / "output.txt"
            with patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}), \
                    patch("sys.argv", ["trial", "--mode", "check", "--request", str(req)]), \
                    patch("fukaji.nnue_sampling_trial.prepare") as prepare:
                main()
                prepare.assert_not_called()
            self.assertEqual(output.read_text(), "enabled=false\n")
            with patch("sys.argv", ["trial", "--mode", "prepare", "--request", str(req), "--out", tmp]):
                with self.assertRaises(ValueError):
                    main()


if __name__ == "__main__":
    unittest.main()
