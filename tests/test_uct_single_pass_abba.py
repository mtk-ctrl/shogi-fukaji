"""Offline protocol and ABBA audit tests; no source downloads or runner use."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fukaji.uct_single_pass_abba import REQUEST, ORDER, summarize, validate_request


class SinglePassABBAProtocolTest(unittest.TestCase):
    def test_disabled_and_explicit_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "request.json"
            with patch.dict(os.environ, {"GITHUB_OUTPUT": ""}):
                for enabled in (False, True):
                    path.write_text(json.dumps(dict(REQUEST, enabled=enabled)))
                    self.assertIs(validate_request(path), enabled)

    def test_reject_unapproved_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "request.json"
            for change in ({"runner_jobs": 20}, {"movetime_ms": 1000},
                           {"research_only": False}, {"model_adoption_allowed": True},
                           {"source_run": 0}, {"unexpected": True}):
                path.write_text(json.dumps({**REQUEST, **change}))
                with self.assertRaises(ValueError):
                    validate_request(path)

    def test_summary_and_pairwise_identity(self):
        records = []
        for label, flag in ORDER:
            for i in range(12):
                records.append({
                    "pass": label, "single_pass": flag,
                    "pair": i, "target_ply": (12, 50, 110)[i // 4],
                    "sfen": "unique_" + str(i), "move": "7g7f",
                    "iterations": 100 if flag else 80,
                    "elapsed_ms": 190.0 if flag else 205.0,
                })
        result = summarize(records)
        self.assertEqual(result["off_over_200ms"], 24)
        self.assertEqual(result["on_over_200ms"], 0)
        self.assertEqual(result["on_faster_positions"], 12)
        self.assertEqual(result["on_iterations"], 2400)
        records[24]["sfen"] = "changed"
        with self.assertRaises(ValueError):
            summarize(records)

    def test_reject_short_and_unresolved(self):
        with self.assertRaises(ValueError):
            summarize([])
        records = []
        for label, flag in ORDER:
            for i in range(12):
                records.append({
                    "pass": label, "single_pass": flag,
                    "pair": i, "target_ply": (12, 50, 110)[i // 4],
                    "sfen": "unique_" + str(i), "move": "resign",
                    "iterations": 0, "elapsed_ms": 0,
                })
        with self.assertRaises(ValueError):
            summarize(records)


if __name__ == "__main__":
    unittest.main()
