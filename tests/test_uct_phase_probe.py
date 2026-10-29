"""Offline guards for the UCT phase diagnostic; no real research run."""
import json
import tempfile
import unittest
from pathlib import Path

from fukaji.uct_phase_probe import EXPECTED, percentile, summarize_trace, validate_request


class UctPhaseProbeTests(unittest.TestCase):
    def test_disabled_and_enabled_only_with_exact_frozen_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "request.json"
            for enabled in (False, True):
                path.write_text(json.dumps(dict(EXPECTED, enabled=enabled)), encoding="utf-8")
                self.assertEqual(validate_request(path), enabled)
            for changed in ({"movetime_ms": 201}, {"parent_indices": [6]}, {"model_adoption_allowed": True}):
                path.write_text(json.dumps({**EXPECTED, **changed}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_request(path)

    def test_percentile_interpolation(self):
        self.assertEqual(percentile([1, 2, 3], 50), 2)
        self.assertEqual(percentile([1, 3], 90), 2.8)
        with self.assertRaises(ValueError):
            percentile([], 50)

    def test_phase_totals_and_unexpected_long_iteration(self):
        trace = []
        for i, ms in enumerate((1.0, 8.0, 2.0)):
            trace.append({
                "total_ms": ms, "reserve_ms": 3.0,
                "selection_ms": ms / 5, "expansion_ms": ms / 5,
                "leaf_ms": ms / 5, "backup_ms": ms / 5,
                "undo_ms": ms / 5,
            })
        result = summarize_trace(trace)
        self.assertEqual(result["iterations"], 3)
        self.assertEqual(result["unexpected_long_iterations"], 1)
        self.assertAlmostEqual(result["phase_sum_ms"]["leaf_ms"], 2.2)

    def test_workflow_is_default_off_and_main_only(self):
        workflow = Path(".github/workflows/uct-phase-profile.yml").read_text(encoding="utf-8")
        self.assertIn("github.ref == 'refs/heads/main'", workflow)
        self.assertIn("steps.protocol.outputs.enabled == 'true'", workflow)
        self.assertIn("permissions:\n  contents: read\n  actions: read", workflow)
        self.assertNotIn("schedule:", workflow)
        self.assertIn("cancel-in-progress: false", workflow)


if __name__ == "__main__":
    unittest.main()
