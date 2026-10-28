"""Offline protocol and ABBA audit tests; no source downloads or runner use."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fukaji.uct_cleanup_reserve_abba import REQUEST, ORDER, summarize, validate_request


class CleanupReserveABBAProtocolTest(unittest.TestCase):
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
            for change in ({"runner_jobs": 20}, {"single_pass_legal_moves": False},
                           {"total_budget_from_call": False}, {"linear_root_choice": False},
                           {"variable": "other"}, {"movetime_ms": 1000},
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
                    "pass": label, "single_pass": True, "total_budget": True,
                    "linear_root_choice": True, "cleanup_reserve": flag,
                    "pair": i, "target_ply": (12, 50, 110)[i // 4],
                    "sfen": "unique_" + str(i), "move": "7g7f",
                    "iterations": 100 if flag else 80, "internal_elapsed_ms": 180.0,
                    "elapsed_ms": 190.0 if flag else 205.0,
                })
        result = summarize(records)
        self.assertEqual(result["off_over_200ms"], 24)
        self.assertEqual(result["on_over_200ms"], 0)
        self.assertEqual(result["on_faster_positions"], 12)
        self.assertEqual(result["on_iterations"], 2400)
        self.assertEqual(result["on_zero_iteration_rows"], 0)
        self.assertTrue(result["single_pass_legal_moves_fixed"])
        self.assertTrue(result["total_budget_from_call_fixed"])
        self.assertTrue(result["linear_root_choice_fixed"])
        self.assertTrue(result["cleanup_aware_reserve_isolated"])
        records[24]["sfen"] = "changed"
        with self.assertRaises(ValueError):
            summarize(records)

    def test_zero_iterations_allowed_only_with_legal_move(self):
        records = []
        for label, flag in ORDER:
            for i in range(12):
                records.append({
                    "pass": label, "single_pass": True, "total_budget": True,
                    "linear_root_choice": True, "cleanup_reserve": flag,
                    "pair": i, "target_ply": (12, 50, 110)[i // 4],
                    "sfen": "unique_" + str(i), "move": "7g7f",
                    "iterations": 0 if flag else 1,
                    "elapsed_ms": 4.0, "internal_elapsed_ms": 4.0,
                })
        result = summarize(records)
        self.assertEqual(result["on_zero_iteration_rows"], 24)
        records[12]["iterations"] = -1
        with self.assertRaises(ValueError):
            summarize(records)

    def test_reject_short_and_unresolved(self):
        with self.assertRaises(ValueError):
            summarize([])
        records = []
        for label, flag in ORDER:
            for i in range(12):
                records.append({
                    "pass": label, "single_pass": True, "total_budget": True,
                    "linear_root_choice": True, "cleanup_reserve": flag,
                    "pair": i, "target_ply": (12, 50, 110)[i // 4],
                    "sfen": "unique_" + str(i), "move": "resign",
                    "iterations": 0, "elapsed_ms": 0, "internal_elapsed_ms": 0,
                })
        with self.assertRaises(ValueError):
            summarize(records)

    def test_mixed_parent_timing_is_counted_without_strength_claim(self):
        records = []
        for label, flag in ORDER:
            for i in range(12):
                records.append({
                    "pass": label, "single_pass": True, "total_budget": True,
                    "linear_root_choice": True, "cleanup_reserve": flag,
                    "pair": i, "target_ply": (12, 50, 110)[i // 4],
                    "sfen": "unique_" + str(i), "move": "7g7f",
                    "iterations": 10, "internal_elapsed_ms": 180.0,
                    "elapsed_ms": (210.0 if i in (1, 7) else 190.0) if flag else 200.0,
                })
        result = summarize(records)
        self.assertEqual(result["on_faster_positions"], 10)
        self.assertEqual(result["on_over_200ms"], 4)
        self.assertEqual(result["off_over_200ms"], 0)
        self.assertFalse(result["model_promoted"])


    def test_fixed_conditions_and_reserve_identity_are_required(self):
        rows = []
        for label, flag in ORDER:
            for i in range(12):
                rows.append({
                    "pass": label, "single_pass": True, "total_budget": True,
                    "linear_root_choice": True, "cleanup_reserve": flag,
                    "pair": i, "target_ply": (12, 50, 110)[i // 4],
                    "sfen": "fixed_" + str(i), "move": "7g7f",
                    "iterations": 1, "elapsed_ms": 199.0,
                    "internal_elapsed_ms": 198.0,
                })
        self.assertEqual(summarize(rows)["on_over_200ms"], 0)
        for field, value in (
            ("single_pass", False), ("total_budget", False),
            ("linear_root_choice", False), ("cleanup_reserve", False),
        ):
            changed = [dict(row) for row in rows]
            changed[12][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize(changed)


if __name__ == "__main__":
    unittest.main()
