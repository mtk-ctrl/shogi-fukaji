"""Holdout-only ABBA protocol tests. No downloads, 200ms run, or games."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fukaji.uct_holdout_cleanup_abba import (
    REQUEST, ORDER, validate_request, prepare_holdout_positions,
)
from fukaji.uct_cleanup_reserve_abba import summarize, MODEL_SHA, SOURCE_SHA
from fukaji.uct_generalization_audit import validate_holdout_abba


class HoldoutReserveProtocolTests(unittest.TestCase):
    def test_request_default_off_and_exact_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "request.json"
            with patch.dict(os.environ, {"GITHUB_OUTPUT": ""}):
                for enabled in (False, True):
                    path.write_text(json.dumps(dict(REQUEST, enabled=enabled)))
                    self.assertIs(validate_request(path), enabled)
                for key, value in (
                    ("split", "development"), ("positions", 24),
                    ("runner_jobs", 20), ("movetime_ms", 250),
                    ("holdout_manifest", "docs/other.json"),
                    ("single_pass_legal_moves", False),
                    ("total_budget_from_call", False),
                    ("linear_root_choice", False),
                    ("variable", "other"), ("research_only", False),
                    ("model_adoption_allowed", True),
                ):
                    path.write_text(json.dumps({**REQUEST, key: value}))
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        validate_request(path)

    def test_synthetic_48_rows_holdout_only(self):
        parents = []
        for i in range(12):
            parents.append({
                "parent_id": f"holdout-{i}", "game_id": f"game-{i}",
                "pair": i + 12, "ply": (12, 50, 110)[i // 4],
                "sfen": f"unique-sfen-{i}", "seed": 20279128 + i,
                "source_sha256": SOURCE_SHA,
            })
        rows = []
        for name, flag in ORDER:
            for p in parents:
                rows.append({
                    "pass": name, "cleanup_reserve": flag,
                    "single_pass": True, "total_budget": True,
                    "linear_root_choice": True,
                    "pair": p["pair"], "target_ply": p["ply"],
                    "parent_id": p["parent_id"], "game_id": p["game_id"],
                    "seed": p["seed"], "sfen": p["sfen"],
                    "source_sha256": SOURCE_SHA, "model_sha256": MODEL_SHA,
                    "movetime_ms": 200, "move": "7g7f", "legal": True,
                    "board_restored": True, "iterations": 0 if flag else 3,
                    "external_ms": 198.0 if flag else 201.0,
                    "elapsed_ms": 198.0 if flag else 201.0,
                    "internal_elapsed_ms": 196.0,
                })
        self.assertEqual(validate_holdout_abba(
            rows, parents, model_sha256=MODEL_SHA,
            source_sha256=SOURCE_SHA)["observations"], 48)
        self.assertEqual(summarize(rows)["on_over_200ms"], 0)
        self.assertEqual(summarize(rows)["off_over_200ms"], 24)
        for key, bad in (("seed", 0), ("game_id", "dev-game"),
                         ("board_restored", False), ("legal", False),
                         ("source_sha256", "bad")):
            tampered = [dict(r) for r in rows]
            tampered[12][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_holdout_abba(tampered, parents,
                                      model_sha256=MODEL_SHA,
                                      source_sha256=SOURCE_SHA)
        tampered = [dict(r) for r in rows]
        tampered[12]["cleanup_reserve"] = False
        with self.assertRaises(ValueError):
            summarize(tampered)

    def test_workflow_preflight_references_existing_entrypoints(self):
        workflow = Path(".github/workflows/uct-holdout-reserve-abba.yml").read_text()
        readme = Path(".github/workflows/README.md").read_text()
        self.assertIn("uct-holdout-reserve-abba.yml", readme)
        self.assertIn("test_uct_holdout_cleanup_abba.py", workflow)
        self.assertIn("test_uct_generalization_audit.py", workflow)
        self.assertNotIn("scripts/check_match_workflows.py", workflow)
        self.assertNotIn("scripts/check_workflow_inventory.py", workflow)
        self.assertIn("steps.protocol.outputs.enabled == 'true'", workflow)
        self.assertIn("cancel-in-progress: false", workflow)

    def test_reject_manifest_overlap_before_any_search(self):
        bad = {"holdout": [{"pair": i, "seed": 0} for i in range(12)]}
        with self.assertRaises(ValueError):
            prepare_holdout_positions([], bad)


if __name__ == "__main__":
    unittest.main()
