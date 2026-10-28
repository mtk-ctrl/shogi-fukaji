"""Lightweight regression for frozen UCT holdout metadata (no games)."""
import json
import unittest
from pathlib import Path
from fukaji.uct_generalization_audit import validate_parent_split

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf"

class TestUCTHoldoutManifest(unittest.TestCase):
    def test_disjoint_frozen_source(self):
        m=json.loads((ROOT/"docs/UCT_HOLDOUT_12_PARENTS_20261029.json").read_text())
        result=validate_parent_split(m["development"],m["holdout"],source_sha256=SOURCE)
        self.assertEqual(result["holdout_by_ply"],{12:4,50:4,110:4})
        self.assertEqual([r["pair"] for r in m["development"]],list(range(12)))
        self.assertEqual([r["pair"] for r in m["holdout"]],list(range(12,24)))
        self.assertEqual([r["seed"] for r in m["holdout"]],list(range(20279128,20279140)))
        self.assertFalse(m["measurement_started"])
        self.assertFalse(m["unresolved_used_as_teacher"])
        self.assertEqual(m["source_sha256"],SOURCE)
        for group in ("development","holdout"):
            for row in m[group]:
                self.assertTrue(row["g8_black"])
                self.assertEqual(row["game_row_index"],2*row["pair"])
                self.assertNotIn("winner",row)
                self.assertNotIn("training_label",row)
