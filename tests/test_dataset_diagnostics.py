import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import shogi

from fukaji.dataset_diagnostics import _split_stats, execute, turn_feature_probe
from fukaji.learning import FEATURES, features


class DatasetDiagnosticTests(unittest.TestCase):
    def test_turn_flip_cannot_reverse_piece_contributions(self):
        p = turn_feature_probe()
        self.assertTrue(p["same_board_non_turn_features_equal"])
        self.assertTrue(p["turn_feature_sign_reverses"])
        self.assertFalse(p["relative_piece_feature_representation"])

    def test_label_balance_and_loss(self):
        x = features(shogi.Board())
        w = np.zeros(FEATURES, dtype=np.float32)
        result = _split_stats([(np.stack([x, x]), np.array([1, 0], dtype=np.float32))], w)
        self.assertEqual(result["positions"], 2)
        self.assertEqual(result["positive_labels"], 1)
        self.assertAlmostEqual(result["brier_score"], 0.25)
        self.assertAlmostEqual(result["log_loss"], np.log(2), places=6)

    def test_sha_mismatch_blocks_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "games.jsonl").write_text("{}\n")
            np.savez_compressed(root / "model.npz", weight=np.zeros(FEATURES, dtype=np.float32))
            (root / "request.json").write_text(json.dumps({
                "source_run_id": 1, "dataset_sha256": "0" * 64, "model_sha256": "0" * 64}))
            with self.assertRaisesRegex(ValueError, "dataset_sha256"):
                execute(root / "games.jsonl", root / "model.npz",
                        root / "request.json", root / "out")


if __name__ == "__main__":
    unittest.main()
