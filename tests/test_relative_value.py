import tempfile
import unittest
from pathlib import Path

import numpy as np
import shogi

from fukaji.learning import FEATURES, features
from fukaji.relative_value import RelativeValue, relative_features
from fukaji.arena import run


class RelativeValueTests(unittest.TestCase):
    def test_turn_normalization_changes_piece_encoding_not_just_bias(self):
        b = shogi.Board()
        words = b.sfen().split()
        words[1] = "w"
        opposite = shogi.Board(" ".join(words))
        a = relative_features(b)
        z = relative_features(opposite)
        self.assertEqual(a.shape, (FEATURES,))
        self.assertEqual(float(a[-1]), 1.0)
        self.assertEqual(float(z[-1]), 1.0)
        self.assertFalse(np.array_equal(a[:-1], z[:-1]))
        self.assertTrue(np.array_equal(features(b)[:-1], features(opposite)[:-1]))

    def test_sparse_relative_inference_matches_dense(self):
        rng = np.random.default_rng(2027)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "relative.npz"
            weights = rng.normal(0, 0.1, FEATURES).astype(np.float32)
            np.savez_compressed(p, weight=weights)
            model = RelativeValue(p, mix_percent=25)
            b = shogi.Board()
            for ply in range(30):
                dense = float(relative_features(b) @ weights)
                expected = int(round(max(-10.0, min(10.0, dense)) * 100))
                self.assertLessEqual(abs(model.score(b) - expected), 1, f"ply={ply}")
                moves = list(b.legal_moves)
                if not moves or b.is_checkmate():
                    break
                b.push(moves[int(rng.integers(len(moves)))])

    def test_unknown_model_schema_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown model schema"):
            run("does-not-exist.npz", "/tmp/unused.jsonl", model_type="not-a-schema")


if __name__ == "__main__":
    unittest.main()
