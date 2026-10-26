import unittest
import numpy as np
from fukaji.generation_train import incremental_fit, score


class GenerationTrainTest(unittest.TestCase):
    def test_parent_is_not_mutated(self):
        parent = np.zeros(3, dtype=np.float32)
        x = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32)
        y = np.array([1, 0], dtype=np.float32)
        candidate = incremental_fit([(x, y)], parent)
        self.assertTrue(np.array_equal(parent, np.zeros(3)))
        self.assertTrue(np.isfinite(candidate).all())
        self.assertGreater(float(np.linalg.norm(candidate-parent)), 0)

    def test_empty_train_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            incremental_fit([], np.zeros(3, dtype=np.float32))

    def test_score_counts(self):
        x = np.array([[1, 0], [0, 1]], dtype=np.float32)
        s = score([(x, np.array([1, 0]))], np.zeros(2, dtype=np.float32))
        self.assertEqual(s["games"], 1)
        self.assertEqual(s["positions"], 2)
        self.assertAlmostEqual(s["log_loss"], np.log(2))


if __name__ == "__main__":
    unittest.main()
