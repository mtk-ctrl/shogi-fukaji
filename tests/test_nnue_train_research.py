"""Training isolation and deterministic research-only NNUE regression tests."""
import unittest

import numpy as np

from fukaji.nnue_research import INPUTS, HIDDEN
from fukaji.nnue_train_research import (
    expected_request, validate_request, frozen_parts, fit_train_only, loss,
)


class NnueTrainResearchTests(unittest.TestCase):
    def sample(self, rows=4, feature=10, label=1):
        x = np.zeros((rows, INPUTS + 1), dtype=np.float32)
        x[:, feature] = 1.0
        x[:, -1] = 1.0
        y = np.full(rows, label, dtype=np.float32)
        return x, y

    def test_explicit_request_default_off_and_no_promotion(self):
        self.assertFalse(validate_request(expected_request(False)))
        self.assertTrue(validate_request(expected_request(True)))
        self.assertFalse(expected_request(True)["model_adoption_allowed"])
        self.assertFalse(expected_request(True)["engine_integration_allowed"])
        for key, value in (("epochs", 5), ("holdout_used_for_training", True),
                           ("unresolved_used_as_teacher", True),
                           ("model_adoption_allowed", True),
                           ("source_artifact_id", 1)):
            with self.subTest(key=key):
                cfg = expected_request(True)
                cfg[key] = value
                with self.assertRaises(ValueError):
                    validate_request(cfg)

    def test_frozen_parent_groups_and_split_counts(self):
        mapping = {f"tr-{i}": "train" for i in range(767)}
        mapping.update({f"va-{i}": "validation" for i in range(95)})
        mapping.update({f"te-{i}": "test" for i in range(95)})
        rows = [(f"tr-{i}", *self.sample()) for i in range(665)]
        rows += [(f"va-{i}", *self.sample()) for i in range(81)]
        rows += [(f"te-{i}", *self.sample()) for i in range(84)]
        parts = frozen_parts(rows, mapping)
        self.assertEqual([len(parts[k]) for k in ("train", "validation", "test")],
                         [665, 81, 84])
        with self.assertRaises(ValueError):
            frozen_parts(rows + [rows[0]], mapping)
        with self.assertRaises(ValueError):
            frozen_parts(rows, {**mapping, "unknown": "train"})
        with self.assertRaises(ValueError):
            frozen_parts(rows[:-1] + [("not-in-map", *self.sample())], mapping)

    def test_train_only_deterministic_and_finite(self):
        train = [self.sample(5, feature=3, label=1),
                 self.sample(5, feature=15, label=0)]
        a, a_stats = fit_train_only(train, epochs=2, seed=71)
        b, b_stats = fit_train_only(train, epochs=2, seed=71)
        self.assertEqual(a_stats, b_stats)
        self.assertEqual(a_stats["optimizer_steps"], 4)
        self.assertEqual(a_stats["sampled_positions_total"], 20)
        self.assertEqual(a.hidden_weights.shape, (INPUTS, HIDDEN))
        self.assertTrue(np.array_equal(a.hidden_weights, b.hidden_weights))
        self.assertTrue(np.array_equal(a.output_weights, b.output_weights))
        self.assertTrue(np.isfinite(a.output_bias))
        self.assertTrue(np.isfinite(loss(train, a)["log_loss"]))

    def test_holdout_is_not_passed_to_optimizer(self):
        train = [self.sample(3, feature=7, label=1)]
        model, _ = fit_train_only(train, epochs=2, seed=17)
        val_a = [self.sample(3, feature=19, label=0)]
        val_b = [self.sample(3, feature=19, label=1)]
        self.assertNotEqual(loss(val_a, model)["log_loss"],
                            loss(val_b, model)["log_loss"])
        model2, _ = fit_train_only(train, epochs=2, seed=17)
        self.assertTrue(np.array_equal(model.hidden_weights, model2.hidden_weights))

    def test_reject_invalid_train_labels(self):
        x, y = self.sample()
        y[0] = 0.5
        with self.assertRaises(ValueError):
            fit_train_only([(x, y)], epochs=1)


if __name__ == "__main__":
    unittest.main()
