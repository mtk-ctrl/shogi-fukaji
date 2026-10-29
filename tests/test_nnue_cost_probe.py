"""D6-1: synthetic NNUE inference/update overhead; no training."""
import json
import unittest

from fukaji.nnue_cost_probe import fixed_positions, measure


class NNUECostGateTests(unittest.TestCase):
    def test_fixed_positions_and_bounded_measurement(self):
        positions = fixed_positions()
        self.assertEqual([ply for ply, _ in positions], [0, 20, 60])
        report = measure(repeats=12)
        self.assertEqual(report["position_count"], 3)
        self.assertTrue(report["synthetic_untrained_weights"])
        self.assertFalse(report["training"])
        self.assertFalse(report["sealed_d5_parents_accessed"])
        for case in report["cases"]:
            self.assertTrue(case["board_restored"])
            for name in ("classical_eval_ms", "nnue_full_eval_ms",
                         "nnue_cached_eval_ms", "board_pushpop_ms",
                         "nnue_pushpop_ms"):
                self.assertGreater(case[name], 0)
        print("D6_NNUE_COST " + json.dumps(report, sort_keys=True))

    def test_rejects_unbounded_work(self):
        with self.assertRaises(ValueError):
            measure(repeats=10000)


if __name__ == "__main__":
    unittest.main()
