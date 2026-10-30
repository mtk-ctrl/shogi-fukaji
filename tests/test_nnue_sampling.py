import unittest

import numpy as np

from fukaji.nnue_sampling import paired_sample_indices, paired_training_rows


class PairedSamplingTests(unittest.TestCase):
    def test_long_game_invariants_across_parities_and_caps(self):
        for cap in (2, 4, 16, 32, 64):
            for length in range(cap + 1, 601):
                ix = paired_sample_indices(length, cap)
                self.assertEqual(len(ix), cap)
                self.assertTrue(np.all(np.diff(ix) > 0))
                self.assertGreaterEqual(int(ix[0]), 0)
                self.assertEqual(int(ix[-1]), length - 1)
                self.assertEqual(int(ix[-2]), length - 2)
                self.assertEqual(int((ix % 2 == 0).sum()), cap // 2)
                self.assertEqual(int((ix % 2 == 1).sum()), cap // 2)
                if cap >= 4:
                    np.testing.assert_array_equal(ix[:2], [0, 1])

    def test_short_game_keeps_every_position(self):
        for length in range(1, 17):
            np.testing.assert_array_equal(paired_sample_indices(length),
                                          np.arange(length))

    def test_241_plies_avoid_single_parity_alias(self):
        old = np.linspace(0, 240, 16, dtype=np.int64)
        self.assertTrue(np.all(old % 2 == 0))
        new = paired_sample_indices(241)
        self.assertEqual(int((new % 2 == 0).sum()), 8)
        self.assertIn(239, new)
        self.assertIn(240, new)

    def test_holdouts_unchanged_and_training_data_not_mutated(self):
        x = np.arange(80).reshape(40, 2)
        y = np.arange(40) % 2
        original_x, original_y = x.copy(), y.copy()
        mapping = {"a": "train", "b": "validation", "c": "test"}
        rows = list(paired_training_rows([(g, x, y) for g in mapping], mapping))
        self.assertEqual(rows[0][1].shape, (16, 2))
        self.assertEqual(int(rows[0][2].sum()), 8)
        for _, hold_x, hold_y in rows[1:]:
            self.assertIs(hold_x, x)
            self.assertIs(hold_y, y)
        np.testing.assert_array_equal(x, original_x)
        np.testing.assert_array_equal(y, original_y)
        with self.assertRaises(ValueError):
            list(paired_training_rows([("missing", x, y)], mapping))

    def test_invalid_parameters_rejected(self):
        for length, cap in ((0, 16), (-1, 16), (True, 16), (2.0, 16),
                            (40, 0), (40, 1), (40, 15), (40, True)):
            with self.subTest(length=length, cap=cap):
                with self.assertRaises(ValueError):
                    paired_sample_indices(length, cap)


if __name__ == "__main__":
    unittest.main()
