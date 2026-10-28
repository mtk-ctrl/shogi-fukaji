"""Research G8-vs-G7 request and 20-runner pair allocation tests."""
import unittest
from fukaji.g8_paired_arena import expected_request, validate_request, pairs_for_shard, wilson


class G8PairedTests(unittest.TestCase):
    def test_disabled_and_enabled_request(self):
        self.assertFalse(validate_request(expected_request(False)))
        self.assertTrue(validate_request(expected_request(True)))
        self.assertFalse(expected_request(True)["training_allowed"])
        self.assertFalse(expected_request(True)["model_adoption_allowed"])

    def test_no_unapproved_240_game_changes(self):
        for key, value in (("pairs", 200), ("shards", 19),
                           ("movetime_ms", 100), ("seed", 20276127),
                           ("source_games_used_as_openings", True)):
            with self.subTest(key=key):
                req = expected_request(True)
                req[key] = value
                with self.assertRaises(ValueError):
                    validate_request(req)

    def test_20_shards_exact_120_pairs_and_swapped_240_games(self):
        parts = [list(pairs_for_shard(i)) for i in range(20)]
        self.assertTrue(all(len(p) == 6 for p in parts))
        self.assertEqual([i for p in parts for i in p], list(range(120)))
        self.assertEqual(sum(2*len(p) for p in parts), 240)
        for bad in (-1, 20, 0.5, True):
            with self.assertRaises(ValueError):
                list(pairs_for_shard(bad))

    def test_wilson_decisive_only(self):
        self.assertIsNone(wilson(0, 0))
        self.assertEqual(wilson(0, 10)[0], 0)
        lo, hi = wilson(50, 50)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)


if __name__ == "__main__":
    unittest.main()
