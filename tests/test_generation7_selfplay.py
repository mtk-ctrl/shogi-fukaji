"""Frozen G7 400-game research input protocol tests (no actual long games)."""
import unittest

from fukaji.generation7_selfplay import (
    GAMES, SHARDS, MAX_PLIES, MODEL_SHA, SOURCE_RUN, SEED,
    expected_request, parent_id, validate_request,
)
from fukaji.arena import opening


class Generation7SelfplayProtocolTests(unittest.TestCase):
    def test_frozen_request_and_disabled_rejection(self):
        cfg = expected_request()
        self.assertEqual((GAMES, SHARDS, GAMES // SHARDS), (400, 20, 20))
        self.assertEqual((SOURCE_RUN, SEED, MAX_PLIES), (40434466661, 20276127, 240))
        self.assertEqual(MODEL_SHA, "417d18e7b64494c9febdf6d217afbe4e4979e68d04528f16db8b45c0bcde2a0e")
        self.assertEqual((cfg["movetime_ms"], cfg["depth"], cfg["mix_percent"]), (200, 2, 25))
        self.assertTrue(cfg["enabled"])
        self.assertFalse(cfg["training_allowed"])
        self.assertFalse(cfg["model_adoption_allowed"])
        self.assertEqual(validate_request(dict(cfg)), cfg)
        for change in ({"enabled": False}, {"games": 399}, {"shards": 8},
                       {"training_allowed": True}, {"seed": 20273427},
                       {"model_sha256": "0" * 64}):
            bad = dict(cfg)
            bad.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_request(bad)

    def test_twenty_equal_disjoint_shards_and_parent_ids(self):
        groups = set()
        ids = set()
        for shard in range(SHARDS):
            for i in range(shard * (GAMES // SHARDS), (shard + 1) * (GAMES // SHARDS)):
                self.assertNotIn(i, ids)
                ids.add(i)
                head = opening(SEED + i, 6)
                self.assertEqual(len(head), 6)
                group = parent_id(SEED + i, head)
                self.assertNotIn(group, groups)
                groups.add(group)
        self.assertEqual(ids, set(range(400)))
        self.assertEqual(len(groups), 400)

    def test_distinct_training_seed_from_prior_g6(self):
        self.assertGreater(SEED, 20273427 + 256)


if __name__ == "__main__":
    unittest.main()
