"""D13 cross-engine A/B research protocol regression (no model adoption)."""
import json
import tempfile
import unittest
from pathlib import Path

from fukaji.arena import opening
from fukaji.kumoji_cross200 import (
    ARMS, G8_SHA, KUMOJI_COMMIT, NEW_SEED, OLD_SEED, PAIRS,
    SHARDS, check_request, config, exact_sign_p, indices, matched_seed
)


class KumojiCross200Tests(unittest.TestCase):
    def test_frozen_one_difference_protocol(self):
        p = config()
        self.assertEqual(p["arms"], ["baseline", "fastlegal"])
        self.assertEqual(p["only_difference"], "FastLegalGuard")
        self.assertEqual(p["fukaji_g8_sha256"], G8_SHA)
        self.assertEqual(p["kumoji_commit"], KUMOJI_COMMIT)
        self.assertEqual(p["max_plies"], 300)
        self.assertEqual(p["games"], 200)
        self.assertFalse(p["formal_adoption"])
        with tempfile.TemporaryDirectory() as tmp:
            name = Path(tmp) / "request.json"
            name.write_text(json.dumps(p))
            self.assertEqual(check_request(name), p)
            p["max_plies"] = 100
            name.write_text(json.dumps(p))
            with self.assertRaises(ValueError):
                check_request(name)

    def test_disjoint_openings_and_even_shard_assignment(self):
        old = {tuple(opening(OLD_SEED + p, 6)) for p in range(PAIRS)}
        fresh = {tuple(opening(NEW_SEED + p, 6)) for p in range(PAIRS)}
        self.assertEqual(len(old), PAIRS)
        self.assertEqual(len(fresh), PAIRS)
        self.assertFalse(old & fresh)
        ids = [i for shard in range(SHARDS) for i in indices(shard)]
        self.assertEqual(sorted(ids), list(range(PAIRS)))

    def test_opponent_random_seed_is_identical_within_matched_color(self):
        for pair in range(PAIRS):
            for seat in (0, 1):
                expected = NEW_SEED + pair * 16 + seat * 2
                self.assertEqual(matched_seed(pair, seat), expected)
                self.assertEqual(matched_seed(pair, seat), matched_seed(pair, seat))
        self.assertNotEqual(matched_seed(0, 0), matched_seed(0, 1))
        with self.assertRaises(ValueError):
            matched_seed(0, 2)

    def test_sign_p_uses_non_tied_paired_openings_only(self):
        self.assertEqual(exact_sign_p(0, 0), None)
        self.assertEqual(exact_sign_p(3, 0), 0.25)
        self.assertEqual(exact_sign_p(5, 5), 1.0)

if __name__ == "__main__":
    unittest.main()
