import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fukaji.parent_arena import (MODEL_SHA, SEEDS, validate_request, aggregate,
                                 execute)


class ParentArenaTest(unittest.TestCase):
    def request(self):
        return {
            "research_only": True, "training_allowed": False,
            "model_adoption_allowed": False, "model_sha256": MODEL_SHA,
            "model_runs": {"g1": 40359123574, "g2": 40404769521,
                           "g3": 40414745515},
            "seeds": SEEDS, "pairs_per_comparison": 16, "shards": 8,
            "opening_plies": 6, "max_plies": 160, "movetime_ms": 200,
            "depth": 2, "mix_percent": 25, "fast_relative": True,
            "root_alpha_beta_both": True,
        }

    def test_frozen_protocol(self):
        cfg = self.request()
        self.assertEqual(validate_request(cfg), cfg)
        for key, bad in (("model_adoption_allowed", True),
                         ("training_allowed", True),
                         ("movetime_ms", 1), ("root_alpha_beta_both", False),
                         ("seeds", {"g2": 20271827, "g3": 20271827})):
            broken = self.request()
            broken[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_request(broken)

    def test_missing_shards_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "eight"):
                aggregate(tmp, tmp)

    def test_invalid_model_hash_rejected_before_play(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = root / "request.json"
            request.write_text(json.dumps(self.request()), encoding="utf-8")
            paths = {g: root / (g + ".npz") for g in MODEL_SHA}
            for path in paths.values():
                path.write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                execute(request, paths, root / "out", 0)

    def test_invalid_shard_rejected_before_model_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / "request.json"
            request.write_text(json.dumps(self.request()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "shard"):
                execute(request, {}, Path(tmp) / "out", 8)


if __name__ == "__main__":
    unittest.main()
