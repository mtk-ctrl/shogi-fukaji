import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fukaji.g7_fast_legal_guard_arena import (
    G7_SHA, PAIRS, SEED, SHARDS, MAX_PLIES,
    aggregate, execute, expected_request, validate_request,
)


class G7FastLegalGuardArenaTests(unittest.TestCase):
    def test_exact_frozen_request(self):
        cfg = expected_request()
        self.assertEqual(validate_request(cfg), cfg)
        self.assertEqual((PAIRS, SHARDS, MAX_PLIES), (128, 20, 240))
        self.assertEqual(cfg["depth"], 4)
        sizes = [len(range(shard, PAIRS, SHARDS)) for shard in range(SHARDS)]
        self.assertEqual((min(sizes), max(sizes), sum(sizes)), (6, 7, 128))
        self.assertEqual(cfg["movetime_ms"], 200)
        self.assertTrue(cfg["fast_move_list_both"])
        self.assertTrue(cfg["fast_legal_guard_candidate"])
        self.assertFalse(cfg["fast_legal_guard_baseline"])
        self.assertFalse(cfg["training_allowed"])
        self.assertFalse(cfg["model_adoption_allowed"])
        self.assertEqual(cfg["model_sha256"], G7_SHA)
        self.assertNotIn(SEED, (20274627, 20274327, 20271727, 20272227, 20272327,
                                20272427, 20272627, 20272827,
                                20273127, 20273427, 20273727,
                                20274027))

    def test_reject_any_protocol_tampering(self):
        changes = {
            "training_allowed": True, "model_adoption_allowed": True,
            "model_sha256": "0" * 64, "depth": 2,
            "movetime_ms": 50, "seed": 20273727, "pairs": 32,
            "max_plies": 160, "fast_relative": False,
            "root_alpha_beta_both": False,
            "fast_move_list_both": False,
            "fast_legal_guard_candidate": False,
            "fast_legal_guard_baseline": True,
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                cfg = copy.deepcopy(expected_request())
                cfg[key] = value
                with self.assertRaises(ValueError):
                    validate_request(cfg)
        cfg = expected_request()
        cfg["unexpected"] = "x"
        with self.assertRaises(ValueError):
            validate_request(cfg)

    def test_missing_shards_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "twenty"):
                aggregate(temp, temp)

    def test_reject_invalid_shard_and_model_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = root / "request.json"
            request.write_text(json.dumps(expected_request()), encoding="utf-8")
            model = root / "model.npz"
            model.write_bytes(b"not a frozen model")
            with self.assertRaisesRegex(ValueError, "shard"):
                execute(request, model, root / "out", 20)
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                execute(request, model, root / "out", 0)

    def test_workflow_is_bounded_and_separate(self):
        path = Path(".github/workflows/g7-fast-legal-guard-arena.yml")
        text = path.read_text(encoding="utf-8")
        self.assertIn("max-parallel: 20", text)
        self.assertEqual(text.count(".github/fukaji-g7-fast-legal-guard-arena-request.json"), 3)
        self.assertNotIn(".github/fukaji-g7-fast-move-list-arena-request.json", text)
        self.assertIn("shard: [" + ", ".join(str(i) for i in range(20)) + "]", text)
        self.assertIn("fail-fast: false", text)
        self.assertIn("cancel-in-progress: false", text)
        self.assertIn("run-id: 40434466661", text)
        self.assertIn("fukaji.g7_fast_legal_guard_arena", text)
        self.assertIn("workflow_dispatch:", text)
        self.assertIn("aggregate:", text)
        self.assertIn("if-no-files-found: error", text)


if __name__ == "__main__":
    unittest.main()
