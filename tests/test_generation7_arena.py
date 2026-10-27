import json
import tempfile
import unittest
from pathlib import Path
from fukaji.generation7_arena import MODEL_SHA, expected_request, validate_request, execute, aggregate

class Generation6ArenaTest(unittest.TestCase):
    def test_frozen_request(self):
        cfg = expected_request()
        self.assertEqual(validate_request(cfg), cfg)
        for key, value in (("training_allowed", True),
                           ("model_adoption_allowed", True),
                           ("max_plies", 240), ("movetime_ms", 50),
                           ("seed", 20272427)):
            with self.subTest(key=key):
                bad = expected_request()
                bad[key] = value
                with self.assertRaises(ValueError):
                    validate_request(bad)

    def test_workflow_uses_correct_frozen_model_sources(self):
        workflow = Path(".github/workflows/generation7-arena.yml").read_text(encoding="utf-8")
        self.assertIn("name: fukaji-generation6-research-40429892376\\n          run-id: 40429892376".replace("\\n", "\n"), workflow)
        self.assertIn("name: fukaji-generation7-research-40434466661\\n          run-id: 40434466661".replace("\\n", "\n"), workflow)

    def test_missing_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "eight"):
                aggregate(tmp, tmp)

    def test_rejects_missing_or_wrong_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = root / "request.json"
            request.write_text(json.dumps(expected_request()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing frozen"):
                execute(request, {}, root / "out", 0)
            models = {k: root / (k + ".npz") for k in MODEL_SHA}
            for p in models.values():
                p.write_bytes(b"not a model")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                execute(request, models, root / "out", 0)

    def test_rejects_invalid_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / "request.json"
            request.write_text(json.dumps(expected_request()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "shard"):
                execute(request, {}, Path(tmp) / "out", 8)

if __name__ == "__main__":
    unittest.main()
