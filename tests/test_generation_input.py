"""Generation-input provenance and fail-closed behavior."""
import tempfile
import unittest
from pathlib import Path
from fukaji.generation_input import SOURCES, MODEL_SHA, read_source, sha


class GenerationInputTest(unittest.TestCase):
    def test_frozen_sources(self):
        self.assertEqual(set(SOURCES), {40387257648, 40395505536})
        self.assertEqual(len(MODEL_SHA), 64)
        self.assertNotEqual(SOURCES[40387257648]["games_sha256"],
                            SOURCES[40395505536]["games_sha256"])

    def test_wrong_hash_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as d:
            game, report = Path(d) / "games.jsonl", Path(d) / "report.json"
            game.write_text("{}\n")
            report.write_text("{}")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                read_source(40395505536, game, report)

    def test_unknown_run_rejected(self):
        with self.assertRaisesRegex(ValueError, "unapproved"):
            read_source(1, "missing", "missing")

    def test_hash_function(self):
        self.assertEqual(sha(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")


if __name__ == "__main__":
    unittest.main()
