"""Historical-root provenance and one-factor speed-probe preconditions."""
import tempfile
import unittest
from pathlib import Path
from fukaji.kumoji_search_speed_probe import positions, G8_SHA


class SpeedProbeTests(unittest.TestCase):
    def test_original_games_are_sha_locked(self):
        with tempfile.TemporaryDirectory() as dirname:
            file=Path(dirname)/"fake.jsonl"
            file.write_text('{"games":[]}\n')
            with self.assertRaisesRegex(ValueError,"wrong frozen"):
                positions(file)

    def test_pinned_model_identity(self):
        self.assertEqual(len(G8_SHA),64)
        self.assertTrue(all(ch in "0123456789abcdef" for ch in G8_SHA))


if __name__=="__main__":unittest.main()
