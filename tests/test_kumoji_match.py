"""Contract tests for the read-only KUMOJI USI research match."""
import unittest

from fukaji.kumoji_match import audit_game, validate_request


class KumojiMatchTests(unittest.TestCase):
    def request(self):
        return {
            "kumoji_commit": "a" * 40,
            "relative_model_sha256": "b" * 64,
            "relative_source_run_id": 40359123574,
            "games": 2,
            "seed": 20271027,
            "max_plies": 160,
            "movetime_ms": 200,
            "research_only": True,
        }

    def test_rejects_nonresearch_and_invalid_timing(self):
        for key, value in (("research_only", False), ("movetime_ms", 500),
                           ("games", 1), ("games", 9), ("seed", -1),
                           ("kumoji_commit", "main")):
            with self.subTest(key=key, value=value):
                cfg = self.request()
                cfg[key] = value
                with self.assertRaises(ValueError):
                    validate_request(cfg)
        self.assertEqual(validate_request(self.request())["games"], 2)

    def test_replay_legal_unresolved_and_illegal_rejected(self):
        row = {"moves": ["7g7f", "3c3d"], "reason": "move_limit",
               "winner": None, "a_black": True}
        self.assertEqual(audit_game(row), 2)
        with self.assertRaises(ValueError):
            audit_game({**row, "winner": "A"})
        with self.assertRaises(ValueError):
            audit_game({**row, "moves": ["7g7f", "7g7f"]})
        with self.assertRaises(ValueError):
            audit_game({**row, "reason": "checkmate", "winner": "A"})


if __name__ == "__main__":
    unittest.main()
