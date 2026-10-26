"""Same-opening KUMOJI paired research protocol tests."""
import unittest
from fukaji.kumoji_paired import validate_request, summarize_pairs


class PairedKumojiTests(unittest.TestCase):
    def config(self):
        return {
            "research_only": True,
            "arena_seed": 20271028,
            "pairs": 2,
            "max_plies": 120,
            "opening_plies": 6,
            "movetime_ms": 200,
            "arms": ["equal_200ms", "kumoji_formal"],
            "relative_source_run_id": 40359123574,
            "kumoji_commit": "a" * 40,
            "relative_model_sha256": "b" * 64,
        }

    def test_fixed_protocol_rejects_unsafe_changes(self):
        self.assertEqual(validate_request(self.config())["pairs"], 2)
        for key, value in (("research_only", False), ("pairs", 0), ("pairs", 100),
                           ("movetime_ms", 1), ("opening_plies", 0),
                           ("arms", ["equal_200ms"]), ("max_plies", 1000),
                           ("kumoji_commit", "main")):
            with self.subTest(key=key):
                c = self.config()
                c[key] = value
                with self.assertRaises(ValueError):
                    validate_request(c)

    def test_paired_openings_and_unresolved_not_wins(self):
        head = ["7g7f", "3c3d"]
        rows = [
            {"opening": head, "a_black": True, "winner": "A"},
            {"opening": head, "a_black": False, "winner": None},
            {"opening": head, "a_black": True, "winner": "B"},
            {"opening": head, "a_black": False, "winner": None},
        ]
        self.assertEqual(summarize_pairs(rows, 2),
                         {"a_advantage": 1, "b_advantage": 1})
        with self.assertRaises(ValueError):
            summarize_pairs([rows[1], rows[0], *rows[2:]], 2)
        with self.assertRaises(ValueError):
            summarize_pairs(rows[:-1], 2)


if __name__ == "__main__":
    unittest.main()
