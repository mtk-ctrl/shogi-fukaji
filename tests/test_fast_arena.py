"""Bounded paired 200ms fast-leaf evaluation research protocol tests."""
import unittest

import shogi

from fukaji.fast_arena import summarize_pairs, validate_request


class FastArenaTests(unittest.TestCase):
    def config(self):
        return {
            "research_only": True,
            "relative_source_run_id": 40359123574,
            "relative_model_sha256": "a" * 64,
            "arena_seed": 20271127,
            "arena_pairs": 8,
            "max_plies": 120,
            "opening_plies": 6,
            "movetime_ms": 200,
            "depth": 2,
            "mix_percent": 25,
            "fast_eval_default": False,
        }

    def test_fixed_protocol_rejects_unsafe_changes(self):
        self.assertEqual(validate_request(self.config())["arena_pairs"], 8)
        for name, value in (
            ("research_only", False),
            ("relative_source_run_id", 1),
            ("relative_model_sha256", "broken"),
            ("arena_pairs", 100),
            ("max_plies", 2000),
            ("opening_plies", 12),
            ("movetime_ms", 1),
            ("depth", 4),
            ("mix_percent", 100),
            ("fast_eval_default", True),
        ):
            with self.subTest(name=name):
                c = self.config()
                c[name] = value
                with self.assertRaises(ValueError):
                    validate_request(c)

    def test_unresolved_is_not_a_win_and_colors_are_paired(self):
        opening = ["7g7f", "3c3d"]
        rows = [
            {"opening": opening, "fast_black": True,
             "status": "checkmate", "winner": shogi.BLACK},
            {"opening": opening, "fast_black": False,
             "status": "move_limit_unresolved", "winner": None},
            {"opening": opening, "fast_black": True,
             "status": "checkmate", "winner": shogi.WHITE},
            {"opening": opening, "fast_black": False,
             "status": "move_limit_unresolved", "winner": None},
        ]
        self.assertEqual(summarize_pairs(rows, 2),
                         {"fast_advantage": 1, "baseline_advantage": 1})
        with self.assertRaises(ValueError):
            summarize_pairs(rows[:-1], 2)
        with self.assertRaises(ValueError):
            summarize_pairs([rows[1], rows[0], *rows[2:]], 2)


if __name__ == "__main__":
    unittest.main()
