"""D5-3 temporal audit: orientation, legal replay, split, history safety."""
import unittest
from unittest.mock import patch

import shogi

from fukaji.d5_eval_timeline import (
    TARGET_PAIRS, CHECKPOINTS, development_cases, trace_game,
)


def fake_game(pair=49, status="checkmate", moves=None):
    return {
        "pair": pair, "status": status,
        "opening": [], "moves": moves or ["7g7f", "3c3d"],
        "winner": int(shogi.BLACK), "g8_black": True,
    }


class D5TimelineTests(unittest.TestCase):
    def test_targets_are_frozen_development_only(self):
        self.assertEqual(TARGET_PAIRS, (49, 52, 109))
        self.assertEqual(CHECKPOINTS, (20, 40, 60, 80, 100))
        rows = [fake_game(49), fake_game(52), fake_game(109)]
        rows += [fake_game(61, "unresolved")] * 237
        self.assertEqual(len(development_cases(rows)), 3)
        self.assertEqual({r["pair"] for r in development_cases(rows)},
                         set(TARGET_PAIRS))

    def test_missing_decisive_parent_is_not_silently_skipped(self):
        rows = [fake_game(49), fake_game(52)]
        rows += [fake_game(61, "unresolved")] * 238
        with self.assertRaises(ValueError):
            development_cases(rows)

    def test_black_perspective_and_history_restoration(self):
        # After black's move white is to play: score must be negated.
        with patch("fukaji.d5_eval_timeline.evaluate",
                   side_effect=[10, 20, 30, 40]):
            result = trace_game(fake_game(), object(), require_mate=False)
        a, b = result["full_move_trace"]
        self.assertEqual([a["g8_black_cp"], b["g8_black_cp"]], [-20, 40])
        self.assertEqual(b["g8_swing_cp"], 60)
        self.assertEqual([a["classical_black_cp"], b["classical_black_cp"]],
                         [-10, 30])
        self.assertTrue(result["board_history_restored"])
        self.assertEqual(result["checkpoints"], [])

    def test_illegal_move_rejected(self):
        with self.assertRaises(ValueError):
            trace_game(fake_game(moves=["7g7g"]), object(), require_mate=False)

    def test_non_mate_not_claimed_as_decisive(self):
        with patch("fukaji.d5_eval_timeline.evaluate",
                   side_effect=[10, 20, 30, 40]):
            with self.assertRaises(ValueError):
                trace_game(fake_game(), object(), require_mate=True)


if __name__ == "__main__":
    unittest.main()
