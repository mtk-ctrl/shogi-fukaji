import unittest
from unittest.mock import patch

import shogi

from fukaji.d9_clock_arena import (
    expected_request, validate_request, pair_indices, play_game, summarize, sign_p,
)


class D9ArenaTests(unittest.TestCase):
    def test_frozen_protocol_rejects_changed_conditions_and_boolean_numbers(self):
        self.assertFalse(validate_request(expected_request(False)))
        self.assertTrue(validate_request(expected_request(True)))
        for key, value in (("enabled", 1), ("training_allowed", True),
                           ("movetime_ms", 201), ("candidate_max_depth", 6), ("candidate_time_check_each_node", False),
                           ("deadline_from_call", False), ("screen_pairs", 21)):
            cfg = expected_request(True)
            cfg[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_request(cfg)

    def test_pair_distribution_has_no_missing_duplicate_or_uneven_lanes(self):
        for stage, count, per_lane in (("screen", 20, 1), ("final", 120, 6)):
            lanes = [pair_indices(stage, s) for s in range(20)]
            self.assertEqual(sorted(p for lane in lanes for p in lane), list(range(count)))
            self.assertEqual({len(x) for x in lanes}, {per_lane})
        with self.assertRaises(ValueError):
            pair_indices("screen", True)

    def test_actual_play_arms_share_fast_list_and_only_change_clock_check_frequency(self):
        options = []

        def choose(board, **kwargs):
            options.append(kwargs)
            return next(iter(board.legal_moves)).usi(), 1, 2

        with patch("fukaji.d9_clock_arena.select_move", side_effect=choose):
            row = play_game([], True, None, max_plies=2)
        self.assertEqual(row["status"], "move_limit_unresolved")
        self.assertIsNone(row["winner"])
        self.assertEqual([x["candidate"] for x in row["played"]], [True, False])
        self.assertEqual([x["fast_move_list"] for x in options], [True, True])
        self.assertEqual([x["max_depth"] for x in options], [3, 3])
        self.assertEqual([x["time_check_each_node"] for x in options], [True, False])
        for opt in options:
            self.assertTrue(opt["deadline_from_call"])
            self.assertEqual(opt["movetime_ms"], 200)

    def test_illegal_move_rejected_and_mate_winner_verified(self):
        with patch("fukaji.d9_clock_arena.select_move", return_value=("9a9b", 1, 1)):
            with self.assertRaisesRegex(ValueError, "illegal selected move"):
                play_game([], True, None, max_plies=1)
        with patch("fukaji.d9_clock_arena.select_move", return_value=("2c2b+", 1, 1)):
            row = play_game([], True, None, max_plies=1,
                            start_sfen="8k/9/7R1/7L1/9/9/9/9/8K b - 1")
        self.assertEqual((row["status"], row["winner"]), ("checkmate", shogi.BLACK))

    @staticmethod
    def row(pair, candidate_black, winner, candidate_ms=100):
        return {"pair": pair, "candidate_black": candidate_black,
                "status": "checkmate" if winner is not None else "move_limit_unresolved",
                "winner": winner,
                "played": [{"candidate": False, "elapsed_ms": 100, "nodes": 2, "depth": 2},
                           {"candidate": True, "elapsed_ms": candidate_ms, "nodes": 3, "depth": 2}]}

    def test_unresolved_not_draw_and_pair_signs_are_paired(self):
        rows = [self.row(0, True, shogi.BLACK), self.row(0, False, shogi.WHITE),
                self.row(1, True, None), self.row(1, False, None)]
        result = summarize(rows)
        self.assertEqual((result["candidate_wins"], result["baseline_wins"], result["unresolved"]), (2, 0, 2))
        self.assertEqual(result["pair_comparison"], {"candidate": 1, "equal": 1})
        self.assertEqual(result["paired_sign_p_two_sided"], 1.0)
        self.assertTrue(result["screen_proceed"])

    def test_latency_growth_tail_and_obvious_regression_block_followup(self):
        slow = summarize([self.row(0, True, None, 106), self.row(0, False, None, 106)])
        self.assertFalse(slow["screen_proceed"])
        tail = summarize([self.row(0, True, None, 301), self.row(0, False, None, 301)])
        self.assertFalse(tail["screen_proceed"])
        self.assertFalse(slow["candidate_mean_time_within_5percent"])
        losses = [self.row(p, seat, shogi.WHITE if seat else shogi.BLACK)
                  for p in range(20) for seat in (True, False)]
        result = summarize(losses)
        self.assertTrue(result["screen_obvious_regression"])
        self.assertFalse(result["screen_proceed"])
        self.assertIsNone(sign_p(0, 0))


if __name__ == "__main__":
    unittest.main()

