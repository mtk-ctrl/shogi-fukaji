"""Frozen FastLegalGuard pairing safety before using multiple runners."""
import unittest
import shogi
from fukaji.arena import opening
from fukaji.direct_arena import replay_check
from fukaji.kumoji_fast_legal_arena import (
    STAGES, expected_request, validate_request, play_game,
)


class FastLegalArenaTests(unittest.TestCase):
    def test_immutable_one_factor_protocol_and_independent_cohorts(self):
        cfg=expected_request(True)
        self.assertTrue(validate_request(cfg))
        self.assertEqual(cfg["baseline_max_depth"],cfg["candidate_max_depth"])
        self.assertFalse(cfg["baseline_fast_legal_guard"])
        self.assertTrue(cfg["candidate_fast_legal_guard"])
        self.assertEqual(cfg["movetime_ms"],200)
        self.assertEqual(STAGES["screen"][0],20)
        self.assertEqual(STAGES["final"][0],120)
        invalid={**cfg,"baseline_max_depth":2}
        with self.assertRaises(ValueError):
            validate_request(invalid)

    def test_both_colors_replay_legal_without_mutating_input(self):
        head=opening(STAGES["screen"][1],6)
        self.assertEqual(len(head),6)
        for candidate_black in (True,False):
            with self.subTest(candidate_black=candidate_black):
                row=play_game(head,candidate_black,None,max_plies=8)
                replay_check(head,row["moves"],row)
                self.assertLessEqual(len(head)+len(row["moves"]),8)
                for turn in row["played"]:
                    self.assertEqual(turn["max_depth"],3)
                    self.assertIs(turn["fast_legal_guard"],turn["candidate"])
                    self.assertTrue(turn["move"])


if __name__=="__main__":
    unittest.main()
