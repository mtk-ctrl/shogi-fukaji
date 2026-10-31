"""Safety, fallback and budget semantics of the opt-in mate defense guard."""
import time
import unittest
from unittest.mock import patch
import shogi

from fukaji.kumoji_defense_guard import (
    ProbeIncomplete, _enemy_mates_immediately, select_guarded,
)
from fukaji.kumoji_loss_study import snapshot


class GuardTests(unittest.TestCase):
    def baseline(self, *args, **kwargs):
        return "7g7f", 2, 55

    def test_proven_unsafe_choice_is_replaced_by_legally_checked_alternative(self):
        board=shogi.Board()
        before=snapshot(board)
        with patch("fukaji.kumoji_defense_guard._unsafe_root_move",
                   side_effect=lambda b,m,d: m.usi()=="7g7f") as mock:
            result=select_guarded(board,None,movetime_ms=1000,reserve_ms=250,
                                  search_override=self.baseline)
        self.assertEqual(result["original_move"],"7g7f")
        self.assertNotEqual(result["move"],"7g7f")
        self.assertTrue(result["original_allows_mate"])
        self.assertEqual(result["guard_replacements"],1)
        self.assertEqual(result["guard_status"],"confirmed_avoidable_immediate_mate")
        self.assertGreaterEqual(mock.call_count,2)
        self.assertEqual(snapshot(board),before)

    def test_safe_chosen_move_is_retained(self):
        board=shogi.Board()
        before=snapshot(board)
        with patch("fukaji.kumoji_defense_guard._unsafe_root_move",return_value=False):
            result=select_guarded(board,None,movetime_ms=1000,reserve_ms=250,
                                  search_override=self.baseline)
        self.assertEqual(result["move"],"7g7f")
        self.assertEqual(result["guard_status"],"verified_no_immediate_mate")
        self.assertEqual(snapshot(board),before)

    def test_unknown_is_not_falsely_safe_and_does_not_change_move(self):
        board=shogi.Board()
        before=snapshot(board)
        with patch("fukaji.kumoji_defense_guard._unsafe_root_move",
                   side_effect=ProbeIncomplete("budget")):
            result=select_guarded(board,None,movetime_ms=1000,reserve_ms=250,
                                  search_override=self.baseline)
        self.assertIsNone(result["original_allows_mate"])
        self.assertEqual(result["move"],"7g7f")
        self.assertIn("unknown",result["guard_status"])
        self.assertEqual(snapshot(board),before)

    def test_legal_complete_mate_in_one_proof(self):
        board=shogi.Board("8k/9/7R1/7L1/9/9/9/9/8K b - 1")
        before=snapshot(board)
        self.assertTrue(_enemy_mates_immediately(board,time.monotonic()+30))
        self.assertEqual(snapshot(board),before)

    def test_split_budget_rejected(self):
        with self.assertRaises(ValueError):
            select_guarded(shogi.Board(),None,movetime_ms=200,reserve_ms=200,
                           search_override=self.baseline)


if __name__=="__main__":
    unittest.main()
