"""Regression tests for tactical positions selected from legal game histories."""
import unittest

import shogi

from fukaji.tactical_positions import (
    position_key, classify_position, select_tactical_positions,
)


class TacticalPositionTests(unittest.TestCase):
    def test_position_key_excludes_move_number_only(self):
        a = shogi.Board()
        b = shogi.Board(a.sfen().rsplit(" ", 1)[0] + " 77")
        self.assertEqual(position_key(a), position_key(b))

    def test_checkmate_and_check_are_distinct(self):
        mate = shogi.Board("4k4/3GRG3/9/9/9/9/9/9/4K4 w - 1")
        check = shogi.Board("4k4/9/4R4/9/9/9/9/9/4K4 w - 1")
        self.assertIn("checkmate", classify_position(mate, 1))
        self.assertNotIn("in_check", classify_position(mate, 1))
        self.assertIn("in_check", classify_position(check, 1))
        self.assertNotIn("checkmate", classify_position(check, 1))

    def test_capture_produces_hand_and_legal_history(self):
        seq = ["7g7f", "7c7d", "7f7e", "3c3d", "7e7d"]
        rows = [{"pair": 0, "g8_black": True, "opening": seq[:2],
                 "moves": seq[2:], "status": "move_limit_unresolved"}]
        selected, shortage = select_tactical_positions(rows, per_category=1)
        hands = [x for x in selected if x["category"] == "hand_pieces"]
        self.assertEqual(len(hands), 1)
        self.assertEqual(hands[0]["hand_pieces"], 1)
        self.assertEqual(hands[0]["prefix"], seq)
        self.assertEqual(shortage["checkmate"], 1)
        self.assertEqual(shortage["in_check"], 1)
        self.assertEqual(shortage["repetition_risk"], 1)
        board = shogi.Board()
        for word in hands[0]["prefix"]:
            board.push(shogi.Move.from_usi(word))
        self.assertEqual(board.sfen(), hands[0]["sfen"])

    def test_no_fabricated_tactical_samples(self):
        seq = ["7g7f", "3c3d"]
        rows = [{"pair": 0, "g8_black": True, "opening": seq,
                 "moves": [], "status": "move_limit_unresolved"}]
        selected, shortage = select_tactical_positions(rows, per_category=2)
        self.assertEqual(selected, [])
        self.assertEqual(shortage, {key: 2 for key in shortage})

    def test_reject_illegal_source(self):
        rows = [{"pair": 0, "g8_black": True, "opening": ["7g7f", "7g7f"],
                 "moves": [], "status": "move_limit_unresolved"}]
        with self.assertRaises(ValueError):
            select_tactical_positions(rows)

    def test_reject_invalid_budgets(self):
        with self.assertRaises(ValueError):
            select_tactical_positions([], per_category=0)
        with self.assertRaises(ValueError):
            select_tactical_positions([], max_ply=0)


if __name__ == "__main__":
    unittest.main()
