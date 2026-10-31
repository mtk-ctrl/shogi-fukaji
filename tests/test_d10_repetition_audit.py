import unittest

import shogi

from fukaji.d8_depth_extension_arena import board_state
from fukaji.d10_repetition_audit import (
    expected_request, validate_request, game_ids, root_partition,
    compare_position, immediate_mate_replies, summarize, inspect_game,
)


def cycled_board():
    board = shogi.Board()
    for word in ("5i6h", "5a6b", "6h5i", "6b5a"):
        move = shogi.Move.from_usi(word)
        assert move in board.legal_moves
        board.push(move)
    return board


class D10RepetitionAuditTests(unittest.TestCase):
    def test_default_off_and_frozen_request(self):
        self.assertFalse(validate_request(expected_request(False)))
        self.assertTrue(validate_request(expected_request(True)))
        for key, value in (("enabled", 1), ("training_allowed", True), ("games", 241),
                           ("max_depth", 3), ("complete_depth", False)):
            cfg = expected_request(True)
            cfg[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_request(cfg)

    def test_twenty_balanced_disjoint_lanes(self):
        lanes = [game_ids(i) for i in range(20)]
        self.assertEqual(sorted(g for lane in lanes for g in lane), list(range(240)))
        self.assertEqual({len(lane) for lane in lanes}, {12})
        with self.assertRaises(ValueError):
            game_ids(True)

    def test_legal_history_excludes_revisited_child_and_restores_board(self):
        board = cycled_board()
        before = board_state(board)
        part = root_partition(board)
        self.assertIn("5i6h", part["removed"])
        self.assertIn("7g7f", part["kept"])
        self.assertEqual(set(part["kept"] + part["removed"]),
                         {m.usi() for m in board.legal_moves})
        self.assertEqual(board_state(board), before)

    def test_immediate_mate_is_always_retained(self):
        board = shogi.Board("8k/9/7R1/7L1/9/9/9/9/8K b - 1")
        part = root_partition(board)
        self.assertIn("2c2b+", part["mate_moves"])
        self.assertIn("2c2b+", part["kept"])
        self.assertEqual(part["counts"]["2c2b+"], -1)

    def test_real_fixed_depth_comparison_preserves_history_and_differs_only_filter(self):
        board = cycled_board()
        before = board_state(board)
        part = root_partition(board)
        result = compare_position(board, None, part)
        self.assertEqual([x["avoid_repetition"] for x in result["calls"]], [True, False, False, True])
        self.assertTrue(all(x["depth"] == 2 for x in result["calls"]))
        self.assertFalse(result["calls"][0]["chosen_was_filtered"])
        self.assertEqual(board_state(board), before)

    def test_mate_reply_audit_and_invalid_move_preserve_history(self):
        board = shogi.Board()
        before = board_state(board)
        self.assertEqual(immediate_mate_replies(board, "7g7f"), 0)
        self.assertEqual(board_state(board), before)
        with self.assertRaisesRegex(ValueError, "illegal diagnostic choice"):
            immediate_mate_replies(board, "9a9b")

    def test_game_with_no_filtering_reports_absence_not_missing_data(self):
        game = {"game_id": 0, "opening": [], "moves": ["7g7f", "3c3d"],
                "status": "move_limit_unresolved", "winner": None}
        result = inspect_game(game, None)
        self.assertIsNone(result["first_affected"])
        self.assertTrue(result["full_game_legal_replay"])
        summary = summarize([result])
        self.assertEqual(summary["games_with_filtering"], 0)
        self.assertEqual(summary["played_moves_replayed"], 2)
        self.assertIsNone(summary["timing"]["filtered"]["mean_ms"])
        self.assertFalse(summary["strength_proven"])

    def test_timeout_or_incomplete_depth_cannot_be_reported_as_comparison(self):
        board = cycled_board()
        part = root_partition(board)
        with self.assertRaisesRegex(ValueError, "incomplete depth"):
            compare_position(board, None, part, select=lambda *a, **k: (part["kept"][0], 1, 3))


if __name__ == "__main__":
    unittest.main()
