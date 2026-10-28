"""Deterministic protocol, tactical provenance and terminal research checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import shogi

from fukaji.tactical_search_probe import (
    EXPECTED_REQUEST, board_for_selected, measure, validate_request,
)
from fukaji.tactical_positions import position_key, classify_position


class TacticalSearchProbeTests(unittest.TestCase):
    def test_default_disabled_and_only_boolean_enable_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "request.json"
            p.write_text(json.dumps(EXPECTED_REQUEST), encoding="utf-8")
            self.assertFalse(validate_request(p))
            p.write_text(json.dumps(dict(EXPECTED_REQUEST, enabled=True)), encoding="utf-8")
            self.assertTrue(validate_request(p))
            p.write_text(json.dumps(dict(EXPECTED_REQUEST, runner_jobs=20)), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_request(p)

    def test_actual_capture_prefix_preserves_hands(self):
        words = ["7g7f", "7c7d", "7f7e", "3c3d", "7e7d"]
        board = shogi.Board()
        occ = {position_key(board): 1}
        for word in words:
            move = shogi.Move.from_usi(word)
            self.assertIn(move, board.legal_moves)
            board.push(move)
            key = position_key(board)
            occ[key] = occ.get(key, 0) + 1
        item = {"prefix": words, "ply": len(words), "sfen": board.sfen(),
                "category": "hand_pieces", "in_check": bool(board.is_check()),
                "is_checkmate": bool(board.is_checkmate()),
                "position_occurrences": occ[position_key(board)], "hand_pieces": 1}
        restored = board_for_selected(item)
        self.assertEqual(restored.sfen(), board.sfen())
        self.assertEqual(restored.pieces_in_hand[shogi.BLACK][shogi.PAWN], 1)
        with self.assertRaises(ValueError):
            board_for_selected(dict(item, hand_pieces=0))

    def test_terminal_mate_requires_resign_and_is_not_move_comparison(self):
        board = shogi.Board("4k4/3GRG3/9/9/9/9/9/9/4K4 w - 1")
        self.assertTrue(board.is_checkmate())
        before = (board.sfen(), tuple(board.move_stack))
        with patch("fukaji.tactical_search_probe.select_move", return_value=("resign", 0, 0)), \
             patch("fukaji.tactical_search_probe.select_mcts", return_value=("resign", 0, 0.0)):
            result = measure(board, None, 0)
        self.assertTrue(result["terminal"])
        self.assertIsNone(result["g8_eval_cp"])
        self.assertEqual(result["methods"]["alpha_beta"]["move"], "resign")
        self.assertEqual((board.sfen(), tuple(board.move_stack)), before)

    def test_check_position_requires_legal_response_and_history_unchanged(self):
        board = shogi.Board("4k4/9/4R4/9/9/9/9/9/4K4 w - 1")
        self.assertTrue(board.is_check())
        self.assertFalse(board.is_checkmate())
        legal = next(iter(board.legal_moves)).usi()
        before = (board.sfen(), tuple(board.move_stack))
        with patch("fukaji.tactical_search_probe.evaluate", return_value=0), \
             patch("fukaji.tactical_search_probe.select_move", return_value=(legal, 2, 15)), \
             patch("fukaji.tactical_search_probe.select_mcts", return_value=(legal, 5, 20.0)):
            result = measure(board, None, 1)
        self.assertFalse(result["terminal"])
        self.assertTrue(result["same_move"])
        self.assertEqual((board.sfen(), tuple(board.move_stack)), before)

    def test_illegal_move_is_rejected(self):
        board = shogi.Board()
        with patch("fukaji.tactical_search_probe.evaluate", return_value=0), \
             patch("fukaji.tactical_search_probe.select_move", return_value=("resign", 0, 0)):
            with self.assertRaises(ValueError):
                measure(board, None, 0)

    def test_mismatched_tactical_category_rejected(self):
        board = shogi.Board()
        item = {"prefix": [], "ply": 0, "sfen": board.sfen(),
                "category": "in_check", "in_check": False, "is_checkmate": False,
                "position_occurrences": 1, "hand_pieces": 0}
        self.assertNotIn("in_check", classify_position(board, 1))
        with self.assertRaises(ValueError):
            board_for_selected(item)


if __name__ == "__main__":
    unittest.main()
