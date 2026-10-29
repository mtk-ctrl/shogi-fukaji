"""Regression evidence for the current UCT clock-start boundary.

This documents the pre-fix behavior; the total-budget candidate must update
the expected elapsed-time contract in its own research PR.
"""
import unittest
from unittest.mock import patch

import shogi

from fukaji import mcts_research


class UctClockBoundaryTests(unittest.TestCase):
    def test_root_preparation_is_excluded_from_internal_elapsed_today(self):
        board = shogi.Board()
        original = (board.sfen(), tuple(board.move_stack))
        clock = [0.0]
        root_calls = [0]
        real_legal = mcts_research._legal_moves_once

        def measured_legal_moves(position, *, single_pass=False):
            root_calls[0] += 1
            if root_calls[0] == 1:
                clock[0] += 0.012  # Synthetic 12ms root preparation.
            return real_legal(position, single_pass=single_pass)

        def measured_evaluate(*args, **kwargs):
            clock[0] += 0.010
            return 0

        with patch("fukaji.mcts_research.time.monotonic",
                   side_effect=lambda: clock[0]), \
             patch("fukaji.mcts_research._legal_moves_once",
                   side_effect=measured_legal_moves), \
             patch("fukaji.mcts_research.evaluate",
                   side_effect=measured_evaluate):
            move, iterations, internal_ms = mcts_research.select_mcts(
                board, movetime_ms=200, max_iterations=1,
                max_tree_depth=1, seed=17,
                single_pass_legal_moves=True,
            )

        self.assertEqual(root_calls[0], 1)
        self.assertEqual(iterations, 1)
        self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), original)
        self.assertAlmostEqual(internal_ms, 10.0, places=6)
        self.assertAlmostEqual(clock[0] * 1000, 22.0, places=6)
        self.assertAlmostEqual(clock[0] * 1000 - internal_ms, 12.0,
                               places=6)


if __name__ == "__main__":
    unittest.main()
