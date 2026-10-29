"""Research-only UCT total-clock budget regression tests.

The opt-in mode includes root preparation. No hard real-time guarantee or
strength claim is inferred from these deterministic fake-clock checks.
"""
import unittest
from unittest.mock import patch

import shogi

from fukaji import mcts_research


class UctTotalClockBudgetTests(unittest.TestCase):
    def _synthetic_probe(self, enabled, root_cost=0.012, budget=200):
        board = shogi.Board()
        original = (board.sfen(), tuple(board.move_stack))
        clock = [0.0]
        calls = [0]
        evaluations = [0]
        real_legal = mcts_research._legal_moves_once

        def measured_legal_moves(position, *, single_pass=False):
            calls[0] += 1
            if calls[0] == 1:
                clock[0] += root_cost
            return real_legal(position, single_pass=single_pass)

        def measured_evaluate(*args, **kwargs):
            evaluations[0] += 1
            clock[0] += 0.010
            return 0

        with patch("fukaji.mcts_research.time.monotonic",
                   side_effect=lambda: clock[0]), \
             patch("fukaji.mcts_research._legal_moves_once",
                   side_effect=measured_legal_moves), \
             patch("fukaji.mcts_research.evaluate",
                   side_effect=measured_evaluate):
            result = mcts_research.select_mcts(
                board, movetime_ms=budget, max_iterations=1,
                max_tree_depth=1, seed=17,
                single_pass_legal_moves=True,
                total_budget_from_call=enabled,
            )
        self.assertIn(shogi.Move.from_usi(result[0]), board.legal_moves)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), original)
        self.assertEqual(calls[0], 1)
        return result, clock[0] * 1000.0, evaluations[0]

    def test_opt_in_includes_synthetic_root_preparation(self):
        old, old_external, old_evaluations = self._synthetic_probe(False)
        new, new_external, new_evaluations = self._synthetic_probe(True)
        self.assertEqual(old[:2], new[:2])
        self.assertEqual(old_evaluations, 1)
        self.assertEqual(new_evaluations, 1)
        self.assertAlmostEqual(old[2], 10.0, places=6)
        self.assertAlmostEqual(new[2], 22.0, places=6)
        self.assertAlmostEqual(old_external, 22.0, places=6)
        self.assertAlmostEqual(new_external, 22.0, places=6)

    def test_root_preparation_can_exhaust_budget_without_iteration(self):
        result, external_ms, evaluations = self._synthetic_probe(
            True, root_cost=0.050, budget=20)
        self.assertEqual(result[1], 0)
        self.assertEqual(evaluations, 0)
        self.assertAlmostEqual(result[2], 50.0, places=6)
        self.assertAlmostEqual(external_ms, 50.0, places=6)
        # A single root preparation may itself exceed the budget.
        self.assertGreater(result[2], 20.0)

    def test_fixed_iteration_move_and_board_preserved(self):
        board = shogi.Board()
        before = (board.sfen(), tuple(board.move_stack))
        old = mcts_research.select_mcts(
            board, movetime_ms=1000, max_iterations=8,
            max_tree_depth=3, seed=41, single_pass_legal_moves=True)
        new = mcts_research.select_mcts(
            board, movetime_ms=1000, max_iterations=8,
            max_tree_depth=3, seed=41, single_pass_legal_moves=True,
            total_budget_from_call=True)
        self.assertEqual(old[:2], new[:2])
        self.assertEqual((board.sfen(), tuple(board.move_stack)), before)

    def test_checkmate_and_invalid_option(self):
        board = shogi.Board("4k4/3GRG3/9/9/9/9/9/9/4K4 w - 1")
        before = (board.sfen(), tuple(board.move_stack))
        self.assertTrue(board.is_checkmate())
        move, count, elapsed = mcts_research.select_mcts(
            board, total_budget_from_call=True)
        self.assertEqual((move, count), ("resign", 0))
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), before)
        with self.assertRaises(ValueError):
            mcts_research.select_mcts(shogi.Board(),
                                      total_budget_from_call=1)


if __name__ == "__main__":
    unittest.main()
