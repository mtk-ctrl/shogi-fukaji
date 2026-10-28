"""Research-only diagnosis of UCT deadline overruns.

Synthetic delays are causal examples, NOT a replay of the 206.721 ms sample.
No new games, training labels, hard deadline claim, or formal adoption.
"""
import unittest
from unittest.mock import patch

import shogi
import fukaji.mcts_research as mcts


class UctResidualOverrunTraceTests(unittest.TestCase):
    def test_unexpected_long_iteration_can_cross_deadline(self):
        # 40 ms budget. Two 5 ms iterations establish a small reserve;
        # a previously unseen 35 ms third iteration cannot be preempted.
        clock = [0.0]
        costs = iter((0.005, 0.005, 0.035))
        board = shogi.Board()
        original = (board.sfen(), tuple(board.move_stack))
        trace = []

        def delayed_evaluate(*args, **kwargs):
            clock[0] += next(costs)
            return 0

        with patch("fukaji.mcts_research.time.monotonic",
                   side_effect=lambda: clock[0]), \
             patch("fukaji.mcts_research.evaluate",
                   side_effect=delayed_evaluate):
            move, count, internal_ms = mcts.select_mcts(
                board, movetime_ms=40, max_iterations=3, max_tree_depth=1,
                seed=19, single_pass_legal_moves=True,
                total_budget_from_call=True, linear_root_choice=True,
                cleanup_aware_reserve=True, clock_phase_trace=trace)

        self.assertEqual(count, 3)
        self.assertAlmostEqual(internal_ms, 45.0)
        self.assertEqual(len(trace), 1)
        self.assertAlmostEqual(trace[0]["longest_iteration_ms"], 35.0)
        self.assertAlmostEqual(trace[0]["last_iteration_ms"], 35.0)
        self.assertAlmostEqual(trace[0]["last_iteration_start_remaining_ms"], 30.0)
        self.assertAlmostEqual(trace[0]["last_iteration_end_remaining_ms"], -5.0)
        self.assertAlmostEqual(trace[0]["internal_deadline_overrun_ms"], 5.0)
        self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), original)

    def test_injected_pause_after_loop_is_counted_in_internal_clock(self):
        # A synthetic pause in final selection, not a measured OS stall.
        clock = [0.0]
        board = shogi.Board()
        original = (board.sfen(), tuple(board.move_stack))
        trace = []
        original_choice = mcts._best_root_child

        def delayed_evaluate(*args, **kwargs):
            clock[0] += 0.005
            return 0

        def delayed_choice(*args, **kwargs):
            clock[0] += 0.012
            return original_choice(*args, **kwargs)

        with patch("fukaji.mcts_research.time.monotonic",
                   side_effect=lambda: clock[0]), \
             patch("fukaji.mcts_research.evaluate",
                   side_effect=delayed_evaluate), \
             patch("fukaji.mcts_research._best_root_child",
                   side_effect=delayed_choice):
            move, count, internal_ms = mcts.select_mcts(
                board, movetime_ms=10, max_iterations=1, max_tree_depth=1,
                seed=19, single_pass_legal_moves=True,
                total_budget_from_call=True, linear_root_choice=True,
                cleanup_aware_reserve=True, clock_phase_trace=trace)

        self.assertEqual(count, 1)
        self.assertAlmostEqual(internal_ms, 17.0)
        self.assertAlmostEqual(trace[0]["longest_iteration_ms"], 5.0)
        self.assertAlmostEqual(trace[0]["last_iteration_end_remaining_ms"], 5.0)
        self.assertAlmostEqual(trace[0]["root_choice_ms"], 12.0)
        self.assertAlmostEqual(trace[0]["internal_deadline_overrun_ms"], 7.0)
        self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), original)

    def test_zero_iterations_are_explicitly_traced(self):
        clock = [0.0]
        board = shogi.Board()
        original = (board.sfen(), tuple(board.move_stack))
        original_legal = mcts._legal_moves_once
        trace = []

        def delayed_root(position, *, single_pass=False):
            result = original_legal(position, single_pass=single_pass)
            clock[0] += 0.050
            return result

        with patch("fukaji.mcts_research.time.monotonic",
                   side_effect=lambda: clock[0]), \
             patch("fukaji.mcts_research._legal_moves_once",
                   side_effect=delayed_root):
            move, count, internal_ms = mcts.select_mcts(
                board, movetime_ms=20, max_iterations=3,
                single_pass_legal_moves=True, total_budget_from_call=True,
                linear_root_choice=True, cleanup_aware_reserve=True,
                clock_phase_trace=trace)

        self.assertEqual(count, 0)
        self.assertAlmostEqual(internal_ms, 50.0)
        self.assertEqual(trace[0]["longest_iteration_ms"], 0.0)
        for key in ("last_iteration_ms", "last_iteration_start_remaining_ms",
                    "last_iteration_end_remaining_ms"):
            self.assertIsNone(trace[0][key])
        self.assertAlmostEqual(trace[0]["internal_deadline_overrun_ms"], 30.0)
        self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), original)

    def test_real_fixed_position_trace_preserves_move_and_history(self):
        board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b Pp 1")
        original = (board.sfen(), tuple(board.move_stack))
        trace = []
        move, count, internal_ms = mcts.select_mcts(
            board, movetime_ms=2000, max_iterations=8, max_tree_depth=3,
            seed=41, single_pass_legal_moves=True,
            total_budget_from_call=True, linear_root_choice=True,
            cleanup_aware_reserve=True, clock_phase_trace=trace)
        self.assertEqual(count, 8)
        self.assertEqual(len(trace), 1)
        self.assertGreaterEqual(trace[0]["generated_nodes"],
                                trace[0]["root_legal_moves"] + 1)
        self.assertGreaterEqual(trace[0]["longest_iteration_ms"],
                                trace[0]["last_iteration_ms"])
        self.assertGreater(trace[0]["last_iteration_start_remaining_ms"], 0)
        self.assertGreaterEqual(trace[0]["internal_deadline_overrun_ms"], 0)
        self.assertAlmostEqual(trace[0]["internal_elapsed_ms"], internal_ms)
        self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), original)


if __name__ == "__main__":
    unittest.main()
