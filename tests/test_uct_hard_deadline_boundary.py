"""Research-only synthetic proof: the caller deadline includes return cleanup.

No OS-stall measurement, no frozen G8 evaluation, no new games or adoption.
The injected release delay is an artificial clock cost, not observed timing.
"""
import unittest
from unittest.mock import patch

import shogi
import fukaji.mcts_research as mcts


class UctHardDeadlineBoundaryTests(unittest.TestCase):
    def test_return_cleanup_can_exceed_budget_after_internal_clock_stops(self):
        clock = [0.0]
        board = shogi.Board()
        before = (board.sfen(), tuple(board.move_stack))
        released = [0]

        class SyntheticSlowReleaseNode(mcts.Node):
            def __del__(self):
                released[0] += 1
                clock[0] += 0.001  # 1 ms per released Node, synthetic only

        def synthetic_evaluate(*_args, **_kwargs):
            clock[0] += 0.005
            return 0

        with patch("fukaji.mcts_research.Node", SyntheticSlowReleaseNode), \
             patch("fukaji.mcts_research.time.monotonic",
                   side_effect=lambda: clock[0]), \
             patch("fukaji.mcts_research.evaluate",
                   side_effect=synthetic_evaluate):
            move, iterations, internal_ms = mcts.select_mcts(
                board, movetime_ms=20, max_iterations=1, max_tree_depth=1,
                seed=29, single_pass_legal_moves=True,
                total_budget_from_call=True, linear_root_choice=True,
                cleanup_aware_reserve=True)
            caller_ms = clock[0] * 1000.0

        self.assertEqual(iterations, 1)
        self.assertAlmostEqual(internal_ms, 5.0)
        self.assertLessEqual(internal_ms, 20.0)  # internal timer meets its budget
        self.assertGreaterEqual(released[0], 20)
        self.assertGreater(caller_ms, 20.0)
        self.assertGreater(caller_ms - internal_ms, 20.0)
        self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), before)


if __name__ == "__main__":
    unittest.main()
