"""D9 research-only per-node clock checks; no default behavior change."""
import unittest
from unittest.mock import patch

import shogi

from fukaji.engine import INF, negamax, select_move


class D9ClockCheckTests(unittest.TestCase):
    def test_expired_deadline_checked_at_first_node_only_when_opted_in(self):
        board = shogi.Board()
        before = (board.sfen(), tuple(board.move_stack), board.zobrist_hash())
        with patch("fukaji.engine.time.monotonic", return_value=100.0):
            # Original 128-node clock sampling does not check on node 1.
            normal_nodes = [0]
            negamax(board, 0, -INF, INF, 0.0, normal_nodes)
            self.assertEqual(normal_nodes[0], 1)
            with self.assertRaises(TimeoutError):
                negamax(board, 0, -INF, INF, 0.0, [0], time_check_each_node=True)
            # Original boundary remains exactly 128 nodes.
            with self.assertRaises(TimeoutError):
                negamax(board, 0, -INF, INF, 0.0, [127])
        self.assertEqual((board.sfen(), tuple(board.move_stack), board.zobrist_hash()), before)

    def test_fixed_depth_legal_move_nodes_and_history_match(self):
        for sequence in ((), ("7g7f", "3c3d", "2g2f", "8c8d")):
            board = shogi.Board()
            for word in sequence:
                board.push_usi(word)
            before = (board.sfen(), tuple(board.move_stack), board.zobrist_hash())
            args = dict(movetime_ms=1, max_depth=2, complete_depth=True,
                        deadline_from_call=True, fast_move_list=True,
                        fast_relative=True, root_alpha_beta=True)
            baseline = select_move(board, time_check_each_node=False, **args)
            candidate = select_move(board, time_check_each_node=True, **args)
            self.assertEqual(baseline, candidate)
            self.assertIn(shogi.Move.from_usi(candidate[0]), board.legal_moves)
            self.assertEqual((board.sfen(), tuple(board.move_stack), board.zobrist_hash()), before)

    def test_short_real_deadline_keeps_legal_move_and_undo(self):
        board = shogi.Board()
        before = (board.sfen(), tuple(board.move_stack), board.zobrist_hash())
        move, depth, nodes = select_move(
            board, movetime_ms=1, max_depth=3, deadline_from_call=True,
            fast_move_list=True, fast_relative=True, root_alpha_beta=True,
            time_check_each_node=True)
        self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
        self.assertGreaterEqual(depth, 0)
        self.assertGreaterEqual(nodes, 0)
        self.assertEqual((board.sfen(), tuple(board.move_stack), board.zobrist_hash()), before)


if __name__ == "__main__":
    unittest.main()
