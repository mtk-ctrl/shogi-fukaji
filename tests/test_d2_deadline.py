"""D2 research-only shared whole-call deadline and legal-list parity."""
import time
import unittest
from unittest.mock import patch

import shogi

from fukaji.engine import ordered_moves, select_move


class D2DeadlineTests(unittest.TestCase):
    def test_whole_call_deadline_rejects_search_after_slow_root(self):
        board = shogi.Board()
        original = ordered_moves

        def slow_root(*args, **kwargs):
            result = original(*args, **kwargs)
            time.sleep(0.012)
            return result

        with patch("fukaji.engine.ordered_moves", side_effect=slow_root):
            timing = {}
            word, depth, nodes = select_move(
                board, movetime_ms=1, max_depth=2, deadline_from_call=True,
                telemetry=timing)
        self.assertEqual(depth, 0)
        self.assertEqual(nodes, 0)
        self.assertIn(shogi.Move.from_usi(word), board.legal_moves)
        self.assertGreaterEqual(timing["root_preparation_ms"], 10)
        self.assertGreaterEqual(timing["total_call_ms"], timing["root_preparation_ms"])

    def test_single_pass_legal_list_has_same_fixed_depth_one_result(self):
        board = shogi.Board()
        before = (board.sfen(), tuple(board.move_stack))
        a = select_move(board, movetime_ms=1, max_depth=1,
                        complete_depth=True, fast_move_list=False,
                        deadline_from_call=True)
        b = select_move(board, movetime_ms=1, max_depth=1,
                        complete_depth=True, fast_move_list=True,
                        deadline_from_call=True)
        self.assertEqual(a, b)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), before)

    def test_default_deadline_mode_is_unchanged_and_legal(self):
        board = shogi.Board()
        a = select_move(board, movetime_ms=20, max_depth=1)
        b = select_move(board, movetime_ms=20, max_depth=1,
                        deadline_from_call=False)
        for word, depth, nodes in (a, b):
            self.assertIn(shogi.Move.from_usi(word), board.legal_moves)
            self.assertGreaterEqual(depth, 0)
            self.assertGreaterEqual(nodes, 0)


if __name__ == "__main__":
    unittest.main()
