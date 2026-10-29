"""D3 research tests: legal tactical leaves, check evasions, and undo safety."""
import time
import unittest

import shogi

from fukaji.engine import INF, evaluate, ordered_moves, select_move
from fukaji.quiescence import quiesce, tactical_moves


def run_q(board, *, evaluator=evaluate, quiet_left=2, check_left=5):
    return quiesce(
        board, -INF, INF, time.monotonic() + 5, [0], evaluator,
        lambda b: ordered_moves(b), quiet_left=quiet_left,
        check_left=check_left,
    )


class D3QuiescenceTests(unittest.TestCase):
    def test_quiet_startpos_is_unchanged(self):
        board = shogi.Board()
        self.assertFalse(board.is_check())
        self.assertEqual(tactical_moves(board, ordered_moves(board)), [])
        self.assertEqual(run_q(board), evaluate(board))

    def test_independent_one_ply_recapture_oracle(self):
        # Two kings and adjacent opposed rooks: White can capture Black's rook.
        # A full-width one-ply legal-move oracle has a unique winning capture;
        # no deeper reply can immediately recapture it.
        board = shogi.Board("8k/9/4r4/4R4/9/9/9/9/8K w - 1")
        capture = shogi.Move.from_usi("5c5d")
        self.assertIn(capture, board.legal_moves)
        before = (board.sfen(), tuple(board.move_stack))
        quiet_score = evaluate(board)
        board.push(capture)
        try:
            expected = -evaluate(board)
        finally:
            board.pop()
        self.assertGreater(expected, quiet_score)
        self.assertEqual(run_q(board, quiet_left=1), expected)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), before)

    def test_capture_is_selected_from_legal_moves(self):
        board = shogi.Board("4k4/9/9/4p4/4R4/9/9/9/4K4 b - 1")
        capture = shogi.Move.from_usi("5e5d")
        self.assertIn(capture, board.legal_moves)
        self.assertIn(capture, tactical_moves(board, ordered_moves(board)))
        before = (board.sfen(), tuple(board.move_stack))
        run_q(board)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), before)

    def test_in_check_never_stands_pat(self):
        board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 w - 1")
        self.assertTrue(board.is_check())
        before = (board.sfen(), tuple(board.move_stack))
        def guarded_evaluate(position):
            self.assertFalse(position.is_check(), "static evaluation in check")
            return evaluate(position)
        result = run_q(board, evaluator=guarded_evaluate)
        self.assertIsInstance(result, int)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), before)

    def test_unresolved_check_chain_is_unknown_not_static(self):
        board = shogi.Board("4k4/9/9/9/4R4/9/9/9/4K4 w - 1")
        with self.assertRaises(TimeoutError):
            run_q(board, check_left=0)

    def test_exception_restores_full_history(self):
        board = shogi.Board("4k4/9/9/4p4/4R4/9/9/9/4K4 b - 1")
        before = (board.sfen(), tuple(board.move_stack),
                  dict(board.transpositions))
        calls = [0]
        def interrupted(position):
            calls[0] += 1
            if calls[0] >= 2:
                raise TimeoutError("forced leaf interruption")
            return evaluate(position)
        with self.assertRaises(TimeoutError):
            run_q(board, evaluator=interrupted)
        self.assertEqual(
            (board.sfen(), tuple(board.move_stack), dict(board.transpositions)),
            before,
        )

    def test_opt_in_and_legacy_path_return_legal_moves(self):
        board = shogi.Board()
        original = (board.sfen(), tuple(board.move_stack))
        old = select_move(board, 1, max_depth=1, complete_depth=True)
        disabled = select_move(board, 1, max_depth=1, complete_depth=True,
                               quiescence=False)
        enabled = select_move(board, 1, max_depth=1, complete_depth=True,
                              quiescence=True)
        self.assertEqual(old, disabled)
        self.assertIn(shogi.Move.from_usi(enabled[0]), board.legal_moves)
        self.assertEqual(enabled[1], 1)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), original)


if __name__ == "__main__":
    unittest.main()
