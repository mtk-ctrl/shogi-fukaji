"""Bounded correctness tests for the opt-in MCTS research prototype."""
import unittest
from unittest.mock import patch

import shogi

from fukaji.mcts_research import Node, _uct_child, select_mcts


class MctsResearchTests(unittest.TestCase):
    def test_returns_legal_move_without_mutating_start_position(self):
        board = shogi.Board()
        before = (board.sfen(), tuple(board.move_stack))
        move, iterations, elapsed = select_mcts(
            board, movetime_ms=1000, max_iterations=8, seed=42,
            max_tree_depth=3,
        )
        self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
        self.assertEqual(iterations, 8)
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), before)

    def test_restores_nonempty_history_and_hands(self):
        board = shogi.Board()
        for usi in ("7g7f", "7c7d", "7f7e", "3c3d", "7e7d"):
            move = shogi.Move.from_usi(usi)
            self.assertIn(move, board.legal_moves)
            board.push(move)
        self.assertEqual(board.pieces_in_hand[shogi.BLACK][shogi.PAWN], 1)
        before = (board.sfen(), tuple(board.move_stack))
        move, iterations, _ = select_mcts(
            board, movetime_ms=1000, max_iterations=12, seed=11,
            max_tree_depth=4,
        )
        self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
        self.assertEqual(iterations, 12)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), before)

    def test_deterministic_for_fixed_iterations(self):
        a = select_mcts(shogi.Board(), movetime_ms=1000,
                        max_iterations=10, seed=7, max_tree_depth=3)
        b = select_mcts(shogi.Board(), movetime_ms=1000,
                        max_iterations=10, seed=7, max_tree_depth=3)
        self.assertEqual(a[0:2], b[0:2])

    def test_child_value_is_negated_from_parent_viewpoint(self):
        positive_for_opponent = Node(visits=10, value_sum=8)
        negative_for_opponent = Node(visits=10, value_sum=-8)
        root = Node(visits=20, children=[
            positive_for_opponent, negative_for_opponent])
        self.assertIs(_uct_child(root, 1.4), negative_for_opponent)

    def test_checked_king_only_chooses_legal_evasion(self):
        board = shogi.Board("4k4/9/4R4/9/9/9/9/9/4K4 w - 1")
        self.assertTrue(board.is_check())
        self.assertFalse(board.is_checkmate())
        before = (board.sfen(), tuple(board.move_stack))
        move, iterations, _ = select_mcts(
            board, movetime_ms=1000, max_iterations=12, seed=19,
            max_tree_depth=3,
        )
        self.assertGreater(iterations, 0)
        self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), before)

    def test_checkmated_side_resigns_without_mutating_board(self):
        board = shogi.Board("4k4/3GRG3/9/9/9/9/9/9/4K4 w - 1")
        self.assertTrue(board.is_checkmate())
        before = (board.sfen(), tuple(board.move_stack))
        self.assertEqual(select_mcts(board, movetime_ms=200),
                         ("resign", 0, 0.0))
        self.assertEqual((board.sfen(), tuple(board.move_stack)), before)

    def test_adaptive_reserve_avoids_starting_an_unsafe_iteration(self):
        # Deterministic fake clock: every leaf evaluation costs 15 ms.
        # At 40 ms, starting a third iteration would exceed the budget.
        clock = [0.0]
        def fake_evaluate(*args, **kwargs):
            clock[0] += 0.015
            return 0
        with patch("fukaji.mcts_research.time.monotonic",
                   side_effect=lambda: clock[0]), \
             patch("fukaji.mcts_research.evaluate",
                   side_effect=fake_evaluate):
            move, iterations, elapsed = select_mcts(
                shogi.Board(), movetime_ms=40, max_iterations=10,
                seed=4, max_tree_depth=1,
            )
        self.assertEqual(iterations, 2)
        self.assertLessEqual(elapsed, 40.0)
        self.assertNotEqual(move, "resign")

    def test_rejects_invalid_research_parameters(self):
        board = shogi.Board()
        for kwargs in ({"movetime_ms": 0}, {"max_iterations": 0},
                       {"max_tree_depth": 0}, {"exploration": -1},
                       {"exploration": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_mcts(board, **kwargs)


if __name__ == "__main__":
    unittest.main()
