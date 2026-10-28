"""Bounded correctness tests for the opt-in MCTS research prototype."""
import unittest

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
        for usi in ("7g7f", "3c3d", "2g2f", "8c8d", "2f2e", "8d8e"):
            move = shogi.Move.from_usi(usi)
            self.assertIn(move, board.legal_moves)
            board.push(move)
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

    def test_rejects_invalid_research_parameters(self):
        board = shogi.Board()
        for kwargs in ({"movetime_ms": 0}, {"max_iterations": 0},
                       {"max_tree_depth": 0}, {"exploration": -1},
                       {"exploration": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_mcts(board, **kwargs)


if __name__ == "__main__":
    unittest.main()
