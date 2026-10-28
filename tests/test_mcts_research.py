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

    def test_opt_in_phase_trace_is_complete_and_board_is_restored(self):
        board = shogi.Board()
        original = (board.sfen(), tuple(board.move_stack))
        trace = []
        move, count, elapsed = select_mcts(
            board, movetime_ms=1000, max_iterations=5,
            max_tree_depth=3, seed=23, iteration_trace=trace,
        )
        self.assertEqual(count, 5)
        self.assertEqual(len(trace), count)
        self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), original)
        self.assertGreaterEqual(elapsed, 0)
        for index, record in enumerate(trace, start=1):
            self.assertEqual(record['iteration'], index)
            self.assertGreaterEqual(record['depth'], 1)
            self.assertGreater(record['root_legal_moves'], 0)
            for phase in ('selection_ms', 'expansion_ms', 'leaf_ms',
                          'backup_ms', 'undo_ms', 'checkmate_ms',
                          'repetition_ms', 'legal_moves_ms',
                          'child_nodes_ms', 'push_ms'):
                self.assertGreaterEqual(record[phase], 0)
            subphases = ('checkmate_ms', 'repetition_ms',
                         'legal_moves_ms', 'child_nodes_ms', 'push_ms')
            self.assertLessEqual(
                sum(record[p] for p in subphases),
                record['expansion_ms'] + 0.5)
            self.assertGreaterEqual(record['total_ms'], 0)
            self.assertGreaterEqual(record['reserve_ms'], 3)
            self.assertLessEqual(
                sum(record[p] for p in ('selection_ms', 'expansion_ms',
                                       'leaf_ms', 'backup_ms', 'undo_ms')),
                record['total_ms'] + 0.1,
            )

    def test_subphase_trace_preserves_fixed_iteration_move(self):
        untraced = select_mcts(
            shogi.Board(), movetime_ms=1000, max_iterations=6,
            seed=31, max_tree_depth=3)
        trace = []
        traced = select_mcts(
            shogi.Board(), movetime_ms=1000, max_iterations=6,
            seed=31, max_tree_depth=3, iteration_trace=trace)
        self.assertEqual(untraced[:2], traced[:2])
        self.assertEqual(len(trace), 6)
        self.assertTrue(any(row['legal_moves_ms'] > 0 for row in trace))

    def test_no_expansion_records_zero_subphase_times(self):
        trace = []
        select_mcts(shogi.Board(), movetime_ms=1000, max_iterations=4,
                    max_tree_depth=1, seed=9, iteration_trace=trace)
        self.assertEqual(len(trace), 4)
        for row in trace:
            for key in ('checkmate_ms', 'repetition_ms',
                        'legal_moves_ms', 'child_nodes_ms', 'push_ms'):
                self.assertEqual(row[key], 0.0)

    def test_single_pass_avoids_generator_length_prepass(self):
        board = shogi.Board()
        before = (board.sfen(), tuple(board.move_stack))
        # list(LegalMoveGenerator) invokes __len__, while a comprehension
        # iterates once. This must cover both root and tree expansions.
        with patch.object(shogi.LegalMoveGenerator, "__len__",
                          side_effect=AssertionError("unexpected length prepass")):
            move, count, _ = select_mcts(
                board, movetime_ms=2000, max_iterations=8,
                seed=41, max_tree_depth=3, single_pass_legal_moves=True)
        self.assertEqual(count, 8)
        self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), before)

    def test_single_pass_preserves_moves_order_and_undo(self):
        boards = [
            shogi.Board(),
            shogi.Board("4k4/9/4R4/9/9/9/9/9/4K4 w - 1"),
            shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b Pp 1"),
            shogi.Board("4k4/9/3RGR3/9/9/9/9/9/4K4 b P 1"),
            shogi.Board("4k4/3GRG3/9/9/9/9/9/9/4K4 w - 1"),
        ]
        repeated = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        for word in ("5i6i", "5a6a", "6i5i", "6a5a") * 3:
            repeated.push(shogi.Move.from_usi(word))
        self.assertTrue(repeated.is_fourfold_repetition())
        boards.append(repeated)
        for board in boards:
            with self.subTest(sfen=board.sfen()):
                before = (board.sfen(), tuple(board.move_stack))
                self.assertEqual(
                    [move.usi() for move in list(board.legal_moves)],
                    [move.usi() for move in board.legal_moves])
                if board.is_checkmate():
                    self.assertEqual(list(board.legal_moves), [])
                if board.sfen().startswith("4k4/9/3RGR3"):
                    self.assertNotIn(shogi.Move.from_usi("P*5b"), board.legal_moves)
                original = select_mcts(
                    board, movetime_ms=2000, max_iterations=8,
                    max_tree_depth=3, seed=41)
                candidate = select_mcts(
                    board, movetime_ms=2000, max_iterations=8,
                    max_tree_depth=3, seed=41, single_pass_legal_moves=True)
                self.assertEqual(original[:2], candidate[:2])
                self.assertEqual((board.sfen(), tuple(board.move_stack)), before)

    def test_single_pass_rejects_non_boolean_flag(self):
        with self.assertRaises(ValueError):
            select_mcts(shogi.Board(), single_pass_legal_moves=1)

    def test_opt_in_return_clock_trace_preserves_fixed_iteration_move(self):
        for total_budget in (False, True):
            with self.subTest(total_budget=total_budget):
                board = shogi.Board()
                original = (board.sfen(), tuple(board.move_stack))
                baseline = select_mcts(
                    board, movetime_ms=1000, max_iterations=5, seed=17,
                    max_tree_depth=3, total_budget_from_call=total_budget)
                trace = []
                observed = select_mcts(
                    board, movetime_ms=1000, max_iterations=5, seed=17,
                    max_tree_depth=3, total_budget_from_call=total_budget,
                    clock_phase_trace=trace)
                self.assertEqual(baseline[:2], observed[:2])
                self.assertEqual(len(trace), 1)
                self.assertEqual(trace[0]['root_legal_moves'],
                                 len([m for m in board.legal_moves]))
                self.assertEqual(trace[0]['iterations'], 5)
                self.assertEqual(trace[0]['total_budget_from_call'],
                                 total_budget)
                self.assertFalse(trace[0]['terminal_no_legal_moves'])
                for key in ('root_setup_ms', 'search_loop_ms',
                            'restore_verify_ms', 'root_choice_ms',
                            'return_clock_ms', 'internal_elapsed_ms'):
                    self.assertGreaterEqual(trace[0][key], 0)
                self.assertEqual((board.sfen(), tuple(board.move_stack)),
                                 original)

    def test_return_clock_trace_handles_no_legal_move(self):
        board = shogi.Board("4k4/3GRG3/9/9/9/9/9/9/4K4 w - 1")
        trace = []
        move, count, elapsed = select_mcts(
            board, clock_phase_trace=trace)
        self.assertEqual((move, count, elapsed), ('resign', 0, 0.0))
        self.assertEqual(len(trace), 1)
        self.assertTrue(trace[0]['terminal_no_legal_moves'])
        self.assertEqual(trace[0]['root_legal_moves'], 0)

    def test_return_clock_trace_rejects_invalid_sink(self):
        with self.assertRaises(ValueError):
            select_mcts(shogi.Board(), clock_phase_trace=object())

    def test_rejects_invalid_research_parameters(self):
        board = shogi.Board()
        for kwargs in ({"movetime_ms": 0}, {"max_iterations": 0},
                       {"max_tree_depth": 0}, {"exploration": -1},
                       {"exploration": float("nan")},
                       {"iteration_trace": object()}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_mcts(board, **kwargs)


if __name__ == "__main__":
    unittest.main()
