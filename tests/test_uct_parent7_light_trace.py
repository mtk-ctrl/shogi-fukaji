"""Bounded trace of the frozen parent-7 position (research only).

The 200 ms traces are INSTRUMENTED and use the default heuristic, not the
frozen G8 model. Never compare their timings directly with the untraced G8
ABBA run, infer playing strength, or claim a hard real-time guarantee.
No games, labels, training, or formal adoption.
"""
import json
import time
import unittest

import shogi

from fukaji.mcts_research import select_mcts


PARENT7_SFEN = (
    "3g5/1rskg3l/2p2s1pn/4p1p2/ln+B3n1p/+b1P1PP1P1/"
    "3PS1P1P/3SGK1R1/5G1NL b 5Plp 51"
)
SEED = 20279034


class UctParent7LightTrace(unittest.TestCase):
    def test_parent7_200ms_clock_boundary_four_light_calls(self):
        # ABBA order is for instrumentation sanity, NOT a strength or
        # untraced time-compliance comparison. The frozen G8 model is absent.
        for pass_name, reserve in (
            ("off_a", False), ("on_a", True),
            ("on_b", True), ("off_b", False)
        ):
            with self.subTest(pass_name=pass_name):
                board = shogi.Board(PARENT7_SFEN)
                before = (board.sfen(), tuple(board.move_stack))
                phase = []
                caller_started = time.perf_counter()
                move, iterations, internal_ms = select_mcts(
                    board, movetime_ms=200, model=None, fast_relative=True,
                    max_tree_depth=12, seed=SEED,
                    single_pass_legal_moves=True,
                    total_budget_from_call=True, linear_root_choice=True,
                    cleanup_aware_reserve=reserve,
                    clock_phase_trace=phase)
                external_ms = (time.perf_counter() - caller_started) * 1000.0
                self.assertEqual(len(phase), 1)
                entry = phase[0]
                self.assertFalse(entry["terminal_no_legal_moves"])
                self.assertEqual(entry["iterations"], iterations)
                self.assertAlmostEqual(entry["internal_elapsed_ms"], internal_ms)
                self.assertGreaterEqual(entry["generated_nodes"],
                                        entry["root_legal_moves"] + 1)
                self.assertGreaterEqual(entry["longest_iteration_ms"], 0.0)
                self.assertGreaterEqual(entry["internal_deadline_overrun_ms"], 0.0)
                for phase_name in ("root_setup_ms", "search_loop_ms",
                                   "restore_verify_ms", "root_choice_ms",
                                   "return_clock_ms"):
                    self.assertGreaterEqual(entry[phase_name], 0.0)
                if iterations:
                    self.assertGreaterEqual(entry["longest_iteration_ms"],
                                            entry["last_iteration_ms"])
                    self.assertIsNotNone(entry["last_iteration_end_remaining_ms"])
                else:
                    self.assertIsNone(entry["last_iteration_end_remaining_ms"])
                self.assertEqual((board.sfen(), tuple(board.move_stack)), before)
                self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
                print("UCT_PARENT7_CLOCK " + json.dumps({
                    "pass": pass_name, "reserve": reserve,
                    "move": move, "iterations": iterations,
                    "external_ms": round(external_ms, 3),
                    "external_minus_internal_ms": round(external_ms - internal_ms, 3),
                    "root_setup_ms": round(entry["root_setup_ms"], 3),
                    "search_loop_ms": round(entry["search_loop_ms"], 3),
                    "restore_verify_ms": round(entry["restore_verify_ms"], 3),
                    "root_choice_ms": round(entry["root_choice_ms"], 3),
                    "return_clock_ms": round(entry["return_clock_ms"], 3),
                    "generated_nodes": entry["generated_nodes"],
                    "longest_iteration_ms": round(entry["longest_iteration_ms"], 3),
                    "last_iteration_start_remaining_ms": entry[
                        "last_iteration_start_remaining_ms"],
                    "last_iteration_end_remaining_ms": entry[
                        "last_iteration_end_remaining_ms"],
                    "internal_deadline_overrun_ms": round(
                        entry["internal_deadline_overrun_ms"], 3),
                }, sort_keys=True), flush=True)

    def test_parent7_fixed_24_iterations_detailed_phase_trace(self):
        # Separate fixed-iteration phase diagnosis; not a 200 ms observation.
        board = shogi.Board(PARENT7_SFEN)
        before = (board.sfen(), tuple(board.move_stack))
        phases, iterations_trace = [], []
        move, count, _ = select_mcts(
            board, movetime_ms=5000, max_iterations=24, model=None,
            fast_relative=True, max_tree_depth=12, seed=SEED,
            single_pass_legal_moves=True, total_budget_from_call=True,
            linear_root_choice=True, cleanup_aware_reserve=True,
            clock_phase_trace=phases, iteration_trace=iterations_trace)
        self.assertEqual(count, 24)
        self.assertEqual(len(iterations_trace), count)
        self.assertEqual(len(phases), 1)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), before)
        self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
        slowest = max(iterations_trace, key=lambda r: r["total_ms"])
        print("UCT_PARENT7_PHASE " + json.dumps({
            "iterations": count,
            "generated_nodes": phases[0]["generated_nodes"],
            "longest_iteration_ms": round(slowest["total_ms"], 3),
            "slowest_iteration": slowest["iteration"],
            "selection_ms": round(slowest["selection_ms"], 3),
            "expansion_ms": round(slowest["expansion_ms"], 3),
            "legal_moves_ms": round(slowest["legal_moves_ms"], 3),
            "leaf_ms": round(slowest["leaf_ms"], 3),
            "backup_ms": round(slowest["backup_ms"], 3),
            "undo_ms": round(slowest["undo_ms"], 3),
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    unittest.main()
