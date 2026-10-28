"""Bounded UCT return-frame cleanup diagnosis; no games or adoption."""
import json
import statistics
import time
import unittest

import shogi
import fukaji.mcts_research as mcts


class UctReturnCleanupProbe(unittest.TestCase):
    def test_fixed_positions_phase_and_tree_lifetime(self):
        boards = (
            ("initial", shogi.Board()),
            ("in_check", shogi.Board("4k4/9/4R4/9/9/9/9/9/4K4 w - 1")),
            ("hands", shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b Pp 1")),
        )
        original_node = mcts.Node
        for name, board in boards:
            before = (board.sfen(), tuple(board.move_stack))
            samples = []
            # ABBA: same Node factory, only root lifetime differs.
            for retain_root in (False, True, True, False):
                held_roots = []

                def node_factory(*args, **kwargs):
                    node = original_node(*args, **kwargs)
                    if retain_root and kwargs.get("expanded") is True:
                        held_roots.append(node)
                    return node

                trace = []
                mcts.Node = node_factory
                try:
                    caller_start = time.perf_counter()
                    move, iterations, internal_ms = mcts.select_mcts(
                        board, movetime_ms=5000, max_iterations=12,
                        max_tree_depth=4, seed=29,
                        single_pass_legal_moves=True,
                        total_budget_from_call=True,
                        linear_root_choice=True,
                        clock_phase_trace=trace,
                    )
                    caller_ms = (time.perf_counter() - caller_start) * 1000.0
                finally:
                    mcts.Node = original_node

                self.assertEqual(iterations, 12)
                self.assertEqual(len(trace), 1)
                self.assertEqual((board.sfen(), tuple(board.move_stack)), before)
                self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
                self.assertEqual(len(held_roots), int(retain_root))
                record = trace[0]
                cleanup_start = time.perf_counter()
                held_roots.clear()
                cleanup_ms = (time.perf_counter() - cleanup_start) * 1000.0
                samples.append({
                    "retain_root": retain_root, "move": move,
                    "iterations": iterations,
                    "root_legal_moves": record["root_legal_moves"],
                    "root_setup_ms": round(record["root_setup_ms"], 4),
                    "search_loop_ms": round(record["search_loop_ms"], 4),
                    "restore_verify_ms": round(record["restore_verify_ms"], 4),
                    "root_choice_ms": round(record["root_choice_ms"], 4),
                    "return_clock_ms": round(record["return_clock_ms"], 4),
                    "internal_elapsed_ms": round(internal_ms, 4),
                    "caller_elapsed_ms": round(caller_ms, 4),
                    "caller_minus_internal_ms": round(caller_ms - internal_ms, 4),
                    "explicit_release_ms": round(cleanup_ms, 4),
                })
            self.assertEqual(len({x["move"] for x in samples}), 1)
            print("UCT_RETURN_CLEANUP_PROBE " + json.dumps({
                "position": name, "observations": samples,
                "gap_median_without_retention_ms": round(statistics.median(
                    x["caller_minus_internal_ms"] for x in samples
                    if not x["retain_root"]), 4),
                "gap_median_with_retention_ms": round(statistics.median(
                    x["caller_minus_internal_ms"] for x in samples
                    if x["retain_root"]), 4),
                "release_median_retained_ms": round(statistics.median(
                    x["explicit_release_ms"] for x in samples
                    if x["retain_root"]), 4),
            }, sort_keys=True), flush=True)


if __name__ == "__main__":
    unittest.main()
