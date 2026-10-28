"""Bounded research-only node-count versus return-frame cleanup probe.

Uses fixed iterations and diagnostic clocks. Not a 200ms compliance or strength test.
"""
import json
import statistics
import time
import unittest

import shogi
import fukaji.mcts_research as mcts


class UctNodeCleanupScalingProbe(unittest.TestCase):
    def test_fixed_positions_node_count_and_release(self):
        positions = (
            ("initial", "startpos"),
            ("in_check", "4k4/9/4R4/9/9/9/9/9/4K4 w - 1"),
            ("hands", "4k4/9/9/9/9/9/9/9/4K4 b Pp 1"),
        )
        original_node = mcts.Node
        for name, sfen in positions:
            prior_generated_nodes = 0
            for budget in (12, 48, 96):
                board = shogi.Board() if sfen == "startpos" else shogi.Board(sfen)
                before = (board.sfen(), tuple(board.move_stack))
                samples = []
                for retain_root in (False, True, True, False):
                    retained = []
                    node_count = [0]

                    def node_factory(*args, **kwargs):
                        node_count[0] += 1
                        node = original_node(*args, **kwargs)
                        if retain_root and kwargs.get("expanded") is True:
                            retained.append(node)
                        return node

                    trace = []
                    iterations_trace = []
                    mcts.Node = node_factory
                    try:
                        caller_started = time.perf_counter()
                        move, iterations, internal_ms = mcts.select_mcts(
                            board, movetime_ms=5000, max_iterations=budget,
                            max_tree_depth=4, seed=29,
                            single_pass_legal_moves=True,
                            total_budget_from_call=True,
                            linear_root_choice=True,
                            clock_phase_trace=trace,
                            iteration_trace=iterations_trace,
                        )
                        caller_ms = (time.perf_counter() - caller_started) * 1000.0
                    finally:
                        mcts.Node = original_node

                    self.assertEqual(iterations, budget)
                    self.assertEqual(len(trace), 1)
                    self.assertEqual(len(iterations_trace), budget)
                    self.assertEqual((board.sfen(), tuple(board.move_stack)), before)
                    self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
                    self.assertEqual(len(retained), int(retain_root))
                    root_legal = trace[0]["root_legal_moves"]
                    self.assertGreaterEqual(node_count[0], root_legal + 1)
                    release_started = time.perf_counter()
                    retained.clear()
                    release_ms = (time.perf_counter() - release_started) * 1000.0
                    samples.append({
                        "retain_root": retain_root,
                        "move": move,
                        "iterations": iterations,
                        "node_count": node_count[0],
                        "root_children": root_legal,
                        "longest_iteration_ms": round(max(
                            x["total_ms"] for x in iterations_trace), 4),
                        "caller_minus_internal_ms": round(
                            caller_ms - internal_ms, 4),
                        "explicit_release_ms": round(release_ms, 4),
                    })
                self.assertEqual(len({x["move"] for x in samples}), 1)
                self.assertEqual(len({x["node_count"] for x in samples}), 1)
                self.assertEqual(len({x["root_children"] for x in samples}), 1)
                # A longer fixed-seed prefix cannot undo earlier node creation.
                self.assertGreaterEqual(samples[0]["node_count"], prior_generated_nodes)
                prior_generated_nodes = samples[0]["node_count"]
                print("UCT_NODE_CLEANUP_SCALING " + json.dumps({
                    "position": name, "fixed_iterations": budget,
                    "observations": samples,
                    "median_gap_off_ms": round(statistics.median(
                        x["caller_minus_internal_ms"] for x in samples
                        if not x["retain_root"]), 4),
                    "median_gap_on_ms": round(statistics.median(
                        x["caller_minus_internal_ms"] for x in samples
                        if x["retain_root"]), 4),
                    "median_release_on_ms": round(statistics.median(
                        x["explicit_release_ms"] for x in samples
                        if x["retain_root"]), 4),
                }, sort_keys=True), flush=True)


if __name__ == "__main__":
    unittest.main()
