"""Research-only read-only C++ evaluation snapshot equivalence and rollback."""
import ctypes
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import shogi

from fukaji import engine
from fukaji.learning import FEATURES
from fukaji.relative_value import RelativeValue
from native.g7_bridge_diagnosis import search
from native.g7_stateful_bridge import (
    StatefulRules, SyncedBoard, build_state_shared, stateful_order_search,
)


class NativeEvalSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.tmp.name) / "state.so"
        build_state_shared(cls.binary)
        model_path = Path(cls.tmp.name) / "synthetic.npz"
        np.savez_compressed(model_path, weight=np.linspace(
            -0.03, 0.03, FEATURES, dtype=np.float32))
        cls.model = RelativeValue(model_path, mix_percent=25)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_snapshot_piece_hand_and_turn_after_reversible_moves(self):
        rng = random.Random(20271028)
        sfens = (
            shogi.Board().sfen(),
            "4k4/9/9/9/9/9/9/9/4K4 b RBGSNLP 1",
            "4k4/9/9/9/9/9/9/9/4K4 w rbgsnlp 1",
            "4k4/9/9/9/9/9/9/4+P4/4K4 b - 1",
        )
        for sfen in sfens:
            with self.subTest(sfen=sfen):
                board = shogi.Board(sfen)
                original = board.sfen()
                state = StatefulRules(self.binary, board, audit=True)
                proxy = SyncedBoard(board, state)
                try:
                    state.eval_snapshot(proxy)
                    played = 0
                    for _ in range(18):
                        moves = list(board.legal_moves)
                        if not moves:
                            break
                        proxy.push(rng.choice(moves))
                        state.eval_snapshot(proxy)
                        self.assertEqual(
                            state.evaluate_relative_snapshot(proxy, self.model),
                            engine.evaluate_relative_fused(board, self.model))
                        played += 1
                    for _ in range(played):
                        proxy.pop()
                        state.eval_snapshot(proxy)
                    proxy.assert_restored(original)
                finally:
                    state.close()

    def test_snapshot_rejects_short_buffer_and_unknown_handle(self):
        board = shogi.Board()
        state = StatefulRules(self.binary, board)
        try:
            original = state._text("sfen")
            small = (ctypes.c_int16 * 95)()
            self.assertLess(state.lib.fukaji_state_eval_snapshot(
                state.handle.value, small, len(small)), 0)
            large = (ctypes.c_int16 * 96)()
            self.assertLess(state.lib.fukaji_state_eval_snapshot(
                999999999, large, len(large)), 0)
            self.assertEqual(state._text("sfen"), original)
            self.assertEqual(state.eval_snapshot(board)[95], int(board.turn))
        finally:
            state.close()

    def test_fixed_depth_equivalence_and_exception_rollback(self):
        for sfen in (
            shogi.Board().sfen(),
            "4k4/9/9/9/9/9/9/4G4/4K4 b P 1",
            "4k4/9/9/9/9/9/9/9/4K4 w p 1",
        ):
            with self.subTest(sfen=sfen):
                board = shogi.Board(sfen)
                original = board.sfen()
                baseline = search(board, self.model, fixed=True)
                state = StatefulRules(self.binary, board, audit=False)
                proxy = SyncedBoard(board, state)
                old_evaluate = engine.evaluate
                try:
                    with stateful_order_search(
                        state, cached_moves=True, native_scored=True,
                        native_snapshot_eval=True):
                        self.assertEqual(search(proxy, self.model, fixed=True), baseline)
                    self.assertIs(engine.evaluate, old_evaluate)
                    proxy.assert_restored(original)
                    self.assertGreater(state.eval_snapshot_calls, 0)
                    # The evaluator must reject a different native board handle.
                    other = StatefulRules(self.binary, board)
                    try:
                        with stateful_order_search(
                            state, native_scored=True, native_snapshot_eval=True):
                            with self.assertRaises(RuntimeError):
                                engine.evaluate(SyncedBoard(board, other), self.model, True)
                    finally:
                        other.close()
                    self.assertIs(engine.evaluate, old_evaluate)
                    proxy.assert_restored(original)
                finally:
                    state.close()


if __name__ == "__main__":
    unittest.main()
