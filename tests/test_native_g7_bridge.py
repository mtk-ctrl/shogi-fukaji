"""Isolated regression tests for the opt-in native G7 legal-move bridge."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import shogi

from fukaji import engine
from native.g7_bridge_diagnosis import (
    NativeBridge, build_native, expected_request, native_search,
    ordered_native, validate_request,
)


@unittest.skipUnless(shutil.which("g++"), "C++ compiler required")
class NativeG7BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.tmp.name) / "probe"
        build_native(cls.binary)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fixed_legal_order_and_search(self):
        sfens = [
            shogi.Board().sfen(),
            "4r3k/9/9/9/9/9/9/4G4/4K4 b - 1",
            "4k4/9/9/9/9/9/9/9/4K4 b PGLNS 1",
            "3pkp3/3p1p3/4G4/9/9/9/9/9/4K4 b P 1",
        ]
        bridge = NativeBridge(self.binary)
        try:
            for sfen in sfens:
                board = shogi.Board(sfen)
                before = board.sfen()
                expected = [m.usi() for m in engine.ordered_moves(
                    board, fast_move_list=True, fast_legal_guard=True)]
                native = [m.usi() for m in ordered_native(board, bridge)]
                self.assertEqual(expected, native)
                self.assertEqual(before, board.sfen())
            board = shogi.Board()
            before = board.sfen()
            original = engine.select_move(
                board, max_depth=2, complete_depth=True,
                fast_move_list=True, fast_legal_guard=True,
                root_alpha_beta=True,
            )
            with native_search(bridge):
                candidate = engine.select_move(
                    board, max_depth=2, complete_depth=True,
                    fast_move_list=True, fast_legal_guard=True,
                    root_alpha_beta=True,
                )
            self.assertEqual(original, candidate)
            self.assertEqual(before, board.sfen())
        finally:
            bridge.close()

    def test_timed_native_search_returns_legal_move(self):
        bridge = NativeBridge(self.binary)
        try:
            board = shogi.Board()
            before = board.sfen()
            with native_search(bridge):
                word, depth, nodes = engine.select_move(
                    board, movetime_ms=200, max_depth=4,
                    fast_move_list=True, fast_legal_guard=True,
                    root_alpha_beta=True,
                )
            self.assertIn(shogi.Move.from_usi(word), board.legal_moves)
            self.assertEqual(before, board.sfen())
            self.assertGreater(nodes, 0)
            self.assertGreaterEqual(depth, 0)
            self.assertGreater(bridge.calls, 0)
        finally:
            bridge.close()

    def test_context_restored_and_strict_request(self):
        original = engine.ordered_moves
        bridge = NativeBridge(self.binary)
        try:
            with self.assertRaisesRegex(ValueError, "requires both"):
                with native_search(bridge):
                    engine.ordered_moves(shogi.Board())
            self.assertIs(engine.ordered_moves, original)
        finally:
            bridge.close()
        request = expected_request()
        self.assertTrue(validate_request(request))
        self.assertFalse(validate_request({**request, "enabled": False}))
        with self.assertRaises(ValueError):
            validate_request({**request, "training_allowed": True})


if __name__ == "__main__":
    unittest.main()
