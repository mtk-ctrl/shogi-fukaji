"""Safety and equivalence checks for research-only C ABI / 20-way diagnosis."""
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

import shogi
from fukaji import engine
from native.g7_inprocess_diagnosis import (
    InProcessBridge, build_shared, expected_request, validate_request,
    aggregate, ordered_native_direct, native_order_search, SHARDS, SAMPLES, VARIANTS,
)
from native.g7_bridge_diagnosis import native_search, ordered_native


@unittest.skipUnless(shutil.which("g++"), "g++ required")
class InProcessRulesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.tmp.name) / "rules.so"
        build_shared(cls.binary)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_native_legal_set_and_order(self):
        bridge = InProcessBridge(self.binary)
        sfens = [
            shogi.Board().sfen(),
            "4r3k/9/9/9/9/9/9/4G4/4K4 b - 1",
            "4k4/9/9/9/9/9/9/9/4K4 b PGLNS 1",
            "3pkp3/3p1p3/4G4/9/9/9/9/9/4K4 b P 1",
        ]
        for sfen in sfens:
            board = shogi.Board(sfen)
            canonical = board.sfen()
            self.assertEqual(bridge.legal(board),
                             sorted(m.usi() for m in board.legal_moves))
            self.assertEqual(
                [m.usi() for m in ordered_native(board, bridge)],
                [m.usi() for m in engine.ordered_moves(
                    board, fast_move_list=True, fast_legal_guard=True)])
            self.assertEqual(bridge.legal_pseudo_order(board),
                             [m.usi() for m in board.legal_moves])
            self.assertEqual(
                [m.usi() for m in ordered_native_direct(board, bridge)],
                [m.usi() for m in engine.ordered_moves(
                    board, fast_move_list=True, fast_legal_guard=True)])
            self.assertEqual(board.sfen(), canonical)

    def test_native_pseudo_order_reachable_positions(self):
        bridge = InProcessBridge(self.binary)
        for seed in range(8):
            board = shogi.Board()
            rng = random.Random(20271027 + seed)
            for _ in range(24):
                expected = [m.usi() for m in board.legal_moves]
                self.assertEqual(bridge.legal_pseudo_order(board), expected)
                self.assertEqual([m.usi() for m in ordered_native_direct(board, bridge)],
                                 [m.usi() for m in engine.ordered_moves(
                                     board, fast_move_list=True, fast_legal_guard=True)])
                legal = list(board.legal_moves)
                if not legal:
                    break
                board.push(rng.choice(legal))

    def test_fixed_depth_same_search_and_restore(self):
        bridge = InProcessBridge(self.binary)
        board = shogi.Board()
        before = board.sfen()
        original = engine.ordered_moves
        args = dict(max_depth=2, complete_depth=True, root_alpha_beta=True,
                    fast_move_list=True, fast_legal_guard=True)
        expected = engine.select_move(board, **args)
        with native_search(bridge):
            self.assertEqual(engine.select_move(board, **args), expected)
        with native_order_search(bridge):
            self.assertEqual(engine.select_move(board, **args), expected)
        self.assertIs(engine.ordered_moves, original)
        self.assertEqual(board.sfen(), before)

    def test_reject_invalid_ffi_and_request(self):
        bridge = InProcessBridge(self.binary)
        buf = __import__("ctypes").create_string_buffer(16)
        self.assertLess(bridge.function(b"invalid", buf, len(buf)), 0)
        self.assertLess(bridge.function(b"", buf, 0), 0)
        request = expected_request()
        self.assertTrue(validate_request(request))
        self.assertFalse(validate_request({**request, "enabled": False}))
        with self.assertRaises(ValueError):
            validate_request({**request, "samples": 16})


class AggregateTests(unittest.TestCase):
    def test_all_20_shards_required_and_exact_80_positions(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            for shard in range(SHARDS):
                records = []
                for sample in range(shard, SAMPLES, SHARDS):
                    for variant in VARIANTS:
                        records.append(dict(sample=sample, shard=shard,
                            variant=variant, fixed_move="7g7f", fixed_nodes=100,
                            nodes=120, depth=2, elapsed_ms=201.0, calls=0,
                            native_ms=0, sfen_ms=0, ipc_ms=0))
                (folder / f"part-{shard:02d}.json").write_text(
                    json.dumps(dict(shard=shard, positions_sha256="fixed",
                        games_sha256="g", model_sha256="m", seed=20271027,
                        records=records)))
            result = aggregate(folder, folder / "out")
            self.assertEqual(len(result["records"]), SAMPLES * len(VARIANTS))
            self.assertEqual(result["summary"]["native_inprocess"]["total_nodes"], 9600)
            (folder / "part-19.json").unlink()
            with self.assertRaises(ValueError):
                aggregate(folder, folder / "out")


if __name__ == "__main__":
    unittest.main()
