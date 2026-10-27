"""Research-only G7 state synchronization and rollback regression."""
import random
import shutil
import tempfile
import unittest
from pathlib import Path

import shogi
from fukaji import engine
from native.g7_bridge_diagnosis import search
from native.g7_stateful_bridge import StatefulRules, SyncedBoard, build_state_shared, stateful_order_search


@unittest.skipUnless(shutil.which("g++"), "g++ required")
class StatefulG7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.temp.name) / "libstate.so"
        build_state_shared(cls.binary)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_random_and_special_positions_push_pop(self):
        sfens = [
            shogi.Board().sfen(),
            "4r3k/9/9/9/9/9/9/4G4/4K4 b - 1",
            "3pkp3/3p1p3/4G4/9/9/9/9/9/4K4 b P 1",
            "4k4/9/9/9/9/9/9/9/4K4 b RBGSNLP 1",
            "4k4/9/9/9/9/9/9/9/4K4 w rbgsnlp 1",
        ]
        for series in range(8):
            rng = random.Random(20271027 + series)
            board = shogi.Board(sfens[series % len(sfens)])
            original = board.sfen()
            state = StatefulRules(self.binary, board, audit=True)
            synced = SyncedBoard(board, state)
            try:
                for _ in range(14):
                    moves = list(board.legal_moves)
                    if not moves:
                        break
                    synced.push(rng.choice(moves))
                for _ in range(len(synced.stack)):
                    synced.pop()
                synced.assert_restored(original)
            finally:
                state.close()

    def test_fixed_depth_search_equivalence_and_cleanup(self):
        args = dict(max_depth=2, complete_depth=True, root_alpha_beta=True,
                    fast_move_list=True, fast_legal_guard=True)
        for seed in range(4):
            rng = random.Random(20271027 + seed)
            board = shogi.Board()
            for _ in range(5 + seed):
                board.push(rng.choice(list(board.legal_moves)))
            original = board.sfen()
            baseline = engine.select_move(board, **args)
            state = StatefulRules(self.binary, board, audit=False)
            synced = SyncedBoard(board, state)
            saved = engine.ordered_moves
            try:
                with stateful_order_search(state):
                    candidate = engine.select_move(synced, **args)
                self.assertEqual(candidate, baseline)
                self.assertIs(engine.ordered_moves, saved)
                synced.assert_restored(original)
            finally:
                state.close()
            self.assertEqual(board.sfen(), original)

    def test_cached_move_parsing_preserves_order_and_undo(self):
        board = shogi.Board()
        original = board.sfen()
        baseline = [m.usi() for m in engine.ordered_moves(
            board, fast_move_list=True, fast_legal_guard=True)]
        state = StatefulRules(self.binary, board, audit=True)
        synced = SyncedBoard(board, state)
        saved = engine.ordered_moves
        try:
            with stateful_order_search(state, cached_moves=True):
                for _ in range(3):
                    moves = engine.ordered_moves(
                        synced, fast_move_list=True, fast_legal_guard=True)
                    self.assertEqual([m.usi() for m in moves], baseline)
                    synced.push(moves[0])
                    synced.pop()
                self.assertIsNot(engine.ordered_moves, saved)
            self.assertIs(engine.ordered_moves, saved)
            self.assertGreater(state.parse_cache_hits, 0)
            self.assertGreater(state.parse_cache_misses, 0)
            synced.assert_restored(original)
        finally:
            state.close()

    def test_native_scored_stable_order_and_fixed_depth(self):
        board = shogi.Board("4k4/9/9/9/9/9/9/4R4/4K4 b P 1")
        original = board.sfen()
        baseline_order = [m.usi() for m in engine.ordered_moves(
            board, fast_move_list=True, fast_legal_guard=True)]
        state = StatefulRules(self.binary, board, audit=True)
        proxy = SyncedBoard(board, state)
        try:
            with stateful_order_search(state, cached_moves=True, native_scored=True):
                for _ in range(3):
                    moves = engine.ordered_moves(
                        proxy, fast_move_list=True, fast_legal_guard=True)
                    self.assertEqual([m.usi() for m in moves], baseline_order)
                    proxy.push(moves[0])
                    proxy.pop()
            proxy.assert_restored(original)
            self.assertGreater(state.scored_ns, 0)
        finally:
            state.close()

    def test_invalid_native_move_preserves_state(self):
        board = shogi.Board()
        state = StatefulRules(self.binary, board, audit=True)
        synced = SyncedBoard(board, state)
        try:
            before = board.sfen()
            with self.assertRaises(RuntimeError):
                synced.push(shogi.Move.from_usi("9a9a"))
            self.assertEqual(board.sfen(), before)
            self.assertEqual(synced.stack, [])
            state.assert_equal(board, full=True)
            with self.assertRaises(RuntimeError):
                synced.pop()
        finally:
            state.close()

    def test_timeout_restores_native_and_python(self):
        board = shogi.Board()
        original = board.sfen()
        state = StatefulRules(self.binary, board)
        synced = SyncedBoard(board, state)
        try:
            with stateful_order_search(state):
                move, depth, nodes = engine.select_move(
                    synced, movetime_ms=1, max_depth=4,
                    root_alpha_beta=True, fast_move_list=True,
                    fast_legal_guard=True)
            self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
            self.assertGreaterEqual(nodes, 0)
            synced.assert_restored(original)
        finally:
            state.close()


if __name__ == "__main__":
    unittest.main()
