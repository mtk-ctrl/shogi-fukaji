"""Independent equivalence tests for the bounded stateful C++ research ABI.

No production Engine patch, training, model adoption or match dispatch.
"""
from __future__ import annotations

import ctypes
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import shogi
from fukaji import engine
from native.g7_stateful_bridge import packed_to_move

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("g++"), "g++ required")
class NativeStateFFITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        binary = Path(cls.tmp.name) / "libfukaji_state.so"
        build = subprocess.run(
            ["g++", "-std=c++17", "-O2", "-fPIC", "-shared",
             "-Wall", "-Wextra", "-Werror",
             str(ROOT / "native/rules_state_ffi.cpp"), "-o", str(binary)],
            capture_output=True, text=True, timeout=90,
        )
        if build.returncode:
            raise AssertionError(build.stdout + build.stderr)
        cls.lib = ctypes.CDLL(str(binary))
        specs = {
            "open": ([ctypes.c_char_p, ctypes.c_size_t,
                      ctypes.POINTER(ctypes.c_uint64)], ctypes.c_int),
            "close": ([ctypes.c_uint64], ctypes.c_int),
            "legal": ([ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "scored": ([ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "scored_packed": ([ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t], ctypes.c_int),
            "sfen": ([ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "verdict": ([ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "is_check": ([ctypes.c_uint64], ctypes.c_int),
            "push": ([ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t], ctypes.c_int),
            "pop": ([ctypes.c_uint64], ctypes.c_int),
            "profile": ([ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
        }
        for name, (args, result) in specs.items():
            fn = getattr(cls.lib, "fukaji_state_" + name)
            fn.argtypes = args
            fn.restype = result

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def open_board(self, board):
        data = board.sfen().encode("ascii")
        handle = ctypes.c_uint64()
        status = self.lib.fukaji_state_open(data, len(data), ctypes.byref(handle))
        self.assertEqual(status, 0)
        self.assertGreater(handle.value, 0)
        return handle.value

    def read(self, handle, name, capacity=16384):
        buf = ctypes.create_string_buffer(capacity)
        size = getattr(self.lib, "fukaji_state_" + name)(handle, buf, len(buf))
        self.assertGreaterEqual(size, 0, (name, size))
        return buf.raw[:size].decode("ascii")

    def read_packed(self, handle, capacity=1024):
        output = (ctypes.c_uint32 * capacity)()
        count = self.lib.fukaji_state_scored_packed(handle, output, capacity)
        self.assertGreaterEqual(count, 0, count)
        self.assertLessEqual(count, capacity)
        return [packed_to_move(int(code)).usi() for code in output[:count]]

    def check(self, handle, board):
        self.assertEqual(self.read(handle, "sfen"), board.sfen())
        self.assertEqual(self.lib.fukaji_state_is_check(handle), int(board.is_check()))
        self.assertEqual(
            self.read(handle, "legal").split(",") if list(board.legal_moves) else [],
            [move.usi() for move in board.legal_moves],
            board.sfen(),
        )
        expected_scored = [move.usi() for move in engine.ordered_moves(
            board, fast_move_list=True, fast_legal_guard=True)]
        self.assertEqual(
            self.read(handle, "scored").split(",") if expected_scored else [],
            expected_scored, board.sfen())
        self.assertEqual(self.read_packed(handle), expected_scored, board.sfen())

    def profile(self, handle):
        fields = self.read(handle, "profile").split(";")
        return {key: int(value) for key, value in
                (field.split("=", 1) for field in fields)}

    def test_validated_cache_parent_sibling_and_stale_child(self):
        board = shogi.Board()
        handle = self.open_board(board)
        try:
            # No preceding legal query: full validation is mandatory.
            first = next(iter(board.legal_moves))
            word = first.usi().encode("ascii")
            self.assertEqual(self.lib.fukaji_state_push(handle, word, len(word)), 0)
            board.push(first)
            self.assertEqual(self.profile(handle)["fallback_pushes"], 1)
            self.assertEqual(self.lib.fukaji_state_pop(handle), 0)
            board.pop()
            self.check(handle, board)

            before = self.profile(handle)["legal_generations"]
            self.read(handle, "legal")
            self.assertEqual(self.profile(handle)["legal_generations"], before)
            root_moves = list(board.legal_moves)
            self.assertGreaterEqual(len(root_moves), 2)

            # Invalid move is rejected even when the cache is populated.
            snapshot = self.read(handle, "sfen")
            counters = self.profile(handle)
            self.assertEqual(self.lib.fukaji_state_push(handle, b"9a9a", 4), -2)
            self.assertEqual(self.read(handle, "sfen"), snapshot)
            self.assertEqual(self.profile(handle), counters)

            for chosen in root_moves[:2]:
                encoded = chosen.usi().encode("ascii")
                self.assertEqual(self.lib.fukaji_state_push(
                    handle, encoded, len(encoded)), 0)
                board.push(chosen)
                self.check(handle, board)
                self.assertEqual(self.lib.fukaji_state_pop(handle), 0)
                board.pop()
                self.assertEqual(self.read(handle, "sfen"), snapshot)
            # Both siblings reuse the root cache across child push/pop.
            self.assertEqual(self.profile(handle)["cache_hits"], 2)

            # A child cache must never be reused for a different sibling.
            child = root_moves[0]
            encoded = child.usi().encode("ascii")
            self.assertEqual(self.lib.fukaji_state_push(handle, encoded, len(encoded)), 0)
            board.push(child)
            self.read(handle, "legal")
            self.assertEqual(self.lib.fukaji_state_pop(handle), 0)
            board.pop()
            other = root_moves[1]
            encoded = other.usi().encode("ascii")
            self.assertEqual(self.lib.fukaji_state_push(handle, encoded, len(encoded)), 0)
            board.push(other)
            child_move = next(iter(board.legal_moves))
            encoded = child_move.usi().encode("ascii")
            previous = self.profile(handle)["fallback_pushes"]
            self.assertEqual(self.lib.fukaji_state_push(handle, encoded, len(encoded)), 0)
            board.push(child_move)
            self.assertEqual(self.profile(handle)["fallback_pushes"], previous + 1)
            self.check(handle, board)
            for _ in range(2):
                self.assertEqual(self.lib.fukaji_state_pop(handle), 0)
                board.pop()
            self.check(handle, board)
        finally:
            self.assertEqual(self.lib.fukaji_state_close(handle), 0)

    def test_scored_buffer_failure_does_not_publish_cache(self):
        board = shogi.Board()
        handle = self.open_board(board)
        try:
            tiny = ctypes.create_string_buffer(2)
            self.assertEqual(self.lib.fukaji_state_scored(handle, tiny, len(tiny)), -2)
            self.assertEqual(self.profile(handle)["legal_generations"], 0)
            move = next(iter(board.legal_moves)).usi().encode("ascii")
            self.assertEqual(self.lib.fukaji_state_push(handle, move, len(move)), 0)
            self.assertEqual(self.profile(handle)["fallback_pushes"], 1)
        finally:
            self.assertEqual(self.lib.fukaji_state_close(handle), 0)

    def test_packed_capacity_failure_does_not_publish_cache(self):
        board = shogi.Board()
        handle = self.open_board(board)
        try:
            tiny = (ctypes.c_uint32 * 1)()
            self.assertEqual(self.lib.fukaji_state_scored_packed(handle, tiny, 1), -2)
            self.assertEqual(self.profile(handle)["legal_generations"], 0)
            self.assertEqual(self.read(handle, "sfen"), board.sfen())
            move = next(iter(board.legal_moves)).usi().encode("ascii")
            self.assertEqual(self.lib.fukaji_state_push(handle, move, len(move)), 0)
            self.assertEqual(self.profile(handle)["fallback_pushes"], 1)
        finally:
            self.assertEqual(self.lib.fukaji_state_close(handle), 0)

    def test_failed_legal_buffer_does_not_publish_cache(self):
        board = shogi.Board()
        handle = self.open_board(board)
        try:
            tiny = ctypes.create_string_buffer(2)
            self.assertEqual(self.lib.fukaji_state_legal(handle, tiny, len(tiny)), -2)
            self.assertEqual(self.profile(handle)["legal_generations"], 0)
            move = next(iter(board.legal_moves)).usi().encode("ascii")
            self.assertEqual(self.lib.fukaji_state_push(handle, move, len(move)), 0)
            self.assertEqual(self.profile(handle)["fallback_pushes"], 1)
        finally:
            self.assertEqual(self.lib.fukaji_state_close(handle), 0)

    def test_special_positions_and_pawn_drop_mate(self):
        sfens = (
            shogi.Board().sfen(),
            "4r3k/9/9/9/9/9/9/4G4/4K4 b - 1",
            "3pkp3/3p1p3/4G4/9/9/9/9/9/4K4 b P 1",
            "4k4/9/9/9/9/9/9/9/4K4 b RBGSNLP 1",
            "4k4/9/9/9/9/9/9/9/4K4 w rbgsnlp 1",
        )
        for sfen in sfens:
            with self.subTest(sfen=sfen):
                board = shogi.Board(sfen)
                handle = self.open_board(board)
                try:
                    self.check(handle, board)
                    if "3pkp3" in sfen:
                        self.assertNotIn("P*5b", self.read(handle, "legal").split(","))
                finally:
                    self.assertEqual(self.lib.fukaji_state_close(handle), 0)

    def test_random_reachable_push_pop_and_order(self):
        rng = random.Random(20271027)
        for series in range(6):
            board = shogi.Board()
            handle = self.open_board(board)
            try:
                self.check(handle, board)
                count = 0
                for _ in range(50):
                    moves = list(board.legal_moves)
                    if not moves:
                        break
                    move = rng.choice(moves)
                    word = move.usi().encode("ascii")
                    self.assertEqual(self.lib.fukaji_state_push(handle, word, len(word)),
                                     0, (series, board.sfen(), move.usi()))
                    board.push(move)
                    self.check(handle, board)
                    count += 1
                for _ in range(count):
                    self.assertEqual(self.lib.fukaji_state_pop(handle), 0)
                    board.pop()
                    self.check(handle, board)
                self.assertEqual(board.sfen(), shogi.Board().sfen())
                self.assertEqual(self.read(handle, "verdict"), "none")
            finally:
                self.assertEqual(self.lib.fukaji_state_close(handle), 0)

    def test_continuous_check_and_ordinary_repetition(self):
        cases = (
            ("4k4/9/5R3/9/9/9/9/9/4K4 b - 1",
             ("4c5c", "5a4a", "5c4c", "4a5a"), "black_loses"),
            ("4k4/9/9/9/9/9/5r3/9/4K4 w - 1",
             ("4g5g", "5i4i", "5g4g", "4i5i"), "white_loses"),
            ("4k4/9/9/9/9/9/9/9/4K4 b - 1",
             ("5i5h", "5a5b", "5h5i", "5b5a"), "draw"),
        )
        for sfen, cycle, verdict in cases:
            with self.subTest(verdict=verdict):
                board = shogi.Board(sfen)
                handle = self.open_board(board)
                try:
                    for repeat in range(3):
                        for usi in cycle:
                            move = shogi.Move.from_usi(usi)
                            self.assertIn(move, board.legal_moves)
                            self.assertEqual(self.lib.fukaji_state_push(
                                handle, usi.encode("ascii"), len(usi)), 0)
                            board.push(move)
                            self.check(handle, board)
                        self.assertEqual(self.read(handle, "verdict"),
                                         verdict if repeat == 2 else "none")
                    self.assertTrue(board.is_fourfold_repetition())
                    self.assertEqual(self.lib.fukaji_state_pop(handle), 0)
                    board.pop()
                    self.check(handle, board)
                    self.assertEqual(self.read(handle, "verdict"), "none")
                finally:
                    self.assertEqual(self.lib.fukaji_state_close(handle), 0)

    def test_rejected_inputs_and_transactional_state(self):
        board = shogi.Board()
        handle = self.open_board(board)
        try:
            self.assertLess(self.lib.fukaji_state_push(handle, b"9a9a", 4), 0)
            self.assertLess(self.lib.fukaji_state_push(handle, b"x\n", 2), 0)
            self.assertLess(self.lib.fukaji_state_pop(handle), 0)
            self.assertLess(self.lib.fukaji_state_legal(
                handle, ctypes.create_string_buffer(2), 2), 0)
            self.assertLess(self.lib.fukaji_state_is_check(99999999), 0)
            self.check(handle, board)
        finally:
            self.assertEqual(self.lib.fukaji_state_close(handle), 0)
        self.assertLess(self.lib.fukaji_state_is_check(handle), 0)
        invalid = ctypes.c_uint64(77)
        self.assertLess(self.lib.fukaji_state_open(
            b"invalid", 7, ctypes.byref(invalid)), 0)
        self.assertEqual(invalid.value, 0)

    def test_handle_capacity_and_no_reuse(self):
        board = shogi.Board()
        handles = [self.open_board(board) for _ in range(16)]
        try:
            data = board.sfen().encode("ascii")
            rejected = ctypes.c_uint64(0)
            self.assertLess(self.lib.fukaji_state_open(
                data, len(data), ctypes.byref(rejected)), 0)
            self.assertEqual(rejected.value, 0)
        finally:
            for handle in handles:
                self.assertEqual(self.lib.fukaji_state_close(handle), 0)
        new_handle = self.open_board(board)
        self.assertNotIn(new_handle, handles)
        self.assertEqual(self.lib.fukaji_state_close(new_handle), 0)

    def test_512_move_boundary_and_failed_push_rollback(self):
        # Reversible king cycle exercises the full bounded history without
        # declaring fourfold repetition a win or silently truncating it.
        board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        initial = board.sfen()
        handle = self.open_board(board)
        cycle = ("5i5h", "5a5b", "5h5i", "5b5a")
        try:
            for step in range(512):
                usi = cycle[step % len(cycle)]
                move = shogi.Move.from_usi(usi)
                self.assertIn(move, board.legal_moves)
                encoded = usi.encode("ascii")
                self.assertEqual(self.lib.fukaji_state_push(
                    handle, encoded, len(encoded)), 0)
                board.push(move)
            self.check(handle, board)
            snapshot = self.read(handle, "sfen")
            verdict = self.read(handle, "verdict")
            self.assertEqual(verdict, "draw")
            self.assertEqual(self.lib.fukaji_state_push(handle, b"5i5h", 4), -2)
            self.assertEqual(self.lib.fukaji_state_push(handle, b"9a9a", 4), -2)
            self.assertEqual(self.read(handle, "sfen"), snapshot)
            self.assertEqual(self.read(handle, "verdict"), verdict)
            self.assertEqual(self.lib.fukaji_state_pop(handle), 0)
            board.pop()
            self.check(handle, board)
            encoded = cycle[511 % len(cycle)].encode("ascii")
            self.assertEqual(self.lib.fukaji_state_push(
                handle, encoded, len(encoded)), 0)
            board.push(shogi.Move.from_usi(cycle[511 % len(cycle)]))
            for _ in range(512):
                self.assertEqual(self.lib.fukaji_state_pop(handle), 0)
                board.pop()
            self.check(handle, board)
            self.assertEqual(board.sfen(), initial)
            self.assertEqual(self.read(handle, "verdict"), "none")
        finally:
            self.assertEqual(self.lib.fukaji_state_close(handle), 0)


if __name__ == "__main__":
    unittest.main()
