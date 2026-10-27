"""Research-only lockstep Python/C++ board for G7 diagnosis.

The production USI and Python engine remain untouched. The Python board owns
search/evaluation/repetition; a bounded C++ state mirrors every push/pop.
Any native failure aborts the experiment rather than falling back silently.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import shutil
import subprocess
import time
from contextlib import contextmanager
from fukaji import engine
from pathlib import Path

import shogi

ROOT = Path(__file__).resolve().parents[1]


def build_state_shared(destination):
    if not shutil.which("g++"):
        raise RuntimeError("g++ required")
    result = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-fPIC", "-shared",
         "-Wall", "-Wextra", "-Werror",
         str(ROOT / "native/rules_state_ffi.cpp"), "-o", str(destination)],
        capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError("stateful C++ build failed: " + result.stderr[-3000:])


class StatefulRules:
    def __init__(self, binary, board, *, audit=False):
        self.lib = ctypes.CDLL(str(binary))
        signatures = {
            "open": ([ctypes.c_char_p, ctypes.c_size_t,
                      ctypes.POINTER(ctypes.c_uint64)], ctypes.c_int),
            "close": ([ctypes.c_uint64], ctypes.c_int),
            "legal": ([ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "sfen": ([ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "verdict": ([ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "is_check": ([ctypes.c_uint64], ctypes.c_int),
            "push": ([ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t], ctypes.c_int),
            "pop": ([ctypes.c_uint64], ctypes.c_int),
            "profile": ([ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, "fukaji_state_" + name)
            fn.argtypes = args
            fn.restype = result
        self.handle = ctypes.c_uint64(0)
        self.audit = audit
        self.calls = 0
        self.legal_ns = 0
        self.push_ns = 0
        self.pop_ns = 0
        self.move_ns = 0
        self.sort_ns = 0
        self.parse_cache_hits = 0
        self.parse_cache_misses = 0
        encoded = board.sfen().encode("ascii")
        rc = self.lib.fukaji_state_open(encoded, len(encoded), ctypes.byref(self.handle))
        if rc or not self.handle.value:
            raise RuntimeError(f"stateful open rejected: {rc}")
        self.assert_equal(board, full=True)

    def _text(self, name):
        buf = ctypes.create_string_buffer(16384)
        result = getattr(self.lib, "fukaji_state_" + name)(
            self.handle.value, buf, len(buf))
        if result < 0 or result >= len(buf):
            raise RuntimeError(f"native state {name} rejected: {result}")
        return buf.raw[:result].decode("ascii")

    def native_profile(self):
        """Counters are diagnostic only; no effect on legality or move choice."""
        fields = self._text("profile").split(";")
        return {key: int(value) for key, value in
                (field.split("=", 1) for field in fields)}

    def legal_pseudo_order(self, board):
        start = time.perf_counter_ns()
        words = self._text("legal")
        self.legal_ns += time.perf_counter_ns() - start
        self.calls += 1
        result = words.split(",") if words else []
        if len(result) != len(set(result)):
            raise RuntimeError("duplicate C++ legal move")
        if self.audit:
            self.assert_equal(board, full=True)
        return result

    def push(self, word):
        encoded = word.encode("ascii")
        start = time.perf_counter_ns()
        rc = self.lib.fukaji_state_push(self.handle.value, encoded, len(encoded))
        self.push_ns += time.perf_counter_ns() - start
        if rc:
            raise RuntimeError(f"native state push rejected {word}: {rc}")

    def pop(self):
        start = time.perf_counter_ns()
        rc = self.lib.fukaji_state_pop(self.handle.value)
        self.pop_ns += time.perf_counter_ns() - start
        if rc:
            raise RuntimeError(f"native state pop rejected: {rc}")

    def assert_equal(self, board, *, full=False):
        if self._text("sfen") != board.sfen():
            raise RuntimeError("Python/C++ board SFEN desynchronized")
        if full:
            check = self.lib.fukaji_state_is_check(self.handle.value)
            if check != int(board.is_check()):
                raise RuntimeError("Python/C++ check status mismatch")
            expected = [m.usi() for m in board.legal_moves]
            legal = self._text("legal")
            actual = legal.split(",") if legal else []
            if actual != expected:
                raise RuntimeError("Python/C++ ordered legal moves mismatch")
            native_verdict = self._text("verdict")
            if (native_verdict != "none") != board.is_fourfold_repetition():
                raise RuntimeError("Python/C++ fourfold repetition mismatch")

    def close(self):
        if self.handle.value:
            rc = self.lib.fukaji_state_close(self.handle.value)
            self.handle.value = 0
            if rc:
                raise RuntimeError(f"stateful close failed: {rc}")


class SyncedBoard:
    """Delegates evaluation/repetition to Python, mirrors each search move to C++."""
    def __init__(self, board, state):
        self.board = board
        self.state = state
        self.stack = []

    def __getattr__(self, name):
        return getattr(self.board, name)

    def push(self, move):
        word = move.usi()
        self.state.push(word)
        try:
            self.board.push(move)
        except BaseException:
            self.state.pop()
            raise
        self.stack.append(word)
        if self.state.audit:
            self.state.assert_equal(self.board, full=True)

    def pop(self):
        if not self.stack:
            raise RuntimeError("research board pop below starting position")
        word = self.stack[-1]
        self.state.pop()
        try:
            move = self.board.pop()
        except BaseException:
            self.state.push(word)
            raise
        self.stack.pop()
        if self.state.audit:
            self.state.assert_equal(self.board, full=True)
        return move

    def assert_restored(self, original):
        if self.stack or self.board.sfen() != original:
            raise RuntimeError("search failed to restore Python board")
        self.state.assert_equal(self.board, full=True)

@contextmanager
def stateful_order_search(state, *, cached_moves=False):
    """Research-only bounded move parsing cache; ordering and legality unchanged."""
    original = engine.ordered_moves
    # A USI move's coordinates/promotion are position-independent. Cache only
    # the immutable parsed representation, never legal status or board state.
    parse_move = (lru_cache(maxsize=8192)(shogi.Move.from_usi)
                  if cached_moves else shogi.Move.from_usi)

    def ordered(board, fast_move_list=False, fast_legal_guard=False):
        if not (fast_move_list and fast_legal_guard):
            raise RuntimeError("stateful research requires frozen G7 options")
        words = state.legal_pseudo_order(board)
        started = time.perf_counter_ns()
        moves = [parse_move(word) for word in words]
        state.move_ns += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        def score(move):
            target = board.piece_at(move.to_square)
            return (engine.VALUES.get(target.piece_type, 0) if target else 0) + (
                200 if move.promotion else 0)
        moves.sort(key=score, reverse=True)
        state.sort_ns += time.perf_counter_ns() - started
        return moves

    engine.ordered_moves = ordered
    try:
        yield
    finally:
        engine.ordered_moves = original
        if cached_moves:
            info = parse_move.cache_info()
            state.parse_cache_hits += info.hits
            state.parse_cache_misses += info.misses
