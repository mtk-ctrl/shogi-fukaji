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
from fukaji.relative_value import RelativeValue
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
            "scored": ([ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "scored_packed": ([ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t], ctypes.c_int),
            "sfen": ([ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "verdict": ([ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "is_check": ([ctypes.c_uint64], ctypes.c_int),
            "push": ([ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t], ctypes.c_int),
            "pop": ([ctypes.c_uint64], ctypes.c_int),
            "profile": ([ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
            "eval_snapshot": ([ctypes.c_uint64, ctypes.POINTER(ctypes.c_int16), ctypes.c_size_t], ctypes.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, "fukaji_state_" + name)
            fn.argtypes = args
            fn.restype = result
        self.handle = ctypes.c_uint64(0)
        self.audit = audit
        self.calls = 0
        self.legal_ns = 0
        self.scored_ns = 0
        self.packed_ns = 0
        self.packed_decode_ns = 0
        self.python_push_ns = 0
        self.python_pop_ns = 0
        self.eval_snapshot_ns = 0
        self.eval_arithmetic_ns = 0
        self.eval_snapshot_calls = 0
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

    def scored_order(self, board):
        """Native stable capture/promotion scoring, legal-cache authority unchanged."""
        start = time.perf_counter_ns()
        words = self._text("scored")
        self.scored_ns += time.perf_counter_ns() - start
        self.calls += 1
        result = words.split(",") if words else []
        if len(result) != len(set(result)):
            raise RuntimeError("duplicate C++ scored legal move")
        if self.audit:
            self.assert_equal(board, full=True)
        return result

    def scored_packed_order(self, board):
        """Bounded compact move transport, still generated from frame-validated legal moves."""
        output = (ctypes.c_uint32 * 1024)()
        start = time.perf_counter_ns()
        count = self.lib.fukaji_state_scored_packed(self.handle.value, output, len(output))
        self.packed_ns += time.perf_counter_ns() - start
        self.calls += 1
        if count < 0 or count > len(output):
            raise RuntimeError(f"native packed scored moves rejected: {count}")
        result = list(output[:count])
        if len(result) != len(set(result)):
            raise RuntimeError("duplicate native packed legal move")
        if self.audit:
            self.assert_equal(board, full=True)
        return result

    def eval_snapshot(self, board):
        """Read a bounded C++ snapshot; never treat it as a legal-move token."""
        output = (ctypes.c_int16 * 96)()
        started = time.perf_counter_ns()
        count = self.lib.fukaji_state_eval_snapshot(self.handle.value, output, len(output))
        self.eval_snapshot_ns += time.perf_counter_ns() - started
        self.eval_snapshot_calls += 1
        if count != 96:
            raise RuntimeError(f"native evaluation snapshot rejected: {count}")
        values = tuple(output)
        if values[95] != int(board.turn):
            raise RuntimeError("native evaluation snapshot turn mismatch")
        if self.audit:
            for sq in range(81):
                piece = board.piece_at(sq)
                expected = (piece.piece_type * (1 if piece.color == shogi.BLACK else -1)
                            if piece else 0)
                if values[sq] != expected:
                    raise RuntimeError("native evaluation snapshot square mismatch")
            for color in (shogi.BLACK, shogi.WHITE):
                for kind in range(1, 8):
                    if values[81 + color * 7 + kind - 1] != board.pieces_in_hand[color].get(kind, 0):
                        raise RuntimeError("native evaluation snapshot hand mismatch")
        return values

    def evaluate_relative_snapshot(self, board, model):
        """Exactly reproduce Python fused evaluation using native square data.

        Hand iteration stays on Python's board to preserve floating addition
        order. Python owns repetition and search; C++ only supplies squares.
        """
        if not isinstance(model, RelativeValue) or model.mix <= 0:
            raise RuntimeError("snapshot requires frozen relative evaluation")
        values = self.eval_snapshot(board)
        started = time.perf_counter_ns()
        turn = int(board.turn)
        classical = 0
        logit = model._turn_weight
        for sq in range(81):
            code = values[sq]
            if not code:
                continue
            kind = abs(code)
            if not 1 <= kind <= 14:
                raise RuntimeError("invalid snapshot piece type")
            color = shogi.BLACK if code > 0 else shogi.WHITE
            val = engine.VALUES[kind]
            if kind != shogi.KING:
                rank, file = sq % 9, sq // 9
                val += 2 * (4 - abs(file - 4)) + (4 - abs(rank - 4))
            classical += val if color == board.turn else -val
            owner = int(color) ^ turn
            relative_sq = sq if turn == shogi.BLACK else 80 - sq
            logit += model._piece_weights[(owner * 14 + kind - 1) * 81 + relative_sq]
        for color in (shogi.BLACK, shogi.WHITE):
            owner = int(color) ^ turn
            factor = 1 if color == board.turn else -1
            for kind, count in board.pieces_in_hand[color].items():
                classical += factor * int(count) * engine.VALUES[kind]
                logit += model._hand_prefix[owner, kind][min(20, int(count))]
        learned = int(round(max(-10.0, min(10.0, logit)) * model.weight_cp))
        result = int(round((1 - model.mix) * classical + model.mix * learned))
        self.eval_arithmetic_ns += time.perf_counter_ns() - started
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
            started = time.perf_counter_ns()
            try:
                self.board.push(move)
            finally:
                self.state.python_push_ns += time.perf_counter_ns() - started
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
            started = time.perf_counter_ns()
            try:
                move = self.board.pop()
            finally:
                self.state.python_pop_ns += time.perf_counter_ns() - started
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

def packed_to_move(code):
    """Decode only compact coordinates, never treat them as a legality certificate."""
    if not isinstance(code, int) or code < 0 or code >= (1 << 18):
        raise ValueError("invalid packed move")
    to_square = code & 127
    from_square = (code >> 7) & 127
    promotion = bool((code >> 14) & 1)
    drop = (code >> 15) & 7
    if to_square >= 81 or (from_square >= 81 and from_square != 127):
        raise ValueError("invalid packed square")
    if from_square == 127:
        if not 1 <= drop <= 7 or promotion:
            raise ValueError("invalid packed drop")
        return shogi.Move(None, to_square, False, drop)
    if drop:
        raise ValueError("invalid packed ordinary move")
    return shogi.Move(from_square, to_square, promotion, None)


@contextmanager
def stateful_order_search(state, *, cached_moves=False, native_scored=False,
                          native_packed=False, native_snapshot_eval=False):
    """Research-only bounded move parsing cache; ordering and legality unchanged."""
    if native_packed and not native_scored:
        raise ValueError("packed transport requires native scored legal order")
    original = engine.ordered_moves
    original_evaluate = engine.evaluate
    # A USI move's coordinates/promotion are position-independent. Cache only
    # the immutable parsed representation, never legal status or board state.
    parse_move = (lru_cache(maxsize=8192)(shogi.Move.from_usi)
                  if cached_moves else shogi.Move.from_usi)

    parse_packed = (lru_cache(maxsize=8192)(packed_to_move)
                    if cached_moves else packed_to_move)

    def ordered(board, fast_move_list=False, fast_legal_guard=False):
        if not (fast_move_list and fast_legal_guard):
            raise RuntimeError("stateful research requires frozen G7 options")
        if native_packed:
            codes = state.scored_packed_order(board)
            started = time.perf_counter_ns()
            moves = [parse_packed(code) for code in codes]
            elapsed = time.perf_counter_ns() - started
            state.packed_decode_ns += elapsed
            state.move_ns += elapsed
        else:
            words = (state.scored_order(board) if native_scored
                     else state.legal_pseudo_order(board))
            started = time.perf_counter_ns()
            moves = [parse_move(word) for word in words]
            state.move_ns += time.perf_counter_ns() - started
        if not native_scored:
            started = time.perf_counter_ns()
            def score(move):
                target = board.piece_at(move.to_square)
                return (engine.VALUES.get(target.piece_type, 0) if target else 0) + (
                    200 if move.promotion else 0)
            moves.sort(key=score, reverse=True)
            state.sort_ns += time.perf_counter_ns() - started
        return moves

    engine.ordered_moves = ordered
    if native_snapshot_eval:
        def evaluate_with_snapshot(board, model=None, fast_relative=False):
            if not fast_relative or not isinstance(model, RelativeValue) or model.mix <= 0:
                return original_evaluate(board, model, fast_relative)
            if not isinstance(board, SyncedBoard) or board.state is not state:
                raise RuntimeError("snapshot evaluation requires matching synced board")
            return state.evaluate_relative_snapshot(board, model)
        engine.evaluate = evaluate_with_snapshot
    try:
        yield
    finally:
        engine.ordered_moves = original
        engine.evaluate = original_evaluate
        if cached_moves:
            info = (parse_packed if native_packed else parse_move).cache_info()
            state.parse_cache_hits += info.hits
            state.parse_cache_misses += info.misses
