"""Research-only persistent C++ legal-move bridge for frozen G7 search.

Not a production engine option. The native H protocol returns sorted USI moves;
reconstruct python-shogi's pseudo-legal iteration order to preserve stable
capture/promotion ordering. Python still owns board, evaluation and repetition.
The bridge intentionally measures all SFEN conversion, IPC and ordering costs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import shogi

from fukaji import engine
from fukaji.g7_depth_diagnosis import (
    G7_SHA, GAMES_SHA, MODEL_RUN, SOURCE_RUN, digest, sample_positions,
)
from fukaji.relative_value import RelativeValue

ROOT = Path(__file__).resolve().parents[1]
SEED = 20271027
SAMPLES = 16


def expected_request():
    return {
        "enabled": True, "research_only": True, "training_allowed": False,
        "adoption_allowed": False, "source_run": SOURCE_RUN,
        "model_run": MODEL_RUN, "games_sha256": GAMES_SHA,
        "model_sha256": G7_SHA, "samples": SAMPLES,
        "movetime_ms": 200, "max_depth": 4,
        "fixed_depth": 2, "mix_percent": 25,
        "fast_relative": True, "root_alpha_beta": True,
        "fast_move_list": True, "fast_legal_guard": True,
        "variants": ["python_guard", "native_bridge"],
    }


def validate_request(data):
    expected = expected_request()
    if data not in (expected, {**expected, "enabled": False}):
        raise ValueError("unapproved native G7 bridge request")
    return bool(data["enabled"])


class NativeBridge:
    """One native subprocess per research run; fail closed on any mismatch."""

    def __init__(self, binary):
        self.proc = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self.calls = 0
        self.io_ns = 0

    def legal(self, board):
        before = board.sfen()
        start = time.perf_counter_ns()
        self.proc.stdin.write("H " + before + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        self.io_ns += time.perf_counter_ns() - start
        self.calls += 1
        if not line:
            raise RuntimeError("native bridge terminated without response")
        words = line.rstrip("\n").split(",") if line.strip() else []
        if words != sorted(set(words)):
            raise ValueError("native response not a sorted unique USI list")
        if board.sfen() != before:
            raise ValueError("native bridge mutated Python board")
        return words

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        if self.proc.returncode:
            raise RuntimeError("native bridge exit: " + self.proc.stderr.read()[-1000:])


def ordered_native(board, bridge):
    words = bridge.legal(board)
    allowed = set(words)
    # Pseudo-legal iteration order is necessary because Python's stable sort
    # keeps original order among moves with the same capture/promotion score.
    moves = [move for move in board.generate_pseudo_legal_moves()
             if move.usi() in allowed]
    if len(moves) != len(words) or {m.usi() for m in moves} != allowed:
        raise ValueError("native/Python pseudo-legal set mismatch")
    def order(move):
        piece = board.piece_at(move.to_square)
        return (engine.VALUES.get(piece.piece_type, 0) if piece else 0) + (
            200 if move.promotion else 0
        )
    moves.sort(key=order, reverse=True)
    return moves


@contextmanager
def native_search(bridge):
    original = engine.ordered_moves
    def wrapper(board, fast_move_list=False, fast_legal_guard=False):
        if not (fast_move_list and fast_legal_guard):
            raise ValueError("bridge requires both frozen G7 move options")
        return ordered_native(board, bridge)
    engine.ordered_moves = wrapper
    try:
        yield
    finally:
        engine.ordered_moves = original


def build_native(destination):
    if not shutil.which("g++"):
        raise RuntimeError("g++ required")
    command = [
        "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
        str(ROOT / "native/rules_probe.cpp"), "-o", str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=90)


def search(board, model, *, fixed=False):
    return engine.select_move(
        board, movetime_ms=200, max_depth=2 if fixed else 4,
        complete_depth=fixed, model=model, fast_relative=True,
        root_alpha_beta=True, fast_move_list=True, fast_legal_guard=True,
    )


def execute(request, games, model, out):
    if not validate_request(json.loads(Path(request).read_text(encoding="utf-8"))):
        raise ValueError("disabled request cannot start an experiment")
    if digest(games) != GAMES_SHA or digest(model) != G7_SHA:
        raise ValueError("frozen source/model SHA mismatch")
    rows = [json.loads(x) for x in Path(games).read_text(encoding="utf-8").splitlines() if x]
    positions = sample_positions(rows)[:SAMPLES]
    if len(positions) != SAMPLES or len({p["sfen"] for p in positions}) != SAMPLES:
        raise ValueError("invalid frozen positions")
    frozen = RelativeValue(model, mix_percent=25)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "rules_probe"
        build_native(binary)
        bridge = NativeBridge(binary)
        try:
            for index, position in enumerate(positions):
                board = shogi.Board(position["sfen"])
                before = board.sfen()
                native = bridge.legal(board)
                original = [m.usi() for m in board.legal_moves]
                if sorted(original) != native:
                    raise ValueError(f"native legal set mismatch at {index}")
                baseline_order = [m.usi() for m in engine.ordered_moves(
                    board, fast_move_list=True, fast_legal_guard=True)]
                native_order = [m.usi() for m in ordered_native(board, bridge)]
                if baseline_order != native_order:
                    raise ValueError(f"native legal ordering mismatch at {index}")
                fixed_python = search(board, frozen, fixed=True)
                with native_search(bridge):
                    fixed_native = search(board, frozen, fixed=True)
                if fixed_python != fixed_native or board.sfen() != before:
                    raise ValueError(f"fixed depth move/nodes mismatch at {index}")
                modes = (["python_guard", "native_bridge"] if index % 2 == 0
                         else ["native_bridge", "python_guard"])
                for mode in modes:
                    start_calls, start_ns = bridge.calls, bridge.io_ns
                    started = time.perf_counter()
                    if mode == "native_bridge":
                        with native_search(bridge):
                            move, depth, nodes = search(board, frozen)
                    else:
                        move, depth, nodes = search(board, frozen)
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    if board.sfen() != before or move == "resign" or (
                        shogi.Move.from_usi(move) not in board.legal_moves
                    ):
                        raise ValueError(f"illegal or mutated timed search {index}")
                    results.append({
                        "sample": index, "variant": mode, "move": move,
                        "depth": depth, "nodes": nodes,
                        "elapsed_ms": round(elapsed_ms, 3),
                        "native_calls": bridge.calls - start_calls,
                        "native_ipc_ms": round((bridge.io_ns - start_ns) / 1e6, 3),
                        "fixed_move": fixed_python[0],
                        "fixed_nodes": fixed_python[2],
                    })
        finally:
            bridge.close()
    summary = {}
    for mode in ("python_guard", "native_bridge"):
        group = [r for r in results if r["variant"] == mode]
        summary[mode] = {
            "total_nodes": sum(r["nodes"] for r in group),
            "median_nodes": statistics.median(r["nodes"] for r in group),
            "median_elapsed_ms": statistics.median(r["elapsed_ms"] for r in group),
            "depth_distribution": dict(Counter(r["depth"] for r in group)),
            "total_native_calls": sum(r["native_calls"] for r in group),
            "total_native_ipc_ms": round(sum(r["native_ipc_ms"] for r in group), 3),
        }
    result = {
        "status": "research_only_native_bridge_not_strength",
        "source_run": SOURCE_RUN, "model_run": MODEL_RUN,
        "games_sha256": GAMES_SHA, "model_sha256": G7_SHA,
        "positions_sha256": hashlib.sha256(
            ("\n".join(p["sfen"] for p in positions) + "\n").encode()
        ).hexdigest(),
        "samples": SAMPLES, "fixed_depth_equivalent": SAMPLES,
        "legal_set_and_order_equivalent": SAMPLES,
        "movetime_ms": 200, "summary": summary, "records": results,
        "caution": "Native IPC+SFEN+Python pseudo-order costs included. "
                   "Native is NOT an integrated C++ search core. "
                   "No training, adoption, or playing-strength inference.",
    }
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    (target / "diagnosis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (target / "positions.json").write_text(
        json.dumps(positions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("request", "games", "model", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    execute(args.request, args.games, args.model, args.out)


if __name__ == "__main__":
    main()
