"""Reproducible research-only legal-move enumeration timing.

Native Q times only the C++ legal generator; Python times list(board.legal_moves).
Both exclude process startup, SFEN parsing, protocol I/O and compilation.
These are distinct language implementations, NOT a 200ms engine benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

import shogi

ROOT = Path(__file__).resolve().parents[1]
SEED = 20271031


def frozen_positions(limit: int):
    fixed = [
        shogi.Board(),
        shogi.Board("4r3k/9/9/9/9/9/9/4G4/4K4 b - 1"),
        shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b PGLNS 1"),
        shogi.Board("3pkp3/3p1p3/4G4/9/9/9/9/9/4K4 b P 1"),
        shogi.Board("4k4/9/4R4/9/9/9/9/9/4K4 w - 1"),
    ]
    rng = random.Random(SEED)
    for _ in range(10):
        board = shogi.Board()
        for ply in range(80):
            if ply % 8 == 0:
                fixed.append(shogi.Board(board.sfen()))
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
    unique, seen = [], set()
    for board in fixed:
        if board.sfen() not in seen:
            unique.append(board)
            seen.add(board.sfen())
        if len(unique) == limit:
            break
    if len(unique) < limit:
        raise ValueError("insufficient distinct frozen positions")
    return unique


def native_run(binary: str, lines: list[str]):
    process = subprocess.run([binary], input="\n".join(lines) + "\n",
                             capture_output=True, text=True, timeout=180)
    if process.returncode:
        raise RuntimeError("native probe failed: " + process.stderr[-2000:])
    output = process.stdout.splitlines()
    if len(output) != len(lines):
        raise RuntimeError("native response count mismatch")
    return output


def benchmark(positions: int, repeats: int, rounds: int):
    if not (1 <= positions <= 128 and 1 <= repeats <= 5000 and 1 <= rounds <= 10):
        raise ValueError("benchmark parameters out of bounds")
    if not shutil.which("g++"):
        raise RuntimeError("g++ required")
    boards = frozen_positions(positions)
    sfens = [board.sfen() for board in boards]
    sha = hashlib.sha256(("\n".join(sfens) + "\n").encode()).hexdigest()
    with tempfile.TemporaryDirectory() as tmp:
        binary = str(Path(tmp) / "rules_probe")
        flags = ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror"]
        build = subprocess.run(flags + [str(ROOT / "native/rules_probe.cpp"),
                                       "-o", binary], capture_output=True,
                               text=True, timeout=60)
        if build.returncode:
            raise RuntimeError(build.stderr)
        expected = [sorted(m.usi() for m in b.legal_moves) for b in boards]
        actual = native_run(binary, ["H " + s for s in sfens])
        for i, (line, moves) in enumerate(zip(actual, expected)):
            if (line.split(",") if line else []) != moves:
                raise AssertionError("legal move mismatch at position " + str(i))
        # Warm both generators outside the timed loops.
        native_run(binary, ["Q 1 " + s for s in sfens])
        for board in boards:
            list(board.legal_moves)
        trials = []
        for trial in range(rounds):
            native = native_run(binary, [f"Q {repeats} {s}" for s in sfens])
            native_ns = []
            for index, line in enumerate(native):
                count, elapsed = map(int, line.split())
                if count != len(expected[index]) * repeats or elapsed <= 0:
                    raise AssertionError("native count/time mismatch")
                native_ns.append(elapsed)
            python_ns = []
            for index, board in enumerate(boards):
                count = 0
                start = time.perf_counter_ns()
                for _ in range(repeats):
                    count += len(list(board.legal_moves))
                elapsed = time.perf_counter_ns() - start
                if count != len(expected[index]) * repeats or elapsed <= 0:
                    raise AssertionError("python count/time mismatch")
                python_ns.append(elapsed)
            trials.append({
                "trial": trial + 1,
                "native_ns": sum(native_ns),
                "python_ns": sum(python_ns),
                "ratio_python_over_native": sum(python_ns) / sum(native_ns),
                "native_per_position_ns": native_ns,
                "python_per_position_ns": python_ns,
            })
    return {
        "scope": "legal move enumeration only, not engine/search speed or playing strength",
        "native_scope": "pseudo_moves(board,true) including string encoding and sorting; excludes SFEN parse and I/O",
        "python_scope": "list(board.legal_moves); excludes SFEN parse and I/O",
        "compiler_flags": flags[1:],
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_shogi_version": "1.1.1 (requirements pin)",
        "seed": SEED,
        "positions": len(sfens),
        "sfens_sha256": sha,
        "repeats_per_position": repeats,
        "rounds": rounds,
        "legal_sets_equal": True,
        "legal_move_counts": [len(x) for x in expected],
        "median_ratio_python_over_native": median(t["ratio_python_over_native"] for t in trials),
        "trials": trials,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = benchmark(args.positions, args.repeats, args.rounds)
    data = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(data, encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "positions", "sfens_sha256", "repeats_per_position",
        "rounds", "legal_sets_equal", "median_ratio_python_over_native"
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
