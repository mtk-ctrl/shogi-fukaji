"""Research-only in-process C++ legal move diagnosis, 80 positions / 20 shards.

All variants share the frozen G7 model and 200ms limit. The Python board,
search, repetition, evaluation and move ordering remain unchanged.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import random
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
from native.g7_bridge_diagnosis import NativeBridge, native_search, ordered_native, search

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = 80
SHARDS = 20
SEED = 20271027
VARIANTS = ("python_guard", "native_ipc", "native_inprocess")


def expected_request():
    return dict(enabled=True, research_only=True, training_allowed=False,
                adoption_allowed=False, source_run=SOURCE_RUN, model_run=MODEL_RUN,
                games_sha256=GAMES_SHA, model_sha256=G7_SHA, samples=SAMPLES,
                shards=SHARDS, movetime_ms=200, max_depth=4, fixed_depth=2,
                mix_percent=25, fast_relative=True, root_alpha_beta=True,
                fast_move_list=True, fast_legal_guard=True,
                variants=list(VARIANTS), seed=SEED)


def validate_request(request):
    expected = expected_request()
    if request not in (expected, {**expected, "enabled": False}):
        raise ValueError("unapproved in-process diagnosis request")
    return bool(request["enabled"])


def build_shared(destination):
    if not shutil.which("g++"):
        raise RuntimeError("g++ required")
    result = subprocess.run([
        "g++", "-std=c++17", "-O2", "-fPIC", "-shared",
        "-Wall", "-Wextra", "-Werror",
        str(ROOT / "native/rules_ffi.cpp"), "-o", str(destination),
    ], capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError("native shared build failed: " + result.stderr[-3000:])


class InProcessBridge:
    def __init__(self, binary):
        self.library = ctypes.CDLL(str(binary))
        self.function = self.library.fukaji_legal_usi
        self.function.argtypes = (ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t)
        self.function.restype = ctypes.c_int
        self.calls = 0
        self.sfen_ns = 0
        self.native_ns = 0

    def legal(self, board):
        started = time.perf_counter_ns()
        before = board.sfen()
        self.sfen_ns += time.perf_counter_ns() - started
        if len(before) > 1024:
            raise ValueError("SFEN too large")
        buffer = ctypes.create_string_buffer(16384)
        started = time.perf_counter_ns()
        length = self.function(before.encode("ascii"), buffer, len(buffer))
        self.native_ns += time.perf_counter_ns() - started
        self.calls += 1
        if length < 0 or length >= len(buffer):
            raise ValueError("native ABI rejected SFEN or output capacity")
        words = buffer.raw[:length].decode("ascii").split(",") if length else []
        if words != sorted(set(words)):
            raise ValueError("native ABI response is not a sorted unique list")
        if board.sfen() != before:
            raise ValueError("native ABI mutated Python board")
        return words


def frozen_positions(rows):
    base = sample_positions(rows)
    seen = {p["sfen"] for p in base}
    extras = []
    for row in rows:
        history = row["opening"] + row["moves"]
        for ply in (18, 22, 26, 30):
            # Choose a matching side-to-move after the fixed opening.
            target = ply if row["g7_black"] else ply + 1
            if len(history) <= target:
                continue
            board = shogi.Board()
            for token in history[:target]:
                move = shogi.Move.from_usi(token)
                if move not in board.legal_moves:
                    raise ValueError("illegal source history")
                board.push(move)
            if board.is_checkmate():
                continue
            sfen = board.sfen()
            if sfen not in seen:
                seen.add(sfen)
                extras.append(dict(pair=row["pair"], g7_black=row["g7_black"],
                                   ply=target, sfen=sfen))
    random.Random(SEED).shuffle(extras)
    positions = base + extras[:SAMPLES - len(base)]
    if len(positions) != SAMPLES or len({p["sfen"] for p in positions}) != SAMPLES:
        raise ValueError("insufficient distinct frozen legal positions")
    return positions


def run_shard(request, games, model, out, shard):
    if not 0 <= shard < SHARDS:
        raise ValueError("invalid shard")
    if not validate_request(json.loads(Path(request).read_text(encoding="utf-8"))):
        raise ValueError("disabled request")
    if digest(games) != GAMES_SHA or digest(model) != G7_SHA:
        raise ValueError("frozen artifact SHA mismatch")
    rows = [json.loads(x) for x in Path(games).read_text(encoding="utf-8").splitlines() if x]
    positions = frozen_positions(rows)
    position_hash = hashlib.sha256(
        ("\n".join(p["sfen"] for p in positions) + "\n").encode()
    ).hexdigest()
    frozen = RelativeValue(model, mix_percent=25)
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        from native.g7_bridge_diagnosis import build_native
        build_native(tmp / "probe")
        build_shared(tmp / "libfukaji_rules.so")
        ipc = NativeBridge(tmp / "probe")
        direct = InProcessBridge(tmp / "libfukaji_rules.so")
        try:
            for index in range(shard, SAMPLES, SHARDS):
                board = shogi.Board(positions[index]["sfen"])
                before = board.sfen()
                original = sorted(m.usi() for m in board.legal_moves)
                baseline_order = [m.usi() for m in engine.ordered_moves(
                    board, fast_move_list=True, fast_legal_guard=True)]
                if ipc.legal(board) != original or direct.legal(board) != original:
                    raise ValueError(f"legal set mismatch at {index}")
                if ([m.usi() for m in ordered_native(board, ipc)] != baseline_order
                    or [m.usi() for m in ordered_native(board, direct)] != baseline_order):
                    raise ValueError(f"legal ordering mismatch at {index}")
                fixed = search(board, frozen, fixed=True)
                for bridge in (ipc, direct):
                    with native_search(bridge):
                        if search(board, frozen, fixed=True) != fixed:
                            raise ValueError(f"fixed-depth search mismatch at {index}")
                # Rotate order to reduce warm-up / frequency drift.
                modes = list(VARIANTS)
                modes = modes[index % 3:] + modes[:index % 3]
                for mode in modes:
                    bridge = {"native_ipc": ipc, "native_inprocess": direct}.get(mode)
                    calls = bridge.calls if bridge else 0
                    native_ns = direct.native_ns
                    sfen_ns = direct.sfen_ns
                    ipc_ns = ipc.io_ns
                    start = time.perf_counter()
                    if bridge:
                        with native_search(bridge):
                            move, depth, nodes = search(board, frozen)
                    else:
                        move, depth, nodes = search(board, frozen)
                    elapsed = (time.perf_counter() - start) * 1000
                    if (board.sfen() != before or move == "resign"
                        or shogi.Move.from_usi(move) not in board.legal_moves):
                        raise ValueError(f"illegal timed move at {index}")
                    records.append(dict(
                        sample=index, shard=shard, variant=mode, move=move,
                        depth=depth, nodes=nodes, elapsed_ms=round(elapsed, 3),
                        fixed_move=fixed[0], fixed_nodes=fixed[2],
                        calls=bridge.calls - calls if bridge else 0,
                        native_ms=round((direct.native_ns-native_ns)/1e6, 3)
                            if mode=="native_inprocess" else 0,
                        sfen_ms=round((direct.sfen_ns-sfen_ns)/1e6, 3)
                            if mode=="native_inprocess" else 0,
                        ipc_ms=round((ipc.io_ns-ipc_ns)/1e6, 3)
                            if mode=="native_ipc" else 0,
                    ))
        finally:
            ipc.close()
    result = dict(shard=shard, shards=SHARDS, samples=SAMPLES,
                  positions_sha256=position_hash, games_sha256=GAMES_SHA,
                  model_sha256=G7_SHA, seed=SEED, records=records,
                  caution="Exploratory search throughput, not playing strength; "
                          "in-process ABI still uses SFEN and Python pseudo-order.")
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    (target / f"part-{shard:02d}.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)+"\n",
        encoding="utf-8")
    return result


def aggregate(parts, out):
    paths = sorted(Path(parts).rglob("part-*.json"))
    if len(paths) != SHARDS:
        raise ValueError(f"expected {SHARDS} shard files, got {len(paths)}")
    items = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    if sorted(x["shard"] for x in items) != list(range(SHARDS)):
        raise ValueError("missing/duplicate shard")
    keys = ("positions_sha256", "games_sha256", "model_sha256", "seed")
    if any(any(x[k] != items[0][k] for k in keys) for x in items):
        raise ValueError("mixed frozen inputs")
    records = [r for item in items for r in item["records"]]
    expected = {(i, mode) for i in range(SAMPLES) for mode in VARIANTS}
    if len(records) != SAMPLES * len(VARIANTS) or {
        (r["sample"], r["variant"]) for r in records} != expected:
        raise ValueError("missing or duplicated measurements")
    if any(r["shard"] != r["sample"] % SHARDS for r in records):
        raise ValueError("incorrect shard allocation")
    by_sample = {}
    for r in records:
        by_sample.setdefault(r["sample"], []).append(r)
    if any(len({(r["fixed_move"],r["fixed_nodes"]) for r in group}) != 1
           for group in by_sample.values()):
        raise ValueError("fixed-depth mismatch")
    summary = {}
    for mode in VARIANTS:
        group = [r for r in records if r["variant"] == mode]
        summary[mode] = dict(total_nodes=sum(r["nodes"] for r in group),
            median_nodes=statistics.median(r["nodes"] for r in group),
            median_elapsed_ms=statistics.median(r["elapsed_ms"] for r in group),
            depth_distribution=dict(Counter(str(r["depth"]) for r in group)),
            total_calls=sum(r["calls"] for r in group),
            native_ms=round(sum(r["native_ms"] for r in group),3),
            sfen_ms=round(sum(r["sfen_ms"] for r in group),3),
            ipc_ms=round(sum(r["ipc_ms"] for r in group),3))
    result = dict(status="research_only_no_training_no_adoption",
                  samples=SAMPLES, shards=SHARDS, positions_sha256=items[0]["positions_sha256"],
                  games_sha256=GAMES_SHA, model_sha256=G7_SHA,
                  summary=summary, records=sorted(records, key=lambda r:(r["sample"],r["variant"])),
                  caution="No claim of stronger play or 10x full-search speed.")
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    (target / "aggregate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request")
    p.add_argument("--games")
    p.add_argument("--model")
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int)
    p.add_argument("--parts")
    args = p.parse_args()
    if args.parts:
        aggregate(args.parts, args.out)
    else:
        if args.shard is None or not args.request or not args.games or not args.model:
            p.error("shard mode requires request/games/model/shard")
        run_shard(args.request,args.games,args.model,args.out,args.shard)


if __name__ == "__main__":
    main()
