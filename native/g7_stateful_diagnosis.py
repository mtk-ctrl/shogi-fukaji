"""Research-only 80-position, 20-runner G7 stateful synchronization benchmark.

Three modes share frozen G7 and identical search options. Stateful C++ mirrors
Python push/pop and generates ordered legal moves without per-node SFEN. The
production USI, models, training, and KUMOJI repository are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

import shogi
from fukaji import engine
from fukaji.g7_depth_diagnosis import G7_SHA, GAMES_SHA, MODEL_RUN, SOURCE_RUN, digest
from fukaji.relative_value import RelativeValue
from native.g7_bridge_diagnosis import search
from native.g7_inprocess_diagnosis import (
    InProcessBridge, build_shared, frozen_positions, native_order_search,
)
from native.g7_stateful_bridge import (
    StatefulRules, SyncedBoard, build_state_shared, stateful_order_search,
)

SAMPLES = 80
SHARDS = 20
SEED = 20271027
VARIANTS = ("python_guard", "native_ordered", "native_stateful")


def expected_request():
    return dict(enabled=True, research_only=True, training_allowed=False,
                adoption_allowed=False, source_run=SOURCE_RUN, model_run=MODEL_RUN,
                games_sha256=GAMES_SHA, model_sha256=G7_SHA,
                samples=SAMPLES, shards=SHARDS, movetime_ms=200,
                max_depth=4, fixed_depth=2, mix_percent=25,
                fast_relative=True, root_alpha_beta=True,
                fast_move_list=True, fast_legal_guard=True,
                variants=list(VARIANTS), seed=SEED)


def validate_request(data):
    expected = expected_request()
    if data not in (expected, {**expected, "enabled": False}):
        raise ValueError("unapproved stateful G7 research request")
    return bool(data["enabled"])


def run_shard(request, games, model, out, shard):
    if shard not in range(SHARDS):
        raise ValueError("invalid shard")
    if not validate_request(json.loads(Path(request).read_text(encoding="utf-8"))):
        raise ValueError("disabled research request")
    if digest(games) != GAMES_SHA or digest(model) != G7_SHA:
        raise ValueError("frozen source/model SHA mismatch")
    rows = [json.loads(line) for line in Path(games).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    positions = frozen_positions(rows)
    positions_sha = hashlib.sha256(
        ("\n".join(p["sfen"] for p in positions) + "\n").encode()).hexdigest()
    frozen = RelativeValue(model, mix_percent=25)
    records = []
    with tempfile.TemporaryDirectory() as td:
        directory = Path(td)
        build_shared(directory / "rules.so")
        build_state_shared(directory / "state.so")
        direct = InProcessBridge(directory / "rules.so")
        for index in range(shard, SAMPLES, SHARDS):
            board = shogi.Board(positions[index]["sfen"])
            original = board.sfen()
            python_legal = [m.usi() for m in board.legal_moves]
            python_order = [m.usi() for m in engine.ordered_moves(
                board, fast_move_list=True, fast_legal_guard=True)]
            if direct.legal_pseudo_order(board) != python_legal:
                raise RuntimeError(f"C++ ordered legal mismatch {index}")
            baseline = search(board, frozen, fixed=True)
            with native_order_search(direct):
                if search(board, frozen, fixed=True) != baseline:
                    raise RuntimeError(f"C++ ordered fixed-depth mismatch {index}")
            open_start = time.perf_counter_ns()
            state = StatefulRules(directory / "state.so", board)
            open_ms = (time.perf_counter_ns() - open_start) / 1e6
            proxy = SyncedBoard(board, state)
            try:
                if state.legal_pseudo_order(proxy) != python_legal:
                    raise RuntimeError(f"stateful legal mismatch {index}")
                with stateful_order_search(state):
                    if search(proxy, frozen, fixed=True) != baseline:
                        raise RuntimeError(f"stateful fixed-depth mismatch {index}")
                proxy.assert_restored(original)
                modes = list(VARIANTS)
                modes = modes[index % len(modes):] + modes[:index % len(modes)]
                for mode in modes:
                    native_calls = direct.calls
                    sfen_ns = direct.sfen_ns
                    native_ns = direct.native_ns
                    state_calls = state.calls
                    legal_ns = state.legal_ns
                    push_ns = state.push_ns
                    pop_ns = state.pop_ns
                    move_ns = state.move_ns
                    sort_ns = state.sort_ns
                    start = time.perf_counter()
                    if mode == "native_ordered":
                        with native_order_search(direct):
                            move, depth, nodes = search(board, frozen)
                    elif mode == "native_stateful":
                        with stateful_order_search(state):
                            move, depth, nodes = search(proxy, frozen)
                    else:
                        move, depth, nodes = search(board, frozen)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    if (board.sfen() != original or move == "resign" or
                        shogi.Move.from_usi(move) not in board.legal_moves):
                        raise RuntimeError(f"illegal timed search at {index}")
                    if mode == "native_stateful":
                        proxy.assert_restored(original)
                    records.append(dict(
                        sample=index, shard=shard, variant=mode,
                        move=move, depth=depth, nodes=nodes,
                        elapsed_ms=round(elapsed_ms, 3),
                        fixed_move=baseline[0], fixed_nodes=baseline[2],
                        native_calls=direct.calls-native_calls if mode=="native_ordered" else 0,
                        sfen_ms=round((direct.sfen_ns-sfen_ns)/1e6,3) if mode=="native_ordered" else 0,
                        abi_ms=round((direct.native_ns-native_ns)/1e6,3) if mode=="native_ordered" else 0,
                        state_calls=state.calls-state_calls if mode=="native_stateful" else 0,
                        state_legal_ms=round((state.legal_ns-legal_ns)/1e6,3) if mode=="native_stateful" else 0,
                        state_push_ms=round((state.push_ns-push_ns)/1e6,3) if mode=="native_stateful" else 0,
                        state_pop_ms=round((state.pop_ns-pop_ns)/1e6,3) if mode=="native_stateful" else 0,
                        move_construct_ms=round((state.move_ns-move_ns)/1e6,3) if mode=="native_stateful" else 0,
                        order_sort_ms=round((state.sort_ns-sort_ns)/1e6,3) if mode=="native_stateful" else 0,
                        state_open_ms=round(open_ms,3) if mode=="native_stateful" else 0,
                    ))
            finally:
                state.close()
    result = dict(shard=shard, shards=SHARDS, samples=SAMPLES, seed=SEED,
                  positions_sha256=positions_sha, games_sha256=GAMES_SHA,
                  model_sha256=G7_SHA, records=records,
                  caution="Stateful ABI uses full-state copy per push/pop; "
                          "search throughput only, not strength or 10x engine speed.")
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    (target / f"part-{shard:02d}.json").write_text(
        json.dumps(result, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    return result


def aggregate(parts, out):
    paths = sorted(Path(parts).rglob("part-*.json"))
    if len(paths) != SHARDS:
        raise ValueError(f"expected 20 shards, got {len(paths)}")
    items = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    if sorted(item["shard"] for item in items) != list(range(SHARDS)):
        raise ValueError("missing or duplicated shard")
    keys = ("positions_sha256", "games_sha256", "model_sha256", "seed")
    if any(any(item[k] != items[0][k] for k in keys) for item in items):
        raise ValueError("mixed frozen input")
    records = [record for item in items for record in item["records"]]
    expected = {(index, mode) for index in range(SAMPLES) for mode in VARIANTS}
    actual = {(record["sample"], record["variant"]) for record in records}
    if len(records) != len(expected) or actual != expected:
        raise ValueError("missing/duplicate measurement rows")
    if any(record["shard"] != record["sample"] % SHARDS for record in records):
        raise ValueError("incorrect 20-runner allocation")
    for index in range(SAMPLES):
        group = [record for record in records if record["sample"] == index]
        if len({(r["fixed_move"],r["fixed_nodes"]) for r in group}) != 1:
            raise ValueError(f"fixed-depth mismatch at {index}")
    summary = {}
    for mode in VARIANTS:
        group = [r for r in records if r["variant"] == mode]
        summary[mode] = dict(
            total_nodes=sum(r["nodes"] for r in group),
            median_nodes=statistics.median(r["nodes"] for r in group),
            median_elapsed_ms=statistics.median(r["elapsed_ms"] for r in group),
            depth_distribution=dict(Counter(str(r["depth"]) for r in group)),
            totals_ms={key:round(sum(r[key] for r in group),3) for key in
                       ("sfen_ms","abi_ms","state_legal_ms","state_push_ms",
                        "state_pop_ms","move_construct_ms","order_sort_ms","state_open_ms")})
    result = dict(status="research_only_no_training_no_adoption",
                  samples=SAMPLES, shards=SHARDS, positions_sha256=items[0]["positions_sha256"],
                  games_sha256=GAMES_SHA, model_sha256=G7_SHA,
                  records=sorted(records,key=lambda r:(r["sample"],r["variant"])),
                  summary=summary,
                  caution="No playing-strength or 10x search claim; state_open measured separately.")
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    (target / "aggregate.json").write_text(
        json.dumps(result, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request")
    parser.add_argument("--games")
    parser.add_argument("--model")
    parser.add_argument("--shard", type=int)
    parser.add_argument("--parts")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.parts:
        aggregate(args.parts,args.out)
    else:
        if args.shard is None or not args.request or not args.games or not args.model:
            parser.error("shard mode requires request/games/model/shard")
        run_shard(args.request,args.games,args.model,args.out,args.shard)


if __name__ == "__main__":
    main()
