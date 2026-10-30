"""D9 research-only one-runner G8 clock-frequency probe (no training or matches)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import time
from zipfile import ZipFile

import shogi

from .arena import opening
from .d8_depth_extension_arena import SOURCE_ID, SOURCE_NAME, SOURCE_RUN, SOURCE_ZIP_SHA, G8_SHA, board_state
from .engine import select_move
from .g8_artifact_selection import artifact_bytes
from .generation8_train import safe_read
from .relative_value import RelativeValue

REQUEST_PATH = ".github/fukaji-d9-clock-probe-request.json"
SEEDS = (202640900, 202640901, 202640902)
ORDER = (False, True, True, False)


def expected_request(enabled=False):
    return {
        "enabled": enabled, "research_only": True,
        "training_allowed": False, "model_adoption_allowed": False,
        "source_run": SOURCE_RUN, "source_artifact_id": SOURCE_ID,
        "source_zip_sha256": SOURCE_ZIP_SHA, "model_sha256": G8_SHA,
        "seeds": list(SEEDS), "order": list(ORDER),
        "movetime_ms": 200, "max_depth": 3, "mix_percent": 25,
        "fast_move_list": True, "fast_relative": True,
        "root_alpha_beta": True, "deadline_from_call": True,
        "max_single_call_ms": 1000,
    }


def validate_request(cfg):
    for enabled in (False, True):
        if json.dumps(cfg, sort_keys=True) == json.dumps(expected_request(enabled), sort_keys=True):
            return enabled
    raise ValueError("D9 clock probe differs from frozen exact request")


def load_model():
    raw = artifact_bytes(SOURCE_ID, SOURCE_NAME, SOURCE_RUN, SOURCE_ZIP_SHA, os.environ["GH_TOKEN"])
    with ZipFile(io.BytesIO(raw)) as z:
        model_bytes = safe_read(z, "mover-relative-generation8-research.npz")
    if hashlib.sha256(model_bytes).hexdigest() != G8_SHA:
        raise ValueError("G8 model SHA mismatch")
    return model_bytes


def run_probe(model, select=select_move):
    rows = []
    for seed in SEEDS:
        head = opening(seed, 6)
        board = shogi.Board()
        for word in head:
            move = shogi.Move.from_usi(word)
            if move not in board.legal_moves:
                raise ValueError("illegal opening")
            board.push(move)
        before = board_state(board)
        for enabled in ORDER:
            telemetry = {}
            began = time.perf_counter()
            word, depth, nodes = select(
                board, movetime_ms=200, max_depth=3, model=model,
                fast_move_list=True, fast_relative=True, root_alpha_beta=True,
                deadline_from_call=True, time_check_each_node=enabled, telemetry=telemetry)
            elapsed = (time.perf_counter() - began) * 1000
            if board_state(board) != before:
                raise ValueError("search did not restore board/history")
            if word == "resign" or shogi.Move.from_usi(word) not in board.legal_moves:
                raise ValueError("illegal selected move")
            if not math.isfinite(elapsed) or elapsed < 0 or elapsed > 1000:
                raise ValueError("probe exceeded fault ceiling")
            rows.append({"seed": seed, "opening": head, "each_node": enabled,
                         "move": word, "depth": depth, "nodes": nodes,
                         "elapsed_ms": elapsed,
                         "root_preparation_ms": telemetry.get("root_preparation_ms")})
    return rows


def summarize(rows):
    out = {}
    for enabled in (False, True):
        r = [x for x in rows if x["each_node"] == enabled]
        if len(r) != len(SEEDS) * 2:
            raise ValueError("missing ABBA observations")
        times = sorted(x["elapsed_ms"] for x in r)
        out["each_node" if enabled else "every_128_nodes"] = {
            "calls": len(r), "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times), "max_ms": max(times),
            "over_200ms": sum(x > 200 for x in times),
            "over_300ms": sum(x > 300 for x in times),
            "depth3_complete": sum(x["depth"] >= 3 for x in r)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check", "run"), required=True)
    parser.add_argument("--request", default=REQUEST_PATH)
    parser.add_argument("--out", default="build/d9/probe")
    args = parser.parse_args()
    cfg = json.loads(Path(args.request).read_text())
    enabled = validate_request(cfg)
    if args.mode == "check":
        print("D9 enabled:", enabled)
        if os.getenv("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write("enabled=" + str(enabled).lower() + "\n")
        return
    if not enabled:
        raise ValueError("D9 probe is disabled")
    model_bytes = load_model()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "g8.npz").write_bytes(model_bytes)
    model = RelativeValue(str(out / "g8.npz"), mix_percent=25)
    rows = run_probe(model)
    summary = {"run_sha": os.getenv("GITHUB_SHA"), "model_sha256": G8_SHA,
               "protocol": cfg, "observations": len(rows),
               "diagnostic_only": True, "strength_proven": False,
               "independent_game_comparison": False, "timing": summarize(rows)}
    (out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
