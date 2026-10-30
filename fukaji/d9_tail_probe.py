"""D9-3: diagnostic replay of frozen D8 extreme-latency positions.

The D8 outcomes are NOT used for model training, selection, or independent
strength claims. This only tests whether the deadline-check stride explains
the previously observed latency tail.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from zipfile import ZipFile

import shogi

from .d8_depth_extension_arena import G8_SHA, board_state
from .d9_clock_probe import ORDER, load_model, summarize
from .engine import select_move
from .g8_artifact_selection import artifact_bytes
from .relative_value import RelativeValue

D8_RUN = 41048053976
D8_ARTIFACT = 12973279499
D8_NAME = "fukaji-d8-final-summary-41048053976"
D8_ZIP_SHA = "f25b7be57645af7c8e6b00387b243009252a9b471c9606a207a654537e7d9c41"
D8_GAMES_SHA = "55d3873c609bab023c3f5507dc890c583178adc646e282dd1ecd5b385af523ef"
TAILS = ((36, 225), (44, 171), (160, 225))
REQUEST_PATH = ".github/fukaji-d9-tail-probe-request.json"


def expected_request(enabled=False):
    return {"enabled": enabled, "research_only": True,
            "training_allowed": False, "model_adoption_allowed": False,
            "model_sha256": G8_SHA, "d8_source_run": D8_RUN,
            "d8_source_artifact": D8_ARTIFACT, "d8_zip_sha256": D8_ZIP_SHA,
            "d8_games_sha256": D8_GAMES_SHA,
            "tails": [list(x) for x in TAILS], "order": list(ORDER),
            "movetime_ms": 200, "max_depth": 3, "mix_percent": 25,
            "fast_move_list": True, "fast_relative": True,
            "root_alpha_beta": True, "deadline_from_call": True,
            "max_single_call_ms": 1000}


def validate_request(cfg):
    for enabled in (False, True):
        if json.dumps(cfg, sort_keys=True) == json.dumps(expected_request(enabled), sort_keys=True):
            return enabled
    raise ValueError("D9 tail diagnostic differs from frozen exact request")


def load_d8_rows():
    raw = artifact_bytes(D8_ARTIFACT, D8_NAME, D8_RUN, D8_ZIP_SHA, os.environ["GH_TOKEN"])
    with ZipFile(io.BytesIO(raw)) as z:
        games_raw = z.read("all_games.jsonl")
    if hashlib.sha256(games_raw).hexdigest() != D8_GAMES_SHA:
        raise ValueError("D8 games SHA mismatch")
    games = {g["game_id"]: g for g in (json.loads(line) for line in games_raw.splitlines())}
    rows = []
    for game_id, ply in TAILS:
        game = games[game_id]
        target = game["played"][ply - 1]
        if not target["candidate"] or target["elapsed_ms"] < 700:
            raise ValueError("D8 frozen extreme-latency target changed")
        rows.append((game_id, ply, game))
    return rows


def run_tail_probe(model, source_rows, select=select_move):
    observations = []
    for game_id, ply, game in source_rows:
        board = shogi.Board()
        for word in game["opening"] + game["moves"][:ply - 1]:
            move = shogi.Move.from_usi(word)
            if move not in board.legal_moves:
                raise ValueError("D8 source cannot be legally replayed")
            board.push(move)
        before = board_state(board)
        for enabled in ORDER:
            telemetry = {}
            started = time.perf_counter()
            word, depth, nodes = select(
                board, movetime_ms=200, max_depth=3, model=model,
                fast_move_list=True, fast_relative=True, root_alpha_beta=True,
                deadline_from_call=True, time_check_each_node=enabled, telemetry=telemetry)
            elapsed = (time.perf_counter() - started) * 1000
            if board_state(board) != before:
                raise ValueError("D9 tail search did not restore board/history")
            if word == "resign" or shogi.Move.from_usi(word) not in board.legal_moves:
                raise ValueError("D9 tail selected illegal move")
            if not math.isfinite(elapsed) or elapsed < 0 or elapsed > 1000:
                raise ValueError("D9 tail probe exceeded fault ceiling")
            observations.append({"game_id": game_id, "ply": ply,
                                 "original_d8_ms": game["played"][ply - 1]["elapsed_ms"],
                                 "each_node": enabled, "move": word, "depth": depth,
                                 "nodes": nodes, "elapsed_ms": elapsed,
                                 "root_preparation_ms": telemetry.get("root_preparation_ms")})
    return observations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check", "run"), required=True)
    parser.add_argument("--request", default=REQUEST_PATH)
    parser.add_argument("--out", default="build/d9/tail")
    args = parser.parse_args()
    cfg = json.loads(Path(args.request).read_text())
    enabled = validate_request(cfg)
    if args.mode == "check":
        print("D9 tail enabled:", enabled)
        if os.getenv("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write("enabled=" + str(enabled).lower() + "\n")
        return
    if not enabled:
        raise ValueError("D9 tail probe disabled")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_bytes = load_model()
    if hashlib.sha256(model_bytes).hexdigest() != G8_SHA:
        raise ValueError("G8 SHA mismatch")
    (out / "g8.npz").write_bytes(model_bytes)
    model = RelativeValue(str(out / "g8.npz"), mix_percent=25)
    source_rows = load_d8_rows()
    rows = run_tail_probe(model, source_rows)
    summary = {"run_sha": os.getenv("GITHUB_SHA"), "protocol": cfg,
               "diagnostic_only": True, "strength_proven": False,
               "original_D8_not_independent": True,
               "timing": summarize(rows)}
    (out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
