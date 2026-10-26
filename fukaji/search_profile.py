"""Bounded, read-only profile of FUKAJI search on independently recorded KUMOJI games.

The frozen model and legal match records are read-only. This is a diagnostic,
not a match, model selection, teacher-label generation, or adoption.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import statistics
import time
from collections import defaultdict
from pathlib import Path

import shogi

from .engine import evaluate, select_move
from .kumoji_match import audit_game, sha256
from .relative_value import RelativeValue


def validate_request(c):
    if (c.get("research_only") is not True
            or c.get("paired_source_run_id") != 40375047387
            or c.get("relative_source_run_id") != 40359123574
            or type(c.get("max_samples")) is not int
            or not 4 <= c["max_samples"] <= 16
            or c.get("movetime_ms") != 200
            or c.get("max_depth") != 2
            or c.get("model_mixes") != [0, 25, 100]):
        raise ValueError("invalid fixed search profiling protocol")
    for name in ("paired_games_sha256", "relative_model_sha256"):
        value = c.get(name)
        if (not isinstance(value, str) or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)):
            raise ValueError("invalid " + name)
    return c


def sample_positions(games, limit):
    """Use both colors from each independent opening; never use result labels."""
    by_pair = defaultdict(dict)
    for row in games:
        if row.get("arm") != "equal_200ms":
            continue
        audit_game(row)
        key = row.get("pair")
        color = row.get("a_black")
        if type(key) is not int or type(color) is not bool or color in by_pair[key]:
            raise ValueError("invalid/duplicate pair color")
        by_pair[key][color] = row
    if len(by_pair) != 4 or any(set(x) != {True, False} for x in by_pair.values()):
        raise ValueError("missing four independent color-swapped opening pairs")
    for rows in by_pair.values():
        if rows[True]["opening"] != rows[False]["opening"]:
            raise ValueError("opening mismatch")
    samples, seen = [], set()
    # First collect an early FUKAJI turn in each color, then midgame turns.
    for phase in ("early", "middle"):
        for pair in sorted(by_pair):
            for a_black in (True, False):
                row = by_pair[pair][a_black]
                moves = row["moves"]
                first = 6 if a_black else 7
                k = first if phase == "early" else first + 8
                if k >= len(moves):
                    continue
                board = shogi.Board()
                for word in moves[:k]:
                    move = shogi.Move.from_usi(word)
                    if move not in board.legal_moves:
                        raise ValueError("illegal source move")
                    board.push(move)
                if board.turn != (shogi.BLACK if a_black else shogi.WHITE):
                    raise ValueError("incorrect FUKAJI side-to-move")
                if board.is_checkmate():
                    continue
                position = board.sfen()
                if position not in seen:
                    samples.append({"pair": pair, "a_black": a_black,
                                    "phase": phase, "plies": k, "sfen": position})
                    seen.add(position)
                if len(samples) >= limit:
                    return samples
    if len(samples) < 4:
        raise ValueError("insufficient independent profiling positions")
    return samples


def execute(request_path, games_path, report_path, model_path, out_dir):
    c = validate_request(json.loads(Path(request_path).read_text(encoding="utf-8")))
    games_path, report_path, model_path, out = map(
        Path, (games_path, report_path, model_path, out_dir))
    if sha256(games_path) != c["paired_games_sha256"]:
        raise ValueError("paired games SHA mismatch")
    if sha256(model_path) != c["relative_model_sha256"]:
        raise ValueError("relative model SHA mismatch")
    paired = json.loads(report_path.read_text(encoding="utf-8"))
    if (paired.get("games_sha256") != sha256(games_path)
            or paired.get("fukaji_relative_model_sha256") != sha256(model_path)
            or paired.get("kumoji_commit") != "807db500ee2739c1eb6a527b614eeea40a5993d6"):
        raise ValueError("paired source provenance mismatch")
    games = [json.loads(s) for s in games_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(games) != 16:
        raise ValueError("unexpected paired game count")
    for row in games:
        audit_game(row)
    positions = sample_positions(games, c["max_samples"])
    models = {mix: RelativeValue(model_path, mix_percent=mix) for mix in c["model_mixes"]}
    records = []
    for index, sample in enumerate(positions):
        board = shogi.Board(sample["sfen"])
        # Rotate run order to reduce bias from runner load and CPU warming.
        order = c["model_mixes"][index % 3:] + c["model_mixes"][:index % 3]
        for mix in order:
            t0 = time.perf_counter()
            move, depth, nodes = select_move(
                board, movetime_ms=c["movetime_ms"],
                max_depth=c["max_depth"], model=models[mix])
            elapsed = (time.perf_counter() - t0) * 1000
            if move == "resign" or shogi.Move.from_usi(move) not in board.legal_moves:
                raise ValueError("profiling returned an illegal move")
            records.append({**{k: sample[k] for k in ("pair", "a_black", "phase", "plies")},
                            "mix_percent": mix, "move": move, "complete_depth": depth,
                            "nodes": nodes, "elapsed_ms": round(elapsed, 3)})
    # cProfile is intentionally separated from wall-clock measurements:
    # instrumentation changes speed, so it must not be treated as game time.
    profiler = cProfile.Profile()
    profiler.enable()
    select_move(shogi.Board(positions[0]["sfen"]), movetime_ms=1200,
                max_depth=3, model=models[25])
    profiler.disable()
    buf = io.StringIO()
    pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(25)
    by_mix = {}
    for mix in c["model_mixes"]:
        subset = [r for r in records if r["mix_percent"] == mix]
        by_mix[str(mix)] = {
            "positions": len(subset),
            "total_nodes": sum(r["nodes"] for r in subset),
            "median_nodes": statistics.median(r["nodes"] for r in subset),
            "depth_zero": sum(r["complete_depth"] == 0 for r in subset),
            "depth_two": sum(r["complete_depth"] == 2 for r in subset),
            "median_elapsed_ms": statistics.median(r["elapsed_ms"] for r in subset),
        }
    out.mkdir(parents=True, exist_ok=True)
    (out / "search_profile.txt").write_text(buf.getvalue(), encoding="utf-8")
    (out / "search_positions.json").write_text(
        json.dumps(positions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {
        "status": "research_profile_only_no_promotion",
        "paired_source_run_id": c["paired_source_run_id"],
        "relative_source_run_id": c["relative_source_run_id"],
        "paired_games_sha256": sha256(games_path),
        "relative_model_sha256": sha256(model_path),
        "positions": len(positions),
        "movetime_ms": c["movetime_ms"],
        "max_depth": c["max_depth"],
        "results": by_mix,
        "samples": records,
        "note": "Wall-clock 200ms samples are noisy; cProfile is instrumented and separate. No strength claim.",
    }
    (out / "profile_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ("request", "games", "paired-report", "model", "out"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    execute(a.request, a.games, a.paired_report, a.model, a.out)


if __name__ == "__main__":
    main()
