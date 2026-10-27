"""Read-only cProfile on frozen G7 legal positions: locate search hot paths.

Instrumented time is NOT comparable to uninstrumented 200ms play.
No training, game result labels, model update or official KUMOJI changes.
"""
from __future__ import annotations
import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path
import shogi
from .engine import select_move
from .g7_depth_diagnosis import (G7_SHA, GAMES_SHA, SOURCE_RUN, MODEL_RUN,
                                 sample_positions, digest)
from .relative_value import RelativeValue

def expected_request():
    return dict(research_only=True, training_allowed=False, adoption_allowed=False,
                source_run=SOURCE_RUN, model_run=MODEL_RUN,
                games_sha256=GAMES_SHA, model_sha256=G7_SHA,
                samples=16, fixed_depth=2, mix_percent=25,
                fast_relative=True, root_alpha_beta=True, instrumented=True)

def validate_request(c):
    if c != expected_request():
        raise ValueError("unapproved G7 frozen CPU hotspot protocol")
    return c

def execute(request, games, model, out):
    """Profile both legal-move collection modes on identical frozen positions."""
    validate_request(json.loads(Path(request).read_text(encoding="utf-8")))
    if digest(games) != GAMES_SHA or digest(model) != G7_SHA:
        raise ValueError("frozen source/model SHA mismatch")
    rows = [json.loads(s) for s in Path(games).read_text(encoding="utf-8").splitlines() if s.strip()]
    positions = sample_positions(rows)[:16]
    if len(positions) != 16 or len({x["sfen"] for x in positions}) != 16:
        raise ValueError("wrong frozen position set")
    frozen = RelativeValue(model, mix_percent=25)
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    variants = {}
    for variant, fast_list in (("legacy", False), ("fast_list", True)):
        profiler = cProfile.Profile()
        records = []
        profiler.enable()
        for i, sample in enumerate(positions):
            board = shogi.Board(sample["sfen"])
            before = board.sfen()
            t0 = time.perf_counter()
            move, depth, nodes = select_move(
                board, movetime_ms=200, max_depth=2, complete_depth=True,
                model=frozen, fast_relative=True, root_alpha_beta=True,
                fast_move_list=fast_list)
            elapsed = round((time.perf_counter() - t0) * 1000, 3)
            if (board.sfen() != before or move == "resign"
                    or shogi.Move.from_usi(move) not in board.legal_moves or depth != 2):
                raise ValueError("profile search modified position or failed fixed depth")
            records.append(dict(sample=i, pair=sample["pair"],
                                g7_black=sample["g7_black"], move=move,
                                depth=depth, nodes=nodes, instrumented_ms=elapsed))
        profiler.disable()
        buf = io.StringIO()
        stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
        stats.print_stats(65)
        top = []
        for (filename, line, func), (primitive, calls, selftime, cumulative, callers) in sorted(
                stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:100]:
            top.append(dict(function=func, file=filename.rsplit("/", 1)[-1],
                            line=line, calls=calls, primitive_calls=primitive,
                            self_seconds=round(selftime, 6),
                            cumulative_seconds=round(cumulative, 6)))
        (target / ("profile_" + variant + ".txt")).write_text(buf.getvalue(), encoding="utf-8")
        variants[variant] = dict(total_nodes=sum(x["nodes"] for x in records),
                                 total_instrumented_ms=round(sum(x["instrumented_ms"] for x in records), 3),
                                 top_functions=top, records=records)
    old = variants["legacy"]["records"]
    fast = variants["fast_list"]["records"]
    if [(r["sample"], r["move"], r["depth"], r["nodes"]) for r in old] != [
            (r["sample"], r["move"], r["depth"], r["nodes"]) for r in fast]:
        raise ValueError("FastMoveList changed fixed-depth search result or nodes")
    result = dict(status="read_only_instrumented_fast_list_hotspot_comparison_not_speed_or_strength",
                  source_run=SOURCE_RUN, model_run=MODEL_RUN,
                  games_sha256=GAMES_SHA, model_sha256=G7_SHA,
                  samples=16, fixed_depth=2, fixed_depth_equivalent=16,
                  variants=variants,
                  caution="cProfile instrumented cumulative times overlap and cannot be summed; compare function hotspots, not actual playing speed.")
    (target / "profile.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (target / "positions.json").write_text(json.dumps(positions, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "variants"}, indent=2))
    return result

def main():
    p = argparse.ArgumentParser()
    for name in ("request", "games", "model", "out"):
        p.add_argument("--"+name, required=True)
    a = p.parse_args()
    execute(a.request, a.games, a.model, a.out)

if __name__ == "__main__":
    main()
