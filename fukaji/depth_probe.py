"""Read-only timed depth probe on frozen, legally replayed 32-game positions.

This measures depth and move sensitivity, not correctness or playing strength.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
import shogi
from .direct_arena import replay_check
from .engine import select_move
from .relative_value import RelativeValue

SOURCE_RUN = 40378086136
MODEL_RUN = 40359123574
MODEL_SHA = "881af10b5bc0cde591e37a9aee883213fa7dd602345b786b9b93d69c7e6ed239"

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate_request(c):
    if (c.get("research_only") is not True or c.get("source_run_id") != SOURCE_RUN
            or c.get("model_run_id") != MODEL_RUN or c.get("model_sha256") != MODEL_SHA
            or c.get("source_seed") != 20271227 or c.get("source_pairs") != 16
            or c.get("max_samples") != 12 or c.get("time_ms") != [200, 1000]
            or c.get("max_depths") != [2, 3] or c.get("mix_percent") != 25
            or c.get("training_allowed") is not False):
        raise ValueError("invalid frozen depth-probe protocol")
    for key in ("games_sha256", "report_sha256"):
        val = c.get(key)
        if not isinstance(val, str) or len(val) != 64 or any(x not in "0123456789abcdef" for x in val):
            raise ValueError("invalid digest: " + key)
    return c

def sample_positions(rows, limit=12):
    """Use frozen unresolved positions at plies 120/160, unique board states."""
    selected, seen = [], set()
    for phase in (120, 160):
        for row in rows:
            if row["status"] != "move_limit_unresolved" or row["fast_black"] is not True:
                continue
            moves = row["opening"] + row["moves"]
            if len(moves) < phase:
                continue
            board = shogi.Board()
            for word in moves[:phase]:
                move = shogi.Move.from_usi(word)
                if move not in board.legal_moves:
                    raise ValueError("illegal frozen source move")
                board.push(move)
            if board.is_checkmate() or board.sfen() in seen:
                continue
            seen.add(board.sfen())
            selected.append({"pair": row["pair"], "phase_ply": phase,
                             "moves": moves[:phase], "sfen": board.sfen()})
            if len(selected) >= limit:
                return selected
    if len(selected) < 8:
        raise ValueError("insufficient independent positions")
    return selected

def execute(request_path, games_path, report_path, model_path, out_dir):
    c = validate_request(json.loads(Path(request_path).read_text()))
    if (sha256(games_path) != c["games_sha256"]
            or sha256(report_path) != c["report_sha256"]
            or sha256(model_path) != c["model_sha256"]):
        raise ValueError("immutable artifact SHA mismatch")
    report = json.loads(Path(report_path).read_text())
    if (report.get("arena_seed") != c["source_seed"]
            or report.get("games_sha256") != sha256(games_path)
            or report.get("model_sha256") != sha256(model_path)
            or report.get("games") != 32 or report.get("pairs") != 16):
        raise ValueError("frozen source provenance mismatch")
    rows = [json.loads(x) for x in Path(games_path).read_text().splitlines() if x.strip()]
    if len(rows) != 32:
        raise ValueError("source row count mismatch")
    for row in rows:
        replay_check(row["opening"], row["moves"], row)
    samples = sample_positions(rows, c["max_samples"])
    model = RelativeValue(model_path, mix_percent=25)
    records = []
    for index, sample in enumerate(samples):
        order = ((200, 2), (1000, 3)) if index % 2 == 0 else ((1000, 3), (200, 2))
        for ms, depth_limit in order:
            board = shogi.Board()
            for word in sample["moves"]:
                board.push_usi(word)
            before = board.sfen()
            t0 = time.perf_counter()
            move, depth, nodes = select_move(
                board, movetime_ms=ms, max_depth=depth_limit,
                model=model, fast_relative=True)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
            if board.sfen() != before or move == "resign" or shogi.Move.from_usi(move) not in board.legal_moves:
                raise ValueError("search changed board or returned illegal move")
            records.append({"sample": index, "pair": sample["pair"],
                            "phase_ply": sample["phase_ply"], "time_ms": ms,
                            "max_depth": depth_limit, "move": move,
                            "complete_depth": depth, "nodes": nodes,
                            "elapsed_ms": elapsed_ms})
    paired = {}
    for i in range(len(samples)):
        short = next(r for r in records if r["sample"] == i and r["time_ms"] == 200)
        long = next(r for r in records if r["sample"] == i and r["time_ms"] == 1000)
        paired[i] = (short, long)
    result = {
        "status": "read_only_timed_depth_probe_no_promotion",
        "source_run_id": SOURCE_RUN, "model_sha256": MODEL_SHA,
        "games_sha256": sha256(games_path), "report_sha256": sha256(report_path),
        "samples": len(samples),
        "move_changed": sum(a["move"] != b["move"] for a, b in paired.values()),
        "short_depth2_complete": sum(a["complete_depth"] == 2 for a, _ in paired.values()),
        "long_depth3_complete": sum(b["complete_depth"] == 3 for _, b in paired.values()),
        "median_short_nodes": statistics.median(a["nodes"] for a, _ in paired.values()),
        "median_long_nodes": statistics.median(b["nodes"] for _, b in paired.values()),
        "positions": samples, "records": records,
        "note": "Time/depth change is not a correctness label. No training, adoption or engine modification."
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "depth_probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("positions","records")}, indent=2))
    return result

def main():
    p = argparse.ArgumentParser()
    for x in ("request","games","report","model","out"):
        p.add_argument("--"+x,required=True)
    a = p.parse_args()
    execute(a.request,a.games,a.report,a.model,a.out)

if __name__ == "__main__":
    main()
