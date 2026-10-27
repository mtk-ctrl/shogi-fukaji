"""Read-only G7 depth-budget diagnosis on frozen G7/KUMOJI games.

Uses only legal positions, never outcome labels or opponent moves as teachers.
Compares the identical frozen G7 model at the same 200ms deadline with
maximum completed depths 2 versus 4. This is NOT a playing-strength claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import shogi

from .engine import select_move
from .kumoji_compare import verify
from .relative_value import RelativeValue

SOURCE_RUN = 40436859527
MODEL_RUN = 40434466661
G7_SHA = "417d18e7b64494c9febdf6d217afbe4e4979e68d04528f16db8b45c0bcde2a0e"
GAMES_SHA = "6cb09fe88e68d9389cd04dfafdf35a1dbc16d135d78ae52bfce1a6b3c81170c6"
PAIRS = 32
SAMPLES = 32

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def expected_request():
    return dict(research_only=True, training_allowed=False,
                adoption_allowed=False, source_run=SOURCE_RUN,
                model_run=MODEL_RUN, games_sha256=GAMES_SHA,
                model_sha256=G7_SHA, source_pairs=PAIRS,
                samples=SAMPLES, movetime_ms=200,
                max_depths=[2, 4], mix_percent=25,
                fast_relative=True, root_alpha_beta=True)

def validate_request(cfg):
    if cfg != expected_request():
        raise ValueError("unapproved frozen depth diagnosis request")
    return cfg

def sample_positions(rows):
    if len(rows) != 64:
        raise ValueError("wrong source game count")
    by_pair = {}
    for row in rows:
        verify(row)
        pair, color = row.get("pair"), row.get("g7_black")
        if type(pair) is not int or pair not in range(PAIRS) or type(color) is not bool:
            raise ValueError("invalid pair or color")
        key = (pair, color)
        if key in by_pair:
            raise ValueError("duplicate pair/color")
        by_pair[key] = row
    if len(by_pair) != 64:
        raise ValueError("missing paired source")
    result, seen = [], set()
    # One early/middle position for each seat from 16 independent opening pairs.
    for pair in range(16):
        black, white = by_pair[(pair, True)], by_pair[(pair, False)]
        if black["opening"] != white["opening"]:
            raise ValueError("source opening mismatch")
        for color, row in ((True, black), (False, white)):
            # After six opening moves, black moves next. Preserve G7's turn.
            ply = 14 if color else 15
            history = row["opening"] + row["moves"]
            if len(history) <= ply:
                raise ValueError("insufficient source moves")
            board = shogi.Board()
            for token in history[:ply]:
                move = shogi.Move.from_usi(token)
                if move not in board.legal_moves:
                    raise ValueError("illegal sampled history")
                board.push(move)
            if bool(board.turn) != color or board.is_checkmate():
                raise ValueError("wrong side or terminal sample")
            sfen = board.sfen()
            if sfen in seen:
                raise ValueError("duplicate sample position")
            seen.add(sfen)
            result.append(dict(pair=pair, g7_black=color, ply=ply, sfen=sfen))
    if len(result) != SAMPLES:
        raise ValueError("incorrect diagnostic sample count")
    return result

def execute(request, games, model, out):
    validate_request(json.loads(Path(request).read_text(encoding="utf-8")))
    if digest(games) != GAMES_SHA or digest(model) != G7_SHA:
        raise ValueError("frozen source/model SHA mismatch")
    rows = [json.loads(s) for s in Path(games).read_text(encoding="utf-8").splitlines() if s.strip()]
    samples = sample_positions(rows)
    frozen = RelativeValue(model, mix_percent=25)
    records = []
    for index, sample in enumerate(samples):
        # Alternate order to limit systematic warm-up effects.
        for max_depth in ([2, 4] if index % 2 == 0 else [4, 2]):
            board = shogi.Board(sample["sfen"])
            before = board.sfen()
            t0 = time.perf_counter()
            word, depth, nodes = select_move(
                board, movetime_ms=200, max_depth=max_depth,
                model=frozen, fast_relative=True, root_alpha_beta=True)
            elapsed = round((time.perf_counter() - t0) * 1000, 3)
            if board.sfen() != before or word == "resign" or shogi.Move.from_usi(word) not in board.legal_moves:
                raise ValueError("illegal or state-mutating search")
            records.append(dict(sample=index, pair=sample["pair"],
                                g7_black=sample["g7_black"], max_depth=max_depth,
                                move=word, complete_depth=depth,
                                nodes=nodes, elapsed_ms=elapsed))
    pairs = []
    for i in range(SAMPLES):
        x = next(r for r in records if r["sample"] == i and r["max_depth"] == 2)
        y = next(r for r in records if r["sample"] == i and r["max_depth"] == 4)
        pairs.append((x, y))
    summary = {}
    for max_depth in (2, 4):
        subset = [r for r in records if r["max_depth"] == max_depth]
        summary[str(max_depth)] = dict(
            median_nodes=statistics.median(r["nodes"] for r in subset),
            median_elapsed_ms=statistics.median(r["elapsed_ms"] for r in subset),
            under_150ms=sum(r["elapsed_ms"] < 150 for r in subset),
            depth_distribution=dict(Counter(r["complete_depth"] for r in subset)),
            total_nodes=sum(r["nodes"] for r in subset))
    result = dict(status="read_only_diagnosis_no_training_or_adoption",
                  source_run=SOURCE_RUN, model_run=MODEL_RUN,
                  games_sha256=GAMES_SHA, model_sha256=G7_SHA,
                  samples=SAMPLES, movetime_ms=200,
                  move_changed=sum(x["move"] != y["move"] for x, y in pairs),
                  extra_depth_completed=sum(y["complete_depth"] > x["complete_depth"] for x, y in pairs),
                  summary=summary, records=records,
                  caution="Different max depth at same wall-clock budget is a diagnostic, not evidence of stronger play. No labels from KUMOJI.")
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    (target/"diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (target/"positions.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="records"}, indent=2))
    return result

def main():
    p = argparse.ArgumentParser()
    for name in ("request", "games", "model", "out"):
        p.add_argument("--"+name, required=True)
    a = p.parse_args()
    execute(a.request, a.games, a.model, a.out)

if __name__ == "__main__":
    main()
