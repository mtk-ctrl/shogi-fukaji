"""Frozen G7 read-only conservative legality-guard comparison."""
from __future__ import annotations
import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
import shogi
from .engine import fast_legal_moves, select_move
from .g7_depth_diagnosis import G7_SHA, GAMES_SHA, SOURCE_RUN, MODEL_RUN, sample_positions, digest
from .relative_value import RelativeValue

def expected_request():
    return dict(research_only=True, training_allowed=False, adoption_allowed=False,
                source_run=SOURCE_RUN, model_run=MODEL_RUN,
                games_sha256=GAMES_SHA, model_sha256=G7_SHA,
                samples=32, movetime_ms=200, max_depth=4,
                fixed_equivalence_depth=2, mix_percent=25,
                fast_relative=True, root_alpha_beta=True,
                fast_move_list=True, variants=["baseline", "guard"])

def validate_request(cfg):
    if cfg != expected_request():
        raise ValueError("unapproved frozen legal-guard protocol")
    return cfg

def execute(request, games, model, out):
    validate_request(json.loads(Path(request).read_text(encoding="utf-8")))
    if digest(games) != GAMES_SHA or digest(model) != G7_SHA:
        raise ValueError("frozen source/model SHA mismatch")
    rows = [json.loads(s) for s in Path(games).read_text(encoding="utf-8").splitlines() if s.strip()]
    positions = sample_positions(rows)
    if len(positions) != 32 or len({p["sfen"] for p in positions}) != 32:
        raise ValueError("frozen positions mismatch")
    frozen = RelativeValue(model, mix_percent=25)
    records = []
    for index, sample in enumerate(positions):
        board = shogi.Board(sample["sfen"])
        before = board.sfen()
        original = [m.usi() for m in board.legal_moves]
        guarded = [m.usi() for m in fast_legal_moves(board)]
        if original != guarded or board.sfen() != before:
            raise ValueError("legal set/order/state mismatch at sample " + str(index))
        fixed = []
        for guard in (False, True):
            fixed.append(select_move(
                board, movetime_ms=200, max_depth=2, complete_depth=True,
                model=frozen, fast_relative=True, root_alpha_beta=True,
                fast_move_list=True, fast_legal_guard=guard))
        if fixed[0] != fixed[1] or board.sfen() != before:
            raise ValueError("fixed-depth result/nodes mismatch at sample " + str(index))
        for guard in ((False, True) if index % 2 == 0 else (True, False)):
            t0 = time.perf_counter()
            move, depth, nodes = select_move(
                board, movetime_ms=200, max_depth=4,
                model=frozen, fast_relative=True, root_alpha_beta=True,
                fast_move_list=True, fast_legal_guard=guard)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
            if board.sfen() != before or move == "resign" or shogi.Move.from_usi(move) not in board.legal_moves:
                raise ValueError("timed search illegal or mutated board")
            records.append(dict(sample=index, pair=sample["pair"], g7_black=sample["g7_black"],
                                guard=guard, move=move, depth=depth, nodes=nodes,
                                elapsed_ms=elapsed_ms, legal_count=len(original),
                                fixed_move=fixed[0][0], fixed_nodes=fixed[0][2]))
    summary = {}
    for guard, name in ((False, "baseline"), (True, "guard")):
        group = [r for r in records if r["guard"] == guard]
        summary[name] = dict(total_nodes=sum(r["nodes"] for r in group),
                             median_nodes=statistics.median(r["nodes"] for r in group),
                             median_ms=statistics.median(r["elapsed_ms"] for r in group),
                             depth_distribution=dict(Counter(r["depth"] for r in group)))
    a = {r["sample"]: r for r in records if not r["guard"]}
    b = {r["sample"]: r for r in records if r["guard"]}
    result = dict(status="read_only_opt_in_legality_guard_diagnosis_not_strength",
                  source_run=SOURCE_RUN, model_run=MODEL_RUN,
                  games_sha256=GAMES_SHA, model_sha256=G7_SHA,
                  legal_set_and_order_equivalent=32, fixed_depth_equivalent=32,
                  timed_samples=32, movetime_ms=200,
                  move_changed=sum(a[i]["move"] != b[i]["move"] for i in range(32)),
                  deeper_completed=sum(b[i]["depth"] > a[i]["depth"] for i in range(32)),
                  summary=summary, records=records,
                  caution="Timed results depend on runner scheduling; exact legality and fixed-depth equivalence do not prove strength.")
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    (target / "diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (target / "positions.json").write_text(json.dumps(positions, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="records"}, ensure_ascii=False, indent=2))
    return result

def main():
    p = argparse.ArgumentParser()
    for name in ("request", "games", "model", "out"):
        p.add_argument("--"+name, required=True)
    a = p.parse_args()
    execute(a.request, a.games, a.model, a.out)

if __name__ == "__main__":
    main()
