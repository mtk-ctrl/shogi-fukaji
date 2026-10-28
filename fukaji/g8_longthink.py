"""G8 read-only 60-position long-think research. No training or adoption.

Input: SHA-pinned, legally replayed 100-position G8 artifact. Each of twenty
workers analyzes exactly three unique training-group positions. The 20-second
move is a hypothesis, NOT a verified best move or a training label.
"""
import argparse
import hashlib
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import shogi

from .engine import select_move
from .relative_value import RelativeValue
from .g8_artifact_selection import artifact_bytes

SOURCE_RUN = 40624316138
SOURCE_ID = 12788612272
SOURCE_NAME = "fukaji-g8-verified-selection-40624316138"
SOURCE_SHA = "5206f1ffbf9cd0e8a71ac65f8d6d2f9828853b408eaf5c3189ed285b8e47db52"
CANDIDATES_SHA = "54192b94cf4e60347790907262b3e516f04ca78dd367d6fdc6d5cb310e63f2be"
MODEL_SHA = "417d18e7b64494c9febdf6d217afbe4e4979e68d04528f16db8b45c0bcde2a0e"
SEED = "fukaji-g8-longthink60-v1|20261028|40624316138"
SELECTED_SHA = "db318cc1ae757bb768b0dec130b64e44f14cdda7e323be48506d3163198dcedd"
QUOTAS = {"mate_tail": 24, "mate_middle": 12,
          "unresolved_late": 12, "early_mixed": 12}
SHARDS, PER_SHARD = 20, 3
BASE_MS, LONG_MS = 200, 20000


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def expected_request(enabled=False):
    return dict(enabled=enabled, research_only=True, training_allowed=False,
                adoption_allowed=False, source_run=SOURCE_RUN,
                source_artifact_id=SOURCE_ID, source_zip_sha256=SOURCE_SHA,
                candidates_sha256=CANDIDATES_SHA, model_sha256=MODEL_SHA,
                selection_seed=SEED, positions=60, shards=20,
                baseline_ms=BASE_MS, long_ms=LONG_MS,
                baseline_depth=2, long_depth=5)


def validate_request(cfg):
    if cfg == expected_request(False):
        return False
    if cfg == expected_request(True):
        return True
    raise ValueError("unexpected G8 long-think research request")


def read_jsonl(raw):
    return [json.loads(s) for s in raw.decode("utf-8").splitlines() if s.strip()]


def state(board):
    return " ".join(board.sfen().split()[:3])


def history_board(row):
    board = shogi.Board()
    for usi in row["history_usi"]:
        move = shogi.Move.from_usi(usi)
        if move not in board.legal_moves:
            raise ValueError("illegal candidate history")
        board.push(move)
    if (state(board) != row["position_sfen_state"] or
            row["position_before_ply"] != len(row["history_usi"]) + 1 or
            board.is_checkmate()):
        raise ValueError("candidate board/ply/terminal mismatch")
    played = shogi.Move.from_usi(row["next_played_move_usi"])
    if played not in board.legal_moves:
        raise ValueError("source played move illegal")
    return board


def validate_candidates(rows):
    if len(rows) != 100 or {r.get("candidate_id") for r in rows} != set(range(100)):
        raise ValueError("candidate source must have exactly 100 IDs")
    if (len({r["parent_id"] for r in rows}) != 100 or
            len({r["position_sfen_state"] for r in rows}) != 100 or
            Counter(r["reason_category"] for r in rows) !=
            {"mate_tail": 40, "mate_middle": 20,
             "unresolved_late": 20, "early_mixed": 20}):
        raise ValueError("candidate source categories or dedup mismatch")
    for row in rows:
        if (row["split"] != "train" or
                row["analysis_label"] != "candidate_only_not_proven_error" or
                (row["terminal_status"] != "checkmate" and
                 row["winner_for_proven_mates_only"] is not None)):
            raise ValueError("holdout leakage or unverified outcome label")
        history_board(row)


def select_sixty(rows):
    """Predeclared 24/12/12/12 categories and 30/30 side-to-move balance."""
    if len(rows) != 100:
        raise ValueError("wrong candidate source count")
    chosen = []
    for category, count in QUOTAS.items():
        for parity in (0, 1):
            pool = [r for r in rows if r["reason_category"] == category and
                    r["position_before_ply"] % 2 == parity]
            pool.sort(key=lambda r: sha((SEED + "|" + str(r["candidate_id"]) +
                                         "|" + r["parent_id"]).encode()))
            if len(pool) < count // 2:
                raise ValueError("insufficient side-balanced candidates")
            chosen.extend(pool[:count // 2])
    chosen.sort(key=lambda r: (r["reason_category"], r["candidate_id"]))
    if (len(chosen) != 60 or len({r["parent_id"] for r in chosen}) != 60 or
            len({r["position_sfen_state"] for r in chosen}) != 60 or
            Counter(r["position_before_ply"] % 2 for r in chosen) != {0: 30, 1: 30}):
        raise ValueError("selected set not independent or balanced")
    return [dict(row, analysis_index=i) for i, row in enumerate(chosen)]


def prepare(request, out):
    if not validate_request(json.loads(Path(request).read_text())):
        raise ValueError("request is disabled")
    if os.environ.get("GITHUB_REPOSITORY") != "mtk-ctrl/shogi-fukaji":
        raise ValueError("wrong repository")
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise ValueError("missing read-only GitHub token")
    raw = artifact_bytes(SOURCE_ID, SOURCE_NAME, SOURCE_RUN, SOURCE_SHA, token)
    with ZipFile(__import__("io").BytesIO(raw)) as z:
        candidates = z.read("selected/analysis_candidates_100.jsonl")
        manifest = json.loads(z.read("selected/selection_manifest.json"))
    if (sha(candidates) != CANDIDATES_SHA or
            manifest.get("candidate_sha256") != CANDIDATES_SHA or
            manifest.get("all_400_games_full_legal_replay_verified") is not True or
            manifest.get("all_candidates_in_train") is not True or
            manifest.get("candidate_count") != 100 or
            manifest.get("g7_model_sha256") != MODEL_SHA):
        raise ValueError("frozen 100-candidate provenance mismatch")
    rows = read_jsonl(candidates)
    validate_candidates(rows)
    chosen = select_sixty(rows)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n"
                      for x in chosen).encode()
    if sha(payload) != SELECTED_SHA:
        raise ValueError("frozen 60-position selection SHA mismatch")
    (root / "selected60.jsonl").write_bytes(payload)
    receipt = dict(status="read_only_research_prepared", source_run=SOURCE_RUN,
                   source_artifact_id=SOURCE_ID, source_zip_sha256=SOURCE_SHA,
                   candidates_sha256=CANDIDATES_SHA, model_sha256=MODEL_SHA,
                   seed=SEED, positions=60, shards=SHARDS,
                   selected_sha256=sha(payload),
                   category_counts=dict(Counter(r["reason_category"] for r in chosen)),
                   side_counts=dict(Counter("black" if r["position_before_ply"] % 2
                                            else "white" for r in chosen)),
                   all_100_legal_histories_verified=True,
                   no_training_or_adoption=True)
    (root / "selection_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def read_selected(root):
    root = Path(root)
    raw = (root / "selected60.jsonl").read_bytes()
    receipt = json.loads((root / "selection_receipt.json").read_text())
    if (receipt.get("selected_sha256") != SELECTED_SHA or sha(raw) != SELECTED_SHA or
            receipt.get("source_zip_sha256") != SOURCE_SHA or
            receipt.get("model_sha256") != MODEL_SHA or
            receipt.get("positions") != 60 or receipt.get("shards") != 20):
        raise ValueError("selected data or receipt mismatch")
    rows = read_jsonl(raw)
    if (len(rows) != 60 or [r["analysis_index"] for r in rows] != list(range(60))
            or len({r["parent_id"] for r in rows}) != 60):
        raise ValueError("selected input IDs or groups invalid")
    return rows, receipt


def live_history(board):
    return {k: v for k, v in board.transpositions.items() if v > 0}


def search(board, model, ms, depth):
    before, before_history = board.sfen(), live_history(board)
    t0 = time.perf_counter()
    try:
        move, completed, nodes = select_move(
            board, movetime_ms=ms, max_depth=depth, model=model,
            fast_relative=True, root_alpha_beta=True)
    finally:
        if board.sfen() != before or live_history(board) != before_history:
            raise ValueError("search mutated board or repetition history")
    elapsed = round((time.perf_counter() - t0) * 1000, 3)
    if (move == "resign" or shogi.Move.from_usi(move) not in board.legal_moves
            or completed < 0 or nodes < 0):
        raise ValueError("illegal or incomplete search result")
    return dict(move=move, completed_depth=completed, nodes=nodes,
                elapsed_ms=elapsed, requested_ms=ms, depth_cap=depth)


def run_shard(input_dir, model_file, shard, out):
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("invalid shard index")
    if sha(Path(model_file).read_bytes()) != MODEL_SHA:
        raise ValueError("frozen G7 model SHA mismatch")
    rows, receipt = read_selected(input_dir)
    model = RelativeValue(model_file, mix_percent=25)
    results = []
    for row in rows[shard * PER_SHARD:(shard + 1) * PER_SHARD]:
        board = history_board(row)
        baseline = search(board, model, BASE_MS, 2)
        deep = search(board, model, LONG_MS, 5)
        results.append(dict(analysis_index=row["analysis_index"],
                            candidate_id=row["candidate_id"],
                            parent_id=row["parent_id"],
                            reason_category=row["reason_category"],
                            source_played_move=row["next_played_move_usi"],
                            baseline=baseline, long_think=deep,
                            changed_from_baseline=baseline["move"] != deep["move"],
                            changed_from_played=deep["move"] != row["next_played_move_usi"],
                            research_label="long_think_hypothesis_not_proven_best_move"))
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(x, sort_keys=True) + "\n" for x in results).encode()
    (root / "longthink_rows.jsonl").write_bytes(raw)
    report = dict(shard=shard, positions=PER_SHARD,
                  rows_sha256=sha(raw), selected_sha256=receipt["selected_sha256"],
                  model_sha256=MODEL_SHA, baseline_ms=BASE_MS, long_ms=LONG_MS,
                  research_only=True, training_executed=False)
    (root / "shard_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def aggregate(input_dir, parts, out):
    selected, receipt = read_selected(input_dir)
    reports = list(Path(parts).glob("*/shard_report.json"))
    if len(reports) != SHARDS:
        raise ValueError("missing long-think shard")
    seen, results = set(), []
    for path in reports:
        report = json.loads(path.read_text())
        shard = report["shard"]
        if (shard in seen or not 0 <= shard < SHARDS or
                report.get("selected_sha256") != receipt["selected_sha256"] or
                report.get("model_sha256") != MODEL_SHA or
                report.get("positions") != PER_SHARD or
                report.get("baseline_ms") != BASE_MS or
                report.get("long_ms") != LONG_MS or
                report.get("training_executed") is not False):
            raise ValueError("invalid or duplicate shard receipt")
        seen.add(shard)
        raw = (path.parent / "longthink_rows.jsonl").read_bytes()
        if sha(raw) != report["rows_sha256"]:
            raise ValueError("shard rows SHA mismatch")
        sub = read_jsonl(raw)
        if [r["analysis_index"] for r in sub] != list(
                range(shard * PER_SHARD, (shard + 1) * PER_SHARD)):
            raise ValueError("shard index coverage mismatch")
        for row in sub:
            source = selected[row["analysis_index"]]
            if (row["parent_id"] != source["parent_id"] or
                    row["candidate_id"] != source["candidate_id"] or
                    row["source_played_move"] != source["next_played_move_usi"]):
                raise ValueError("shard source mismatch")
            board = history_board(source)
            for key in ("baseline", "long_think"):
                r = row[key]
                if (r["move"] == "resign" or
                        shogi.Move.from_usi(r["move"]) not in board.legal_moves):
                    raise ValueError("invalid reported analysis move")
            if (row["changed_from_baseline"] !=
                    (row["baseline"]["move"] != row["long_think"]["move"])):
                raise ValueError("inconsistent change flag")
        results.extend(sub)
    results.sort(key=lambda r: r["analysis_index"])
    if [r["analysis_index"] for r in results] != list(range(60)):
        raise ValueError("missing or duplicated analysis row")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(x, sort_keys=True) + "\n" for x in results).encode()
    (root / "all_longthink_rows.jsonl").write_bytes(raw)
    summary = dict(status="read_only_60_longthink_completed_not_training",
                   positions=60, shards=20, per_shard=3,
                   source_zip_sha256=SOURCE_SHA,
                   selected_sha256=receipt["selected_sha256"],
                   results_sha256=sha(raw), model_sha256=MODEL_SHA,
                   baseline_ms=BASE_MS, long_ms=LONG_MS,
                   changed_from_baseline=sum(r["changed_from_baseline"] for r in results),
                   changed_from_played=sum(r["changed_from_played"] for r in results),
                   depth_gain=sum(r["long_think"]["completed_depth"] >
                                  r["baseline"]["completed_depth"] for r in results),
                   median_long_elapsed_ms=statistics.median(
                       r["long_think"]["elapsed_ms"] for r in results),
                   caution="Move changes and search depth do not prove correctness, teacher labels, or playing strength.",
                   training_executed=False, model_promoted=False)
    (root / "longthink_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("check", "prepare", "shard", "aggregate"), required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--out")
    p.add_argument("--input")
    p.add_argument("--model")
    p.add_argument("--parts")
    p.add_argument("--shard", type=int)
    a = p.parse_args()
    enabled = validate_request(json.loads(Path(a.request).read_text()))
    if a.mode == "check":
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write("enabled=" + str(enabled).lower() + "\n")
        print("G8 long-think request enabled:", enabled)
        return
    if not enabled:
        raise ValueError("research request disabled")
    if a.mode == "prepare":
        result = prepare(a.request, a.out)
    elif a.mode == "shard":
        result = run_shard(a.input, a.model, a.shard, a.out)
    else:
        result = aggregate(a.input, a.parts, a.out)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
