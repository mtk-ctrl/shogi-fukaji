"""Frozen G8 40 changed-move follow-up; research only, never training labels.

Input comes from two SHA-pinned artifacts of the completed 60-position run.
The 20s repeat, 30s independent search, and 5s opponent replies are
diagnostic observations, not proof that a move is better.
"""
import argparse
import hashlib
import io
import json
import os
import statistics
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import shogi

from .g8_artifact_selection import artifact_bytes
from .g8_longthink import MODEL_SHA, history_board, live_history, search, sha
from .relative_value import RelativeValue

RUN = 40630630178
INPUT_ID = 12791352643
INPUT_NAME = "fukaji-g8-longthink-input-40630630178"
INPUT_ZIP_SHA = "0d11ac42493bb595b9a74769781a364703f8d4f521c66c8f9ba78aa17583672c"
AGG_ID = 12791512324
AGG_NAME = "fukaji-g8-longthink-aggregate-40630630178"
AGG_ZIP_SHA = "cfb8278806940f7e845e6ae0475b4500777308a0d077537c6d2f4b8134d279d5"
SELECTED_SHA = "db318cc1ae757bb768b0dec130b64e44f14cdda7e323be48506d3163198dcedd"
RESULTS_SHA = "ef6a73d03f3ce6f28436e6ca45b5a3724325aa67ff7b6420b408a35028116e30"
FOLLOWUP_SHA = "5a801a692b0cdaf9d97a7622da0d5676b553a01ff933183fc0018b07f1a4d6bb"
SHARDS, PER_SHARD = 20, 2
REPEAT_MS, INDEPENDENT_MS, REPLY_MS = 20000, 30000, 5000


def expected_request(enabled=False):
    return {
        "enabled": enabled, "research_only": True,
        "training_allowed": False, "adoption_allowed": False,
        "source_run": RUN, "input_artifact_id": INPUT_ID,
        "input_zip_sha256": INPUT_ZIP_SHA,
        "aggregate_artifact_id": AGG_ID,
        "aggregate_zip_sha256": AGG_ZIP_SHA,
        "selected_sha256": SELECTED_SHA, "results_sha256": RESULTS_SHA,
        "followup_sha256": FOLLOWUP_SHA, "model_sha256": MODEL_SHA,
        "positions": 40, "shards": SHARDS,
        "repeat_ms": REPEAT_MS, "independent_ms": INDEPENDENT_MS,
        "reply_ms": REPLY_MS, "max_depth": 5, "reply_depth": 4,
    }


def validate_request(cfg):
    if cfg == expected_request(False):
        return False
    if cfg == expected_request(True):
        return True
    raise ValueError("unapproved G8 follow-up protocol")


def read_rows(raw):
    return [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]


def select_changed(selected, results):
    """All 60 identities must match before taking exactly 40 changed moves."""
    if len(selected) != 60 or len(results) != 60:
        raise ValueError("source 60-position count mismatch")
    if ([r.get("analysis_index") for r in selected] != list(range(60)) or
            [r.get("analysis_index") for r in results] != list(range(60))):
        raise ValueError("source 60-position index mismatch")
    if (len({r.get("parent_id") for r in selected}) != 60 or
            len({r.get("position_sfen_state") for r in selected}) != 60 or
            any(r.get("split") != "train" for r in selected)):
        raise ValueError("source parents, positions or training isolation invalid")
    chosen = []
    for src, obs in zip(selected, results):
        if (src["candidate_id"] != obs.get("candidate_id") or
                src["parent_id"] != obs.get("parent_id") or
                src["next_played_move_usi"] != obs.get("source_played_move") or
                obs.get("baseline", {}).get("move") != src["next_played_move_usi"] or
                obs.get("baseline", {}).get("depth_cap") != 2 or
                obs.get("long_think", {}).get("depth_cap") != 5 or
                obs.get("changed_from_baseline") is not
                (obs["baseline"]["move"] != obs["long_think"]["move"]) or
                obs.get("changed_from_played") is not
                (obs["long_think"]["move"] != src["next_played_move_usi"])):
            raise ValueError("source observation/played move mismatch")
        if obs["changed_from_baseline"]:
            chosen.append(dict(
                **src, followup_index=len(chosen),
                baseline=obs["baseline"], long_think=obs["long_think"],
                source_played_move=obs["source_played_move"],
                changed_from_baseline=True))
    if len(chosen) != 40:
        raise ValueError("expected exactly forty changed moves")
    return chosen


def encoded(rows):
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                   for r in rows).encode("utf-8")


def prepare(out):
    if os.environ.get("GITHUB_REPOSITORY") != "mtk-ctrl/shogi-fukaji":
        raise ValueError("wrong GitHub repository")
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise ValueError("missing read-only GitHub token")
    input_zip = artifact_bytes(INPUT_ID, INPUT_NAME, RUN, INPUT_ZIP_SHA, token)
    aggregate_zip = artifact_bytes(AGG_ID, AGG_NAME, RUN, AGG_ZIP_SHA, token)
    with ZipFile(io.BytesIO(input_zip)) as z:
        selected_raw = z.read("selected60.jsonl")
        selection_receipt = json.loads(z.read("selection_receipt.json"))
    with ZipFile(io.BytesIO(aggregate_zip)) as z:
        results_raw = z.read("all_longthink_rows.jsonl")
        summary = json.loads(z.read("longthink_summary.json"))
    if (sha(selected_raw) != SELECTED_SHA or sha(results_raw) != RESULTS_SHA or
            selection_receipt.get("selected_sha256") != SELECTED_SHA or
            summary.get("results_sha256") != RESULTS_SHA or
            summary.get("changed_from_baseline") != 40 or
            summary.get("training_executed") is not False or
            summary.get("model_promoted") is not False):
        raise ValueError("frozen source SHA or read-only provenance mismatch")
    selected = read_rows(selected_raw)
    results = read_rows(results_raw)
    # Reconstruct all 60 actual histories, not just the 40 selected.
    for src, obs in zip(selected, results):
        board = history_board(src)
        for key in ("baseline", "long_think"):
            if shogi.Move.from_usi(obs[key]["move"]) not in board.legal_moves:
                raise ValueError("source analysis move is illegal")
    chosen = select_changed(selected, results)
    raw = encoded(chosen)
    if sha(raw) != FOLLOWUP_SHA:
        raise ValueError("frozen forty-position selection SHA mismatch")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "followup40.jsonl").write_bytes(raw)
    receipt = {
        "status": "read_only_frozen_40_positions",
        "source_run": RUN, "input_zip_sha256": INPUT_ZIP_SHA,
        "aggregate_zip_sha256": AGG_ZIP_SHA,
        "selected_sha256": SELECTED_SHA, "results_sha256": RESULTS_SHA,
        "followup_sha256": FOLLOWUP_SHA, "model_sha256": MODEL_SHA,
        "all_60_histories_legally_replayed": True,
        "positions": 40, "shards": 20,
        "category_counts": dict(Counter(r["reason_category"] for r in chosen)),
        "training_executed": False, "adopted": False,
    }
    (root / "followup_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def read_input(root):
    root = Path(root)
    raw = (root / "followup40.jsonl").read_bytes()
    receipt = json.loads((root / "followup_receipt.json").read_text())
    if (sha(raw) != FOLLOWUP_SHA or
            receipt.get("followup_sha256") != FOLLOWUP_SHA or
            receipt.get("model_sha256") != MODEL_SHA or
            receipt.get("positions") != 40 or
            receipt.get("training_executed") is not False):
        raise ValueError("follow-up input SHA/receipt mismatch")
    rows = read_rows(raw)
    if ([r.get("followup_index") for r in rows] != list(range(40)) or
            len({r["parent_id"] for r in rows}) != 40 or
            len({r["position_sfen_state"] for r in rows}) != 40):
        raise ValueError("follow-up coverage or duplicate mismatch")
    return rows


def opponent_reply(board, usi, model):
    before = board.sfen()
    history = live_history(board)
    move = shogi.Move.from_usi(usi)
    if move not in board.legal_moves:
        raise ValueError("candidate reply branch is illegal")
    board.push(move)
    try:
        if board.is_checkmate():
            result = {"terminal": "checkmate_after_candidate", "reply": None}
        else:
            result = {"terminal": "nonterminal", "reply":
                      search(board, model, REPLY_MS, 4)}
    finally:
        board.pop()
        if board.sfen() != before or live_history(board) != history:
            raise ValueError("opponent branch did not restore board/history")
    return result


def run_shard(input_dir, model_file, shard, out):
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("invalid shard index")
    if sha(Path(model_file).read_bytes()) != MODEL_SHA:
        raise ValueError("frozen G7 model SHA mismatch")
    rows = read_input(input_dir)
    model = RelativeValue(model_file, mix_percent=25)
    results = []
    for src in rows[shard * PER_SHARD:(shard + 1) * PER_SHARD]:
        board = history_board(src)
        before = board.sfen()
        history = live_history(board)
        repeat = search(board, model, REPEAT_MS, 5)
        independent = search(board, model, INDEPENDENT_MS, 5)
        baseline_reply = opponent_reply(board, src["baseline"]["move"], model)
        deep_reply = opponent_reply(board, src["long_think"]["move"], model)
        if board.sfen() != before or live_history(board) != history:
            raise ValueError("follow-up mutated source board/history")
        results.append({
            "followup_index": src["followup_index"],
            "analysis_index": src["analysis_index"],
            "candidate_id": src["candidate_id"],
            "parent_id": src["parent_id"],
            "reason_category": src["reason_category"],
            "source_played_move": src["source_played_move"],
            "old_long_move": src["long_think"]["move"],
            "repeat20": repeat, "independent30": independent,
            "baseline_opponent_reply5": baseline_reply,
            "old_long_opponent_reply5": deep_reply,
            "repeat_matches_old": repeat["move"] == src["long_think"]["move"],
            "independent_matches_old": independent["move"] == src["long_think"]["move"],
            "repeat_matches_independent": repeat["move"] == independent["move"],
            "research_label": "diagnostic_not_best_move_or_teacher",
        })
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    raw = encoded(results)
    (root / "followup_rows.jsonl").write_bytes(raw)
    report = {"shard": shard, "positions": PER_SHARD,
              "rows_sha256": sha(raw), "followup_sha256": FOLLOWUP_SHA,
              "model_sha256": MODEL_SHA, "training_executed": False}
    (root / "shard_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def aggregate(input_dir, parts, out):
    selected = read_input(input_dir)
    paths = list(Path(parts).glob("*/shard_report.json"))
    if len(paths) != SHARDS:
        raise ValueError("missing follow-up shard")
    seen, rows = set(), []
    for path in paths:
        report = json.loads(path.read_text())
        shard = report["shard"]
        if (shard in seen or not 0 <= shard < SHARDS or
                report.get("positions") != PER_SHARD or
                report.get("followup_sha256") != FOLLOWUP_SHA or
                report.get("model_sha256") != MODEL_SHA or
                report.get("training_executed") is not False):
            raise ValueError("duplicate or invalid follow-up shard")
        seen.add(shard)
        raw = (path.parent / "followup_rows.jsonl").read_bytes()
        if sha(raw) != report["rows_sha256"]:
            raise ValueError("follow-up shard SHA mismatch")
        sub = read_rows(raw)
        if [r["followup_index"] for r in sub] != list(
                range(shard * PER_SHARD, (shard + 1) * PER_SHARD)):
            raise ValueError("follow-up shard index mismatch")
        for row in sub:
            src = selected[row["followup_index"]]
            if (row["parent_id"] != src["parent_id"] or
                    row["candidate_id"] != src["candidate_id"] or
                    row["old_long_move"] != src["long_think"]["move"] or
                    row["source_played_move"] != src["source_played_move"]):
                raise ValueError("follow-up result identity mismatch")
            board = history_board(src)
            for key in ("repeat20", "independent30"):
                if shogi.Move.from_usi(row[key]["move"]) not in board.legal_moves:
                    raise ValueError("illegal reanalysis move")
            for key, candidate in (("baseline_opponent_reply5", src["baseline"]["move"]),
                                   ("old_long_opponent_reply5", src["long_think"]["move"])):
                board.push(shogi.Move.from_usi(candidate))
                try:
                    observed = row[key]
                    if observed["terminal"] == "checkmate_after_candidate":
                        if not board.is_checkmate() or observed["reply"] is not None:
                            raise ValueError("invalid terminal reply")
                    elif observed["terminal"] == "nonterminal":
                        if (board.is_checkmate() or
                                shogi.Move.from_usi(observed["reply"]["move"])
                                not in board.legal_moves):
                            raise ValueError("illegal opponent reply")
                    else:
                        raise ValueError("unknown opponent reply state")
                finally:
                    board.pop()
            if (row["repeat_matches_old"] !=
                    (row["repeat20"]["move"] == row["old_long_move"]) or
                    row["independent_matches_old"] !=
                    (row["independent30"]["move"] == row["old_long_move"]) or
                    row["repeat_matches_independent"] !=
                    (row["repeat20"]["move"] == row["independent30"]["move"])):
                raise ValueError("inconsistent stability flags")
        rows.extend(sub)
    rows.sort(key=lambda r: r["followup_index"])
    if [r["followup_index"] for r in rows] != list(range(40)):
        raise ValueError("missing or duplicate follow-up results")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    raw = encoded(rows)
    (root / "all_followup_rows.jsonl").write_bytes(raw)
    summary = {
        "status": "read_only_40_reanalysis_completed_not_training",
        "positions": 40, "shards": SHARDS, "per_shard": PER_SHARD,
        "followup_sha256": FOLLOWUP_SHA, "results_sha256": sha(raw),
        "model_sha256": MODEL_SHA,
        "repeat_matches_old": sum(r["repeat_matches_old"] for r in rows),
        "independent_matches_old": sum(r["independent_matches_old"] for r in rows),
        "repeat_matches_independent": sum(r["repeat_matches_independent"] for r in rows),
        "median_independent_ms": statistics.median(
            r["independent30"]["elapsed_ms"] for r in rows),
        "training_executed": False, "model_promoted": False,
        "caution": "Search agreement and opponent replies are not teacher labels or proof of stronger play.",
    }
    (root / "followup_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("check", "prepare", "shard", "aggregate"),
                   required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--out")
    p.add_argument("--input")
    p.add_argument("--model")
    p.add_argument("--parts")
    p.add_argument("--shard", type=int)
    args = p.parse_args()
    enabled = validate_request(json.loads(Path(args.request).read_text()))
    if args.mode == "check":
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write("enabled=" + str(enabled).lower() + "\n")
        print("G8 40-position follow-up enabled:", enabled)
        return
    if not enabled:
        raise ValueError("research request disabled")
    if args.mode == "prepare":
        result = prepare(args.out)
    elif args.mode == "shard":
        result = run_shard(args.input, args.model, args.shard, args.out)
    else:
        result = aggregate(args.input, args.parts, args.out)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
