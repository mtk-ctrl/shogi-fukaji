"""D4 read-only probe on frozen original G8/G7 paired-game histories.

Research only; no new games, no training labels, no USI wiring/adoption.
The 24 source parent pairs are disjoint from D4 artificial development cases.
Once results are inspected this is no longer a fresh holdout for tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import statistics
import time
from pathlib import Path
from zipfile import ZipFile

import shogi

from .d4_mate_assist import select_with_mate_assist
from .d4_three_only_probe import snapshot
from .engine import select_move
from .g8_artifact_selection import artifact_bytes
from .g8_paired_arena import G8_RUN, G8_ID, G8_NAME, G8_ZIP_SHA, G8_SHA
from .mate_oracle import ProofIncomplete, prove_mate_up_to_three
from .relative_value import RelativeValue

SOURCE_RUN = 40637486947
SOURCE_ID = 12793918544
SOURCE_NAME = "fukaji-g8-paired-aggregate-40637486947"
SOURCE_ZIP_SHA = "a3a8bee0a63478b296c7362cc33adb33c31d37407ae3b28527b0e1c9688c3b14"
SOURCE_GAMES_SHA = "60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf"
SEED = "D4-20261030-frozen-real-games-v1"
STAGES = ("terminal_minus_1", "terminal_minus_3", "terminal_minus_2", "midgame_ply_40")
PER_STAGE = 6
SAMPLES = PER_STAGE * len(STAGES)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def expected_request(enabled=False):
    return {"enabled": enabled, "research_only": True, "adoption_allowed": False,
            "training_allowed": False, "source_run": SOURCE_RUN,
            "source_artifact_id": SOURCE_ID, "source_zip_sha256": SOURCE_ZIP_SHA,
            "source_games_sha256": SOURCE_GAMES_SHA, "g8_run": G8_RUN,
            "g8_artifact_id": G8_ID, "g8_zip_sha256": G8_ZIP_SHA,
            "g8_model_sha256": G8_SHA, "sample_parent_pairs": SAMPLES,
            "selection_seed": SEED, "movetime_ms": 200,
            "proof_budget_ms": 25, "search_max_depth": 4, "abba": True}


def validate_request(cfg):
    if cfg == expected_request(False):
        return False
    if cfg == expected_request(True):
        return True
    raise ValueError("D4 request differs from frozen read-only protocol")


def source_rows(source_zip):
    if digest(source_zip) != SOURCE_ZIP_SHA:
        raise ValueError("source ZIP SHA mismatch")
    with ZipFile(io.BytesIO(source_zip)) as z:
        if sorted(z.namelist()) != ["all_paired_games.jsonl", "paired_aggregate_report.json"]:
            raise ValueError("source ZIP entries mismatch")
        raw = z.read("all_paired_games.jsonl")
        report = json.loads(z.read("paired_aggregate_report.json"))
    if (digest(raw) != SOURCE_GAMES_SHA or
            report.get("games_sha256") != SOURCE_GAMES_SHA or
            report.get("games") != 240 or report.get("pairs") != 120 or
            report.get("legal_replay_verified") is not True or
            report.get("candidate_promoted") is not False):
        raise ValueError("source game SHA/metadata mismatch")
    games = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(games) != 240 or len({(r["pair"], r["g8_black"]) for r in games}) != 240:
        raise ValueError("missing or duplicate game rows")
    return games


def freeze_cases(games):
    """Choose one completed game per parent; never use played moves as labels."""
    by_pair = {}
    for row in games:
        if row["status"] == "checkmate" and len(row["moves"]) >= 48:
            by_pair.setdefault(row["pair"], []).append(row)
    order = sorted(by_pair, key=lambda pair: digest((SEED + "|" + str(pair)).encode()))
    if len(order) < SAMPLES:
        raise ValueError("not enough distinct completed parent groups")
    cases = []
    for index, pair in enumerate(order[:SAMPLES]):
        stage = STAGES[index // PER_STAGE]
        game = sorted(by_pair[pair],
                      key=lambda row: digest((SEED + str(row["g8_black"])).encode()))[0]
        full = game["opening"] + game["moves"]
        cutoff = 40 if stage == "midgame_ply_40" else len(full) - int(stage[-1])
        prefix = full[:cutoff]
        cases.append({"id": "R%02d" % index, "pair": pair,
                      "g8_black": game["g8_black"], "stage": stage,
                      "source_full_plies": len(full), "prefix_plies": len(prefix),
                      "source_row_sha256": digest(json.dumps(
                          game, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")),
                      "moves": prefix, "_full_moves": full})
    return cases


def board_from_moves(moves):
    board = shogi.Board()
    for word in moves:
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal original move: " + word)
        board.push(move)
    return board


def verify_proof(board, proof):
    """Recheck positive witnesses against complete defender reply sets."""
    before = snapshot(board)
    for first in proof["mate_in_one"]:
        move = shogi.Move.from_usi(first)
        if move not in board.legal_moves:
            raise AssertionError("illegal mate-in-one witness")
        board.push(move)
        try:
            if not board.is_checkmate():
                raise AssertionError("mate-in-one witness does not mate")
        finally:
            board.pop()
    for branch in proof["mate_in_three"]:
        first = shogi.Move.from_usi(branch["first"])
        if first not in board.legal_moves:
            raise AssertionError("illegal three-ply first")
        board.push(first)
        try:
            if not board.is_check():
                raise AssertionError("three-ply first is not check")
            defenses = {x["reply"]: x["mates"] for x in branch["defenses"]}
            if set(defenses) != {m.usi() for m in board.legal_moves} or not defenses:
                raise AssertionError("not all defender replies covered")
            for reply, finishes in defenses.items():
                board.push(shogi.Move.from_usi(reply))
                try:
                    if not finishes:
                        raise AssertionError("no mating finish")
                    for finish in finishes:
                        m = shogi.Move.from_usi(finish)
                        if m not in board.legal_moves:
                            raise AssertionError("illegal mating finish")
                        board.push(m)
                        try:
                            if not board.is_checkmate():
                                raise AssertionError("non-mating finish")
                        finally:
                            board.pop()
                finally:
                    board.pop()
        finally:
            board.pop()
    if snapshot(board) != before:
        raise AssertionError("proof witness altered board/history")


def probe_case(case, model):
    board = board_from_moves(case["moves"])
    full_board = board_from_moves(case["_full_moves"])
    if not full_board.is_checkmate():
        raise ValueError("original completed game not checkmate")
    before = snapshot(board)
    oracle_start = time.monotonic()
    try:
        oracle = prove_mate_up_to_three(
            board, max_legal_moves=5000, deadline=oracle_start + 0.100)
        verify_proof(board, oracle)
        oracle_status = ("mate_one" if oracle["mate_in_one"] else
                         "mate_three" if oracle["mate_in_three"] else "no_short_mate")
        oracle_moves = list(oracle["mate_in_one"]) or [
            b["first"] for b in oracle["mate_in_three"]]
    except ProofIncomplete:
        oracle_status, oracle_moves = "unknown", []
    if snapshot(board) != before:
        raise AssertionError("oracle changed board/history")
    observations = []
    for mode in ("off", "on", "on", "off"):
        if mode == "off":
            telemetry = {}
            move, depth, nodes = select_move(
                board, movetime_ms=200, max_depth=4, model=model,
                deadline_from_call=True, telemetry=telemetry)
            elapsed = telemetry["total_call_ms"]
            status, proof_elapsed = "disabled", 0
        else:
            result = select_with_mate_assist(
                board, movetime_ms=200, max_depth=4, model=model,
                proof_ms=25, max_legal_moves=1500)
            move, depth, nodes = (result["move"], result["search_depth"],
                                  result["nodes"])
            elapsed, status, proof_elapsed = (
                result["total_ms"], result["status"], result["proof_ms"])
        if snapshot(board) != before:
            raise AssertionError("search changed board/history")
        if move != "resign" and shogi.Move.from_usi(move) not in board.legal_moves:
            raise AssertionError("illegal chosen move")
        observations.append({"mode": mode, "move": move, "depth": depth,
                             "nodes": nodes, "total_ms": round(elapsed, 3),
                             "proof_ms": round(proof_elapsed, 3),
                             "proof_status": status, "over_200ms": elapsed > 200,
                             "oracle_match": (move in oracle_moves
                                              if oracle_status != "unknown" else None)})
    return {k: v for k, v in case.items() if k not in ("moves", "_full_moves")} | {
        "position_sfen": board.sfen(), "in_check": board.is_check(),
        "oracle_status": oracle_status, "oracle_first_moves": oracle_moves,
        "oracle_ms": round((time.monotonic() - oracle_start) * 1000, 3),
        "observations": observations}


def run(out):
    if os.environ.get("GITHUB_REPOSITORY") != "mtk-ctrl/shogi-fukaji":
        raise ValueError("wrong Actions repository")
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise ValueError("missing read-only Actions token")
    source = artifact_bytes(SOURCE_ID, SOURCE_NAME, SOURCE_RUN, SOURCE_ZIP_SHA, token)
    model_zip = artifact_bytes(G8_ID, G8_NAME, G8_RUN, G8_ZIP_SHA, token)
    with ZipFile(io.BytesIO(model_zip)) as z:
        raw_model = z.read("mover-relative-generation8-research.npz")
    if digest(raw_model) != G8_SHA:
        raise ValueError("G8 internal model SHA mismatch")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    model_file = root / "frozen-g8.npz"
    model_file.write_bytes(raw_model)
    model = RelativeValue(model_file, mix_percent=25)
    cases = freeze_cases(source_rows(source))
    results = [probe_case(case, model) for case in cases]
    data = {"scope": "D4_new_source_not_training_not_general_strength",
            "source_run": SOURCE_RUN, "source_zip_sha256": SOURCE_ZIP_SHA,
            "source_games_sha256": SOURCE_GAMES_SHA, "g8_model_sha256": G8_SHA,
            "seed": SEED, "cases": results}
    raw = (json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    (root / "frozen_realgame_results.json").write_bytes(raw)
    by_stage = {}
    for stage in STAGES:
        sub = [r for r in results if r["stage"] == stage]
        by_stage[stage] = {"cases": len(sub),
                           "oracle_status": {s: sum(r["oracle_status"] == s for r in sub)
                                             for s in ("mate_one", "mate_three",
                                                       "no_short_mate", "unknown")},
                           "off_over_200": sum(o["over_200ms"] for r in sub
                                               for o in r["observations"] if o["mode"] == "off"),
                           "on_over_200": sum(o["over_200ms"] for r in sub
                                              for o in r["observations"] if o["mode"] == "on")}
    report = {"status": "research_only_not_adopted", "cases": len(results),
              "unique_parent_pairs": len({r["pair"] for r in results}),
              "source_zip_sha256": SOURCE_ZIP_SHA, "source_games_sha256": SOURCE_GAMES_SHA,
              "model_sha256": G8_SHA, "results_sha256": digest(raw),
              "stages": by_stage, "model_promoted": False, "training_executed": False}
    (root / "frozen_realgame_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "run"), required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    enabled = validate_request(json.loads(Path(args.request).read_text()))
    if args.mode == "validate":
        path = os.environ.get("GITHUB_OUTPUT")
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write("enabled=" + str(enabled).lower() + "\n")
        print("D4 realgame research request valid; enabled:", enabled)
        return
    if not enabled or not args.out:
        raise ValueError("D4 realgame run is disabled or missing output")
    run(args.out)


if __name__ == "__main__":
    main()
