"""Outcome-blind diagnostic of root repetition filtering on frozen D8 games.

No new games, learning, engine changes or automatic adoption. Selected alternative
moves are temporary diagnostic observations, not played-move training records.
"""
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

from .d8_depth_extension_arena import G8_SHA, board_state
from .d9_clock_probe import load_model
from .d9_tail_probe import D8_RUN, D8_ARTIFACT, D8_NAME, D8_ZIP_SHA, D8_GAMES_SHA
from .direct_arena import replay_check
from .engine import select_move
from .g8_artifact_selection import artifact_bytes
from .relative_value import RelativeValue

SHARDS = 20
ORDER = (True, False, False, True)
REQUEST = ".github/fukaji-d10-repetition-request.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def expected_request(enabled=False):
    return {"enabled": enabled, "research_only": True, "training_allowed": False,
            "model_adoption_allowed": False, "d8_run": D8_RUN,
            "d8_artifact": D8_ARTIFACT, "d8_zip_sha256": D8_ZIP_SHA,
            "d8_games_sha256": D8_GAMES_SHA, "model_sha256": G8_SHA,
            "games": 240, "shards": SHARDS, "first_affected_position_per_game": 1,
            "order": list(ORDER), "max_depth": 2, "complete_depth": True,
            "mix_percent": 25, "fast_relative": True, "root_alpha_beta": True,
            "fast_move_list": True, "time_check_each_node": False,
            "max_single_call_ms": 5000}


def validate_request(cfg):
    for enabled in (False, True):
        if json.dumps(cfg, sort_keys=True) == json.dumps(expected_request(enabled), sort_keys=True):
            return enabled
    raise ValueError("D10 request differs from fixed diagnostic protocol")


def game_ids(shard):
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("invalid D10 shard")
    return list(range(shard, 240, SHARDS))


def root_partition(board):
    """Reproduce only the current root filter using the complete legal set."""
    before = board_state(board)
    counts, mates = {}, []
    for move in board.legal_moves:
        board.push(move)
        try:
            mate = board.is_checkmate()
            counts[move.usi()] = -1 if mate else board.transpositions[board.zobrist_hash()]
            if mate:
                mates.append(move.usi())
        finally:
            board.pop()
    if board_state(board) != before:
        raise ValueError("partition changed board/history")
    least = min(counts.values()) if counts else None
    kept = sorted(word for word, count in counts.items() if count == least)
    removed = sorted(set(counts) - set(kept))
    return {"kept": kept, "removed": removed, "counts": counts, "mate_moves": mates}


def immediate_mate_replies(board, word):
    """Full legal enumeration; independent of the search's numeric score."""
    before = board_state(board)
    move = shogi.Move.from_usi(word)
    if move not in board.legal_moves:
        raise ValueError("illegal diagnostic choice")
    board.push(move)
    try:
        count = 0
        for reply in board.legal_moves:
            board.push(reply)
            try:
                count += int(board.is_checkmate())
            finally:
                board.pop()
    finally:
        board.pop()
    if board_state(board) != before:
        raise ValueError("mate-reply audit changed board/history")
    return count


def compare_position(board, model, partition, select=None):
    select = select or select_move
    before = board_state(board)
    calls = []
    for avoid in ORDER:
        telemetry = {}
        started = time.perf_counter()
        word, depth, nodes = select(
            board, max_depth=2, complete_depth=True, model=model,
            avoid_repetition=avoid, fast_relative=True, root_alpha_beta=True,
            fast_move_list=True, time_check_each_node=False, telemetry=telemetry)
        elapsed = (time.perf_counter() - started) * 1000
        if board_state(board) != before:
            raise ValueError("D10 search changed board/history")
        if not math.isfinite(elapsed) or not 0 <= elapsed <= 5000 or depth != 2:
            raise ValueError(
                f"D10 incomplete depth or fault time ceiling: avoid_repetition={avoid} "
                f"depth={depth} elapsed_ms={elapsed!r} nodes={nodes} "
                f"root_preparation_ms={telemetry.get('root_preparation_ms')!r}")
        if word == "resign" or shogi.Move.from_usi(word) not in board.legal_moves:
            raise ValueError("D10 illegal search move")
        if avoid and word not in partition["kept"]:
            raise ValueError("diagnostic filter differs from engine root filter")
        calls.append({"avoid_repetition": avoid, "move": word, "depth": depth,
                      "nodes": nodes, "elapsed_ms": elapsed,
                      "root_preparation_ms": telemetry.get("root_preparation_ms"),
                      "chosen_was_filtered": word in partition["removed"]})
    for avoid in (True, False):
        if len({x["move"] for x in calls if x["avoid_repetition"] == avoid}) != 1:
            raise ValueError("fixed-depth repeat changed its chosen move")
    choices = {str(avoid): next(x["move"] for x in calls if x["avoid_repetition"] == avoid)
               for avoid in (True, False)}
    mate_replies = {word: immediate_mate_replies(board, word) for word in set(choices.values())}
    return {"in_check": board.is_check(), "legal_count": len(partition["counts"]),
            "kept_count": len(partition["kept"]), "removed_count": len(partition["removed"]),
            "calls": calls, "changed_move": choices["True"] != choices["False"],
            "filtered_choice_mate_replies": mate_replies[choices["True"]],
            "unfiltered_choice_mate_replies": mate_replies[choices["False"]]}


def inspect_game(game, model):
    board = shogi.Board()
    for word in game["opening"]:
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal D8 opening")
        board.push(move)
    chosen, inspected = None, 0
    for ply, word in enumerate(game["moves"], 1):
        # Select before consulting the recorded outcome or selected move.
        if chosen is None:
            part = root_partition(board)
            inspected += 1
            if part["removed"]:
                try:
                    comparison = compare_position(board, model, part)
                except ValueError as exc:
                    raise ValueError(
                        f"D10 game_id={game['game_id']} played_ply={ply} "
                        f"legal={len(part['counts'])} kept={len(part['kept'])}: {exc}") from exc
                chosen = {"game_id": game["game_id"], "played_ply": ply, **comparison}
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal D8 played move")
        board.push(move)
    replay_check(game["opening"], game["moves"], game)
    return {"game_id": game["game_id"], "played_moves": len(game["moves"]),
            "inspected_prefix_positions": inspected, "first_affected": chosen,
            "full_game_legal_replay": True}


def prepare(out):
    raw = artifact_bytes(D8_ARTIFACT, D8_NAME, D8_RUN, D8_ZIP_SHA, os.environ["GH_TOKEN"])
    with ZipFile(io.BytesIO(raw)) as archive:
        games_raw = archive.read("all_games.jsonl")
    if sha(games_raw) != D8_GAMES_SHA:
        raise ValueError("D8 game bytes changed")
    games = [json.loads(x) for x in games_raw.splitlines()]
    if sorted(g["game_id"] for g in games) != list(range(240)):
        raise ValueError("D8 game identity/count changed")
    model_raw = load_model()
    if sha(model_raw) != G8_SHA:
        raise ValueError("G8 model changed")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "g8.npz").write_bytes(model_raw)
    (root / "games.jsonl").write_bytes(games_raw)
    receipt = {"protocol": expected_request(True), "code_commit": os.getenv("GITHUB_SHA"),
               "engine_sha256": sha(Path("fukaji/engine.py").read_bytes()),
               "audit_sha256": sha(Path(__file__).read_bytes())}
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def verify_input(input_dir):
    root = Path(input_dir)
    receipt = json.loads((root / "receipt.json").read_text())
    if (receipt.get("protocol") != expected_request(True) or
            receipt.get("code_commit") != os.getenv("GITHUB_SHA") or
            receipt.get("engine_sha256") != sha(Path("fukaji/engine.py").read_bytes()) or
            receipt.get("audit_sha256") != sha(Path(__file__).read_bytes()) or
            sha((root / "g8.npz").read_bytes()) != G8_SHA or
            sha((root / "games.jsonl").read_bytes()) != D8_GAMES_SHA):
        raise ValueError("D10 input/source identity mismatch")
    return receipt


def run_shard(input_dir, out, shard):
    ids = game_ids(shard)
    verify_input(input_dir)
    root = Path(input_dir)
    model = RelativeValue(str(root / "g8.npz"), mix_percent=25)
    games = {g["game_id"]: g for g in map(json.loads, (root / "games.jsonl").read_text().splitlines())}
    rows = []
    for game_id in ids:
        rows.append(inspect_game(games[game_id], model))
        print("completed game replay", game_id, "cases", int(rows[-1]["first_affected"] is not None), flush=True)
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(rows, indent=2) + "\n").encode()
    (target / "rows.json").write_bytes(raw)
    (target / "report.json").write_text(json.dumps({"shard": shard, "game_ids": ids,
        "rows_sha256": sha(raw), "receipt_sha256": sha((root / "receipt.json").read_bytes())}, indent=2) + "\n")


def summarize(rows):
    cases = [x["first_affected"] for x in rows if x["first_affected"] is not None]
    calls = [c for case in cases for c in case["calls"]]
    timing = {}
    for avoid in (True, False):
        values = [c["elapsed_ms"] for c in calls if c["avoid_repetition"] == avoid]
        timing["filtered" if avoid else "unfiltered"] = {
            "calls": len(values), "mean_ms": statistics.mean(values) if values else None,
            "median_ms": statistics.median(values) if values else None,
            "max_ms": max(values) if values else None}
    return {"games_replayed": len(rows), "played_moves_replayed": sum(r["played_moves"] for r in rows),
            "prefix_positions_examined": sum(r["inspected_prefix_positions"] for r in rows),
            "games_with_filtering": len(cases), "selected_in_check": sum(c["in_check"] for c in cases),
            "changed_choices": sum(c["changed_move"] for c in cases),
            "unfiltered_selected_excluded_move": sum(c["calls"][1]["chosen_was_filtered"] for c in cases),
            "filtered_only_allows_mate_in_one": sum(c["filtered_choice_mate_replies"] > 0 and
                c["unfiltered_choice_mate_replies"] == 0 for c in cases),
            "unfiltered_only_allows_mate_in_one": sum(c["unfiltered_choice_mate_replies"] > 0 and
                c["filtered_choice_mate_replies"] == 0 for c in cases),
            "timing": timing, "strength_proven": False, "training_performed": False,
            "limitations": ["D8 cohort is reused for diagnosis, not an independent strength test.",
                "Only the first affected position in each game is compared; later positions are not screened.",
                "Fixed depth 2 isolates root filtering, not equal-time playing strength.",
                "Mate audit checks one opponent move only; absence is not proof of safety.",
                "Search-internal repetition scores remain unchanged in both arms."]}


def aggregate(input_dir, parts, out):
    verify_input(input_dir)
    receipt_sha = sha((Path(input_dir) / "receipt.json").read_bytes())
    rows, seen = [], set()
    reports = list(Path(parts).glob("*/report.json"))
    if len(reports) != SHARDS:
        raise ValueError("D10 requires twenty complete shards")
    for path in reports:
        report = json.loads(path.read_text())
        shard = report["shard"]
        ids = game_ids(shard)
        raw = (path.parent / "rows.json").read_bytes()
        if (shard in seen or report["game_ids"] != ids or report["rows_sha256"] != sha(raw)
                or report["receipt_sha256"] != receipt_sha):
            raise ValueError("D10 shard identity/hash mismatch")
        part = json.loads(raw)
        if [r["game_id"] for r in part] != ids or not all(r["full_game_legal_replay"] is True for r in part):
            raise ValueError("D10 missing or unverified games")
        for row in part:
            case = row["first_affected"]
            if case is not None and (case["game_id"] != row["game_id"] or
                    [c["avoid_repetition"] for c in case["calls"]] != list(ORDER)):
                raise ValueError("D10 case/order mismatch")
        seen.add(shard)
        rows.extend(part)
    rows.sort(key=lambda x: x["game_id"])
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(rows, indent=2) + "\n").encode()
    (root / "observations.json").write_bytes(raw)
    summary = {"protocol": expected_request(True), "code_commit": os.getenv("GITHUB_SHA"),
               "observations_sha256": sha(raw), **summarize(rows)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("check", "prepare", "audit", "aggregate"), required=True)
    p.add_argument("--request", default=REQUEST)
    p.add_argument("--input")
    p.add_argument("--parts")
    p.add_argument("--shard", type=int)
    p.add_argument("--out")
    a = p.parse_args()
    enabled = validate_request(json.loads(Path(a.request).read_text()))
    if a.mode == "check":
        if os.getenv("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write("enabled=" + str(enabled).lower() + "\n")
        print("D10 enabled:", enabled)
        return
    if not enabled or os.getenv("GITHUB_REPOSITORY") != "mtk-ctrl/shogi-fukaji":
        raise ValueError("D10 disabled or wrong repository")
    if a.mode == "prepare":
        prepare(a.out)
    elif a.mode == "audit":
        run_shard(a.input, a.out, a.shard)
    else:
        aggregate(a.input, a.parts, a.out)


if __name__ == "__main__":
    main()
