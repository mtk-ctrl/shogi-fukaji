"""Research-only G8 depth-3 clock-check 128 versus each-node paired comparison.

Both arms use FastMoveList ON, the same frozen G8 model and whole-call 200ms.
Only the clock-check frequency differs. No learning or automatic adoption.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from zipfile import ZipFile

import shogi

from .arena import opening
from .direct_arena import replay_check
from .engine import select_move
from .g8_artifact_selection import artifact_bytes
from .generation8_train import safe_read
from .relative_value import RelativeValue

SOURCE_RUN, SOURCE_ID = 40636602223, 12793956048
SOURCE_NAME = "fukaji-generation8-research-40636602223"
SOURCE_ZIP_SHA = "9b5c004446f4b93a529d52fa96243ccbd5ac434373fe1ff69191bee9c8e6306e"
G8_SHA = "18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492"
SHARDS, MAX_PLIES = 20, 240
STAGES = {"screen": (20, 202650300), "final": (120, 202651300)}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def expected_request(enabled=False):
    return {"enabled": enabled, "research_only": True,
            "model_adoption_allowed": False, "training_allowed": False,
            "source_run": SOURCE_RUN, "source_artifact_id": SOURCE_ID,
            "source_zip_sha256": SOURCE_ZIP_SHA, "model_sha256": G8_SHA,
            "screen_pairs": 20, "screen_seed": 202650300,
            "final_pairs": 120, "final_seed": 202651300,
            "shards": SHARDS, "opening_plies": 6, "max_plies": MAX_PLIES,
            "movetime_ms": 200, "baseline_max_depth": 3, "candidate_max_depth": 3,
            "baseline_time_check_each_node": False, "candidate_time_check_each_node": True,
            "deadline_from_call": True,
            "mix_percent": 25, "fast_relative": True, "root_alpha_beta": True,
            "baseline_fast_move_list": True, "candidate_fast_move_list": True,
            "max_single_call_ms": 1000,
            "mean_time_growth_within_shared_budget_expected": True,
            "screen_obvious_regression_p": 0.01}


def validate_request(cfg):
    for enabled in (False, True):
        if json.dumps(cfg, sort_keys=True) == json.dumps(expected_request(enabled), sort_keys=True):
            return enabled
    raise ValueError("D9 differs from preregistered frozen protocol")


def pair_indices(stage, shard):
    if stage not in STAGES or type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("invalid stage/shard")
    return list(range(shard, STAGES[stage][0], SHARDS))


def board_state(board):
    return (board.sfen(), len(board.move_stack), board.zobrist_hash(),
            tuple(sorted((k, v) for k, v in board.transpositions.items() if v)))


def play_game(head, candidate_black, model, max_plies=MAX_PLIES, start_sfen=None):
    if type(candidate_black) is not bool:
        raise ValueError("candidate color must be boolean")
    board = shogi.Board(start_sfen) if start_sfen else shogi.Board()
    for word in head:
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal opening")
        board.push(move)
    moves, played = [], []
    status, winner = "move_limit_unresolved", None
    for _ in range(max_plies - len(head)):
        if board.is_checkmate():
            status, winner = "checkmate", int(not board.turn)
            break
        if board.is_fourfold_repetition():
            status = "repetition_unresolved"
            break
        candidate = (board.turn == shogi.BLACK) == candidate_black
        before = board_state(board)
        telemetry = {}
        began = time.perf_counter()
        word, depth, nodes = select_move(
            board, movetime_ms=200, max_depth=3, model=model,
            fast_relative=True, root_alpha_beta=True,
            fast_move_list=True, deadline_from_call=True,
            time_check_each_node=candidate, telemetry=telemetry)
        elapsed = (time.perf_counter() - began) * 1000
        if board_state(board) != before:
            raise ValueError("search did not restore board/history")
        if not math.isfinite(elapsed) or not 0 <= elapsed <= 1000:
            raise ValueError("call exceeded frozen 1000ms fault ceiling")
        if word == "resign":
            status = "no_move_unresolved"
            break
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal selected move")
        played.append({"move": word, "turn": int(board.turn), "candidate": candidate,
                       "depth": int(depth), "max_depth": 3, "time_check_each_node": candidate,
                       "nodes": int(nodes),
                       "elapsed_ms": elapsed, "root_preparation_ms": telemetry.get("root_preparation_ms"),
                       "score": None, "score_status": "not_exposed_by_select_move"})
        moves.append(word)
        board.push(move)
    else:
        if board.is_checkmate():
            status, winner = "checkmate", int(not board.turn)
    return {"status": status, "winner": winner, "moves": moves, "played": played,
            "candidate_black": candidate_black}


def sign_p(candidate_pairs, baseline_pairs):
    n = candidate_pairs + baseline_pairs
    if not n:
        return None
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(candidate_pairs, baseline_pairs) + 1)) / 2**n)


def summarize(rows):
    wins = Counter()
    pairs = {}
    timing = {side: {"turns": 0, "elapsed_ms": 0.0, "nodes": 0,
                     "depth_sum": 0, "depth2_incomplete": 0, "depth3_complete": 0, "over_200ms": 0,
                     "max_elapsed_ms": 0.0, "root_preparation_ms": 0.0,
                     "root_preparation_observations": 0} for side in ("baseline", "candidate")}
    for row in rows:
        key = ("candidate" if (row["winner"] == shogi.BLACK) == row["candidate_black"]
               else "baseline") if row["status"] == "checkmate" else "unresolved"
        wins[key] += 1
        pairs.setdefault(row["pair"], Counter())[key] += 1
        for turn in row["played"]:
            t = timing["candidate" if turn["candidate"] else "baseline"]
            t["turns"] += 1
            t["elapsed_ms"] += turn["elapsed_ms"]
            t["nodes"] += turn["nodes"]
            t["depth_sum"] += turn["depth"]
            t["depth2_incomplete"] += int(turn["depth"] < 2)
            t["depth3_complete"] += int(turn["depth"] >= 3)
            t["over_200ms"] += int(turn["elapsed_ms"] > 200)
            t["max_elapsed_ms"] = max(t["max_elapsed_ms"], turn["elapsed_ms"])
            prep = turn.get("root_preparation_ms")
            if prep is not None and math.isfinite(prep) and prep >= 0:
                t["root_preparation_ms"] += prep
                t["root_preparation_observations"] += 1
    pair_counts = Counter("candidate" if p["candidate"] > p["baseline"] else
                          "baseline" if p["baseline"] > p["candidate"] else "equal"
                          for p in pairs.values())
    for t in timing.values():
        t["mean_elapsed_ms"] = t["elapsed_ms"] / t["turns"] if t["turns"] else None
        t["mean_root_preparation_ms"] = (t["root_preparation_ms"] / t["root_preparation_observations"]
                                           if t["root_preparation_observations"] else None)
    p = sign_p(pair_counts["candidate"], pair_counts["baseline"])
    a, b = timing["baseline"]["mean_elapsed_ms"], timing["candidate"]["mean_elapsed_ms"]
    time_ok = a is not None and b is not None and b <= a * 1.05
    obvious_loss = pair_counts["baseline"] > pair_counts["candidate"] and p is not None and p < 0.01
    return {"games": len(rows), "candidate_wins": wins["candidate"],
            "baseline_wins": wins["baseline"], "unresolved": wins["unresolved"],
            "pair_comparison": dict(pair_counts), "paired_sign_p_two_sided": p,
            "timing": timing, "candidate_mean_time_within_5percent": time_ok,
            "screen_obvious_regression": obvious_loss,
            "screen_proceed": a is not None and b is not None and b <= a * 1.05 and not obvious_loss,
            "strength_proven": False, "model_promoted": False}


def prepare(out):
    raw = artifact_bytes(SOURCE_ID, SOURCE_NAME, SOURCE_RUN, SOURCE_ZIP_SHA, os.environ["GH_TOKEN"])
    with ZipFile(io.BytesIO(raw)) as z:
        model = safe_read(z, "mover-relative-generation8-research.npz")
        old_games = safe_read(z, "merged_verified_mates.jsonl")
    if sha(model) != G8_SHA:
        raise ValueError("frozen G8 model changed")
    old_heads = {tuple(json.loads(x)["moves"][:6]) for x in old_games.splitlines()}
    # D6's 400 evaluation-only games are excluded using seed/opening metadata,
    # without inspecting their outcomes or using them as new test positions.
    old_heads.update(tuple(opening(202610300 + i, 6)) for i in range(400))
    for seed, count in ((202620300, 20), (202621300, 120),
                        (202630300, 20), (202631300, 120), (202640900, 3)):
        old_heads.update(tuple(opening(seed + i, 6)) for i in range(count))
    seen = set()
    for count, seed in STAGES.values():
        for i in range(count):
            head = tuple(opening(seed + i, 6))
            if len(head) != 6 or head in old_heads or head in seen:
                raise ValueError("D9 opening reuses previous source/evaluation")
            seen.add(head)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "g8.npz").write_bytes(model)
    receipt = {"protocol": expected_request(True), "g8_model_sha256": G8_SHA,
               "engine_source_sha256": sha(Path("fukaji/engine.py").read_bytes()),
               "arena_source_sha256": sha(Path(__file__).read_bytes()),
               "code_commit": os.environ.get("GITHUB_SHA"),
               "opening_pairs": len(seen), "model_trained": False}
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def verify_input(input_dir):
    root = Path(input_dir)
    r = json.loads((root / "receipt.json").read_text())
    if (r.get("protocol") != expected_request(True) or
            r.get("g8_model_sha256") != G8_SHA or sha((root / "g8.npz").read_bytes()) != G8_SHA or
            r.get("engine_source_sha256") != sha(Path("fukaji/engine.py").read_bytes()) or
            r.get("arena_source_sha256") != sha(Path(__file__).read_bytes())):
        raise ValueError("D9 input identity changed")
    return r


def play_shard(input_dir, out, stage, shard):
    verify_input(input_dir)
    model = RelativeValue(str(Path(input_dir) / "g8.npz"), mix_percent=25)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "games.jsonl"
    with path.open("w") as f:
        for pair in pair_indices(stage, shard):
            head = opening(STAGES[stage][1] + pair, 6)
            for seat, candidate_black in enumerate((True, False)):
                row = play_game(head, candidate_black, model)
                replay_check(head, row["moves"], row)
                row.update({"pair": pair, "game_id": 2 * pair + seat, "stage": stage,
                            "opening": head})
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(stage, "completed game", row["game_id"], row["status"], flush=True)
    report = {"protocol": expected_request(True), "stage": stage, "shard": shard,
              "games": 2 * len(pair_indices(stage, shard)), "games_sha256": sha(path.read_bytes()),
              "input_receipt_sha256": sha((Path(input_dir) / "receipt.json").read_bytes())}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def aggregate(input_dir, parts, out, stage):
    verify_input(input_dir)
    rows, seen = [], set()
    reports = list(Path(parts).glob("*/report.json"))
    if len(reports) != SHARDS:
        raise ValueError("expected twenty completed D9 shards")
    receipt_sha = sha((Path(input_dir) / "receipt.json").read_bytes())
    for path in reports:
        r = json.loads(path.read_text())
        shard = r.get("shard")
        if (type(shard) is not int or not 0 <= shard < SHARDS or shard in seen or
                r.get("stage") != stage or r.get("protocol") != expected_request(True) or
                r.get("input_receipt_sha256") != receipt_sha):
            raise ValueError("D9 shard identity changed")
        seen.add(shard)
        games_path = path.parent / "games.jsonl"
        if sha(games_path.read_bytes()) != r.get("games_sha256"):
            raise ValueError("D9 shard hash mismatch")
        part = [json.loads(x) for x in games_path.read_text().splitlines()]
        expected_ids = sorted(2 * p + seat for p in pair_indices(stage, shard) for seat in (0, 1))
        if sorted(g.get("game_id") for g in part) != expected_ids or r.get("games") != len(expected_ids):
            raise ValueError("missing/duplicate D9 games")
        for row in part:
            pair, seat = divmod(row["game_id"], 2)
            head = opening(STAGES[stage][1] + pair, 6)
            if (row.get("pair") != pair or row.get("stage") != stage or
                    row.get("opening") != head or type(row.get("candidate_black")) is not bool or
                    row["candidate_black"] != (seat == 0)):
                raise ValueError("D9 color-swapped opening mismatch")
            replay_check(head, row["moves"], row)
            if len(head) + len(row["moves"]) > MAX_PLIES or len(row["played"]) != len(row["moves"]):
                raise ValueError("D9 move/telemetry count mismatch")
            for i, turn in enumerate(row["played"]):
                color = (len(head) + i) % 2
                elapsed = turn["elapsed_ms"]
                if (turn["move"] != row["moves"][i] or turn["turn"] != color or
                        turn["candidate"] != ((color == shogi.BLACK) == row["candidate_black"]) or
                        not math.isfinite(elapsed) or not 0 <= elapsed <= 1000 or
                        type(turn["depth"]) is not int or not 0 <= turn["depth"] <= 3 or
                        turn.get("max_depth") != 3 or
                        turn.get("time_check_each_node") != turn["candidate"] or
                        type(turn["nodes"]) is not int or turn["nodes"] < 0):
                    raise ValueError("D9 played telemetry invalid")
            rows.append(row)
    rows.sort(key=lambda x: x["game_id"])
    summary = {"protocol": expected_request(True), "stage": stage, **summarize(rows),
               "input_receipt_sha256": receipt_sha,
               "limitations": ["Identical 200ms request and deadline origin; actual response times are reported.",
                               "Higher mean time is expected when using spare budget for depth 3; actual time tails must be reviewed.",
                               "Unresolved games have no winner and are not counted as draws.",
                               "No external teacher or learning. Formal adoption requires owner decision."]}
    if stage == "final":
        summary.pop("screen_proceed")
        summary.pop("screen_obvious_regression")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    all_path = root / "all_games.jsonl"
    all_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary["all_games_sha256"] = sha(all_path.read_bytes())
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if stage == "screen" and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("proceed=" + str(summary["screen_proceed"]).lower() + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("check", "prepare", "play", "aggregate"), required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--input")
    p.add_argument("--parts")
    p.add_argument("--stage", choices=tuple(STAGES))
    p.add_argument("--shard", type=int)
    p.add_argument("--out")
    a = p.parse_args()
    enabled = validate_request(json.loads(Path(a.request).read_text()))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("enabled=" + str(enabled).lower() + "\n")
    if a.mode == "check":
        print("D9 enabled:", enabled)
        return
    if not enabled or os.environ.get("GITHUB_REPOSITORY") != "mtk-ctrl/shogi-fukaji":
        raise ValueError("D9 disabled or wrong repository")
    if a.mode == "prepare":
        prepare(a.out)
    elif a.mode == "play":
        play_shard(a.input, a.out, a.stage, a.shard)
    else:
        aggregate(a.input, a.parts, a.out, a.stage)


if __name__ == "__main__":
    main()

