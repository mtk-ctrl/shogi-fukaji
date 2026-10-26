"""Extend SHA-frozen, replay-verified 160-ply unresolved FUKAJI games to 240.

Only additional actual search moves are generated; old moves stay unchanged.
Only a replay-verified checkmate is a learning label. Research-only.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import shogi

from .direct_arena import digest, replay_check
from .engine import select_move
from .generation_input import MODEL_SHA, SOURCES, read_source, sha
from .relative_value import RelativeValue

SOURCE_RUNS = (40387257648, 40395505536)
ORIGINAL_PLIES = 160
FINAL_PLIES = 240


def validate_request(cfg):
    expected = {
        "research_only": True,
        "source_runs": list(SOURCE_RUNS),
        "original_plies": ORIGINAL_PLIES,
        "max_plies": FINAL_PLIES,
        "relative_model_sha256": MODEL_SHA,
        "movetime_ms": 200,
        "depth": 2,
        "mix_percent": 25,
        "fast_relative": True,
        "root_alpha_beta_default": False,
        "root_alpha_beta_candidate": True,
        "training_allowed": False,
        "model_adoption_allowed": False,
    }
    if cfg != expected:
        raise ValueError("unapproved continuation protocol")
    return cfg


def advance(board, candidate_black, model, remaining_plies):
    if not 0 <= remaining_plies <= FINAL_PLIES - ORIGINAL_PLIES:
        raise ValueError("invalid continuation length")
    extra = []
    stats = {side + "_" + metric: 0
             for side in ("candidate", "baseline")
             for metric in ("nodes", "turns", "depth_sum", "depth_zero",
                            "incomplete", "elapsed_ms")}
    status, winner = "move_limit_unresolved", None
    for _ in range(remaining_plies):
        if board.is_checkmate():
            status, winner = "checkmate", int(not board.turn)
            break
        if board.is_fourfold_repetition():
            status = "repetition_unresolved"
            break
        candidate = (board.turn == shogi.BLACK) == candidate_black
        side = "candidate" if candidate else "baseline"
        start = time.perf_counter()
        word, depth, nodes = select_move(
            board, movetime_ms=200, max_depth=2, model=model,
            fast_relative=True, root_alpha_beta=candidate)
        elapsed_ms = (time.perf_counter() - start) * 1000
        stats[side + "_nodes"] += nodes
        stats[side + "_turns"] += 1
        stats[side + "_depth_sum"] += depth
        stats[side + "_elapsed_ms"] += round(elapsed_ms, 3)
        stats[side + "_depth_zero"] += int(depth == 0)
        stats[side + "_incomplete"] += int(depth < 2)
        if word == "resign":
            status = "no_move_unresolved"
            break
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal continuation move")
        extra.append(word)
        board.push(move)
    else:
        if board.is_checkmate():
            status, winner = "checkmate", int(not board.turn)
        elif board.is_fourfold_repetition():
            status = "repetition_unresolved"
    return {"status": status, "winner": winner,
            "continuation_moves": extra, "continuation_stats": stats}


def extend_one(game, run_id, model):
    if (game.get("status") != "move_limit_unresolved" or
            game.get("winner") is not None or
            type(game.get("candidate_black")) is not bool):
        raise ValueError("only frozen unresolved games can be extended")
    prefix = game["opening"] + game["moves"]
    if len(prefix) != ORIGINAL_PLIES:
        raise ValueError("original game must end at 160 plies")
    board = shogi.Board()
    for word in prefix:
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal original move")
        board.push(move)
    if board.is_checkmate():
        raise ValueError("original game already checkmated")
    result = advance(board, game["candidate_black"], model,
                     FINAL_PLIES - len(prefix))
    all_moves = prefix + result["continuation_moves"]
    replay_check([], all_moves, {"status": result["status"],
                                 "winner": result["winner"]})
    if result["status"] == "move_limit_unresolved" and len(all_moves) != FINAL_PLIES:
        raise ValueError("short move-limit continuation")
    parent_id = sha(
        ("fukaji-paired-v1|" + str(run_id) + "|" + str(game["pair"]) + "|" +
         " ".join(game["opening"])).encode())
    return {
        "source_run": run_id,
        "source_games_sha256": SOURCES[run_id]["games_sha256"],
        "parent_id": parent_id,
        "pair": game["pair"],
        "candidate_black": game["candidate_black"],
        "opening": game["opening"],
        "prefix_moves": game["moves"],
        "continuation_moves": result["continuation_moves"],
        "status": result["status"],
        "winner": result["winner"],
        "total_plies": len(all_moves),
        "continuation_stats": result["continuation_stats"],
    }


def execute(request, source_a, report_a, source_b, report_b, model_path, out):
    validate_request(json.loads(Path(request).read_text(encoding="utf-8")))
    if digest(model_path) != MODEL_SHA:
        raise ValueError("frozen relative model SHA mismatch")
    model = RelativeValue(model_path, mix_percent=25)
    sources = [(SOURCE_RUNS[0], source_a, report_a),
               (SOURCE_RUNS[1], source_b, report_b)]
    extended = []
    counts = {}
    for run_id, games_path, report_path in sources:
        _, source_counts = read_source(run_id, games_path, report_path)
        if source_counts.get("unresolved") != 16:
            raise ValueError("expected 16 unresolved games per source")
        rows = [json.loads(line) for line in Path(games_path).read_text(
            encoding="utf-8").splitlines() if line.strip()]
        unresolved = [r for r in rows if r["status"] == "move_limit_unresolved"]
        if len(unresolved) != 16:
            raise ValueError("source unresolved count mismatch")
        counts[str(run_id)] = source_counts
        for row in unresolved:
            extended.append(extend_one(row, run_id, model))
    if len(extended) != 32:
        raise ValueError("wrong continuation count")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    games_path = root / "continuations.jsonl"
    games_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                  for r in extended), encoding="utf-8")
    mates, seen = [], set()
    for row in extended:
        if row["status"] != "checkmate":
            continue
        moves = row["opening"] + row["prefix_moves"] + row["continuation_moves"]
        item = {"start_sfen": shogi.Board().sfen(), "moves": moves,
                "winner": row["winner"], "source": "selfplay",
                "parent_id": row["parent_id"]}
        key = sha((item["start_sfen"] + "|" + " ".join(moves)).encode())
        if key not in seen:
            seen.add(key)
            mates.append(item)
    mate_path = root / "new_verified_mates.jsonl"
    mate_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                 for r in mates), encoding="utf-8")
    from .learning import load_games
    if len(load_games(mate_path)) != len(mates):
        raise ValueError("new mate labels failed learner verification")
    outcomes = Counter(r["status"] for r in extended)
    manifest = {
        "status": "research_only_verified_continuation_no_training_no_adoption",
        "source_runs": list(SOURCE_RUNS),
        "source_game_sha256": [SOURCES[r]["games_sha256"] for r in SOURCE_RUNS],
        "source_counts": counts,
        "frozen_model_sha256": digest(model_path),
        "original_plies": ORIGINAL_PLIES, "max_plies": FINAL_PLIES,
        "movetime_ms": 200, "depth": 2, "mix_percent": 25,
        "fast_relative": True,
        "root_alpha_beta_candidate": True,
        "root_alpha_beta_default": False,
        "extended_games": len(extended),
        "outcomes": dict(outcomes),
        "new_verified_mates": len(mates),
        "continuations_sha256": digest(games_path),
        "new_mates_sha256": digest(mate_path),
        "note": "No win labels for unresolved/repetition. Parent group retained. No promotion.",
    }
    (root / "continuation_report.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser()
    for arg in ("request", "source-a", "report-a", "source-b",
                "report-b", "model", "out"):
        p.add_argument("--" + arg, required=True)
    a = p.parse_args()
    execute(a.request, a.source_a, a.report_a, a.source_b,
            a.report_b, a.model, a.out)


if __name__ == "__main__":
    main()
