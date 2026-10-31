"""Replay old KUMOJI losses to diagnose FUKAJI tactical failure, never copy moves as labels.

Source is the authenticated 100-game paired archive from Actions 41177748192.
All 100 games are legally replayed and checked against the archived summary.
Focused last-turn checks are descriptive; a safe reply only refutes mate-in-one,
not longer forced mate. This does not fit or promote any playing model.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import shogi

from .kumoji_match import audit_game

RUN_ID = 41177748192
GAMES_SHA = "c0874b1b7cb96464385e76544a4fac69454ac3c9ebf681127bbca20c909e019f"
GAME_COUNT = 100
PAIRS = 50
# Every fifth pair is sampled for exhaustive immediate-mate alternatives.
# The sampling rule is fixed without looking at the tactical results.
PROBE_PAIRS = tuple(range(0, PAIRS, 5))


def snapshot(board):
    return (board.sfen(), tuple(m.usi() for m in board.move_stack),
            board.zobrist_hash(),
            tuple(sorted((k, v) for k, v in board.transpositions.items() if v)))


def has_mate_in_one(board):
    """Exact legal reply enumeration for an immediate checkmate."""
    before = snapshot(board)
    for move in tuple(board.legal_moves):
        board.push(move)
        try:
            if board.is_checkmate():
                return True
        finally:
            board.pop()
    if snapshot(board) != before:
        raise ValueError("mate probe damaged position/history")
    return False


def last_defense_probe(row):
    """Reconstruct final FUKAJI root and test each legal move against 1-ply mate."""
    all_moves = row["opening"] + row["moves"]
    if len(all_moves) < 2 or row["move_records"][-1]["engine"] != "B":
        raise ValueError("unexpected game finish")
    board = shogi.Board()
    for word in all_moves[:-2]:
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal earlier game move")
        board.push(move)
    before = snapshot(board)
    original = all_moves[-2]
    if (board.turn == shogi.BLACK) != row["a_black"]:
        raise ValueError("last defense was not FUKAJI")
    choices = tuple(board.legal_moves)
    safe, immediate_mate_choices = [], []
    for move in choices:
        board.push(move)
        try:
            if has_mate_in_one(board):
                immediate_mate_choices.append(move.usi())
            else:
                safe.append(move.usi())
        finally:
            board.pop()
    if snapshot(board) != before:
        raise ValueError("defense audit changed board/history")
    if original not in immediate_mate_choices:
        raise ValueError("recorded last FUKAJI move was not 1-ply mated")
    return {"game_id": row["game_id"], "pair": row["pair"],
            "chosen": original, "total_legal": len(choices),
            "moves_allow_immediate_mate": len(immediate_mate_choices),
            "moves_avoid_immediate_mate": len(safe),
            "example_legal_escape": safe[0] if safe else None,
            "depth_at_last_fukaji_move": [m["search"].get("depth")
                for m in row["move_records"] if m["engine"] == "A"][-1],
            "scope": "mate_in_one_only_not_longer_tactics_or_best_play"}


def audit(games_path, summary_path):
    games_raw = Path(games_path).read_bytes()
    if hashlib.sha256(games_raw).hexdigest() != GAMES_SHA:
        raise ValueError("source 100 games SHA256 mismatch")
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if (summary.get("games") != GAME_COUNT or summary.get("opening_pairs") != PAIRS
        or summary.get("fukaji_wins") != 0 or summary.get("kumoji_wins") != 100
        or summary.get("unresolved") != 0
        or summary.get("all_games_sha256") != GAMES_SHA
        or summary.get("protocol", {}).get("fukaji_g8_sha256")
        != "18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492"):
        raise ValueError("wrong archived source/protocol")
    games = [json.loads(x) for x in games_raw.splitlines()]
    if len(games) != GAME_COUNT or {g.get("game_id") for g in games} != set(range(GAME_COUNT)):
        raise ValueError("missing/duplicate game identity")
    depths = Counter()
    kdepth = Counter()
    f_times, k_times = [], []
    f_check_evasions, k_check_evasions = 0, 0
    f_checks_delivered, k_checks_delivered = 0, 0
    f_ply, k_ply = 0, 0
    targeted = []
    outliers = []
    for row in games:
        index = row["game_id"]
        if (row["pair"] != index//2 or row["a_black"] is not (index%2 == 0)
            or len(row["opening"]) != 6 or row["winner"] != "B"
            or row["reason"] != "checkmate"
            or audit_game(row) != len(row["moves"])
            or len(row["move_records"]) != len(row["played_moves"])):
            raise ValueError("source pair/terminal/telemetry mismatch")
        board = shogi.Board()
        for word in row["opening"]:
            move = shogi.Move.from_usi(word)
            if move not in board.legal_moves:
                raise ValueError("illegal opening")
            board.push(move)
        for j, turn in enumerate(row["move_records"]):
            is_f = (board.turn == shogi.BLACK) == row["a_black"]
            if turn["engine"] != ("A" if is_f else "B"):
                raise ValueError("wrong source engine side")
            if board.is_check():
                if is_f: f_check_evasions += 1
                else: k_check_evasions += 1
            word = row["played_moves"][j]
            if word != turn["move"]:
                raise ValueError("played move telemetry disagrees")
            move = shogi.Move.from_usi(word)
            if move not in board.legal_moves:
                raise ValueError("illegal source move")
            board.push(move)
            if is_f:
                f_ply += 1
                depths[turn["search"].get("depth")] += 1
                if board.is_check(): f_checks_delivered += 1
                f_times.append(turn["search"].get("elapsed_ms"))
            else:
                k_ply += 1
                kdepth[turn["search"].get("depth")] += 1
                if board.is_check(): k_checks_delivered += 1
                k_times.append(turn["search"].get("elapsed_ms"))
        if not board.is_checkmate():
            raise ValueError("terminal mate missing")
        if row["pair"] in PROBE_PAIRS:
            targeted.append(last_defense_probe(row))
        if row["plies"] <= 50:
            outliers.append(index)
    if any(t is None or t < 0 for t in f_times + k_times):
        raise ValueError("missing invalid timing")
    if (f_ply != summary["fukaji_plays"] or k_ply != summary["kumoji_plays"]):
        raise ValueError("mover telemetry totals mismatch")
    def timing(values):
        return {"observed": len(values), "median_ms": round(statistics.median(values), 3),
                "max_ms": round(max(values), 3),
                "over_200_ms": sum(t > 200 for t in values)}
    return {
        "status": "verified_100_old_losses_tactical_development_diagnosis",
        "source_run_id": RUN_ID, "all_games_sha256": GAMES_SHA,
        "games_replayed": 100, "all_terminal_checkmates": 100,
        "source_pairs": 50, "fukaji_wins": 0, "kumoji_wins": 100,
        "fukaji_search_depth_histogram": dict(depths),
        "kumoji_search_depth_histogram": dict(kdepth),
        "fukaji_under_depth_three": sum(v for k, v in depths.items() if k < 3),
        "fukaji_time": timing(f_times), "kumoji_time": timing(k_times),
        "fukaji_moves_while_checked": f_check_evasions,
        "kumoji_moves_while_checked": k_check_evasions,
        "checks_given_by_fukaji": f_checks_delivered,
        "checks_given_by_kumoji": k_checks_delivered,
        "short_games_up_to_50_plies": len(outliers),
        "last_defense_sample_rule": "game pair indices 0,5,10,...45; both colors",
        "last_defense_probes": targeted,
        "last_defense_has_immediate_mate_escape": sum(p["moves_avoid_immediate_mate"] > 0 for p in targeted),
        "limitations": [
            "Previously seen opponent games are development diagnosis; not independent strength test.",
            "Opponent move / score / PV is never a FUKAJI training target.",
            "No official FUKAJI model, default USI, or KUMOJI engine is altered.",
            "A move avoiding immediate mate can still lose to longer tactics.",
            "Wall times and compute resources differ between engines; no equal-compute strength claim."
        ]
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--games",required=True)
    p.add_argument("--summary",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    report=audit(a.games,a.summary)
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    (out/"loss_study.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="last_defense_probes"},ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
