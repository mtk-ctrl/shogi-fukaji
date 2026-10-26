"""Read-only diagnosis of frozen fast-evaluation paired games.

Do not convert unresolved games into wins or training labels.  All moves are
replayed against python-shogi before computing descriptive tactical metrics.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import shogi

from .direct_arena import replay_check
from .engine import VALUES
from .fast_arena import summarize_pairs


SOURCE_RUN = 40378086136
MODEL_SHA = "881af10b5bc0cde591e37a9aee883213fa7dd602345b786b9b93d69c7e6ed239"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_request(c):
    if (c.get("research_only") is not True
            or c.get("source_run_id") != SOURCE_RUN
            or c.get("model_sha256") != MODEL_SHA
            or c.get("source_arena_seed") != 20271227
            or c.get("source_pairs") != 16
            or c.get("source_max_plies") != 160
            or c.get("training_allowed") is not False):
        raise ValueError("invalid frozen diagnostic protocol")
    for key in ("games_sha256", "report_sha256"):
        value = c.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(
                ch not in "0123456789abcdef" for ch in value):
            raise ValueError("invalid source digest " + key)
    return c


def material_balance(board):
    """Black-minus-white piece and hand balance; descriptive, not an evaluation."""
    score = 0
    for square in range(81):
        piece = board.piece_at(square)
        if piece and piece.piece_type != shogi.KING:
            score += VALUES[piece.piece_type] * (1 if piece.color == shogi.BLACK else -1)
    for color in (shogi.BLACK, shogi.WHITE):
        sign = 1 if color == shogi.BLACK else -1
        for kind, count in board.pieces_in_hand[color].items():
            score += sign * VALUES[kind] * count
    return score


def analyze_game(row, max_plies):
    head, moves = row["opening"], row["moves"]
    if not isinstance(head, list) or not isinstance(moves, list):
        raise ValueError("missing move list")
    if len(head) + len(moves) > max_plies:
        raise ValueError("ply limit exceeded")
    replay_check(head, moves, row)
    if row["status"] == "move_limit_unresolved" and len(head) + len(moves) != max_plies:
        raise ValueError("premature move-limit termination")
    board = shogi.Board()
    seen = Counter({board.zobrist_hash(): 1})
    captures = promotions = drops = checks = revisits = late_revisits = 0
    all_moves = head + moves
    for index, word in enumerate(all_moves):
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal source move")
        if board.piece_at(move.to_square):
            captures += 1
        promotions += int(word.endswith("+"))
        drops += int("*" in word)
        board.push(move)
        checks += int(board.is_check())
        key = board.zobrist_hash()
        if seen[key]:
            revisits += 1
            if index >= len(all_moves) - 40:
                late_revisits += 1
        seen[key] += 1
    return {
        "pair": row["pair"], "fast_black": row["fast_black"],
        "status": row["status"], "winner": row["winner"],
        "plies": len(all_moves), "captures": captures,
        "promotions": promotions, "drops": drops, "checks": checks,
        "revisited_positions": revisits, "late_40_revisits": late_revisits,
        "final_abs_material_gap": abs(material_balance(board)),
    }


def diagnose(rows, report, cfg):
    if len(rows) != 32 or report.get("games") != 32 or report.get("pairs") != 16:
        raise ValueError("unexpected game count")
    if (report.get("model_sha256") != cfg["model_sha256"]
            or report.get("arena_seed") != cfg["source_arena_seed"]
            or report.get("max_plies") != cfg["source_max_plies"]
            or report.get("movetime_ms") != 200
            or report.get("depth") != 2
            or report.get("mix_percent") != 25
            or report.get("all_pairs_alternate_color") is not True):
        raise ValueError("source provenance mismatch")
    expected = summarize_pairs(rows, 16)
    if expected != report["pair_comparison"]:
        raise ValueError("pair comparison mismatch")
    diagnostics = []
    for pair in range(16):
        first, second = rows[2 * pair:2 * pair + 2]
        if (first["pair"] != pair or second["pair"] != pair
                or first["fast_black"] is not True
                or second["fast_black"] is not False
                or first["opening"] != second["opening"]
                or len(first["opening"]) != 6):
            raise ValueError("invalid paired opening")
        diagnostics.extend([analyze_game(first, 160), analyze_game(second, 160)])
    wins = sum(r["status"] == "checkmate" and
               (r["winner"] == int(shogi.BLACK)) == r["fast_black"] for r in rows)
    losses = sum(r["status"] == "checkmate" and
                 (r["winner"] == int(shogi.BLACK)) != r["fast_black"] for r in rows)
    if (wins, losses, 32 - wins - losses) != (
            report["fast_wins"], report["baseline_wins"], report["unresolved"]):
        raise ValueError("reported outcomes mismatch")
    by_status = {}
    for group, selected in (
            ("checkmate", [r for r in diagnostics if r["status"] == "checkmate"]),
            ("unresolved", [r for r in diagnostics if r["status"] != "checkmate"])):
        by_status[group] = {
            "games": len(selected),
            "median_plies": statistics.median(r["plies"] for r in selected) if selected else None,
            "median_captures": statistics.median(r["captures"] for r in selected) if selected else None,
            "median_checks": statistics.median(r["checks"] for r in selected) if selected else None,
            "games_with_position_revisits": sum(r["revisited_positions"] > 0 for r in selected),
            "games_with_late_revisits": sum(r["late_40_revisits"] > 0 for r in selected),
            "median_final_abs_material_gap": statistics.median(
                r["final_abs_material_gap"] for r in selected) if selected else None,
        }
    return {"status": "read_only_diagnosis_no_promotion",
            "source_run_id": SOURCE_RUN,
            "model_sha256": MODEL_SHA,
            "outcomes": {"fast": wins, "baseline": losses, "unresolved": 32 - wins - losses},
            "groups": by_status, "per_game": diagnostics,
            "note": "Descriptive diagnostics only. Unresolved outcomes are not wins, draws, or teacher labels."}


def execute(request_path, games_path, report_path, out_dir):
    cfg = validate_request(json.loads(Path(request_path).read_text(encoding="utf-8")))
    if sha256(games_path) != cfg["games_sha256"] or sha256(report_path) != cfg["report_sha256"]:
        raise ValueError("immutable source digest mismatch")
    rows = [json.loads(line) for line in Path(games_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("games_sha256") != sha256(games_path):
        raise ValueError("report does not bind games")
    result = diagnose(rows, report, cfg)
    result["games_sha256"] = sha256(games_path)
    result["report_sha256"] = sha256(report_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_game"}, ensure_ascii=False, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ("request", "games", "report", "out"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    execute(a.request, a.games, a.report, a.out)


if __name__ == "__main__":
    main()
