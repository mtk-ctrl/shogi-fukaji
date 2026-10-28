"""Read-only, SHA-pinned G8 alpha-beta vs UCT 200ms position probe.

Uses positions reached in the independent G8/G7 240-game arena, not training
labels. This measures search throughput and move disagreement, NOT strength.
Neither engine nor the G8 model is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import shogi

from .direct_arena import replay_check
from .engine import evaluate, select_move
from .g8_paired_arena import G8_SHA
from .mcts_research import select_mcts
from .relative_value import RelativeValue

REQUEST_FILE = '.github/fukaji-g8-alpha-uct-probe-request.json'
EXPECTED_REQUEST = {
    'enabled': False, 'research_only': True, 'model_adoption_allowed': False,
    'source_run': 40637486947, 'g8_model_run': 40636602223,
    'positions': 12, 'movetime_ms': 200, 'runner_jobs': 1,
}
TARGET_PLIES = (12, 50, 110)
POSITIONS_PER_STAGE = 4
MODEL_MIX = 25
MOVE_TIME_MS = 200


def validate_request(path):
    request = json.loads(Path(path).read_text(encoding='utf-8'))
    disabled = EXPECTED_REQUEST
    enabled = dict(EXPECTED_REQUEST, enabled=True)
    if request not in (disabled, enabled):
        raise ValueError('unapproved G8 alpha/UCT probe request')
    is_enabled = request['enabled']
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as handle:
            handle.write('enabled=' + str(is_enabled).lower() + '\\n')
    return is_enabled


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def load_source(games_path, report_path):
    raw = Path(games_path).read_bytes()
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if (report.get("status") != "research_only_no_adoption"
            or report.get("games") != 240 or report.get("pairs") != 120
            or report.get("g8_model_sha256") != G8_SHA
            or report.get("games_sha256") != sha256(raw)
            or report.get("legal_replay_verified") is not True):
        raise ValueError("frozen comparison provenance mismatch")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 240:
        raise ValueError("incomplete 240-game source")
    for pair in range(120):
        a, b = rows[2 * pair:2 * pair + 2]
        if (a.get("pair") != pair or b.get("pair") != pair
                or a.get("g8_black") is not True
                or b.get("g8_black") is not False
                or a.get("opening") != b.get("opening")):
            raise ValueError("source color-swap pair mismatch")
        for row in (a, b):
            if row.get("status") != "checkmate" and row.get("winner") is not None:
                raise ValueError("unresolved game falsely labeled with winner")
            replay_check(row["opening"], row["moves"], row)
    return rows, sha256(raw)


def build_positions(rows, targets=TARGET_PLIES, per_stage=POSITIONS_PER_STAGE):
    """Select unique parent games and SFENs from actual legally played moves."""
    if not targets or per_stage < 1:
        raise ValueError("invalid selection settings")
    positions, used_pairs, seen_sfen = [], set(), set()
    for target in targets:
        count = 0
        for row in rows:
            if row["pair"] in used_pairs:
                continue
            sequence = row["opening"] + row["moves"]
            if len(sequence) <= target:
                continue
            board = shogi.Board()
            for word in sequence[:target]:
                move = shogi.Move.from_usi(word)
                if move not in board.legal_moves:
                    raise ValueError("illegal source move in selected position")
                board.push(move)
            if (board.is_checkmate() or board.is_fourfold_repetition()
                    or board.sfen() in seen_sfen):
                continue
            hand_pieces = sum(int(n) for h in board.pieces_in_hand for n in h.values())
            positions.append({
                "pair": row["pair"], "g8_black": row["g8_black"],
                "target_ply": target, "prefix": sequence[:target],
                "sfen": board.sfen(), "in_check": bool(board.is_check()),
                "hand_pieces": hand_pieces,
                "source_game_status": row["status"],
            })
            seen_sfen.add(board.sfen())
            used_pairs.add(row["pair"])
            count += 1
            if count == per_stage:
                break
        if count != per_stage:
            raise ValueError("not enough independent game parents for stage " + str(target))
    return positions


def _board_from_prefix(prefix):
    board = shogi.Board()
    for word in prefix:
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal position history")
        board.push(move)
    return board


def probe(games_path, report_path, model_path, out_path):
    raw_model = Path(model_path).read_bytes()
    if sha256(raw_model) != G8_SHA:
        raise ValueError("wrong G8 model SHA")
    rows, source_sha = load_source(games_path, report_path)
    model = RelativeValue(model_path, mix_percent=MODEL_MIX)
    if model.sha256 != G8_SHA:
        raise ValueError("model loader SHA mismatch")
    positions = build_positions(rows)
    result_rows = []
    for index, position in enumerate(positions):
        board = _board_from_prefix(position["prefix"])
        if board.sfen() != position["sfen"]:
            raise ValueError("position reconstruction mismatch")
        slow_eval = evaluate(board, model, fast_relative=False)
        fast_eval = evaluate(board, model, fast_relative=True)
        if abs(slow_eval - fast_eval) > 1:
            raise ValueError("fast/normal G8 evaluation diverged")
        before = (board.sfen(), tuple(board.move_stack))
        results = {}
        # Alternate order to reduce a fixed first-method timing bias.
        methods = ("alpha_beta", "uct") if index % 2 == 0 else ("uct", "alpha_beta")
        for method in methods:
            start = time.perf_counter()
            if method == "alpha_beta":
                word, depth, nodes = select_move(
                    board, movetime_ms=MOVE_TIME_MS, max_depth=2,
                    model=model, fast_relative=True, root_alpha_beta=True)
                metadata = {"completed_depth": depth, "nodes": nodes}
            else:
                word, iterations, internal_elapsed = select_mcts(
                    board, movetime_ms=MOVE_TIME_MS, model=model,
                    fast_relative=True, max_tree_depth=12,
                    seed=20279028 + index)
                metadata = {"iterations": iterations, "internal_elapsed_ms": round(internal_elapsed, 3)}
            elapsed = (time.perf_counter() - start) * 1000.0
            if (board.sfen(), tuple(board.move_stack)) != before:
                raise AssertionError(method + " mutated board or history")
            if word == "resign" or shogi.Move.from_usi(word) not in board.legal_moves:
                raise ValueError(method + " returned an illegal move")
            results[method] = {"move": word, "elapsed_ms": round(elapsed, 3), **metadata}
        result_rows.append({
            **{k: v for k, v in position.items() if k != "prefix"},
            "g8_eval_cp": fast_eval,
            "methods": results,
            "same_move": results["alpha_beta"]["move"] == results["uct"]["move"],
        })
    summary = {
        "status": "read_only_search_diagnostic_not_strength_or_adoption",
        "source_games_sha256": source_sha, "g8_model_sha256": G8_SHA,
        "source_run": 40637486947, "g8_model_run": 40636602223,
        "positions": len(result_rows), "independent_game_parents": len({r["pair"] for r in result_rows}),
        "stages": {str(p): sum(r["target_ply"] == p for r in result_rows) for p in TARGET_PLIES},
        "movetime_ms_each": MOVE_TIME_MS, "alpha_beta_max_depth": 2,
        "uct_max_tree_depth": 12, "value_mix_percent": MODEL_MIX,
        "move_agreements": sum(r["same_move"] for r in result_rows),
        "move_disagreements": sum(not r["same_move"] for r in result_rows),
        "positions_with_hand_pieces": sum(r["hand_pieces"] > 0 for r in result_rows),
        "positions_in_check": sum(r["in_check"] for r in result_rows),
        "alpha_beta_total_nodes": sum(r["methods"]["alpha_beta"]["nodes"] for r in result_rows),
        "uct_total_iterations": sum(r["methods"]["uct"]["iterations"] for r in result_rows),
        "alpha_beta_completed_depth_2": sum(r["methods"]["alpha_beta"]["completed_depth"] == 2 for r in result_rows),
        "alpha_beta_median_elapsed_ms": round(statistics.median(r["methods"]["alpha_beta"]["elapsed_ms"] for r in result_rows), 3),
        "uct_median_elapsed_ms": round(statistics.median(r["methods"]["uct"]["elapsed_ms"] for r in result_rows), 3),
        "alpha_beta_over_200ms": sum(r["methods"]["alpha_beta"]["elapsed_ms"] > MOVE_TIME_MS for r in result_rows),
        "uct_over_200ms": sum(r["methods"]["uct"]["elapsed_ms"] > MOVE_TIME_MS for r in result_rows),
        "unresolved_source_games_used_as_teacher": False,
        "model_promoted": False, "official_kumoji_changed": False,
        "limitations": [
            "Alpha-beta nodes and UCT iterations are different units and cannot be compared as equal work.",
            "A different move is not proof of a better move.",
            "Selected positions are from completed G8/G7 games, not a randomized representative sample.",
            "Pure UCT uses no policy prior, neural policy, or rollout; its value is a heuristic, not calibrated win probability.",
            "Wall-clock deadline is checked between operations; overrun must be reported.",
            "No head-to-head games or NNUE training are performed.",
        ],
    }
    root = Path(out_path)
    root.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in result_rows)
    (root / "position_diagnostics.jsonl").write_text(lines, encoding="utf-8")
    summary["diagnostics_sha256"] = sha256((root / "position_diagnostics.jsonl").read_bytes())
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-request")
    parser.add_argument("--games")
    parser.add_argument("--source-report")
    parser.add_argument("--model")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.check_request:
        print(json.dumps({"enabled": validate_request(args.check_request)}))
        return
    if not all((args.games, args.source_report, args.model, args.out)):
        parser.error("missing source/model/out arguments")
    if not validate_request(REQUEST_FILE):
        raise ValueError("probe request is disabled")
    print(json.dumps(probe(args.games, args.source_report, args.model, args.out),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
