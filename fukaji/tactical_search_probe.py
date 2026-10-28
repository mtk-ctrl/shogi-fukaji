"""SHA-pinned, research-only G8 alpha-beta / UCT tactical position probe.

No new games, no training labels, no adoption. Terminal checkmate is tested
as resignation, not as a move-choice comparison. Missing source categories
remain missing. Full legal prefixes preserve hands and repetition history.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import shogi

from .engine import evaluate, select_move
from .g8_paired_arena import G8_SHA
from .mcts_research import select_mcts
from .relative_value import RelativeValue
from .search_method_probe import load_source
from .tactical_positions import CATEGORIES, classify_position, position_key, select_tactical_positions

REQUEST_FILE = ".github/fukaji-g8-tactical-uct-request.json"
EXPECTED_REQUEST = {
    "enabled": False, "research_only": True, "model_adoption_allowed": False,
    "source_run": 40637486947, "g8_model_run": 40636602223,
    "positions_per_category": 3, "max_ply": 240,
    "movetime_ms": 200, "runner_jobs": 1,
}
SOURCE_SHA = "60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf"
MOVE_TIME_MS = 200
MODEL_MIX = 25


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_request(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg not in (EXPECTED_REQUEST, dict(EXPECTED_REQUEST, enabled=True)):
        raise ValueError("unapproved G8 tactical search research request")
    enabled = cfg["enabled"]
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
            print("enabled=" + str(enabled).lower(), file=handle)
    return enabled


def board_for_selected(item):
    board = shogi.Board()
    occurrences = defaultdict(int)
    occurrences[position_key(board)] = 1
    for ply, word in enumerate(item["prefix"], start=1):
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal selected prefix at ply " + str(ply))
        board.push(move)
        occurrences[position_key(board)] += 1
    if (board.sfen() != item["sfen"] or
            len(item["prefix"]) != item["ply"] or
            item["category"] not in classify_position(board, occurrences[position_key(board)]) or
            item["in_check"] != bool(board.is_check()) or
            item["is_checkmate"] != bool(board.is_checkmate()) or
            item["position_occurrences"] != occurrences[position_key(board)] or
            item["hand_pieces"] != sum(int(n) for h in board.pieces_in_hand for n in h.values())):
        raise ValueError("selected tactical category or history mismatch")
    return board


def measure(board, model, index, movetime_ms=MOVE_TIME_MS):
    before = (board.sfen(), tuple(board.move_stack))
    terminal = bool(board.is_checkmate())
    if terminal:
        eval_cp = None
    else:
        normal = evaluate(board, model, fast_relative=False)
        fast = evaluate(board, model, fast_relative=True)
        if abs(normal - fast) > 1:
            raise ValueError("G8 fast and normal evaluation mismatch")
        eval_cp = fast
    methods = {}
    order = ("alpha_beta", "uct") if index % 2 == 0 else ("uct", "alpha_beta")
    for method in order:
        start = time.perf_counter()
        if method == "alpha_beta":
            move, depth, nodes = select_move(
                board, movetime_ms=movetime_ms, max_depth=2,
                model=model, fast_relative=True, root_alpha_beta=True)
            stats = {"completed_depth": depth, "nodes": nodes}
        else:
            move, iterations, internal_ms = select_mcts(
                board, movetime_ms=movetime_ms, model=model,
                fast_relative=True, max_tree_depth=12, seed=20279029 + index)
            stats = {"iterations": iterations, "internal_elapsed_ms": round(internal_ms, 3)}
        elapsed = (time.perf_counter() - start) * 1000.0
        if (board.sfen(), tuple(board.move_stack)) != before:
            raise AssertionError(method + " mutated board or repetition history")
        if terminal:
            if move != "resign":
                raise ValueError(method + " failed terminal checkmate")
        elif move == "resign" or shogi.Move.from_usi(move) not in board.legal_moves:
            raise ValueError(method + " returned illegal tactical move")
        methods[method] = {"move": move, "elapsed_ms": round(elapsed, 3), **stats}
    return {"g8_eval_cp": eval_cp, "methods": methods,
            "same_move": methods["alpha_beta"]["move"] == methods["uct"]["move"],
            "terminal": terminal}


def probe(games_path, report_path, model_path, out_path):
    if digest(Path(model_path).read_bytes()) != G8_SHA:
        raise ValueError("G8 model SHA mismatch")
    rows, source_sha = load_source(games_path, report_path)
    if source_sha != SOURCE_SHA:
        raise ValueError("G8/G7 240-game source SHA mismatch")
    model = RelativeValue(model_path, mix_percent=MODEL_MIX)
    if model.sha256 != G8_SHA:
        raise ValueError("G8 model loader SHA mismatch")
    selected, shortages = select_tactical_positions(
        rows, per_category=EXPECTED_REQUEST["positions_per_category"],
        max_ply=EXPECTED_REQUEST["max_ply"])
    if any(shortages[c] < 0 for c in CATEGORIES):
        raise AssertionError("negative category shortage")
    diagnostics = []
    for index, item in enumerate(selected):
        board = board_for_selected(item)
        observed = measure(board, model, index)
        diagnostics.append({**item, **observed})
    if len({r["pair"] for r in diagnostics}) != len(diagnostics):
        raise AssertionError("reused game parent")
    by_category = dict(Counter(r["category"] for r in diagnostics))
    if any(by_category.get(c, 0) + shortages[c] != EXPECTED_REQUEST["positions_per_category"]
           for c in CATEGORIES):
        raise AssertionError("missing tactical shortage accounting")
    nonterminal = [r for r in diagnostics if not r["terminal"]]
    summary = {
        "status": "research_only_no_training_or_adoption",
        "source_run": EXPECTED_REQUEST["source_run"],
        "source_games": 240, "source_pairs": 120,
        "source_games_sha256": source_sha,
        "g8_model_run": EXPECTED_REQUEST["g8_model_run"],
        "g8_model_sha256": G8_SHA,
        "positions_requested_per_category": EXPECTED_REQUEST["positions_per_category"],
        "positions_selected": len(diagnostics),
        "independent_parent_pairs": len({r["pair"] for r in diagnostics}),
        "selected_by_category": {c: by_category.get(c, 0) for c in CATEGORIES},
        "shortages": shortages,
        "terminal_checkmate_positions": len(diagnostics) - len(nonterminal),
        "nonterminal_move_comparisons": len(nonterminal),
        "positions_in_check": sum(r["in_check"] for r in diagnostics),
        "positions_with_hands": sum(r["hand_pieces"] > 0 for r in diagnostics),
        "repetition_risk_positions": by_category.get("repetition_risk", 0),
        "movetime_ms_each": MOVE_TIME_MS, "alpha_beta_max_depth": 2,
        "uct_max_tree_depth": 12, "value_mix_percent": MODEL_MIX,
        "move_agreements_nonterminal": sum(r["same_move"] for r in nonterminal),
        "move_disagreements_nonterminal": sum(not r["same_move"] for r in nonterminal),
        "alpha_beta_total_nodes": sum(r["methods"]["alpha_beta"]["nodes"] for r in diagnostics),
        "uct_total_iterations": sum(r["methods"]["uct"]["iterations"] for r in diagnostics),
        "alpha_beta_over_200ms": sum(r["methods"]["alpha_beta"]["elapsed_ms"] > MOVE_TIME_MS for r in diagnostics),
        "uct_over_200ms": sum(r["methods"]["uct"]["elapsed_ms"] > MOVE_TIME_MS for r in diagnostics),
        "alpha_beta_median_ms": round(statistics.median(r["methods"]["alpha_beta"]["elapsed_ms"] for r in diagnostics), 3) if diagnostics else None,
        "uct_median_ms": round(statistics.median(r["methods"]["uct"]["elapsed_ms"] for r in diagnostics), 3) if diagnostics else None,
        "all_240_games_replayed_legally": True,
        "unresolved_source_games_used_as_teacher": False,
        "official_kumoji_changed": False, "model_promoted": False,
        "limitations": [
            "A missing category is reported as a shortage, never synthesized.",
            "Terminal checkmate is resignation-only and excluded from move agreement.",
            "Different moves do not establish strength; alpha-beta nodes and UCT iterations differ.",
            "Time overruns are measured, not concealed; UCT guard is not hard real-time.",
            "This is a biased selection of real games, not an independent head-to-head.",
        ],
    }
    root = Path(out_path)
    root.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in diagnostics)
    (root / "tactical_diagnostics.jsonl").write_text(raw, encoding="utf-8")
    summary["diagnostics_sha256"] = digest((root / "tactical_diagnostics.jsonl").read_bytes())
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
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
        parser.error("missing games, source-report, model or out")
    if not validate_request(REQUEST_FILE):
        raise ValueError("tactical research request disabled")
    print(json.dumps(probe(args.games, args.source_report, args.model, args.out),
                     ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
