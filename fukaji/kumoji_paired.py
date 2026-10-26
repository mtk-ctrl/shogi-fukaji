"""Same-opening color-swapped FUKAJI versus pinned KUMOJI research matches.

The official KUMOJI engine is built from a frozen read-only checkout. Its
existing USI Engine/Adjudicator classes are imported, not copied or modified.
This is a diagnostic bridge, not an adoption/strength verdict.
"""
import argparse
import importlib.util
import json
import subprocess
from collections import Counter
from pathlib import Path

import shogi

from .arena import opening
from .kumoji_match import sha256, audit_game
from .relative_value import RelativeValue


def validate_request(c):
    if (c.get("research_only") is not True
            or type(c.get("arena_seed")) is not int
            or not 0 <= c["arena_seed"] <= 2147483647 - 100
            or type(c.get("pairs")) is not int or not 1 <= c["pairs"] <= 8
            or c.get("max_plies") not in (80, 120, 160)
            or c.get("opening_plies") != 6
            or c.get("movetime_ms") != 200
            or c.get("arms") != ["equal_200ms", "kumoji_formal"]
            or c.get("relative_source_run_id") != 40359123574):
        raise ValueError("invalid fixed paired comparison protocol")
    for key, size in (("kumoji_commit", 40), ("relative_model_sha256", 64)):
        value = c.get(key)
        if not isinstance(value, str) or len(value) != size or any(x not in "0123456789abcdef" for x in value):
            raise ValueError("invalid " + key)
    return c


def load_kumoji_arena(root):
    path = Path(root) / "benchmarks/arena.py"
    spec = importlib.util.spec_from_file_location("fukaji_pinned_kumoji_arena", path)
    if spec is None or spec.loader is None:
        raise ValueError("KUMOJI arena unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def play_paired_game(arena, a, b, head, a_black, max_plies, seed):
    black, white = (a, b) if a_black else (b, a)
    black.configure_game(seed)
    white.configure_game(seed + 1)
    board = shogi.Board()
    adjudicator = arena.Adjudicator(board)
    moves = []
    for word in head:
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal fixed opening")
        board.push(move)
        moves.append(word)
        if adjudicator.after_move(board) is not None:
            raise ValueError("terminal fixed opening")
    records = []
    winner, reason = None, "move_limit"
    for _ in range(max_plies - len(head)):
        engine = black if board.turn == shogi.BLACK else white
        side = board.turn
        word = engine.bestmove(moves)
        if word == "resign":
            winner, reason = (white.label if engine is black else black.label), "resign"
            break
        if word == "win":
            if not arena.can_declare_win(board):
                raise ValueError("invalid declaration")
            winner, reason = engine.label, "declare_win"
            break
        try:
            move = shogi.Move.from_usi(word)
        except ValueError as exc:
            raise ValueError("invalid engine USI token") from exc
        if move not in board.legal_moves:
            raise ValueError("illegal engine move")
        records.append({
            "ply": len(moves) + 1, "engine": engine.label,
            "side_to_move": int(side), "move": word,
            "search": arena.compact_search_telemetry(engine.last_search, side),
        })
        board.push(move)
        moves.append(word)
        terminal = adjudicator.after_move(board)
        if terminal is not None:
            color, reason = terminal
            winner = None if color is None else (black.label if color == shogi.BLACK else white.label)
            break
        if hasattr(engine, "begin_ponder") and len(moves) < max_plies:
            engine.begin_ponder(moves, board)
    result = {
        "a_black": a_black, "opening": head, "moves": moves,
        "played_moves": moves[len(head):], "move_records": records,
        "winner": winner, "reason": reason, "plies": len(moves),
        "final_sfen": board.sfen(),
    }
    if audit_game(result) != len(moves):
        raise ValueError("replay length mismatch")
    return result


def summarize_pairs(rows, pairs):
    if len(rows) != pairs * 2:
        raise ValueError("incorrect paired game count")
    counts = Counter()
    for i in range(pairs):
        x, y = rows[2 * i:2 * i + 2]
        if x["opening"] != y["opening"] or x["a_black"] is not True or y["a_black"] is not False:
            raise ValueError("paired opening/color mismatch")
        aw = sum(z["winner"] == "A" for z in (x, y))
        bw = sum(z["winner"] == "B" for z in (x, y))
        counts["a_advantage" if aw > bw else "b_advantage" if bw > aw else "equal"] += 1
    return dict(counts)


def execute(request, model, kumoji_root, fukaji_usi, out):
    cfg = validate_request(json.loads(Path(request).read_text(encoding="utf-8")))
    kumoji_root, model, fukaji_usi, out = map(Path, (kumoji_root, model, fukaji_usi, out))
    out.mkdir(parents=True, exist_ok=True)
    if sha256(model) != cfg["relative_model_sha256"]:
        raise ValueError("relative model SHA mismatch")
    RelativeValue(model, mix_percent=25)
    commit = subprocess.check_output(["git", "-C", str(kumoji_root), "rev-parse", "HEAD"], text=True).strip()
    if commit != cfg["kumoji_commit"]:
        raise ValueError("KUMOJI commit mismatch")
    binary = kumoji_root / "build/shogi-ai"
    knowledge = kumoji_root / "position-knowledge-v1.tsv"
    manifest = json.loads((kumoji_root / "position-knowledge-v1.manifest.json").read_text())
    if manifest["adopted_positions"] != 163 or sha256(knowledge) != manifest["output_sha256"]:
        raise ValueError("KUMOJI adopted knowledge SHA/count mismatch")
    arena = load_kumoji_arena(kumoji_root)
    report = {"status": "paired_research_no_promotion",
              "kumoji_commit": commit, "kumoji_binary_sha256": sha256(binary),
              "kumoji_knowledge_sha256": sha256(knowledge),
              "fukaji_relative_model_sha256": sha256(model),
              "seed": cfg["arena_seed"], "pairs": cfg["pairs"],
              "opening_plies": 6, "max_plies": cfg["max_plies"],
              "movetime_ms": 200, "knowledge_asymmetry": "KUMOJI adopted 163 ON; FUKAJI has no equivalent",
              "arms": {}}
    all_rows = []
    for arm in cfg["arms"]:
        formal = arm == "kumoji_formal"
        opts_a = {"ValueModelType": "relative", "ValueModelFile": str(model),
                  "ValueModelMix": 25, "SearchDepth": 2,
                  "PositionKnowledge": False, "RiskVerify": False, "USI_Ponder": False}
        opts_b = {"PositionKnowledge": True, "PositionKnowledgeFile": str(knowledge),
                  "AdaptiveLongThink": formal, "USI_Ponder": formal, "RiskVerify": True}
        a = arena.Engine(fukaji_usi, "A", opts_a, "go movetime 200")
        b = arena.Engine(binary, "B", opts_b, "go movetime 200")
        rows = []
        try:
            for pair in range(cfg["pairs"]):
                head = opening(cfg["arena_seed"] + pair, cfg["opening_plies"])
                if len(head) != 6:
                    raise ValueError("incomplete fixed opening")
                for a_black in (True, False):
                    row = play_paired_game(arena, a, b, head, a_black, cfg["max_plies"],
                                           cfg["arena_seed"] + pair * 4 + int(not a_black) * 2)
                    row.update({"pair": pair, "arm": arm})
                    rows.append(row)
        finally:
            a.close()
            b.close()
        if len(rows) != 2 * cfg["pairs"]:
            raise ValueError("incomplete arm")
        for row in rows:
            audit_game(row)
        by_reason = dict(Counter(x["reason"] for x in rows))
        summary = {
            "games": len(rows),
            "fukaji_wins_as_adjudicated": sum(x["winner"] == "A" for x in rows),
            "kumoji_wins_as_adjudicated": sum(x["winner"] == "B" for x in rows),
            "unresolved": sum(x["winner"] is None for x in rows),
            "checkmate_wins_fukaji": sum(x["winner"] == "A" and x["reason"] == "checkmate" for x in rows),
            "checkmate_wins_kumoji": sum(x["winner"] == "B" and x["reason"] == "checkmate" for x in rows),
            "reasons": by_reason, "pair_comparison": summarize_pairs(rows, cfg["pairs"]),
            "fukaji_nodes": sum(x.get("nodes", 0) for x in a.search_stats),
            "kumoji_nodes": sum(x.get("nodes", 0) for x in b.search_stats),
            "kumoji_knowledge_probes": sum(x.get("knowledge_probes", 0) for x in b.search_stats),
            "kumoji_knowledge_hits": sum(x.get("knowledge_hits", 0) for x in b.search_stats),
            "kumoji_ponder_hits": b.ponder_hits,
            "kumoji_ponder_misses": b.ponder_misses,
            "kumoji_adaptive_longthink": formal,
            "kumoji_ponder_enabled": formal,
            "both_go_movetime_200": True,
        }
        report["arms"][arm] = summary
        all_rows.extend(rows)
    path = out / "paired_games.jsonl"
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in all_rows), encoding="utf-8")
    report["games_sha256"] = sha256(path)
    report["caution"] = ("Exploratory. KUMOJI uses adopted knowledge, FUKAJI has none. "
                         "Formal KUMOJI arm has extra think and ponder. No strength claim.")
    (out / "paired_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    for name in ("request", "model", "kumoji-root", "fukaji-usi", "out"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    execute(a.request, a.model, a.kumoji_root, a.fukaji_usi, a.out)


if __name__ == "__main__":
    main()
