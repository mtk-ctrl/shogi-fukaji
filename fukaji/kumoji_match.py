"""Audited FUKAJI-versus-pinned-KUMOJI USI connectivity trial.

Research-only diagnostic. KUMOJI retains its adopted knowledge; long-think
and ponder are explicitly disabled for a 200ms-per-move timing diagnostic.
This is NOT a formal strength comparison or equal-knowledge experiment.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import shogi

from .relative_value import RelativeValue


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_request(c):
    required = {"kumoji_commit", "relative_model_sha256", "relative_source_run_id",
                "games", "seed", "max_plies", "movetime_ms", "research_only"}
    if not required.issubset(c):
        raise ValueError("missing comparison configuration")
    for key in ("kumoji_commit", "relative_model_sha256"):
        n = 40 if key == "kumoji_commit" else 64
        value = c[key]
        if not isinstance(value, str) or len(value) != n or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError("invalid " + key)
    if (c["research_only"] is not True or type(c["relative_source_run_id"]) is not int
            or c["relative_source_run_id"] <= 0 or type(c["games"]) is not int
            or not 2 <= c["games"] <= 8 or c["games"] % 2 != 0
            or type(c["seed"]) is not int or not 0 <= c["seed"] <= 2147483647 - 2 * c["games"]
            or type(c["max_plies"]) is not int or not 1 <= c["max_plies"] <= 200
            or c["movetime_ms"] != 200):
        raise ValueError("invalid bounded diagnostic settings")
    return c


def audit_game(row):
    board = shogi.Board()
    for word in row["moves"]:
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal recorded move: " + word)
        board.push(move)
    reason, winner = row["reason"], row["winner"]
    if reason == "checkmate":
        if not board.is_checkmate():
            raise ValueError("unverified checkmate")
        black = "A" if row["a_black"] else "B"
        white = "B" if row["a_black"] else "A"
        expected = white if board.turn == shogi.BLACK else black
        if winner != expected:
            raise ValueError("incorrect checkmate winner")
    elif reason in ("move_limit", "fourfold_repetition", "no_legal_moves"):
        if winner is not None:
            raise ValueError("unresolved game mislabelled decisive")
    elif reason == "perpetual_check":
        if winner not in ("A", "B"):
            raise ValueError("missing perpetual-check winner")
    elif reason == "resign":
        if winner not in ("A", "B"):
            raise ValueError("missing resign winner")
    elif reason == "declare_win":
        if winner not in ("A", "B"):
            raise ValueError("missing declaration winner")
    else:
        raise ValueError("illegal/unknown adjudication: " + reason)
    return len(row["moves"])


def execute(request_path, model_path, kumoji_root, fukaji_usi, out_dir):
    cfg = validate_request(json.loads(Path(request_path).read_text(encoding="utf-8")))
    kumoji_root = Path(kumoji_root)
    model_path = Path(model_path)
    fukaji_usi = Path(fukaji_usi)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if sha256(model_path) != cfg["relative_model_sha256"]:
        raise ValueError("frozen FUKAJI model SHA mismatch")
    RelativeValue(model_path, mix_percent=25)  # Validate schema before engine startup.
    source_sha = subprocess.check_output(
        ["git", "-C", str(kumoji_root), "rev-parse", "HEAD"], text=True).strip()
    if source_sha != cfg["kumoji_commit"]:
        raise ValueError("KUMOJI source commit mismatch")
    binary = kumoji_root / "build/shogi-ai"
    knowledge = kumoji_root / "position-knowledge-v1.tsv"
    manifest = kumoji_root / "position-knowledge-v1.manifest.json"
    if not binary.is_file() or not knowledge.is_file() or not manifest.is_file():
        raise ValueError("KUMOJI binary/adopted knowledge missing")
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    if meta.get("adopted_positions") != 163:
        raise ValueError("KUMOJI adopted knowledge count mismatch")
    opts_a = {"ValueModelType": "relative", "ValueModelFile": str(model_path),
              "ValueModelMix": 25, "SearchDepth": 2,
              "PositionKnowledge": False, "RiskVerify": False, "USI_Ponder": False}
    opts_b = {"PositionKnowledge": True, "PositionKnowledgeFile": str(knowledge),
              "AdaptiveLongThink": False, "USI_Ponder": False, "RiskVerify": True}
    match_path = out / "match.json"
    command = [sys.executable, str(kumoji_root / "benchmarks/arena.py"),
               "--engine-a", str(fukaji_usi), "--engine-b", str(binary),
               "--options-a", json.dumps(opts_a), "--options-b", json.dumps(opts_b),
               "--games", str(cfg["games"]), "--max-plies", str(cfg["max_plies"]),
               "--seed", str(cfg["seed"]), "--go-command", "go movetime 200",
               "--output", str(match_path)]
    subprocess.run(command, check=True)
    result = json.loads(match_path.read_text(encoding="utf-8"))
    games = result.get("details")
    if (result.get("games") != cfg["games"] or len(games) != cfg["games"]
            or result.get("illegal_games") != 0):
        raise ValueError("missing/illegal games")
    if [g["a_black"] for g in games] != [i % 2 == 0 for i in range(cfg["games"])]:
        raise ValueError("unbalanced color assignment")
    for g in games:
        if audit_game(g) > cfg["max_plies"]:
            raise ValueError("game over maximum plies")
    k = result["position_knowledge"]
    if (k["A"].get("enabled") is not False or k["B"].get("enabled") is not True
            or k["B"].get("available") is not True
            or k["B"].get("sha256") != sha256(knowledge)):
        raise ValueError("KUMOJI knowledge use not correctly configured")
    if (result.get("go_command") != "go movetime 200"
            or result["options_b"].get("AdaptiveLongThink") is not False
            or result["options_b"].get("USI_Ponder") is not False):
        raise ValueError("timing/ponder protocol mismatch")
    report = {
        "status": "technical_connectivity_only_no_promotion",
        "kumoji_commit": source_sha,
        "kumoji_binary_sha256": sha256(binary),
        "kumoji_knowledge_sha256": sha256(knowledge),
        "kumoji_knowledge_positions": 163,
        "fukaji_relative_model_sha256": sha256(model_path),
        "games": cfg["games"],
        "fukaji_wins_as_adjudicated": result["wins_a"],
        "kumoji_wins_as_adjudicated": result["wins_b"],
        "unresolved": result["draws"],
        "checkmate_wins_fukaji": sum(g["reason"] == "checkmate" and g["winner"] == "A" for g in games),
        "checkmate_wins_kumoji": sum(g["reason"] == "checkmate" and g["winner"] == "B" for g in games),
        "seed": cfg["seed"], "movetime_ms": 200, "max_plies": cfg["max_plies"],
        "all_legal_replay": True, "colors_alternate": True,
        "knowledge_asymmetry": "KUMOJI adopted 163 positions ON; FUKAJI no adopted knowledge",
        "timing": "both go movetime 200; KUMOJI AdaptiveLongThink OFF; ponder OFF",
        "not_same_opening_pairs": True,
        "match_sha256": sha256(match_path),
        "caution": "Exploratory USI smoke only. Not an equal-knowledge or formal strength test.",
    }
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--kumoji-root", required=True)
    p.add_argument("--fukaji-usi", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    execute(a.request, a.model, a.kumoji_root, a.fukaji_usi, a.out)


if __name__ == "__main__":
    main()
