"""Prepare unadopted FUKAJI generation input from two SHA-frozen paired arenas.

Only actual replay-verified checkmates receive labels. Color-swapped games
share a parent group; no unfinished game is rewarded. No model is trained.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import shogi
from .learning import load_games, partition_rows

SOURCES = {
    40387257648: {"seed": 20271427, "games_sha256": "d61a79f9dab9b9a4b069e569a2355f5d2f9a50753226553cdab1cec81e3180fb"},
    40395505536: {"seed": 20271527, "games_sha256": "b5b9d52fdfa6fb8e77000027ef38dd4bcc8344f2afc45b7e4575896b77e1e47c"},
}
MODEL_SHA = "881af10b5bc0cde591e37a9aee883213fa7dd602345b786b9b93d69c7e6ed239"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_source(run_id, games_path, report_path):
    if run_id not in SOURCES:
        raise ValueError("unapproved source run")
    raw = Path(games_path).read_bytes()
    if sha(raw) != SOURCES[run_id]["games_sha256"]:
        raise ValueError("source game SHA mismatch")
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if (report.get("games_sha256") != sha(raw) or
            report.get("model_sha256") != MODEL_SHA or
            report.get("arena_seed") != SOURCES[run_id]["seed"] or
            report.get("games") != 32 or report.get("pairs") != 16 or
            report.get("movetime_ms") != 200 or report.get("depth") != 2 or
            report.get("mix_percent") != 25 or report.get("max_plies") != 160 or
            report.get("root_alpha_beta_default") is not False or
            report.get("all_pairs_alternate_color") is not True or
            report.get("status") != "research_only_no_promotion"):
        raise ValueError("frozen protocol or report mismatch")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 32:
        raise ValueError("not exactly 32 games")
    outcomes = Counter()
    accepted = []
    for pair in range(16):
        a, b = rows[2*pair:2*pair+2]
        if (a.get("pair") != pair or b.get("pair") != pair or
                a.get("candidate_black") is not True or b.get("candidate_black") is not False or
                a.get("opening") != b.get("opening") or
                not isinstance(a.get("opening"), list) or len(a["opening"]) != 6):
            raise ValueError("paired opening or color mismatch")
        group = sha(f"fukaji-paired-v1|{run_id}|{pair}|{' '.join(a['opening'])}".encode())
        for game in (a, b):
            board = shogi.Board()
            all_moves = game["opening"] + game["moves"]
            if len(all_moves) > 160:
                raise ValueError("game exceeds move cap")
            for word in all_moves:
                move = shogi.Move.from_usi(word)
                if move not in board.legal_moves:
                    raise ValueError("illegal replay")
                board.push(move)
            status, winner = game.get("status"), game.get("winner")
            if status == "checkmate":
                if not board.is_checkmate() or winner != int(not board.turn):
                    raise ValueError("invalid mate or winner")
                accepted.append({"start_sfen": shogi.Board().sfen(), "moves": all_moves,
                                 "winner": winner, "source": "selfplay", "parent_id": group})
                outcomes["checkmate"] += 1
            elif status == "move_limit_unresolved":
                if winner is not None or len(all_moves) != 160 or board.is_checkmate():
                    raise ValueError("invalid unresolved game")
                outcomes["unresolved"] += 1
            else:
                raise ValueError("unexpected terminal state")
    if (outcomes["checkmate"] != report.get("candidate_wins", -1) + report.get("baseline_wins", -1) or
            outcomes["unresolved"] != report.get("unresolved", -1)):
        raise ValueError("terminal totals differ from source report")
    return accepted, dict(outcomes)


def execute(source_a, report_a, source_b, report_b, out):
    rows_a, counts_a = read_source(40387257648, source_a, report_a)
    rows_b, counts_b = read_source(40395505536, source_b, report_b)
    all_games = rows_a + rows_b
    seen, games = set(), []
    for game in all_games:
        key = sha((game["start_sfen"] + "|" + " ".join(game["moves"])).encode())
        if key not in seen:
            seen.add(key)
            games.append(game)
    duplicate_games = len(all_games) - len(games)
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    dataset = dest / "verified_mates.jsonl"
    dataset.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in games), encoding="utf-8")
    verified = load_games(dataset)
    if len(verified) != len(games):
        raise ValueError("duplicate games or learner rejection")
    _, split_counts = partition_rows(verified)
    groups = Counter(g["parent_id"] for g in games)
    if len(groups) < 6 or max(groups.values()) > 2 or min(split_counts.values()) < 2:
        raise ValueError("insufficient independent training/validation/test parent groups")
    manifest = {
        "status": "replay_verified_generation_input_only_no_training_no_adoption",
        "source_runs": list(SOURCES),
        "source_game_sha256": [SOURCES[r]["games_sha256"] for r in SOURCES],
        "frozen_model_sha256": MODEL_SHA,
        "source_protocol": "paired_200ms_depth2_mix25_fast_eval",
        "source_counts": {str(SOURCES[r]["seed"]): counts for r, counts in zip(SOURCES, (counts_a, counts_b))},
        "source_checkmate_games": len(all_games), "accepted_checkmate_games": len(games),
        "exact_duplicate_mates_removed": duplicate_games,
        "unresolved_excluded_from_labels": counts_a["unresolved"] + counts_b["unresolved"],
        "independent_parent_groups": len(groups), "largest_group": max(groups.values()),
        "split_parent_groups": split_counts, "dataset_sha256": sha(dataset.read_bytes()),
        "limitations": ["Outcome labels are not independently verified move-quality labels.",
                        "The two search configurations share one frozen value model.",
                        "Do not select hyperparameters on the held-out test split.",
                        "No model has been trained, tested against a parent, or adopted."],
    }
    (dest / "generation_input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    for name in ("source-a", "report-a", "source-b", "report-b", "out"):
        ap.add_argument("--" + name, required=True)
    args = ap.parse_args()
    execute(args.source_a, args.report_a, args.source_b, args.report_b, args.out)


if __name__ == "__main__":
    main()
