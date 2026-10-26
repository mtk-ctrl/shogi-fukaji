"""Replay and SHA-verify all 8 independently executed continuation shards."""
import argparse
import json
from collections import Counter
from pathlib import Path

import shogi

from .direct_arena import digest, replay_check
from .generation_input import MODEL_SHA, SOURCES, sha
from .continue_unresolved import SOURCE_RUNS, FINAL_PLIES, ORIGINAL_PLIES
from .learning import load_games


def aggregate(parts_dir, out):
    parts = sorted(Path(parts_dir).glob("*/continuation_report.json"))
    if len(parts) != 8:
        raise ValueError("expected exactly eight shard reports")
    rows, mate_items, seen_shards, seen_games = [], [], set(), set()
    for report_path in parts:
        base = report_path.parent
        report = json.loads(report_path.read_text(encoding="utf-8"))
        shard = report.get("shard")
        if (type(shard) is not int or shard not in range(8) or
                shard in seen_shards or report.get("shards") != 8 or
                report.get("extended_games") != 4 or
                report.get("frozen_model_sha256") != MODEL_SHA or
                report.get("source_runs") != list(SOURCE_RUNS) or
                report.get("original_plies") != ORIGINAL_PLIES or
                report.get("max_plies") != FINAL_PLIES or
                report.get("status") != "research_only_verified_continuation_no_training_no_adoption"):
            raise ValueError("invalid or duplicate continuation shard")
        seen_shards.add(shard)
        games_path = base / "continuations.jsonl"
        mates_path = base / "new_verified_mates.jsonl"
        if (digest(games_path) != report.get("continuations_sha256") or
                digest(mates_path) != report.get("new_mates_sha256")):
            raise ValueError("shard SHA mismatch")
        shard_rows = [json.loads(line) for line in games_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if len(shard_rows) != 4:
            raise ValueError("shard row count mismatch")
        shard_mates = []
        for row in shard_rows:
            run_id = row["source_run"]
            key = (run_id, row["pair"], row["candidate_black"])
            if (run_id not in SOURCE_RUNS or key in seen_games or
                    type(row["candidate_black"]) is not bool or
                    type(row["pair"]) is not int or row["pair"] not in range(16) or
                    row["source_games_sha256"] != SOURCES[run_id]["games_sha256"]):
                raise ValueError("invalid game identity or duplicate")
            seen_games.add(key)
            original = row["opening"] + row["prefix_moves"]
            moves = original + row["continuation_moves"]
            if (len(original) != ORIGINAL_PLIES or
                    row["total_plies"] != len(moves) or len(moves) > FINAL_PLIES or
                    row["parent_id"] != sha(
                        ("fukaji-paired-v1|" + str(run_id) + "|" + str(row["pair"]) +
                         "|" + " ".join(row["opening"])).encode())):
                raise ValueError("continuation parent or length mismatch")
            replay_check([], moves, {"status": row["status"], "winner": row["winner"]})
            if row["status"] == "move_limit_unresolved" and len(moves) != FINAL_PLIES:
                raise ValueError("incomplete move-limit game")
            if row["status"] == "checkmate":
                shard_mates.append({"start_sfen": shogi.Board().sfen(),
                                    "moves": moves, "winner": row["winner"],
                                    "source": "selfplay", "parent_id": row["parent_id"]})
            rows.append(row)
        original_mates = [json.loads(line) for line in mates_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if original_mates != shard_mates:
            raise ValueError("shard mate list does not match replay")
        mate_items.extend(shard_mates)
    if seen_shards != set(range(8)) or len(rows) != 32:
        raise ValueError("missing continuation shard or game")
    by_source = Counter(row["source_run"] for row in rows)
    if any(by_source[run] != 16 for run in SOURCE_RUNS):
        raise ValueError("source game count mismatch")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (row["source_run"], row["pair"], not row["candidate_black"]))
    games_path = root / "all_continuations.jsonl"
    games_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                          encoding="utf-8")
    unique_mates, seen_mates = [], set()
    for item in mate_items:
        key = sha((item["start_sfen"] + "|" + " ".join(item["moves"])).encode())
        if key not in seen_mates:
            seen_mates.add(key)
            unique_mates.append(item)
    mate_path = root / "new_verified_mates.jsonl"
    mate_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                 for row in unique_mates), encoding="utf-8")
    if len(load_games(mate_path)) != len(unique_mates):
        raise ValueError("aggregate mate label audit failed")
    report = {
        "status": "research_only_replay_verified_no_training_no_adoption",
        "source_runs": list(SOURCE_RUNS), "frozen_model_sha256": MODEL_SHA,
        "shards": 8, "games": len(rows),
        "by_source": {str(k): v for k, v in by_source.items()},
        "outcomes": dict(Counter(row["status"] for row in rows)),
        "new_verified_mates": len(unique_mates),
        "duplicate_mates_removed": len(mate_items) - len(unique_mates),
        "games_sha256": digest(games_path), "mates_sha256": digest(mate_path),
        "legal_replay_verified": True,
        "caution": "No labels for unresolved; no generation or model promotion.",
    }
    (root / "aggregate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    aggregate(a.parts, a.out)


if __name__ == "__main__":
    main()
