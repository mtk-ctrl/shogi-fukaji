"""D13: frozen G8 FastLegalGuard ON/OFF versus the SAME pinned KUMOJI.

This is 50 fresh six-ply openings, each played four times: FUKAJI baseline
and fast-legal each take both colors against the same identical KUMOJI
binary, full research candidate, and published search options.
200 authenticated games, 20 shards, no model training or formal promotion.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path

import shogi

from .arena import opening
from .kumoji_match import audit_game
from .kumoji_paired import play_paired_game
from .kumoji_full100 import G8_SHA, KUMOJI_COMMIT, config as old_config
from .kumoji_full100 import input_files as old_input_files

OLD_RUN = 41177748192
OLD_INPUT_ARTIFACT = 13033349670
OLD_SEED = 202610315
NEW_SEED = 202611015
PAIRS, SHARDS, MAX_PLIES = 50, 20, 300
ARMS = ("baseline", "fastlegal")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def config():
    return {
        "enabled": True, "research_only": True, "formal_adoption": False,
        "training_allowed": False, "kumoji_source_run": OLD_RUN,
        "kumoji_input_artifact_id": OLD_INPUT_ARTIFACT,
        "source_artifact_name": "fukaji-kumoji-full100-input-41177748192",
        "original_kumoji_seed": OLD_SEED,
        "kumoji_commit": KUMOJI_COMMIT, "fukaji_g8_sha256": G8_SHA,
        "seed": NEW_SEED, "pairs": PAIRS, "games": 200, "shards": SHARDS,
        "opening_plies": 6, "max_plies": MAX_PLIES, "movetime_ms": 200,
        "depth_fukaji": 3, "g8_mix_percent": 25,
        "fast_relative": True, "fast_move_list": True,
        "root_alpha_beta": True,
        "arms": list(ARMS), "only_difference": "FastLegalGuard",
        "kumoji_research_candidate": "unchanged_prior_460_positions",
        "kumoji_adaptive_long_think": True, "kumoji_search_threads": 2,
        "kumoji_ponder": True
    }


def check_request(path):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    if c != config():
        raise ValueError("cross200 source model, openings, or engine options changed")
    previous = {tuple(opening(OLD_SEED + pair, 6)) for pair in range(PAIRS)}
    fresh = {tuple(opening(NEW_SEED + pair, 6)) for pair in range(PAIRS)}
    if len(previous) != PAIRS or len(fresh) != PAIRS or previous & fresh:
        raise ValueError("fresh openings overlap earlier 100-game study")
    if any(len(x) != 6 for x in fresh):
        raise ValueError("incomplete nonterminal opening")
    return c


def verify_input(folder):
    # Reuse previous verified full-archive source ONLY. It is not main-adopted data.
    receipt, report = old_input_files(folder)
    if receipt.get("request") != old_config() or report["source_observations"] != 1079:
        raise ValueError("unverified original full research package")
    if report.get("playable_positions") != 460:
        raise ValueError("changed KUMOJI research position count")
    return receipt, report


def matched_seed(pair, seat):
    """Same USI RandomSeed for baseline and candidate at identical pair+color."""
    if type(pair) is not int or type(seat) is not int or not 0 <= pair < PAIRS or seat not in (0, 1):
        raise ValueError("invalid matched game seed coordinate")
    return NEW_SEED + pair * 16 + seat * 2


def indices(shard):
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("bad shard")
    return range(shard, PAIRS, SHARDS)


def play_shard(request_path, input_dir, shard, out):
    check_request(request_path)
    receipt, _ = verify_input(input_dir)
    from benchmarks import arena
    folder = Path(input_dir).resolve()
    launcher = folder / "fukaji-usi"
    launcher.write_text("#!/bin/sh\nexec python3 -m fukaji.engine\n", encoding="utf-8")
    launcher.chmod(0o755)
    common_a = {
        "ValueModelType": "relative", "ValueModelFile": str(folder / "g8.npz"),
        "ValueModelMix": 25, "SearchDepth": 3,
        "FastRelativeEval": True, "RootAlphaBeta": True,
        "FastMoveList": True, "USI_Ponder": False
    }
    engines = {}
    for arm in ARMS:
        engines[arm] = arena.Engine(
            launcher, "A", {**common_a, "FastLegalGuard": arm == "fastlegal"},
            "go movetime 200")
    kumoji = arena.Engine(folder / "kumoji", "B", {
        "PositionKnowledge": True,
        "PositionKnowledgeFile": str(folder / "full-knowledge.tsv"),
        "AdaptiveLongThink": True, "SearchThreads": 2,
        "RiskVerify": True, "USI_Ponder": True
    }, "go movetime 200")
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    played, per_arm = [], Counter()
    path = target / "games.jsonl"
    try:
        with path.open("w", encoding="utf-8") as dst:
            for pair in indices(shard):
                head = opening(NEW_SEED + pair, 6)
                for arm_index, arm in enumerate(ARMS):
                    for seat, black in enumerate((True, False)):
                        row = play_paired_game(
                            arena, engines[arm], kumoji, head, black, MAX_PLIES,
                            matched_seed(pair, seat))
                        row.update({
                            "pair": pair, "arm": arm, "game_id": pair * 4 + arm_index * 2 + seat,
                            "random_seed": matched_seed(pair, seat),
                            "study": "D13_frozen_G8_fastlegal_vs_same_kumoji",
                        })
                        if audit_game(row) != len(row["moves"]):
                            raise ValueError("game source not fully legally replayed")
                        if len(row["move_records"]) != len(row["played_moves"]):
                            raise ValueError("missing per-move search telemetry")
                        dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                        dst.flush()
                        played.append(row)
                        per_arm[arm] += 1
                        print("D13 game", row["game_id"], arm, row["winner"], row["reason"], flush=True)
    finally:
        for engine in engines.values():
            engine.close()
        kumoji.close()
    expected = sum(1 for _ in indices(shard)) * 4
    if len(played) != expected or any(per_arm[arm] != expected // 2 for arm in ARMS):
        raise ValueError("incomplete historical-opponent A/B shard")
    result = {
        "schema": "kumoji-cross200-shard-v1", "request": config(),
        "shard": shard, "games": expected, "ids": [r["game_id"] for r in played],
        "games_sha256": digest(path), "receipt_sha256": digest(folder / "receipt.json"),
        "fukaji_model_sha256": G8_SHA, "kumoji_binary_sha256": receipt["binary_sha256"],
        "knowledge_sha256": receipt["knowledge_sha256"]
    }
    (target / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def exact_sign_p(pos, neg):
    n = pos + neg
    if n == 0:
        return None
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n)


def aggregate(request_path, input_dir, parts, out):
    check_request(request_path)
    receipt, full_report = verify_input(input_dir)
    expected_receipt = digest(Path(input_dir) / "receipt.json")
    paths = list(Path(parts).glob("*/report.json"))
    if len(paths) != SHARDS:
        raise ValueError("expected 20 shards, got " + str(len(paths)))
    seen_shards, rows = set(), []
    for path in paths:
        part = json.loads(path.read_text())
        shard = part["shard"]
        if (shard in seen_shards or type(shard) is not int
            or not 0 <= shard < SHARDS or part["request"] != config()
            or part["receipt_sha256"] != expected_receipt
            or part["fukaji_model_sha256"] != G8_SHA
            or part["kumoji_binary_sha256"] != receipt["binary_sha256"]
            or part["knowledge_sha256"] != receipt["knowledge_sha256"]):
            raise ValueError("invalid cross200 shard provenance")
        seen_shards.add(shard)
        gamefile = path.parent / "games.jsonl"
        if digest(gamefile) != part["games_sha256"]:
            raise ValueError("cross200 source shard SHA mismatch")
        current = [json.loads(line) for line in gamefile.read_text().splitlines() if line.strip()]
        expected_ids = {pair * 4 + idx * 2 + seat
                        for pair in indices(shard) for idx in range(2) for seat in (0, 1)}
        if len(current) != len(expected_ids) or {r["game_id"] for r in current} != expected_ids:
            raise ValueError("missing/duplicate cross200 game")
        for r in current:
            game_id = r["game_id"]
            pair, offset = divmod(game_id, 4)
            arm_index, seat = divmod(offset, 2)
            if (r["pair"] != pair or r["arm"] != ARMS[arm_index]
                or r["a_black"] is not (seat == 0)
                or r.get("random_seed") != matched_seed(pair, seat)
                or r["opening"] != opening(NEW_SEED + pair, 6)
                or r["study"] != "D13_frozen_G8_fastlegal_vs_same_kumoji"
                or len(r["moves"]) > MAX_PLIES or len(r["played_moves"]) != len(r["move_records"])
                or audit_game(r) != len(r["moves"])):
                raise ValueError("A/B matched historical-opponent source not authentic")
        rows.extend(current)
    rows.sort(key=lambda r: r["game_id"])
    if len(rows) != 200 or [r["game_id"] for r in rows] != list(range(200)):
        raise ValueError("not exactly 200 independently replayed games")
    stats = {}
    pair_compare = Counter()
    per_pair = []
    for arm in ARMS:
        s = [r for r in rows if r["arm"] == arm]
        f = [m for r in s for m in r["move_records"] if m["engine"] == "A"]
        k = [m for r in s for m in r["move_records"] if m["engine"] == "B"]
        depths = Counter(m["search"].get("depth") for m in f)
        stats[arm] = {
            "games": len(s), "fukaji_wins": sum(r["winner"] == "A" for r in s),
            "kumoji_wins": sum(r["winner"] == "B" for r in s),
            "unresolved": sum(r["winner"] is None for r in s),
            "reasons": dict(Counter(r["reason"] for r in s)),
            "fukaji_plays": len(f), "kumoji_plays": len(k),
            "fukaji_depth_histogram": {str(key): count for key, count in depths.items()},
            "kumoji_knowledge_hits": sum(int(m["search"].get("knowledge_hits") or 0) for m in k),
            "kumoji_knowledge_probes": sum(int(m["search"].get("knowledge_probes") or 0) for m in k)
        }
    for pair in range(PAIRS):
        rs = rows[4 * pair:4 * pair + 4]
        if ([r["arm"] for r in rs] != ["baseline", "baseline", "fastlegal", "fastlegal"]
            or [r["a_black"] for r in rs] != [True, False, True, False]):
            raise ValueError("not same-opening both-color 4-game pairing")
        old, new = sum(r["winner"] == "A" for r in rs[:2]), sum(r["winner"] == "A" for r in rs[2:])
        label = "fastlegal" if new > old else "baseline" if old > new else "equal"
        pair_compare[label] += 1
        per_pair.append({"pair": pair, "baseline_fukaji_wins": old,
                         "fastlegal_fukaji_wins": new, "comparison": label})
    report = {
        "status": "D13_studied_FUKAJI_G8_fastlegal_v_same_KUMOJI_200_games",
        "research_only": True, "formal_adoption": False,
        "source_run": OLD_RUN, "input_receipt_sha256": expected_receipt,
        "kumoji_commit": KUMOJI_COMMIT,
        "kumoji_binary_sha256": receipt["binary_sha256"],
        "kumoji_knowledge_sha256": receipt["knowledge_sha256"],
        "kumoji_known_research_positions": full_report["playable_positions"],
        "g8_model_sha256": G8_SHA, "new_seed": NEW_SEED,
        "new_openings_disjoint_from_old": True, "pairs": PAIRS, "games": 200,
        "all_games_legally_replayed": True, "same_KUMOJI_binary_settings": True,
        "same_fukaji_model_and_settings_other_than_fast_legal": True,
        "same_KUMOJI_RandomSeed_for_identical_pair_and_color": True,
        "arms": stats, "paired_comparison": dict(pair_compare),
        "paired_sign_p_two_sided": exact_sign_p(pair_compare["fastlegal"], pair_compare["baseline"]),
        "per_pair": per_pair,
        "limitations": [
            "KUMOJI has more compute: 2 threads, optional 1000ms longthink, ponder; FUKAJI has one 200ms search.",
            "KUMOJI uses research-candidate 460-position knowledge, not a formally adopted new KUMOJI release.",
            "FUKAJI G8 is a research candidate. Engine/default/model are not promoted.",
            "Unresolved is neither a draw nor a win; checkmate and other outcomes remain separated.",
            "Against-KUMOJI outcomes are external challenge, not a substitute for frozen G8 independent selfplay."
        ]
    }
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    games_file = output / "all_games.jsonl"
    games_file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    report["all_games_sha256"] = digest(games_file)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_pair"}, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=("check", "prepare", "play", "aggregate"))
    p.add_argument("--request", required=True)
    p.add_argument("--input")
    p.add_argument("--shard", type=int)
    p.add_argument("--parts")
    p.add_argument("--out")
    a = p.parse_args()
    check_request(a.request)
    if a.mode == "check":
        print("D13 50 new openings x 2 arms x 2 colors validated")
    elif a.mode == "prepare":
        if not a.input: p.error("--input required")
        cfg, report = verify_input(a.input)
        print(json.dumps({"source_run": OLD_RUN, "source_positions": report["playable_positions"],
                          "source_binary": cfg["binary_sha256"]}, indent=2))
    elif a.mode == "play":
        if not a.input or a.shard is None or not a.out: p.error("input/shard/out required")
        play_shard(a.request, a.input, a.shard, a.out)
    else:
        if not a.input or not a.parts or not a.out: p.error("input/parts/out required")
        aggregate(a.request, a.input, a.parts, a.out)

if __name__ == "__main__":
    main()
