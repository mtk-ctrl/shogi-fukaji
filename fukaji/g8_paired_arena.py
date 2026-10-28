"""Independent G8-vs-G7 paired 240-game research, no automatic adoption.

A preflight verifies original ZIP bytes and training data lineage. Games use
new openings and color swaps, not training/validation/test source openings.
Unresolved games are never counted as wins or draws.
"""
import argparse
import hashlib
import io
import json
import math
import os
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from .arena import opening
from .direct_arena import replay_check
from .g8_artifact_selection import artifact_bytes
from .generation_arena import play
from .generation8_train import (
    OLD_RUN, OLD_ID, OLD_NAME, OLD_G7_ZIP_SHA, NEW_RUN, NEW_ID, NEW_NAME,
    SOURCE_AGGREGATE_SHA, SELECTION_RUN, SELECTION_ID, SELECTION_NAME,
    SELECTION_ZIP_SHA, NEW_MATES_SHA, ALL_GAMES_SHA, safe_read,
)
from .relative_value import RelativeValue

G7_SHA = "417d18e7b64494c9febdf6d217afbe4e4979e68d04528f16db8b45c0bcde2a0e"
G8_RUN, G8_ID = 40636602223, 12793956048
G8_NAME = "fukaji-generation8-research-40636602223"
G8_ZIP_SHA = "9b5c004446f4b93a529d52fa96243ccbd5ac434373fe1ff69191bee9c8e6306e"
G8_SHA = "18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492"
SEED, PAIRS, SHARDS, MAX_PLIES = 20276827, 120, 20, 240
PER_SHARD_PAIRS = PAIRS // SHARDS
FIELDS = ("nodes", "turns", "depth_sum", "incomplete", "elapsed_ms")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def expected_request(enabled=False):
    return {
        "enabled": enabled, "research_only": True,
        "training_allowed": False, "model_adoption_allowed": False,
        "g7_run": OLD_RUN, "g7_artifact_id": OLD_ID,
        "g7_artifact_zip_sha256": OLD_G7_ZIP_SHA, "g7_model_sha256": G7_SHA,
        "g8_run": G8_RUN, "g8_artifact_id": G8_ID,
        "g8_artifact_zip_sha256": G8_ZIP_SHA, "g8_model_sha256": G8_SHA,
        "seed": SEED, "pairs": PAIRS, "shards": SHARDS,
        "opening_plies": 6, "max_plies": MAX_PLIES,
        "movetime_ms": 200, "depth": 2, "mix_percent": 25,
        "fast_relative": True, "root_alpha_beta_both": True,
        "source_games_used_as_openings": False,
    }


def validate_request(cfg):
    if cfg == expected_request(False):
        return False
    if cfg == expected_request(True):
        return True
    raise ValueError("unapproved G8-vs-G7 paired protocol")


def pairs_for_shard(shard):
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("invalid shard")
    return range(shard * PER_SHARD_PAIRS, (shard + 1) * PER_SHARD_PAIRS)


def preflight(out):
    if os.environ.get("GITHUB_REPOSITORY") != "mtk-ctrl/shogi-fukaji":
        raise ValueError("wrong repository")
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise ValueError("missing read-only artifact token")
    raw_g7 = artifact_bytes(OLD_ID, OLD_NAME, OLD_RUN, OLD_G7_ZIP_SHA, token)
    raw_g8 = artifact_bytes(G8_ID, G8_NAME, G8_RUN, G8_ZIP_SHA, token)
    raw_new = artifact_bytes(NEW_ID, NEW_NAME, NEW_RUN, SOURCE_AGGREGATE_SHA, token)
    raw_sel = artifact_bytes(SELECTION_ID, SELECTION_NAME, SELECTION_RUN,
                             SELECTION_ZIP_SHA, token)
    with ZipFile(io.BytesIO(raw_g7)) as z:
        g7_model = safe_read(z, "mover-relative-generation7-research.npz")
        old_mates = safe_read(z, "merged_verified_mates.jsonl")
        old_splits = json.loads(safe_read(z, "frozen_parent_splits.json"))
    with ZipFile(io.BytesIO(raw_new)) as z:
        new_mates = safe_read(z, "verified_mates.jsonl")
        all_games = safe_read(z, "all_selfplay_games.jsonl")
    with ZipFile(io.BytesIO(raw_sel)) as z:
        new_splits = json.loads(safe_read(z, "selected/new_parent_splits_400.json"))
        selection = json.loads(safe_read(z, "selected/selection_manifest.json"))
    with ZipFile(io.BytesIO(raw_g8)) as z:
        g8_model = safe_read(z, "mover-relative-generation8-research.npz")
        merged = safe_read(z, "merged_verified_mates.jsonl")
        mapping = json.loads(safe_read(z, "frozen_parent_splits.json"))
        report = json.loads(safe_read(z, "generation8_report.json"))
    if (digest(g7_model) != G7_SHA or digest(g8_model) != G8_SHA or
            digest(new_mates) != NEW_MATES_SHA or
            digest(all_games) != ALL_GAMES_SHA or
            merged != old_mates + new_mates or
            mapping != dict(old_splits, **new_splits) or
            len(mapping) != 957 or
            len(old_splits) != 557 or len(new_splits) != 400 or
            Counter(mapping.values()) !=
            {"train": 767, "validation": 95, "test": 95} or
            selection.get("all_400_games_full_legal_replay_verified") is not True or
            report.get("status") != "generation8_research_candidate_not_promoted" or
            report.get("parent_model_sha256") != G7_SHA or
            report.get("candidate_model_sha256") != G8_SHA or
            report.get("merged_verified_mates_sha256") != digest(merged) or
            report.get("split_mate_games") !=
            {"train": 665, "validation": 81, "test": 84} or
            report.get("new_train_mates") != 206 or
            report.get("unresolved_excluded") != 142 or
            report.get("model_promoted") is not False or
            report.get("strength_proven") is not False):
        raise ValueError("G7/G8 candidate or parent isolation audit failed")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "g7.npz").write_bytes(g7_model)
    (root / "g8.npz").write_bytes(g8_model)
    receipt = {
        "status": "sha_pinned_G8_training_artifact_independently_rechecked",
        "g7_zip_sha256": OLD_G7_ZIP_SHA, "g8_zip_sha256": G8_ZIP_SHA,
        "g8_model_sha256": G8_SHA, "g7_model_sha256": G7_SHA,
        "new_mates_sha256": NEW_MATES_SHA,
        "merged_mates_sha256": digest(merged),
        "frozen_parent_groups": 957, "new_train_mates": 206,
        "unresolved_not_teacher": 142, "model_promoted": False,
        "comparison_seed": SEED, "pairs": PAIRS, "games": PAIRS * 2,
    }
    (root / "preflight_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipt


def play_shard(request, g7_file, g8_file, shard, out):
    if not validate_request(json.loads(Path(request).read_text(encoding="utf-8"))):
        raise ValueError("disabled comparison request")
    pairs = pairs_for_shard(shard)
    if digest(Path(g7_file).read_bytes()) != G7_SHA or digest(Path(g8_file).read_bytes()) != G8_SHA:
        raise ValueError("comparison model SHA mismatch")
    g7, g8 = RelativeValue(g7_file, mix_percent=25), RelativeValue(g8_file, mix_percent=25)
    rows = []
    for pair in pairs:
        head = opening(SEED + pair, 6)
        if len(head) != 6:
            raise ValueError("invalid independent opening")
        for black in (True, False):
            row = play(head, black, g8, g7, max_plies=MAX_PLIES)
            row["stats"] = {
                ("g8_" if key.startswith("g3_") else "g7_") + key.split("_", 1)[1]: value
                for key, value in row["stats"].items()
            }
            row.update({"pair": pair, "g8_black": black, "opening": head})
            replay_check(head, row["moves"], row)
            rows.append(row)
    if len(rows) != PER_SHARD_PAIRS * 2:
        raise ValueError("incomplete shard")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    games = root / "paired_games.jsonl"
    games.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                     encoding="utf-8")
    report = {"status": "research_only_not_promoted", "shard": shard,
              "games": len(rows), "pairs": PER_SHARD_PAIRS, "seed": SEED,
              "g7_model_sha256": G7_SHA, "g8_model_sha256": G8_SHA,
              "movetime_ms": 200, "depth": 2, "max_plies": MAX_PLIES,
              "mix_percent": 25, "root_alpha_beta_both": True,
              "legal_replay_verified": True,
              "games_sha256": digest(games.read_bytes())}
    (root / "paired_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def wilson(wins, losses):
    n = wins + losses
    if not n:
        return None
    z = 1.95996398454
    p = wins / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [max(0, c-h), min(1, c+h)]


def aggregate(parts, out):
    paths = list(Path(parts).glob("*/paired_report.json"))
    if len(paths) != SHARDS:
        raise ValueError("missing paired shard")
    seen, rows = set(), []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        shard = report.get("shard")
        if (type(shard) is not int or shard not in range(SHARDS) or shard in seen or
                report.get("status") != "research_only_not_promoted" or
                report.get("games") != PER_SHARD_PAIRS*2 or
                report.get("pairs") != PER_SHARD_PAIRS or
                report.get("seed") != SEED or
                report.get("g7_model_sha256") != G7_SHA or
                report.get("g8_model_sha256") != G8_SHA or
                report.get("movetime_ms") != 200 or
                report.get("depth") != 2 or report.get("max_plies") != MAX_PLIES or
                report.get("mix_percent") != 25 or
                report.get("root_alpha_beta_both") is not True or
                report.get("legal_replay_verified") is not True):
            raise ValueError("invalid paired shard metadata")
        seen.add(shard)
        raw = (path.parent / "paired_games.jsonl").read_bytes()
        if digest(raw) != report.get("games_sha256"):
            raise ValueError("shard game SHA mismatch")
        sub = [json.loads(s) for s in raw.decode("utf-8").splitlines() if s.strip()]
        if len(sub) != PER_SHARD_PAIRS*2:
            raise ValueError("missing paired games")
        for pair in pairs_for_shard(shard):
            games = [g for g in sub if g.get("pair") == pair]
            if (len(games) != 2 or
                    [g.get("g8_black") for g in games] != [True, False] or
                    games[0].get("opening") != games[1].get("opening") or
                    games[0].get("opening") != opening(SEED + pair, 6)):
                raise ValueError("missing or swapped opening pair")
            for game in games:
                replay_check(game["opening"], game["moves"], game)
                if len(game["opening"]) + len(game["moves"]) > MAX_PLIES:
                    raise ValueError("game exceeds move cap")
                if set(game.get("stats", {})) != {
                        side + "_" + field for side in ("g7", "g8") for field in FIELDS}:
                    raise ValueError("missing search telemetry")
                rows.append(game)
    if seen != set(range(SHARDS)) or len(rows) != 240:
        raise ValueError("missing 240 games")
    rows.sort(key=lambda r: (r["pair"], not r["g8_black"]))
    wins = Counter()
    pair_result = Counter()
    for pair in range(PAIRS):
        games = rows[2*pair:2*pair+2]
        g8, g7 = 0, 0
        for game in games:
            if game["status"] != "checkmate":
                wins["unresolved"] += 1
            elif (game["winner"] == 0) == game["g8_black"]:
                wins["g8"] += 1
                g8 += 1
            else:
                wins["g7"] += 1
                g7 += 1
        pair_result["g8_advantage" if g8 > g7 else
                    "g7_advantage" if g7 > g8 else "equal_decisive_wins"] += 1
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    games_file = root / "all_paired_games.jsonl"
    games_file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                          encoding="utf-8")
    report = {
        "status": "research_only_no_adoption", "games": 240, "pairs": 120,
        "g8_checkmate_wins": wins["g8"], "g7_checkmate_wins": wins["g7"],
        "unresolved_not_counted_as_wins_or_draws": wins["unresolved"],
        "pair_decisive_wins_comparison": dict(pair_result),
        "decisive_only_g8_win_rate_95pct_wilson": wilson(wins["g8"], wins["g7"]),
        "g7_model_sha256": G7_SHA, "g8_model_sha256": G8_SHA,
        "seed": SEED, "movetime_ms": 200, "depth": 2,
        "max_plies": MAX_PLIES, "mix_percent": 25,
        "all_pairs_color_swapped": True, "legal_replay_verified": True,
        "games_sha256": digest(games_file.read_bytes()),
        "candidate_promoted": False,
        "limitations": [
            "Unresolved games are NOT draws, losses, or wins.",
            "Decisive-only Wilson interval is conditional on termination, not an overall Elo claim.",
            "One new-seed 240-game comparison is not proof of broad generalization.",
            "G8 is a research candidate; KUMOJI and official model unchanged.",
        ],
    }
    (root / "paired_aggregate_report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("validate", "prepare", "play", "aggregate"), required=True)
    p.add_argument("--request")
    p.add_argument("--g7")
    p.add_argument("--g8")
    p.add_argument("--shard", type=int)
    p.add_argument("--parts")
    p.add_argument("--out")
    a = p.parse_args()
    if a.mode == "aggregate":
        if not a.parts or not a.out:
            p.error("missing aggregate arguments")
        print(json.dumps(aggregate(a.parts, a.out), indent=2))
        return
    if not a.request:
        p.error("missing request")
    enabled = validate_request(json.loads(Path(a.request).read_text(encoding="utf-8")))
    if a.mode == "validate":
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
                f.write("enabled=" + ("true" if enabled else "false") + "\n")
        print("G8 paired request valid; enabled:", enabled)
    elif a.mode == "prepare":
        if not enabled or not a.out:
            p.error("disabled or missing preparation destination")
        print(json.dumps(preflight(a.out), indent=2))
    else:
        if not enabled or not all((a.g7, a.g8, a.out)) or a.shard is None:
            p.error("disabled or missing paired game inputs")
        print(json.dumps(play_shard(a.request, a.g7, a.g8, a.shard, a.out), indent=2))


if __name__ == "__main__":
    main()
