"""G7 FastMoveList versus frozen KUMOJI: equal clock/one search thread.

Independent research only. Both use go movetime 200, no long-think, no ponder,
no position knowledge. Equal time budget does not imply equal hardware work.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from .arena import opening
from .direct_arena import digest
from .kumoji_compare import G7_SHA, KUMOJI_REF, play_one, verify

G7_RUN = 40434466661
SEED, PAIRS, SHARDS, MAX_PLIES = 20274927, 32, 20, 240
FIELDS = ("nodes", "turns", "depth_sum", "elapsed_ms",
          "knowledge_probes", "knowledge_hits", "ponder_hits", "long_think_reports")


def expected_request():
    return dict(research_only=True, training_allowed=False,
                model_adoption_allowed=False, official_engine_changes_allowed=False,
                g7_source_run=G7_RUN, g7_sha256=G7_SHA, kumoji_ref=KUMOJI_REF,
                seed=SEED, pairs=PAIRS, shards=SHARDS, opening_plies=6,
                max_plies=MAX_PLIES, movetime_ms=200, g7_depth=4,
                g7_mix_percent=25, g7_fast_relative=True,
                g7_root_alpha_beta=True, g7_fast_move_list=True,
                kumoji_adaptive_long_think=False, kumoji_ponder=False,
                kumoji_position_knowledge=False, kumoji_search_threads=1,
                comparative_only=True)


def validate_request(cfg):
    if cfg != expected_request():
        raise ValueError("unapproved equal-resource FUKAJI/KUMOJI protocol")
    return cfg


def engine_options(model_path):
    return (
        dict(ValueModelFile=str(Path(model_path).resolve()),
             ValueModelType="relative", ValueModelMix=25, SearchDepth=4,
             FastRelativeEval=True, RootAlphaBeta=True,
             FastMoveList=True, USI_Ponder=False),
        dict(AdaptiveLongThink=False, PositionKnowledge=False,
             SearchThreads=1, USI_Ponder=False, RiskVerify=True),
    )


def play_shard(request, g7_path, kumoji_path, shard, out):
    validate_request(json.loads(Path(request).read_text(encoding="utf-8")))
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("invalid shard")
    if digest(g7_path) != G7_SHA:
        raise ValueError("frozen G7 SHA mismatch")
    from benchmarks.arena import Engine
    aopts, bopts = engine_options(g7_path)
    g7 = Engine("build/fukaji-usi", "g7", aopts, "go movetime 200")
    kumoji = Engine(str(kumoji_path), "kumoji", bopts, "go movetime 200")
    rows = []
    try:
        for pair in range(shard, PAIRS, SHARDS):
            head = opening(SEED + pair, 6)
            if len(head) != 6:
                raise ValueError("incomplete independent opening")
            for g7_black in (True, False):
                row = play_one(head, g7_black, g7, kumoji,
                               SEED + 4 * pair + int(not g7_black))
                row.update(pair=pair, g7_black=g7_black)
                verify(row)
                rows.append(row)
    finally:
        g7.close()
        kumoji.close()
    if len(rows) != 2 * len(range(shard, PAIRS, SHARDS)):
        raise ValueError("incorrect shard game count")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    games = root / "games.jsonl"
    games.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                             for r in rows), encoding="utf-8")
    report = dict(status="research_only_not_adopted", shard=shard,
                  games=len(rows), pairs=len(rows)//2, request=expected_request(),
                  g7_sha256=G7_SHA, kumoji_binary_sha256=digest(kumoji_path),
                  games_sha256=digest(games), legal_replay_verified=True)
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def aggregate(request, parts, out):
    validate_request(json.loads(Path(request).read_text(encoding="utf-8")))
    paths = list(Path(parts).glob("*/report.json"))
    if len(paths) != SHARDS:
        raise ValueError("missing independent shard")
    rows, seen, binaries = [], set(), set()
    for path in paths:
        rep = json.loads(path.read_text())
        shard = rep.get("shard")
        if (type(shard) is not int or shard not in range(SHARDS)
                or shard in seen or rep.get("request") != expected_request()
                or rep.get("g7_sha256") != G7_SHA
                or rep.get("legal_replay_verified") is not True
                or rep.get("games") != 2*len(range(shard, PAIRS, SHARDS))):
            raise ValueError("invalid shard report")
        seen.add(shard)
        binaries.add(rep["kumoji_binary_sha256"])
        game_file = path.parent / "games.jsonl"
        if digest(game_file) != rep["games_sha256"]:
            raise ValueError("shard hash mismatch")
        group = [json.loads(x) for x in game_file.read_text().splitlines()]
        if len(group) != rep["games"]:
            raise ValueError("shard row mismatch")
        for pair in range(shard, PAIRS, SHARDS):
            pair_rows = [r for r in group if r.get("pair") == pair]
            if len(pair_rows) != 2 or {r.get("g7_black") for r in pair_rows} != {True, False}:
                raise ValueError("invalid swapped pair")
            head = opening(SEED + pair, 6)
            for row in pair_rows:
                if row["opening"] != head:
                    raise ValueError("opening mismatch")
                verify(row)
                rows.append(row)
    if len(binaries) != 1 or len(rows) != PAIRS*2:
        raise ValueError("wrong binary count or game count")
    rows.sort(key=lambda r: (r["pair"], not r["g7_black"]))
    if len({tuple(r["opening"]) for r in rows}) != PAIRS:
        raise ValueError("non-independent openings")
    wins, pairs = Counter(), Counter()
    for row in rows:
        if row["winner"] is None:
            wins["unresolved"] += 1
        elif (row["winner"] == 0) == row["g7_black"]:
            wins["g7"] += 1
        else:
            wins["kumoji"] += 1
        if row["stats"]["kumoji_knowledge_hits"] or row["stats"]["kumoji_ponder_hits"] or row["stats"]["kumoji_long_think_reports"]:
            raise ValueError("forbidden KUMOJI resource advantage detected")
    for pair in range(PAIRS):
        two = [r for r in rows if r["pair"] == pair]
        w = sum(r["winner"] is not None and (r["winner"] == 0) == r["g7_black"] for r in two)
        l = sum(r["winner"] is not None and (r["winner"] == 0) != r["g7_black"] for r in two)
        pairs["g7" if w>l else "kumoji" if l>w else "equal"] += 1
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    games = root / "all_games.jsonl"
    games.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = dict(status="research_only_no_adoption", games=len(rows), pairs=PAIRS,
                   g7_wins=wins["g7"], kumoji_wins=wins["kumoji"],
                   unresolved=wins["unresolved"], pair_comparison=dict(pairs),
                   termination=dict(Counter(r["status"] for r in rows)),
                   request=expected_request(), kumoji_binary_sha256=next(iter(binaries)),
                   games_sha256=digest(games), legal_replay_verified=True,
                   stats={k:sum(r["stats"][k] for r in rows) for k in rows[0]["stats"]},
                   caution="Equal movetime and one thread; different implementations/throughput. No knowledge, long think or ponder. No training or adoption.")
    (root / "aggregate_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("play", "aggregate"), required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--g7-model")
    p.add_argument("--kumoji-engine")
    p.add_argument("--shard", type=int)
    p.add_argument("--parts")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.mode == "play":
        if a.shard is None or not a.g7_model or not a.kumoji_engine:
            p.error("missing play args")
        play_shard(a.request, a.g7_model, a.kumoji_engine, a.shard, a.out)
    else:
        if not a.parts:
            p.error("missing parts")
        aggregate(a.request, a.parts, a.out)


if __name__ == "__main__":
    main()
