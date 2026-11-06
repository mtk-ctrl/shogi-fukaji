"""D42: D37 before vs D41 after NNUE, 50 fresh openings x swapped colors.

20 independent GitHub Actions jobs (2 or 3 opening pairs per lane); no learning.
The official model is never changed by this comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shogi
from fukaji.online_cycle import fresh_prefix
from fukaji.online_learning import atomic_json, digest, load_snapshot, validate_game
from fukaji.online_search import Position, terminal
from fukaji.native_nnue import NativeSelector
from scripts import d41_public_20 as prior

BASE = "170d6c1ee41d77b69f0b75722ece3deb85f3eea189f70d9789db477a38266db1"
AFTER = "5cb31550fd9a011cace5b007df1c58c4e1889655e0a2f170869ce8eddb189e73"
STATE_SHA = "41c8f56b76cd5d7aa87bff9131d553b304fa4134bf7448c05d3309458f66760e"
TRAINED_RELEASE_SHA = "0ceb5bc8f419579be064c97b720dde82f21fce267cc9b1bac6edb88438370731"
SELFPLAY_RELEASE_SHA = "8708db460d4d2bcad949d91ab881d9aeb804abcb8e28808dbbb70bda5c21bd9f"
SEED = 202611062114
PAIRS = 50
LANES = 20
MAX_PLIES = 512
NORMAL_MS = 200
LONG_MS = 1000
LONG_LIMIT = 10


def verify(path: Path, sha: str) -> None:
    actual = digest(path.read_bytes())
    if actual != sha:
        raise ValueError(f"SHA256 mismatch {path}: expected {sha}, got {actual}")


def pack(folder: Path, dest: Path) -> None:
    prior.pack_tar(folder, dest)


def restore(bundle: Path, dest: Path) -> None:
    prior.safe_extract(bundle, dest)


def get_models(work: Path):
    before, opt_before, _ = load_snapshot(work / "models", BASE)
    after, opt_after, _ = load_snapshot(work / "models", AFTER)
    if opt_before is None or opt_before["step"] != 2106:
        raise ValueError("D37 baseline optimizer is not Adam2106")
    if opt_after is None or opt_after["step"] != 2144:
        raise ValueError("D41 trained optimizer is not Adam2144")
    if before.inputs != after.inputs or before.inputs != 2548:
        raise ValueError("model input dimensions differ")
    return before, after


def read_d41_parent_ids(asset: Path):
    parents = set()
    with tarfile.open(asset, "r:gz") as tar:
        names = tar.getnames()
        games = sorted(x for x in names if x.startswith("game-lane-") and x.endswith(".json"))
        if len(games) != 40:
            raise ValueError("D41 training source must contain 40 games")
        for name in games:
            game = json.load(tar.extractfile(name))
            if game.get("role") != "train" or game.get("source") != "selfplay":
                raise ValueError("D41 source is not verified selfplay train data")
            parents.add(game["parent_id"])
    if len(parents) != 40:
        raise ValueError("D41 training openings are not 40 independent parents")
    return parents


def prepare(statefile: Path, trained: Path, selfplay: Path,
            historical: Path, output: Path) -> None:
    verify(statefile, STATE_SHA)
    verify(trained, TRAINED_RELEASE_SHA)
    verify(selfplay, SELFPLAY_RELEASE_SHA)
    output.mkdir(parents=True, exist_ok=True)
    work = output / "runtime"
    models = work / "models"
    models.mkdir(parents=True, exist_ok=True)

    with tarfile.open(statefile, "r:gz") as tar:
        state = json.load(tar.extractfile("d17-state/state.json"))
        if state["research_latest"] != BASE:
            raise ValueError("wrong frozen D37 parent")
        for ext in ("npz", "json"):
            member = f"d17-state/models/{BASE}.{ext}"
            (models / f"{BASE}.{ext}").write_bytes(tar.extractfile(member).read())
    with tarfile.open(trained, "r:gz") as tar:
        names = set(tar.getnames())
        if not {f"{AFTER}.npz", f"{AFTER}.json", "report.json"} <= names:
            raise ValueError("D41 training release lacks complete snapshot")
        report = json.load(tar.extractfile("report.json"))
        if (report.get("baseline") != BASE or
                report.get("trained_sha256") != AFTER or
                report.get("baseline_optimizer_step") != 2106 or
                report.get("new_optimizer_step") != 2144 or
                report.get("source_games_release_sha256") != SELFPLAY_RELEASE_SHA or
                report.get("candidate_strength_comparison_completed") is not False):
            raise ValueError("D41 release does not match frozen training provenance")
        for ext in ("npz", "json"):
            (models / f"{AFTER}.{ext}").write_bytes(tar.extractfile(f"{AFTER}.{ext}").read())
    get_models(work)
    historical_ids = json.loads(historical.read_text())
    if not (len(historical_ids) == 50 and len(set(historical_ids)) == 50 and
            all(isinstance(s, str) and len(s) == 64 for s in historical_ids)):
        raise ValueError("invalid D40 comparison parent list")
    d41_ids = read_d41_parent_ids(selfplay)
    exclusions = set(state["excluded_openings"]) | set(state["protected_parents"])
    exclusions.update(historical_ids)
    exclusions.update(d41_ids)
    frozen = set(exclusions)
    families = []
    for index in range(PAIRS):
        prefix, parent = fresh_prefix(SEED + index * 1019, exclusions)
        if parent in frozen or len(prefix) != 6:
            raise ValueError("new audit parent reused old train or comparison parent")
        families.append(dict(index=index, prefix=prefix, parent_id=parent, lane=index % LANES))
    if len({f["parent_id"] for f in families}) != PAIRS:
        raise ValueError("fresh 50 parent families were not independent")
    plan = {
        "schema": "fukaji-d42-before-after-20lane-paired100-v1",
        "before_sha256": BASE, "after_sha256": AFTER,
        "before_adam_step": 2106, "after_adam_step": 2144,
        "training_release_sha256": TRAINED_RELEASE_SHA,
        "source_selfplay_release_sha256": SELFPLAY_RELEASE_SHA,
        "state_sha256": STATE_SHA,
        "seed": SEED, "openings": PAIRS, "games": PAIRS * 2, "lanes": LANES,
        "role": "audit", "heldout_from_d37": True,
        "heldout_from_d40": True, "heldout_from_d41": True,
        "conditions": {
            "backend": "cpp_nnue", "quiescence": 0, "max_depth": 64,
            "normal_ms": NORMAL_MS, "long_ms": LONG_MS,
            "max_long_per_side": LONG_LIMIT, "max_total_plies": MAX_PLIES,
            "exploration_probability": 0, "same_opening_swapped_colors": True,
            "longthink_policy": "D41_check_captures_promotions_complexity",
            "parallel_workers": LANES, "single_job_models": "both"
        },
        "decision_plan": {
            "finish_all_100_before_judgment": True, "no_ad_hoc_early_stop": True,
            "unresolved_reported_separately": True,
            "clear_margin_threshold_wins": 8,
            "unresolved_max_for_clear_decision": 5,
            "if_ambiguous": "keep_D37_and_collect_further_independent_pairs",
            "adoption_not_automatic_in_workflow": True
        },
        "families": families
    }
    atomic_json(work / "plan.json", plan)
    pack(work, output / "runtime.tar.gz")
    atomic_json(output / "prepare-receipt.json",
                {"plan_sha256": digest((work / "plan.json").read_bytes()),
                 "before": BASE, "after": AFTER,
                 "parents": PAIRS, "lanes": LANES,
                 "excluded_counts": {
                     "state": len(state["excluded_openings"]),
                     "D40": len(historical_ids), "D41": len(d41_ids)
                 }})
    print(json.dumps({"prepared": True, "families": PAIRS,
                      "jobs": LANES, "candidate": AFTER}))


def segment(work: Path, index: int, candidate_black: int,
            dest: Path, search_limit: int) -> None:
    plan = json.loads((work / "plan.json").read_text())
    if not (0 <= index < PAIRS and candidate_black in (0, 1)):
        raise ValueError("out-of-bounds pair or side")
    family = plan["families"][index]
    before, after = get_models(work)
    names = [AFTER, BASE] if candidate_black == 1 else [BASE, AFTER]
    models = [after, before] if candidate_black == 1 else [before, after]
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / f"family-{index:03d}-candidate-black{candidate_black}.json"
    partial = dest / f"family-{index:03d}-candidate-black{candidate_black}.partial.json"
    if final.exists():
        return
    if partial.exists():
        state = json.loads(partial.read_text())
        if (state.get("index") != index or state.get("candidate_black") != candidate_black or
                state.get("models") != names or state.get("parent_id") != family["parent_id"]):
            raise ValueError("bad resume checkpoint")
        moves = list(state["moves"])
        trace = list(state["telemetry"])
        used = list(state["longthink_used"])
        if len(trace) != len(moves) - 6 or sum(r["longthink"] for r in trace) != sum(used):
            raise ValueError("checkpoint telemetry mismatch")
        state.update(moves=moves, telemetry=trace, longthink_used=used)
    else:
        moves = list(family["prefix"])
        trace = []
        used = [0, 0]
        state = dict(index=index, candidate_black=candidate_black, models=names,
                     parent_id=family["parent_id"], moves=moves, telemetry=trace,
                     longthink_used=used, status="in_progress")
        atomic_json(partial, state)

    board = shogi.Board()
    for word in moves:
        move = shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError("illegal checkpoint history")
        board.push(move)
    history = Position(board, models[0])
    outcome = None
    processed = 0
    with NativeSelector() as selector:
        selector.register(before)
        selector.register(after)
        while len(moves) < MAX_PLIES:
            legal = list(board.legal_moves)
            outcome = terminal(history, legal)
            if outcome:
                break
            if processed >= search_limit:
                return
            player = int(board.turn)
            longthink = prior.need_long(board, len(moves), legal, used[player])
            budget = LONG_MS if longthink else NORMAL_MS
            row = selector.select(board, models[player], movetime_ms=budget,
                                  max_depth=64, history=history, quiescence=0)
            word = row.get("move")
            if not word:
                raise ValueError("search failed on a nonterminal position")
            move = shogi.Move.from_usi(word)
            if move not in legal:
                raise ValueError("illegal NNUE search move")
            if longthink:
                used[player] += 1
            trace.append({"ply": len(moves)+1, "side": player,
                          "budget_ms": budget, "longthink": longthink, **row})
            history.push(move)
            moves.append(word)
            processed += 1
            atomic_json(partial, state)
        if outcome is None:
            outcome = terminal(history, list(board.legal_moves)) or {
                "status": "move_limit_unresolved", "winner": None}
    game = dict(start_sfen=shogi.STARTING_SFEN, moves=moves, **outcome,
                source="selfplay", role="audit", parent_id=family["parent_id"],
                seed=SEED+index*1019, models=names, learning_start_ply=6,
                telemetry=trace, family_index=index, candidate_side=0 if candidate_black else 1,
                candidate_black=candidate_black, lane=family["lane"],
                conditions={"movetime_ms":NORMAL_MS, "longthink_ms":LONG_MS,
                            "longthink_used_per_side":used,
                            "max_longthink_per_side":LONG_LIMIT,
                            "max_additional_plies":MAX_PLIES-6,
                            "backend_by_side":["cpp_nnue", "cpp_nnue"],
                            "quiescence_by_side":[0,0],
                            "exploration_probability":0,
                            "knowledge":False, "value":"nnue_only"})
    validate_game(game, before)
    if max(used) > LONG_LIMIT or len(trace) != len(moves)-6:
        raise ValueError("invalid clock counts")
    atomic_json(final, game)
    partial.unlink(missing_ok=True)


def worker(runtime: Path, lane: int, output: Path) -> None:
    if not 0 <= lane < LANES:
        raise ValueError("lane must be 0..19")
    work = output / "runtime"
    restore(runtime, work)
    get_models(work)
    games = output / "games"
    games.mkdir(parents=True, exist_ok=True)
    families = list(range(lane, PAIRS, LANES))
    for index in families:
        for candidate_black in (0,1):
            final = games / f"family-{index:03d}-candidate-black{candidate_black}.json"
            if final.exists():
                continue
            stalled = 0
            partial = games / f"family-{index:03d}-candidate-black{candidate_black}.partial.json"
            for attempt in range(24):
                if final.exists():
                    break
                before = len(json.loads(partial.read_text())["moves"]) if partial.exists() else 0
                cmd = [sys.executable, __file__, "segment", "--work", str(work),
                       "--index", str(index), "--candidate-black", str(candidate_black),
                       "--output", str(games), "--search-limit", "36"]
                try:
                    task = subprocess.run(cmd, text=True, capture_output=True, timeout=125)
                    if task.returncode:
                        raise RuntimeError(task.stderr[-1600:])
                except subprocess.TimeoutExpired:
                    pass
                after = len(json.loads(partial.read_text())["moves"]) if partial.exists() else MAX_PLIES
                stalled = stalled+1 if after <= before and not final.exists() else 0
                if stalled >= 3:
                    raise RuntimeError("no game progress across 3 restart attempts")
            if not final.exists():
                raise RuntimeError(f"game not completed: family {index}, side {candidate_black}")
    before_model, _ = get_models(work)
    for path in games.glob("*.json"):
        if path.name.endswith(".partial.json"):
            raise ValueError("partial game remained after worker success")
        game = json.loads(path.read_text())
        validate_game(game, before_model)
    package = output / f"lane-{lane:02d}.tar.gz"
    pack(games, package)
    print(json.dumps({"lane": lane, "families": families,
                      "games": len(families)*2,
                      "sha256": digest(package.read_bytes())}))


def collect(runtime: Path, downloads: Path, output: Path) -> None:
    work = output / "runtime"
    restore(runtime, work)
    before, after = get_models(work)
    plan_path = work / "plan.json"
    plan = json.loads(plan_path.read_text())
    if (plan["before_sha256"] != BASE or plan["after_sha256"] != AFTER or
            plan["games"] != 100 or plan["lanes"] != 20):
        raise ValueError("comparison plan changed before aggregation")
    lanes = sorted(downloads.glob("lane-*.tar.gz"))
    if [p.name for p in lanes] != [f"lane-{x:02d}.tar.gz" for x in range(LANES)]:
        raise ValueError("all 20 independent lane artifacts required")
    files = output / "games"
    files.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    trace = []
    game_hashes = {}
    pair_scores = defaultdict(list)
    total_plies = 0
    seen = set()
    for lane, archive in enumerate(lanes):
        expect = {f"family-{i:03d}-candidate-black{c}.json"
                  for i in range(lane, PAIRS, LANES) for c in (0,1)}
        with tarfile.open(archive, "r:gz") as tar:
            names = [m.name for m in tar.getmembers() if m.isfile()]
            if len(names) != len(expect) or set(names) != expect:
                raise ValueError(f"wrong lane {lane} game partition")
            for name in sorted(names):
                raw = tar.extractfile(name).read()
                game = json.loads(raw)
                index, c = int(name.split("-")[1]), int(name.split("black")[1].split(".")[0])
                family = plan["families"][index]
                expected_models = [AFTER,BASE] if c==1 else [BASE,AFTER]
                expected_side = 0 if c==1 else 1
                if (game["models"] != expected_models or game["parent_id"] != family["parent_id"] or
                        game["moves"][:6] != family["prefix"] or game["role"] != "audit" or
                        game["candidate_side"] != expected_side or game["lane"] != lane):
                    raise ValueError("wrong model, opening, side or role")
                validate_game(game, before)
                if game.get("status") == "checkmate" and game.get("winner") is None:
                    raise ValueError("no winner on checkmate")
                if len(game["telemetry"]) != len(game["moves"])-6 or len(game["moves"]) > MAX_PLIES:
                    raise ValueError("wrong move/trace count")
                used = [0,0]
                for row in game["telemetry"]:
                    if row["budget_ms"] not in (NORMAL_MS,LONG_MS):
                        raise ValueError("changed clock")
                    if row.get("backend") != "cpp_nnue":
                        raise ValueError("wrong engine backend")
                    if row["longthink"]:
                        used[int(row["side"])] += 1
                    trace.append(row)
                if (used != game["conditions"]["longthink_used_per_side"] or
                        max(used) > LONG_LIMIT):
                    raise ValueError("invalid longthink allotment")
                key = digest((game["start_sfen"]+"|"+" ".join(game["moves"])).encode())
                if key in seen:
                    raise ValueError("duplicate audit trajectory")
                seen.add(key)
                if game["winner"] is None:
                    if game["status"] == "repetition_draw":
                        counts["draws"] += 1
                        points = 0.5
                    else:
                        counts["unresolved"] += 1
                        points = None
                elif game["winner"] == expected_side:
                    counts["after_wins"] += 1
                    points = 1
                else:
                    counts["before_wins"] += 1
                    points = 0
                pair_scores[index].append(points)
                total_plies += len(game["moves"])
                game_hashes[name] = digest(raw)
                (files / name).write_bytes(raw)
    if len(game_hashes) != 100 or len(pair_scores) != 50 or any(len(v)!=2 for v in pair_scores.values()):
        raise ValueError("missing paired games")
    resolved = 100-counts["unresolved"]
    score = (counts["after_wins"] + .5*counts["draws"])/resolved if resolved else None
    lo = sum(sum(0 if v is None else v for v in pair_scores[i])/2 for i in range(50))/50
    hi = sum(sum(1 if v is None else v for v in pair_scores[i])/2 for i in range(50))/50
    eps = math.sqrt(math.log(2/.05)/(2*50))
    diff = counts["after_wins"] - counts["before_wins"]
    clear = counts["unresolved"] <= 5 and abs(diff) >= 8
    decision = ("after_advantage" if diff > 0 else "before_advantage") if clear else "inconclusive_keep_before"
    timing = {}
    for budget in (NORMAL_MS,LONG_MS):
        actual = [float(t["elapsed_ms"]) for t in trace if t["budget_ms"]==budget and "elapsed_ms" in t]
        timing[str(budget)] = {"observed_moves":len(actual),
                              "mean_elapsed_ms":statistics.mean(actual) if actual else None,
                              "max_elapsed_ms":max(actual) if actual else None,
                              "over_budget_100ms":sum(t>budget+100 for t in actual)}
    report = {
        "schema": "fukaji-d42-paired100-result-v1",
        "plan_sha256": digest(plan_path.read_bytes()),
        "before_sha256": BASE, "after_sha256": AFTER,
        "planned_games": 100, "completed_games": 100, "complete_pairs": 50,
        "after_wins": counts["after_wins"], "before_wins": counts["before_wins"],
        "draws": counts["draws"], "unresolved": counts["unresolved"],
        "after_resolved_score": score,
        "after_all_games_score_min": lo, "after_all_games_score_max": hi,
        "pair_Hoeffding_95pct_interval": [max(0,lo-eps),min(1,hi+eps)],
        "difference_after_minus_before": diff,
        "predefined_decision": decision, "decision_is_heuristic_not_proof": True,
        "formal_model_changed": False,
        "independent_github_actions_lanes_requested": 20,
        "game_legal_replay_count":len(game_hashes),
        "validated_plies":total_plies,
        "timings": timing,
        "game_sha256": game_hashes,
        "families": [{"index":i, "candidate_score_lower":sum(0 if v is None else v for v in pair_scores[i])/2,
                     "candidate_score_upper":sum(1 if v is None else v for v in pair_scores[i])/2}
                    for i in range(50)]
    }
    atomic_json(output / "report.json", report)
    (output / "plan.json").write_bytes(plan_path.read_bytes())
    pack(output / "games", output / "D42-100games-only.tar.gz")
    package_folder = output / "publication"
    package_folder.mkdir(exist_ok=True)
    (package_folder / "report.json").write_bytes((output/"report.json").read_bytes())
    (package_folder / "plan.json").write_bytes(plan_path.read_bytes())
    for p in files.glob("*.json"):
        dest=package_folder/"games"
        dest.mkdir(exist_ok=True)
        (dest/p.name).write_bytes(p.read_bytes())
    package = output / "D42-D37-before-vs-D41-after-100games.tar.gz"
    pack(package_folder, package)
    atomic_json(output/"receipt.json",
                {"package_sha256":digest(package.read_bytes()),
                 "package_size":package.stat().st_size,
                 "games":100,"plan_sha256":report["plan_sha256"],
                 "before":BASE,"after":AFTER,"formal_changed":False})
    print(json.dumps({k:v for k,v in report.items()
                      if k not in ("game_sha256","families")},ensure_ascii=False))


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("prepare")
    a.add_argument("--state",type=Path,required=True)
    a.add_argument("--trained",type=Path,required=True)
    a.add_argument("--selfplay",type=Path,required=True)
    a.add_argument("--excluded",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    b=sub.add_parser("worker")
    b.add_argument("--runtime",type=Path,required=True)
    b.add_argument("--lane",type=int,required=True)
    b.add_argument("--output",type=Path,required=True)
    c=sub.add_parser("segment")
    c.add_argument("--work",type=Path,required=True)
    c.add_argument("--index",type=int,required=True)
    c.add_argument("--candidate-black",type=int,choices=[0,1],required=True)
    c.add_argument("--output",type=Path,required=True)
    c.add_argument("--search-limit",type=int,default=36)
    d=sub.add_parser("collect")
    d.add_argument("--runtime",type=Path,required=True)
    d.add_argument("--downloads",type=Path,required=True)
    d.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.command=="prepare":
        prepare(args.state,args.trained,args.selfplay,args.excluded,args.output)
    elif args.command=="worker":
        worker(args.runtime,args.lane,args.output)
    elif args.command=="segment":
        segment(args.work,args.index,args.candidate_black,args.output,args.search_limit)
    else:
        collect(args.runtime,args.downloads,args.output)


if __name__=="__main__":
    main()
