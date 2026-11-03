"""D37: prepare, play, verify and learn one private 20-runner generation."""
import argparse
import json
from pathlib import Path
import random
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
import shogi

from fukaji.native_nnue import build_native
from fukaji.online_cycle import fresh_prefix, lock, play, read_games, write_games
from fukaji.online_learning import (
    atomic_json, digest, fit, game_key, implementation_digest, load_snapshot,
    make_samples, save_snapshot, validate_game,
)
from fukaji.online_state import verify

RUNNERS = 20
FAMILIES = 100
SEED = 202611040470
PARENT = "7d74b86bcd5bd1a0e7cefc10f8d01cd2c816822ac4636e6f1603795fab9dcf6b"
FORMAL = "e0cbd2abe53b1d151bb73407f848b4087456c1466121333ad490a96aee131b0a"
MOVETIME = 50
MAX_PLIES = 256
DECIDED = {"checkmate", "perpetual_check", "repetition_draw"}


def opening_families(state):
    excluded = set(state["excluded_openings"]) | set(state["protected_parents"])
    for index in range(200):
        fresh_prefix(202611040370 + index * 1000, excluded)
    result = []
    for index in range(FAMILIES):
        prefix, parent = fresh_prefix(SEED + index * 1000, excluded)
        result.append(dict(index=index, prefix=prefix, parent=parent))
    return result


def prepare(root, package):
    with lock(root):
        verify(root)
        raw = (root / "state.json").read_bytes()
        state = json.loads(raw)
        if (state["research_latest"] != PARENT or state["paused"]
            or state.get("pending_campaign") or state.get("pending_comparison")):
            raise ValueError("idle D36 source required")
        models = package / "models"
        models.mkdir(parents=True)
        for key in (PARENT, FORMAL):
            load_snapshot(root / "models", key)
            for suffix in (".npz", ".json"):
                name = key + suffix
                copyfile(root / "models" / name, models / name)
        plan = dict(schema="fukaji-fresh-train-v1", parent=PARENT, opponent=FORMAL,
                    role="train", planned_games=2*FAMILIES, formal_adoption=False,
                    runners=RUNNERS, seed=SEED, movetime_ms=MOVETIME, max_plies=MAX_PLIES,
                    families=opening_families(state), source_sha256=digest(raw),
                    worker_sha256=digest(Path(__file__).read_bytes()),
                    engine_sha256=implementation_digest())
        atomic_json(package / "plan.json", plan)
    return digest((package / "plan.json").read_bytes())


def checked_plan(package, expected):
    raw = (package / "plan.json").read_bytes()
    if digest(raw) != expected:
        raise ValueError("plan hash mismatch")
    plan = json.loads(raw)
    items = plan["families"]
    if (plan["schema"] != "fukaji-fresh-train-v1" or plan["role"] != "train"
        or plan["formal_adoption"] is not False or plan["planned_games"] != 2*FAMILIES
        or (plan["parent"], plan["opponent"], plan["runners"], plan["seed"],
            plan["movetime_ms"], plan["max_plies"]) != (PARENT, FORMAL, RUNNERS, SEED, MOVETIME, MAX_PLIES)
        or plan["worker_sha256"] != digest(Path(__file__).read_bytes())
        or plan["engine_sha256"] != implementation_digest()
        or [f["index"] for f in items] != list(range(FAMILIES))
        or len({f["parent"] for f in items}) != FAMILIES):
        raise ValueError("plan does not match worker protocol")
    return plan


def side_models(side):
    return [PARENT, FORMAL] if side == 0 else [FORMAL, PARENT]


def run_shard(package, expected, shard, output):
    plan = checked_plan(package, expected)
    build_native()
    models = {key: load_snapshot(package / "models", key)[0] for key in (PARENT, FORMAL)}
    pairs = []
    for family in plan["families"][shard::RUNNERS]:
        games = []
        for side in (0, 1):
            ids = side_models(side)
            games.append(play([models[key] for key in ids], ids, family["prefix"],
                              role="train", parent_id=family["parent"],
                              seed=SEED + 2*family["index"] + side, movetime=MOVETIME,
                              max_plies=MAX_PLIES, quiescence=0, native_backend=True))
        pairs.append(dict(index=family["index"], games=games))
    atomic_json(output, dict(plan_sha256=expected, shard=shard, pairs=pairs))


def collect_verified(root, package, expected, results, state):
    plan = checked_plan(package, expected)
    if (digest((root / "state.json").read_bytes()) != plan["source_sha256"]
        or plan["families"] != opening_families(state)):
        raise ValueError("source state or opening assignment changed")
    parent = load_snapshot(root / "models", PARENT)[0]
    expected_names = {f"shard-{i}" for i in range(RUNNERS)}
    paths = list(results.glob("shard-*/result.json"))
    if {p.parent.name for p in paths} != expected_names:
        raise ValueError("expected all 20 runner results")
    verified = {}
    for i in range(RUNNERS):
        result = json.loads((results / f"shard-{i}/result.json").read_text())
        selected = plan["families"][i::RUNNERS]
        if (result["plan_sha256"] != expected or result["shard"] != i
            or [p["index"] for p in result["pairs"]] != [f["index"] for f in selected]):
            raise ValueError("runner result does not match assignment")
        for family, pair in zip(selected, result["pairs"]):
            if len(pair["games"]) != 2:
                raise ValueError("both colors required")
            for side, game in enumerate(pair["games"]):
                c = game["conditions"]
                if (game["models"] != side_models(side) or game["parent_id"] != family["parent"]
                    or game["start_sfen"] != shogi.STARTING_SFEN or game["role"] != "train"
                    or game["moves"][:6] != family["prefix"] or game["learning_start_ply"] != 6
                    or len(game["moves"]) > 6+MAX_PLIES or game["seed"] != SEED+2*family["index"]+side
                    or c["backend_by_side"] != ["cpp_nnue", "cpp_nnue"] or c["quiescence_by_side"] != [0, 0]
                    or c["movetime_ms"] != MOVETIME or c["max_additional_plies"] != MAX_PLIES
                    or c["exploration_probability"] != .05 or c["exploration_additional_plies"] != 24):
                    raise ValueError("game does not match frozen protocol")
                validate_game(game, parent, training=True, protected=state["protected_parents"])
            verified[family["index"]] = pair["games"]
    return plan, verified


def train(root, package, expected, results, apply):
    with lock(root):
        verify(root)
        state_path = root / "state.json"
        original = state_path.read_bytes()
        state = json.loads(original)
        plan, pairs = collect_verified(root, package, expected, results, state)
        games = [game for i in range(FAMILIES) for game in pairs[i]]
        summary = dict(complete=True, completed_games=len(games), outcomes_match=True,
                       legal_replay_moves=sum(len(g["moves"]) for g in games),
                       plan_sha256=expected, formal_adoption=False)
        if not apply:
            return summary
        out = root / "d37-private-selfplay"
        if out.exists() or state["research_latest"] != PARENT:
            raise ValueError("new D37 research state required")
        model, optimizer, _ = load_snapshot(root / "models", PARENT)
        fresh = {game_key(g): g for g in games if g["status"] in DECIDED}
        if len(fresh) < 32 or optimizer["step"] != 1850:
            raise ValueError("at least 32 decided new games and the D36 optimizer are required")
        old = {}
        for receipt in [state["bootstrap"], *state["train_batches"], *state["supplemental_train_batches"]]:
            for game in read_games(root, receipt):
                key = game_key(game)
                if game["status"] in DECIDED and key not in fresh:
                    old[key] = game
        replay = random.Random(SEED).sample(sorted(old), 128)
        rng = np.random.default_rng(SEED)
        samples = []
        for subset in (list(fresh.values()), [old[k] for k in replay]):
            x, y, _ = make_samples(subset, model, seed=SEED, per_game=32, outcome_weight=.75,
                                   td_lambda=.8, protected=state["protected_parents"])
            draws = rng.integers(len(y), size=4096)
            samples.append((x[draws], y[draws]))
        width = max(x.shape[1] for x, _ in samples)
        x = np.concatenate([np.pad(a, ((0, 0), (0, width-a.shape[1])),
                                   constant_values=model.inputs) for a, _ in samples])
        y = np.concatenate([b for _, b in samples])
        new, adam, stats = fit(model, optimizer, x, y, seed=SEED, epochs=2, lr=.001, batch_size=128)
        if adam["step"] != 1978:
            raise ValueError("optimizer update count differs")
        candidate = save_snapshot(root / "models", new, adam, dict(
            parent=PARENT, purpose="D37 private continuation", seed=SEED, training=stats,
            collection_plan_sha256=expected, replay_game_hashes=replay,
        ))
        _, checked_adam, _ = load_snapshot(root / "models", candidate)
        if checked_adam["step"] != 1978:
            raise ValueError("saved optimizer verification failed")
        with TemporaryDirectory(dir=root) as temporary:
            staging = Path(temporary) / out.name
            staging.mkdir()
            copyfile(package / "plan.json", staging / "plan.json")
            receipts = [write_games(staging, f"family-{i:03d}.jsonl", pairs[i]) for i in range(FAMILIES)]
            summary.update(candidate=candidate, training=stats, resolved_fresh=len(fresh))
            atomic_json(staging / "report.json", {**summary, "receipts": receipts})
            state["supplemental_collections"].append(dict(
                plan=f"{out.name}/plan.json", plan_sha256=expected, report=f"{out.name}/report.json",
                report_sha256=digest((staging / "report.json").read_bytes()),
            ))
            state["supplemental_train_batches"].extend(
                {**r, "path": f"{out.name}/{r['path']}", "cohort": "fresh_d37"} for r in receipts
            )
            state["excluded_openings"] = sorted(set(state["excluded_openings"]) |
                                                {f["parent"] for f in plan["families"]})
            state["history"].append(candidate)
            state["research_latest"] = candidate
            staging.rename(out)
            try:
                atomic_json(state_path, state)
                verify(root)
            except Exception:
                atomic_json(state_path, json.loads(original))
                out.rename(staging)
                raise
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "shard", "verify", "train"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--shard", type=int, choices=range(RUNNERS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    required = {
        "prepare": ("root",),
        "shard": ("plan_sha", "shard", "output"),
        "verify": ("root", "plan_sha", "results"),
        "train": ("root", "plan_sha", "results"),
    }
    if any(getattr(args, key) is None for key in required[args.phase]):
        parser.error("missing arguments for " + args.phase)
    if args.root is not None:
        args.root = args.root.resolve()
    if args.phase == "prepare":
        print(prepare(args.root, args.package))
    elif args.phase == "shard":
        run_shard(args.package, args.plan_sha, args.shard, args.output)
    else:
        print(json.dumps(train(args.root, args.package, args.plan_sha, args.results,
                               apply=args.phase == "train")))


if __name__ == "__main__":
    main()
