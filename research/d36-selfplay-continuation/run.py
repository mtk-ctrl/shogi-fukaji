"""D36 bounded study driver. Run one round at a time on restored D35 state."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import random

import numpy as np

from fukaji.frozen_evaluation import cpu_plan, prepare
from fukaji.native_nnue import build_native
from fukaji.online_cycle import fresh_prefix, lock, play, read_games, write_games
from fukaji.online_learning import (
    atomic_json, digest, fit, game_key, implementation_digest, load_snapshot,
    make_samples, parameters, save_snapshot, validate_game,
)
from fukaji.online_state import verify

START = "db94693a9fcbb178a9d5d660109359c7ef7462c6ff376680ded6e0db151ca9c9"
FORMAL = "e0cbd2abe53b1d151bb73407f848b4087456c1466121333ad490a96aee131b0a"
SEED = 202611033600
DECIDED = {"checkmate", "perpetual_check", "repetition_draw"}
WORKER = {}


def initialize_worker(root, plan, cpus):
    os.sched_setaffinity(0, {cpus.get()})
    ids = (plan["parent"], plan["opponent"])
    WORKER.update(plan=plan, models={key: load_snapshot(root / "models", key)[0] for key in ids})


def generate(family):
    plan = WORKER["plan"]
    games = []
    for side in (0, 1):
        ids = [plan["parent"], plan["opponent"]]
        if side:
            ids.reverse()
        games.append(play(
            [WORKER["models"][key] for key in ids], ids, family["prefix"],
            role="train", parent_id=family["parent"], seed=plan["seed"] + 2 * family["index"] + side,
            movetime=50, max_plies=256, quiescence=0, native_backend=True,
        ))
    return games


def collect(root, out, plan, parent):
    ctx = mp.get_context("spawn")
    cpus = ctx.Queue()
    for cpu in plan["cpu"]["worker_cpus"]:
        cpus.put(cpu)
    try:
        with ProcessPoolExecutor(4, mp_context=ctx, initializer=initialize_worker,
                                 initargs=(root, plan, cpus)) as pool:
            pending = {}
            for family in plan["families"]:
                name = f"family-{family['index']:03d}.jsonl"
                receipt_path = out / (name + ".receipt.json")
                if not receipt_path.exists():
                    if (out / name).exists():
                        raise ValueError("uncommitted pair requires inspection")
                    pending[name] = pool.submit(generate, family)
            completed = []
            for family in plan["families"]:
                name = f"family-{family['index']:03d}.jsonl"
                receipt_path = out / (name + ".receipt.json")
                receipt = json.loads(receipt_path.read_text()) if name not in pending else None
                games = read_games(out, receipt) if receipt else pending[name].result()
                if len(games) != 2 or (receipt and receipt["path"] != name):
                    raise ValueError("pair receipt mismatch")
                for side, game in enumerate(games):
                    ids = [plan["parent"], plan["opponent"]][::1 if side == 0 else -1]
                    if (game["models"] != ids or game["parent_id"] != family["parent"]
                        or game["moves"][:6] != family["prefix"] or game["learning_start_ply"] != 6
                        or game["seed"] != plan["seed"] + 2 * family["index"] + side
                        or game["conditions"]["movetime_ms"] != 50
                        or game["conditions"]["max_additional_plies"] != 256
                        or game["conditions"]["quiescence_by_side"] != [0, 0]
                        or game["conditions"]["backend_by_side"] != ["cpp_nnue", "cpp_nnue"]):
                        raise ValueError("pair protocol mismatch")
                    validate_game(game, parent, training=True, protected=plan["protected_parents"])
                if receipt is None:
                    receipt = write_games(out, name, games)
                    atomic_json(receipt_path, receipt)
                completed.append((receipt, games))
                print(json.dumps({"round": plan["round"], "games_verified": len(completed) * 2}), flush=True)
    finally:
        cpus.close()
    return completed


def sample_group(games, parent, seed, protected):
    x, y, _ = make_samples(games, parent, seed=seed, per_game=32, outcome_weight=.75,
                           td_lambda=.8, protected=protected)
    return x, y


def run(root, number):
    build_native()
    with lock(root):
        verify(root)
        state_path = root / "state.json"
        state = json.loads(state_path.read_text())
        if state["paused"] or state.get("pending_campaign") or state.get("pending_comparison"):
            raise ValueError("idle research state required")
        out = root / f"d36-round-{number}"
        if number == 1:
            parent_id = START
        else:
            prev = json.loads((root / "d36-round-1/report.json").read_text())
            parent_id = prev["candidate"]
        if state["research_latest"] != parent_id or (out / "report.json").exists():
            raise ValueError("unexpected lineage or completed round")
        parent, optimizer, _ = load_snapshot(root / "models", parent_id)
        excluded = set(state["excluded_openings"]) | set(state["protected_parents"])
        seed = SEED + number * 100_000
        families = []
        for i in range(16):
            prefix, group = fresh_prefix(seed + i * 1000, excluded)
            families.append(dict(index=i, prefix=prefix, parent=group))
        plan = dict(schema="fukaji-fresh-train-v1", parent=parent_id, opponent=FORMAL, seed=seed,
                    round=number, families=families, role="train", planned_games=32, cpu=cpu_plan(4),
                    protected_parents=state["protected_parents"], source_sha256=digest(Path(__file__).read_bytes()),
                    engine_sha256=implementation_digest(), state_sha256=digest(state_path.read_bytes()),
                    conditions=dict(movetime_ms=50, max_additional_plies=256, quiescence=0,
                                    backend="cpp_nnue", exploration_probability=.05), formal_adoption=False)
        out.mkdir(exist_ok=True)
        plan_path = out / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("frozen plan changed")
        if not plan_path.exists():
            atomic_json(plan_path, plan)
        pairs = collect(root, out, plan, parent)
        games = [game for _, pair in pairs for game in pair]
        fresh = {game_key(g): g for g in games if g["status"] in DECIDED}
        if len(fresh) < 8:
            raise ValueError("fewer than eight distinct resolved fresh games")
        rng = random.Random(seed)
        old = {}
        for receipt in [state["bootstrap"], *state["train_batches"], *state["supplemental_train_batches"]]:
            for game in read_games(root, receipt):
                key = game_key(game)
                if game["status"] in DECIDED and key not in fresh:
                    old[key] = game
        old_games = [old[key] for key in rng.sample(sorted(old), 64)]
        xa, ya = sample_group(list(fresh.values()), parent, seed, state["protected_parents"])
        xb, yb = sample_group(old_games, parent, seed + 1, state["protected_parents"])
        width = max(xa.shape[1], xb.shape[1])
        rnd = np.random.default_rng(seed)
        samples = []
        for x, y in ((xa, ya), (xb, yb)):
            indices = rnd.integers(len(y), size=2048)
            samples.append((np.pad(x[indices], ((0, 0), (0, width - x.shape[1])),
                                   constant_values=parent.inputs), y[indices]))
        x = np.concatenate([sample[0] for sample in samples])
        y = np.concatenate([sample[1] for sample in samples])
        model, adam, stats = fit(parent, optimizer, x, y, seed=seed, epochs=2, lr=.001, batch_size=128)
        if adam["step"] != optimizer["step"] + 64:
            raise ValueError("optimizer step mismatch")
        key = save_snapshot(root / "models", model, adam, dict(
            parent=parent_id, purpose="D36 bounded self-play continuation", seed=seed, training=stats,
            collection_plan_sha256=digest(plan_path.read_bytes()), replay_game_hashes=[game_key(g) for g in old_games],
        ))
        loaded, loaded_opt, _ = load_snapshot(root / "models", key)
        for left, right in zip(parameters(model), parameters(loaded)):
            np.testing.assert_array_equal(left, right)
        if loaded_opt["step"] != adam["step"] or all(np.array_equal(a, b) for a, b in zip(parameters(model), parameters(parent))):
            raise ValueError("candidate reload or update failed")
        report_path = out / "report.json"
        report = dict(complete=True, completed_games=32, outcomes_match=True, plan_sha256=digest(plan_path.read_bytes()),
                      receipts=[receipt for receipt, _ in pairs], outcomes=dict(Counter(g["status"] for g in games)),
                      legal_replay_moves=sum(len(g["moves"]) for g in games), candidate=key, parent=parent_id,
                      resolved_fresh=len(fresh), old_games=len(old_games), raw_sample_positions=[len(ya), len(yb)],
                      optimizer_before=optimizer["step"], optimizer_after=adam["step"], training=stats,
                      research_update="provisional_finite_run", formal_adoption=False)
        atomic_json(report_path, report)
        state["supplemental_collections"].append(dict(
            plan=plan_path.relative_to(root).as_posix(), plan_sha256=report["plan_sha256"],
            report=report_path.relative_to(root).as_posix(), report_sha256=digest(report_path.read_bytes()),
        ))
        state["supplemental_train_batches"].extend(
            {**receipt, "path": f"{out.name}/{receipt['path']}", "cohort": f"fresh_d36_round_{number}"}
            for receipt in report["receipts"]
        )
        state["excluded_openings"] = sorted(excluded)
        state["history"].append(key)
        state["research_latest"] = key
        previous_state = json.loads(state_path.read_text())
        atomic_json(state_path, state)
        try:
            verify(root)
        except Exception:
            atomic_json(state_path, previous_state)
            raise
        print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("1", "2", "prepare"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase != "prepare":
        run(root, int(args.phase))
        return
    state = json.loads((root / "state.json").read_text())
    last = json.loads((root / "d36-round-2/report.json").read_text())
    assert state["research_latest"] == last["candidate"]
    plan = prepare(root, name="d36-monitor80", pairs=20, seed=202611033699, movetime=50,
                   max_total_plies=512, workers=4, purpose="regression", candidate_id=last["candidate"],
                   opponents=[{"label": "before_D36", "model": START}, {"label": "formal_v030", "model": FORMAL}])
    print(json.dumps({"pairs": plan["pairs"], "opponents": plan["opponents"], "candidate": plan["candidate"]}))


if __name__ == "__main__":
    main()
