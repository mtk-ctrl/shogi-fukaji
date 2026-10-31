"""Bounded, resumable research-only self-improvement. No formal promotion path.

CLI: init --source verified-G8.zip; cycle --root ... --cycles 1
Immutable models/data are committed before an atomic research pointer update.
Repeated/incomplete operations are never silently replayed. A flock prevents
two local writers; the workflow also serializes writers across runners.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import random
import time

import numpy as np
import shogi

from .arena import opening
from .online_learning import (atomic_json, bootstrap_games, digest, fit, fresh_model,
    game_key, load_snapshot, make_samples, opening_key, save_snapshot, validate_game)
from .online_search import Position, select, terminal


@contextmanager
def lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield root
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def write_games(root, name, games):
    raw = ("\n".join(json.dumps(x, sort_keys=True) for x in games)+"\n").encode()
    path = root / name
    if path.exists():
        raise ValueError("immutable game batch already exists")
    path.write_bytes(raw)
    return {"path": name, "sha256": digest(raw), "games": len(games)}


def read_games(root, receipt):
    raw = (root/receipt["path"]).read_bytes()
    if digest(raw) != receipt["sha256"]:
        raise ValueError("game batch checksum mismatch")
    games = [json.loads(line) for line in raw.splitlines()]
    if len(games) != receipt["games"]:
        raise ValueError("game count mismatch")
    return games


def initialize(root, source, *, seed=2026103101, epochs=8):
    with lock(root) as root:
        if (root/"state.json").exists():
            raise ValueError("research state already initialized")
        start = time.monotonic()
        games, protected, excluded = bootstrap_games(source)
        model = fresh_model(seed)
        x, y, data_stats = make_samples(games, model, seed=seed,
                                       outcome_weight=1., protected=protected)
        model, opt, stats = fit(model, None, x, y, epochs=epochs, seed=seed)
        key = save_snapshot(root/"models", model, opt, {
            "parent": None, "purpose": "outcome-only bootstrap from 665 train games",
            "seed": seed, "source_sha256": digest(Path(source).read_bytes()),
            "training": stats})
        receipt = write_games(root, "bootstrap-train.jsonl", games)
        state = {"schema": 1, "research_latest": key, "checkpoint": key,
                 "fixed_anchor": key, "history": [key], "generation": 0,
                 "formal_adoption": False, "bootstrap": receipt, "train_batches": [],
                 "evaluation_batches": [], "protected_parents": sorted(protected),
                 "excluded_openings": sorted(excluded), "used_seeds": [], "cycles": [],
                 "seed": seed, "paused": False}
        atomic_json(root/"bootstrap-report.json", {"training": stats, "data": data_stats,
                    "seconds": time.monotonic()-start, "model": key,
                    "holdouts_evaluated": False, "strength_proven": False})
        atomic_json(root/"state.json", state)
        return state


def play(models, model_ids, prefix, *, role, parent_id, seed, movetime=20,
         max_plies=256, ordering=True):
    b = shogi.Board()
    rng = random.Random(seed)
    for word in prefix:
        move = shogi.Move.from_usi(word)
        if move not in b.legal_moves:
            raise ValueError("invalid opening")
        b.push(move)
    moves, stats = list(prefix), []
    end = None
    # Keep both accumulators current during the actual game; search builds its
    # own reversible state. Exact legal replay below is a separate validator.
    rules = Position(b, models[b.turn])
    for _ in range(max_plies):
        end = terminal(rules, list(b.legal_moves))
        if end:
            break
        row = select(b, models[b.turn], movetime_ms=movetime, ordering=ordering)
        if row["move"] is None:
            raise ValueError("nonterminal search returned no move")
        if role == "train" and len(stats) < 24 and rng.random() < .05:
            row = {**row, "move": rng.choice(list(b.legal_moves)).usi(),
                   "score": None, "decision_source": "exploration"}
        else:
            row["decision_source"] = "nnue_search"
        move = shogi.Move.from_usi(row["move"])
        stats.append({"ply": len(moves)+1, "side": int(b.turn), **row})
        rules.push(move)
        moves.append(row["move"])
    end = end or terminal(rules, list(b.legal_moves)) or {"status": "move_limit_unresolved", "winner": None}
    game = {"start_sfen": shogi.STARTING_SFEN, "moves": moves, **end,
            "source": "selfplay", "role": role, "parent_id": parent_id,
            "seed": seed, "models": model_ids, "learning_start_ply": len(prefix),
            "telemetry": stats, "conditions": {"movetime_ms": movetime,
            "max_additional_plies": max_plies, "ordering": ordering,
            "value": "nnue_only", "knowledge": False,
            "exploration_probability": .05 if role == "train" else 0,
            "exploration_additional_plies": 24 if role == "train" else 0}}
    validate_game(game, models[0])
    return game


def fresh_prefix(seed, excluded):
    for attempt in range(1000):
        words = opening(seed+attempt, 6)
        key = opening_key(shogi.STARTING_SFEN, words)
        if key not in excluded:
            excluded.add(key)
            return words, key
    raise ValueError("no unused opening found")


def compare(root, left, opponents, *, seed, pairs, movetime, max_plies, excluded):
    games = []
    model_a = load_snapshot(root/"models", left)[0]
    for pair in range(pairs):
        right = opponents[pair % len(opponents)]
        model_b = load_snapshot(root/"models", right)[0]
        prefix, parent = fresh_prefix(seed+pair*1000, excluded)
        for black in (True, False):
            models = [model_a, model_b] if black else [model_b, model_a]
            ids = [left, right] if black else [right, left]
            g = play(models, ids, prefix, role="monitor", parent_id=parent,
                     seed=seed+pair, movetime=movetime, max_plies=max_plies)
            g["candidate_side"] = 0 if black else 1
            games.append(g)
    return games


def gate(games, *, periodic=False):
    counts = Counter()
    for g in games:
        if g["winner"] is not None:
            counts["win" if g["winner"] == g["candidate_side"] else "loss"] += 1
        elif g["status"] == "repetition_draw":
            counts["draw"] += 1
        else:
            counts["unresolved"] += 1
    w, l, d, u = (counts[k] for k in ("win", "loss", "draw", "unresolved"))
    # A deliberately coarse fault/collapse detector, not a superiority test.
    collapse = (l >= 4 and w == 0 and d == 0)
    # All-draw runs can hide a broken search which merely repeats moves.
    # Draws remain valid labels, but cannot by themselves clear this pilot gate.
    information = w+l >= 4
    action = "rollback" if collapse else ("pause" if periodic and not information else "provisional")
    return {"wins": w, "losses": l, "draws": d, "unresolved": u,
            "action": action, "strength_proven": False,
            "score_resolved": (w+.5*d)/(w+l+d) if w+l+d else None,
            "reason": "catastrophic screen" if collapse else
                      "insufficient decisive periodic outcomes" if action == "pause" else
                      "bounded research update; superiority unproven"}


def cycle(root, *, pairs=2, movetime=50, max_plies=256, epochs=2,
          outcome_weight=.75, periodic_every=3, audit_pairs=4):
    if not (1 <= pairs <= 16 and 1 <= movetime <= 200 and 16 <= max_plies <= 512 and
            1 <= epochs <= 4 and 1 <= periodic_every <= 5 and 2 <= audit_pairs <= 16):
        raise ValueError("research resource budget exceeded")
    with lock(root) as root:
        state = json.loads((root/"state.json").read_text())
        if state["formal_adoption"] is not False or state["paused"]:
            raise ValueError("research paused or invalid formal state")
        generation = len(state["cycles"])+1
        out = root/f"cycle-{generation:04d}"
        if out.exists():
            raise ValueError("incomplete cycle exists; inspect it before retrying")
        out.mkdir()
        started = time.monotonic()
        atomic_json(out/"status.json", {"stage": "selfplay", "started": time.time()})
        parent_id = state["research_latest"]
        parent, opt, _ = load_snapshot(root/"models", parent_id)
        bootstrap = read_games(root, state["bootstrap"])
        excluded = set(state["excluded_openings"])
        protected = set(state["protected_parents"])
        seed = state["seed"] + generation*100000
        if seed in state["used_seeds"]:
            raise ValueError("seed reused")
        rng = random.Random(seed)
        games = []
        opponent_ids = [parent_id, state["checkpoint"], state["history"][max(0,len(state["history"])-3)]]
        for pair in range(pairs):
            # Half the pairs are continuations of TRAIN parents, preserving
            # full histories. This curriculum reduces cold-start sparse reward.
            if pair % 2 == 0:
                base = rng.choice(bootstrap)
                prefix = base["moves"][:-min(12, len(base["moves"])-1)]
                group = base["parent_id"]
            else:
                prefix, group = fresh_prefix(seed+pair*1000, excluded)
            opponent_id = opponent_ids[pair % len(opponent_ids)]
            opponent = load_snapshot(root/"models", opponent_id)[0]
            for black in (True, False):
                models = [parent, opponent] if black else [opponent, parent]
                ids = [parent_id, opponent_id] if black else [opponent_id, parent_id]
                games.append(play(models, ids, prefix, role="train", parent_id=group,
                                  seed=seed+pair*2+int(black), movetime=movetime, max_plies=max_plies))
        train_receipt = write_games(root, f"cycle-{generation:04d}/selfplay.jsonl", games)
        atomic_json(out/"status.json", {"stage": "training"})
        resolved = list({game_key(g): g for g in games
                        if g["status"] in ("checkmate", "perpetual_check", "repetition_draw")}.values())
        report = {"generation": generation, "parent": parent_id, "train": train_receipt,
                  "new_games": len(games), "new_resolved": len(resolved),
                  "formal_adoption": False, "strength_proven": False,
                  "config": {"pairs": pairs, "movetime_ms": movetime, "max_plies": max_plies,
                             "outcome_weight": outcome_weight, "td_lambda": .8,
                             "epochs": epochs, "periodic_every": periodic_every}}
        state["used_seeds"].append(seed)
        state["train_batches"].append(train_receipt)
        if not resolved:
            report["decision"] = "no_update_no_resolved_experience"
        else:
            # Equal per-game sample caps, 50% recent and 50% historical games.
            # Old raw games are retained even when not selected this cycle.
            replay = list(bootstrap)
            for receipt in state["train_batches"][:-1]:
                replay.extend(g for g in read_games(root, receipt)
                              if g["status"] in ("checkmate", "perpetual_check", "repetition_draw"))
            recent_keys = {game_key(g) for g in resolved}
            replay = list({game_key(g): g for g in replay if game_key(g) not in recent_keys}.values())
            old = rng.sample(replay, min(len(replay), len(resolved)))
            x, y, sample_stats = make_samples(resolved+old, parent, seed=seed,
                outcome_weight=outcome_weight, protected=protected)
            candidate, new_opt, stats = fit(parent, opt, x, y, epochs=epochs, seed=seed)
            key = save_snapshot(root/"models", candidate, new_opt, {
                "parent": parent_id, "purpose": "provisional selfplay update", "seed": seed,
                "training": stats, "train_receipt": train_receipt,
                "replay_game_hashes": [game_key(g) for g in old],
                "outcome_weight": outcome_weight, "td_lambda": .8})
            report.update(candidate=key, training=stats, samples=sample_stats,
                          replay_games=len(old), recent_games=len(resolved))
            atomic_json(out/"status.json", {"stage": "screen", "candidate": key})
            screening = compare(root, key, [parent_id], seed=seed+50000, pairs=2,
                                movetime=movetime, max_plies=max_plies, excluded=excluded)
            periodic = generation % periodic_every == 0
            audits = []
            if periodic:
                atomic_json(out/"status.json", {"stage": "periodic", "candidate": key})
                audit_opponents = list(dict.fromkeys([state["fixed_anchor"], state["checkpoint"],
                    state["history"][max(0, len(state["history"])-2)]]))
                audits = compare(root, key, audit_opponents, seed=seed+70000, pairs=audit_pairs,
                    movetime=movetime, max_plies=max_plies, excluded=excluded)
            evaluation = screening + audits
            receipt = write_games(root, f"cycle-{generation:04d}/monitor.jsonl", evaluation)
            state["evaluation_batches"].append(receipt)
            protected.update(g["parent_id"] for g in evaluation)
            decision = gate(screening)
            if decision["action"] != "rollback" and periodic:
                decision = gate(audits, periodic=True)
            if stats["loss_history"][-1]["loss"] > stats["loss_history"][0]["loss"] + .05:
                decision = {"action": "rollback", "reason": "training loss exploded", "strength_proven": False}
            report.update(screen=gate(screening), periodic=gate(audits, periodic=True) if periodic else None,
                          evaluation=receipt, decision=decision)
            if decision["action"] == "rollback":
                state["research_latest"] = state["checkpoint"]
                state["paused"] = True  # diagnose before producing more bad data
            elif decision["action"] == "pause":
                state["paused"] = True
            else:
                state["research_latest"] = key
                state["generation"] += 1
                state["history"].append(key)
                if periodic:
                    state["checkpoint"] = key  # operational checkpoint, not a champion
        state["excluded_openings"] = sorted(excluded)
        state["protected_parents"] = sorted(protected)
        report["seconds"] = time.monotonic()-started
        report["research_latest"] = state["research_latest"]
        atomic_json(out/"report.json", report)
        state["cycles"].append({"report": f"cycle-{generation:04d}/report.json",
                                "sha256": digest((out/"report.json").read_bytes())})
        atomic_json(out/"status.json", {"stage": "complete", "decision": report["decision"]})
        atomic_json(root/"state.json", state)
        return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["init", "cycle"])
    p.add_argument("--root", required=True)
    p.add_argument("--source")
    p.add_argument("--epochs", type=int)
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--pairs", type=int, default=2)
    p.add_argument("--movetime", type=int, default=50)
    p.add_argument("--max-plies", type=int, default=256)
    p.add_argument("--outcome-weight", type=float, default=.75)
    p.add_argument("--periodic-every", type=int, default=3)
    args = p.parse_args()
    if args.mode == "init":
        if not args.source:
            p.error("init requires --source")
        state = initialize(args.root, args.source, epochs=args.epochs or 8)
        print(json.dumps({"initialized": state["research_latest"]}))
    else:
        if not 1 <= args.cycles <= 5:
            p.error("cycles must be 1..5; no unbounded loop")
        for _ in range(args.cycles):
            result = cycle(args.root, pairs=args.pairs, movetime=args.movetime,
                           max_plies=args.max_plies, epochs=args.epochs or 2,
                           outcome_weight=args.outcome_weight, periodic_every=args.periodic_every)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if json.loads((Path(args.root)/"state.json").read_text())["paused"]:
                break


if __name__ == "__main__":
    main()
