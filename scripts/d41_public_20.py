"""D41: reproducible 20 GitHub Actions workers for approved D37 NNUE selfplay.

No publication or official-model update is performed by this module. The GitHub
workflow publishes a complete cycle receipt after this module validates inputs.
Source of weights is the owner-authorized public D37 baseline release, pinned by SHA.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import shogi

from fukaji.online_cycle import fresh_prefix
from fukaji.online_learning import (
    atomic_json, digest, fit, game_key, load_snapshot, make_samples,
    save_snapshot, validate_game,
)
from fukaji.online_search import Position, terminal
from fukaji.native_nnue import NativeSelector

D37 = "170d6c1ee41d77b69f0b75722ece3deb85f3eea189f70d9789db477a38266db1"
ARCHIVE_SHA = "5540f04d5838bca20ede85c0121acd8f5e5ce7ebc67d6cba4e41b8cf67ebb801"
SEED = 202611051407
PAIRS = 20
MAX_PLIES = 512
NORMAL_MS = 200
LONG_MS = 1000
MAX_LONG_PER_SIDE = 10


def safe_extract(archive: Path, out: Path) -> None:
    """Unpack trusted, SHA-pinned release packages, rejecting unsafe members."""
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as f:
        for member in f.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise ValueError("unsafe archive member")
        f.extractall(out, filter="data")


def verify_archive(archive: Path) -> None:
    if digest(archive.read_bytes()) != ARCHIVE_SHA:
        raise ValueError("D37 release archive content SHA256 differs from approved archive")


def need_long(board, ply: int, legal, used: int) -> bool:
    if used >= MAX_LONG_PER_SIDE or ply < 12:
        return False
    captures = sum(board.piece_type_at(m.to_square) != shogi.NONE for m in legal)
    promotions = sum(m.promotion for m in legal)
    return bool(board.is_check() or captures >= 2 or
                (promotions >= 2 and ply >= 24) or
                (ply >= 48 and len(legal) >= 32 and ply % 16 == 0))


def pack_tar(folder: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as t:
        for item in sorted(folder.rglob("*")):
            if item.is_file():
                t.add(item, arcname=item.relative_to(folder))


def prepare(archive: Path, out: Path, extra_excluded: Path | None) -> None:
    verify_archive(archive)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "runtime"
    work.mkdir(exist_ok=True)
    models = work / "models"
    models.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:gz") as t:
        state = json.load(t.extractfile("d17-state/state.json"))
        if state["research_latest"] != D37 or state["formal_adoption"] is not False:
            raise ValueError("full D37 state does not identify approved baseline")
        for extension in ("npz", "json"):
            name = f"d17-state/models/{D37}.{extension}"
            (models / f"{D37}.{extension}").write_bytes(t.extractfile(name).read())
    model, opt, _ = load_snapshot(models, D37)
    if opt is None or opt["step"] != 2106:
        raise ValueError("D37 optimizer step not 2106")
    excluded = set(state["excluded_openings"])
    protected = set(state["protected_parents"])
    if extra_excluded is not None:
        extra = json.loads(extra_excluded.read_text())
        if not isinstance(extra, list) or any(not isinstance(x, str) or len(x) != 64 for x in extra):
            raise ValueError("invalid historical D40 opening exclusion file")
        excluded.update(extra)
        protected.update(extra)
    families = []
    # Separate openings per game; two identical-model games from the same
    # opening can otherwise yield identical legal move histories and abort
    # the learning audit as duplicate experience.
    for lane in range(PAIRS):
        openings = []
        for game_side in range(2):
            opening, parent = fresh_prefix(SEED + lane * 1019 + game_side * 1000003, excluded)
            openings.append(dict(prefix=opening, parent_id=parent))
        families.append(dict(lane=lane, openings=openings))
    plan = {
        "schema": "fukaji-d41-d37-twenty-runner-selfplay-v1",
        "baseline": D37,
        "baseline_optimizer_step": opt["step"],
        "original_archive_sha256": ARCHIVE_SHA,
        "seed": SEED,
        "runners": PAIRS,
        "games_per_runner": 2,
        "game_count": PAIRS * 2,
        "conditions": {"normal_ms": NORMAL_MS, "long_ms": LONG_MS,
                       "max_long_per_side": MAX_LONG_PER_SIDE, "max_total_plies": MAX_PLIES,
                       "native_search": "cpp_nnue", "quiescence": 0,
                       "explore_probability_first_24": 0.05},
        "protected_parent_count": len(protected),
        "families": families,
    }
    atomic_json(work / "plan.json", plan)
    pack_tar(work, out / "runtime.tar.gz")
    atomic_json(out / "prepare-receipt.json", dict(plan_sha256=digest((work / "plan.json").read_bytes()),
                                                     baseline=D37, game_count=PAIRS * 2))
    print(json.dumps({"prepared": True, "families": len(families), "model": D37}))


def segment(runtime: Path, lane: int, game_side: int, out: Path, search_limit: int) -> None:
    plan = json.loads((runtime / "plan.json").read_text())
    if plan["baseline"] != D37 or len(plan["families"]) != PAIRS:
        raise ValueError("invalid plan")
    family = plan["families"][lane]["openings"][game_side]
    model, opt, _ = load_snapshot(runtime / "models", D37)
    if opt["step"] != 2106:
        raise ValueError("D37 optimizer changed")
    out.mkdir(parents=True, exist_ok=True)
    complete = out / f"game-{game_side}.json"
    partial = out / f"game-{game_side}.partial.json"
    if complete.exists():
        return
    if partial.exists():
        state = json.loads(partial.read_text())
        if state["baseline"] != D37 or state["parent_id"] != family["parent_id"]:
            raise ValueError("unexpected resume state")
        moves = list(state["moves"])
        trace = list(state["telemetry"])
        used = list(state["used"])
        state["moves"] = moves
        state["telemetry"] = trace
        state["used"] = used
    else:
        moves = list(family["prefix"])
        trace = []
        used = [0, 0]
        state = dict(baseline=D37, parent_id=family["parent_id"], moves=moves,
                     telemetry=trace, used=used, status="in_progress")
        atomic_json(partial, state)
    board = shogi.Board()
    for move in moves:
        word = shogi.Move.from_usi(move)
        if word not in board.legal_moves:
            raise ValueError("illegal history checkpoint")
        board.push(word)
    history = Position(board, model)
    end = None
    count = 0
    with NativeSelector() as native:
        native.register(model)
        while len(moves) < MAX_PLIES:
            legal = list(board.legal_moves)
            end = terminal(history, legal)
            if end:
                break
            if count >= search_limit:
                return
            player = int(board.turn)
            long = need_long(board, len(moves), legal, used[player])
            ms = LONG_MS if long else NORMAL_MS
            row = native.select(board, model, movetime_ms=ms, max_depth=64,
                                history=history, quiescence=0)
            if not row.get("move"):
                raise ValueError("native search returned no move in nonterminal position")
            # Exploratory moves are limited to the first 24 newly searched moves.
            explore_rng = random.Random(SEED + 100001 * lane + 4099 * game_side +
                                        len(moves) * 7151)
            explore = (len(moves) < len(family["prefix"]) + 24 and
                       explore_rng.random() < 0.05)
            if explore:
                word = explore_rng.choice(legal).usi()
                decision = "exploration"
            else:
                word = row["move"]
                decision = "nnue_search"
            move = shogi.Move.from_usi(word)
            if move not in legal:
                raise ValueError("illegal selected move")
            if long:
                used[player] += 1
            trace.append({"ply": len(moves) + 1, "side": player, "budget_ms": ms,
                          "longthink": long, "decision_source": decision,
                          **row, "move": word})
            history.push(move)
            moves.append(word)
            count += 1
            atomic_json(partial, state)
        end = end or terminal(history, list(board.legal_moves)) or {
            "status": "move_limit_unresolved", "winner": None,
        }
    game = {
        "start_sfen": shogi.STARTING_SFEN, "moves": moves, **end,
        "source": "selfplay", "role": "train", "parent_id": family["parent_id"],
        "seed": SEED + 100001 * lane + 4099 * game_side,
        "models": [D37, D37], "learning_start_ply": len(family["prefix"]),
        "telemetry": trace,
        "conditions": {"movetime_ms": NORMAL_MS, "longthink_ms": LONG_MS,
            "max_longthink_per_side": MAX_LONG_PER_SIDE,
            "longthink_used_per_side": used, "max_additional_plies": MAX_PLIES - len(family["prefix"]),
            "backend_by_side": ["cpp_nnue", "cpp_nnue"], "quiescence_by_side": [0, 0],
            "exploration_probability": .05, "exploration_additional_plies": 24,
            "knowledge": False, "value": "nnue_only"},
        "lane": lane, "game_side": game_side,
    }
    validate_game(game, model, training=True)
    if len(trace) != len(moves) - len(family["prefix"]) or max(used) > 10:
        raise ValueError("telemetry or longthink count mismatch")
    atomic_json(complete, game)
    partial.unlink(missing_ok=True)


def worker(runtime_file: Path, lane: int, dest: Path) -> None:
    if not 0 <= lane < PAIRS:
        raise ValueError("invalid lane")
    work = dest / "runtime"
    safe_extract(runtime_file, work)
    games = dest / "games"
    games.mkdir(parents=True, exist_ok=True)
    for game_side in (0, 1):
        stalled = 0
        for iteration in range(30):
            complete = games / f"game-{game_side}.json"
            if complete.exists():
                break
            partial = games / f"game-{game_side}.partial.json"
            before = len(json.loads(partial.read_text())["moves"]) if partial.exists() else 0
            command = [sys.executable, __file__, "segment", "--runtime", str(work),
                       "--lane", str(lane), "--game-side", str(game_side),
                       "--out", str(games), "--search-limit", "36"]
            try:
                proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
                if proc.returncode:
                    raise RuntimeError(proc.stderr[-2200:])
            except subprocess.TimeoutExpired:
                pass  # Checkpoint remains available for the following attempt.
            after = len(json.loads(partial.read_text())["moves"]) if partial.exists() else MAX_PLIES
            stalled = stalled + 1 if after <= before and not complete.exists() else 0
            if stalled >= 3:
                raise RuntimeError("three consecutive segment attempts made no progress")
        else:
            raise RuntimeError("maximum segment attempts exhausted")
        if not complete.exists():
            raise RuntimeError("game missing after segmented execution")
    model, _, _ = load_snapshot(work / "models", D37)
    for game_side in (0, 1):
        game = json.loads((games / f"game-{game_side}.json").read_text())
        validate_game(game, model, training=True)
    package = dest / f"lane-{lane:02d}.tar.gz"
    pack_tar(games, package)
    print(json.dumps({"lane": lane, "games": 2, "sha256": digest(package.read_bytes())}))


def read_jsonlines(raw: bytes, expected_sha: str):
    if digest(raw) != expected_sha:
        raise ValueError("replay data differs from saved owner history")
    return [json.loads(x) for x in raw.decode().splitlines() if x.strip()]


def train(archive: Path, downloads: Path, output: Path) -> None:
    verify_archive(archive)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(downloads.glob("lane-*.tar.gz"))
    if len(files) != PAIRS or [int(p.stem.split("-")[1].split(".")[0]) for p in files] != list(range(PAIRS)):
        raise ValueError("all 20 lane artifacts are required")
    with tarfile.open(archive, "r:gz") as t:
        state = json.load(t.extractfile("d17-state/state.json"))
        d37_raw = t.extractfile(f"d17-state/models/{D37}.npz").read()
        d37_meta = t.extractfile(f"d17-state/models/{D37}.json").read()
        old = read_jsonlines(t.extractfile("d17-state/bootstrap-train.jsonl").read(),
                             state["bootstrap"]["sha256"])
    models = output / "models"
    models.mkdir(exist_ok=True)
    (models / f"{D37}.npz").write_bytes(d37_raw)
    (models / f"{D37}.json").write_bytes(d37_meta)
    baseline, opt, _ = load_snapshot(models, D37)
    if opt["step"] != 2106:
        raise ValueError("wrong baseline optimizer")
    games = []
    for lane, path in enumerate(files):
        with tarfile.open(path, "r:gz") as f:
            members = sorted((m for m in f if m.isfile()), key=lambda m: m.name)
            if sorted(m.name for m in members) != ["game-0.json", "game-1.json"]:
                raise ValueError("lane artifact does not contain exactly two finished games")
            for m in members:
                game = json.load(f.extractfile(m))
                if game.get("lane") != lane or game["models"] != [D37, D37]:
                    raise ValueError("wrong source lane or baseline model")
                validate_game(game, baseline, training=True, protected=state["protected_parents"])
                games.append(game)
    if len(games) != PAIRS * 2 or len({game_key(g) for g in games}) != len(games):
        raise ValueError("missing or duplicate selfplay games")
    if len({g["parent_id"] for g in games}) != PAIRS * 2:
        raise ValueError("new selfplay parent openings repeated across lanes")
    resolved = [g for g in games if g["status"] in ("checkmate", "perpetual_check", "repetition_draw")]
    if not resolved:
        raise ValueError("no resolved training games; do not train on unfinished games")
    parents = set(state["protected_parents"])
    for g in old:
        validate_game(g, baseline, training=True, protected=parents)
    old = random.Random(SEED).sample(old, min(len(old), len(resolved)))
    indices, labels, sampling = make_samples(resolved + old, baseline,
        seed=SEED, protected=parents, outcome_weight=.75, td_lambda=.8)
    candidate, candidate_opt, stats = fit(baseline, opt, indices, labels,
                                           epochs=2, lr=.001, seed=SEED)
    key = save_snapshot(models, candidate, candidate_opt, {
        "parent": D37, "purpose": "D41 post-selfplay unadopted comparison input",
        "source": "D37 x D37 public 20-runner GitHub Actions",
        "training": stats, "resolved_games": len(resolved),
        "outcome_weight": .75, "td_lambda": .8,
    })
    report = {"schema": "fukaji-d41-v1", "baseline": D37, "trained_sha256": key,
              "baseline_optimizer_step": opt["step"], "new_optimizer_step": candidate_opt["step"],
              "total_games": len(games), "resolved": len(resolved),
              "unresolved": len(games) - len(resolved), "samples": sampling,
              "training": stats, "formal_model_changed": False,
              "candidate_strength_comparison_completed": False,
              "preexisting_D37_preserved": True}
    atomic_json(output / "report.json", report)
    games_dest = output / "new-selfplay.jsonl"
    games_dest.write_text("".join(json.dumps(g, separators=(",", ":")) + "\n" for g in games))
    model_bytes = (models / f"{key}.npz").read_bytes()
    if digest(model_bytes) != key:
        raise ValueError("trained snapshot hash changed")
    package_dir = output / "publication"
    package_dir.mkdir(exist_ok=True)
    for p in (output / "report.json", games_dest, models / f"{key}.npz", models / f"{key}.json"):
        (package_dir / p.name).write_bytes(p.read_bytes())
    package = output / "D41-selfplay-training-unadopted.tar.gz"
    pack_tar(package_dir, package)
    atomic_json(output / "publication-receipt.json",
        {"sha256": digest(package.read_bytes()), "size_bytes": package.stat().st_size,
         "baseline": D37, "trained": key})
    print(json.dumps(report))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--baseline", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--excluded", type=Path)
    w = sub.add_parser("worker")
    w.add_argument("--runtime-file", type=Path, required=True)
    w.add_argument("--lane", type=int, required=True)
    w.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("segment")
    s.add_argument("--runtime", type=Path, required=True)
    s.add_argument("--lane", type=int, required=True)
    s.add_argument("--game-side", type=int, choices=[0, 1], required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--search-limit", type=int, default=36)
    t = sub.add_parser("train")
    t.add_argument("--baseline", type=Path, required=True)
    t.add_argument("--downloads", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.baseline, args.output, args.excluded)
    elif args.command == "worker":
        worker(args.runtime_file, args.lane, args.output)
    elif args.command == "segment":
        segment(args.runtime, args.lane, args.game_side, args.out, args.search_limit)
    elif args.command == "train":
        train(args.baseline, args.downloads, args.output)

if __name__ == "__main__":
    main()