"""D41 public D37 inference-only 20 GitHub Actions worker adapter.

Public owner-approved D37 weights are used exactly. Optimizer moments are NOT
present in the inference bundle; this workflow generates auditable selfplay
games but cannot claim to have trained a new model. The frozen original
Adam2106 state must be restored to perform the subsequent training step.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from collections import Counter

import numpy as np

from fukaji.nnue_research import SmallNNUE
from fukaji.online_cycle import fresh_prefix
from fukaji.online_learning import atomic_json, digest, game_key, validate_game
from scripts import d41_public_20 as core

BUNDLE_SHA = "d113cdeabaf0d5d5799081b6f385ab0fb6981ae4cb8291ec639ff9e2a1521a7d"
WEIGHTS_SHA = "2a0001826c2df835b063c896f8f52e2e99972e144921b1ceff6b77ed6e104b8e"
OPENINGS_SHA = "76e9cb273335571867459151a5241adfa75cdd7345775d6eb98af5d8f913ccbb"
BASELINE = core.D37
SEED = core.SEED


def read_model(models):
    raw = (Path(models) / "D37-inference.npz").read_bytes()
    if digest(raw) != WEIGHTS_SHA:
        raise ValueError("D37 inference weights SHA256 mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as arr:
        if set(arr.files) != {"w", "b", "v", "o"}:
            raise ValueError("unexpected weights fields")
        return SmallNNUE(arr["w"], arr["b"], arr["v"], arr["o"][0])


def families():
    used = set()
    rows = []
    for lane in range(20):
        opens = []
        for side in range(2):
            prefix, parent = fresh_prefix(SEED + lane * 1019 + side * 1000003, used)
            opens.append({"prefix": prefix, "parent_id": parent})
        rows.append({"lane": lane, "openings": opens})
    canonical = [x["openings"] for x in rows]
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    if digest(raw) != OPENINGS_SHA or len(used) != 40:
        raise ValueError("frozen 40-opening plan drifted")
    return rows


def prepare(bundle, target):
    if digest(bundle.read_bytes()) != BUNDLE_SHA:
        raise ValueError("public D37 release bundle mismatch")
    with tarfile.open(bundle, "r:gz") as tar:
        raw = tar.extractfile("fukaji-v0.4.0/models/fukaji-v0.4.0.nnue.npz").read()
    if digest(raw) != WEIGHTS_SHA:
        raise ValueError("inference weights differ from formal D37")
    dest = target / "runtime"
    (dest / "models").mkdir(parents=True, exist_ok=True)
    (dest / "models/D37-inference.npz").write_bytes(raw)
    plan = {
        "schema": "fukaji-d41-public-selfplay-v1",
        "baseline": BASELINE,
        "optimizer_present": False,
        "weights_sha256": WEIGHTS_SHA,
        "openings_sha256": OPENINGS_SHA,
        "workers": 20,
        "games_per_worker": 2,
        "games": 40,
        "normal_ms": 200,
        "longthink_ms": 1000,
        "max_long_per_side": 10,
        "max_plies": 512,
        "quiescence": 0,
        "families": families(),
    }
    atomic_json(dest / "plan.json", plan)
    read_model(dest / "models")
    core.pack_tar(dest, target / "runtime.tar.gz")
    print(json.dumps({"prepared": True, "workers": 20, "games": 40,
                      "optimizer_loaded": False}))


def inject_model(models):
    # The frozen inference model has no Adam moments. core.segment only needs
    # the model; the step value here is approval metadata, never used to train.
    model = read_model(models)
    def loader(_models, key):
        if key != BASELINE:
            raise ValueError("unexpected model ID")
        return model, {"step": 2106}, {"inference_only": True}
    core.load_snapshot = loader


def segment(runtime, lane, side, output, search_limit):
    inject_model(runtime / "models")
    core.segment(runtime, lane, side, output, search_limit)


def worker(bundle, lane, output):
    if not 0 <= lane < 20:
        raise ValueError("invalid lane")
    runtime = output / "runtime"
    core.safe_extract(bundle, runtime)
    model = read_model(runtime / "models")
    games = output / "games"
    games.mkdir(parents=True, exist_ok=True)
    for side in range(2):
        stalled = 0
        for iteration in range(30):
            done = games / f"game-{side}.json"
            if done.exists():
                break
            partial = games / f"game-{side}.partial.json"
            before = len(json.loads(partial.read_text())["moves"]) if partial.exists() else 0
            command = [sys.executable, __file__, "segment", "--runtime", str(runtime),
                       "--lane", str(lane), "--side", str(side),
                       "--output", str(games), "--search-limit", "36"]
            try:
                task = subprocess.run(command, capture_output=True, text=True, timeout=120)
                if task.returncode:
                    raise RuntimeError(task.stderr[-2000:])
            except subprocess.TimeoutExpired:
                pass
            after = len(json.loads(partial.read_text())["moves"]) if partial.exists() else core.MAX_PLIES
            stalled = stalled + 1 if after <= before and not done.exists() else 0
            if stalled >= 3:
                raise RuntimeError("game stalled across three restarted processes")
        else:
            raise RuntimeError("segment restart allowance exceeded")
        if not done.exists():
            raise ValueError("game not completed")
        game = json.loads(done.read_text())
        validate_game(game, model, training=True)
    package = output / f"lane-{lane:02d}.tar.gz"
    core.pack_tar(games, package)
    print(json.dumps({"lane": lane, "games": 2, "sha256": digest(package.read_bytes())}))


def collect(runtime_bundle, downloads, output):
    runtime = output / "runtime"
    core.safe_extract(runtime_bundle, runtime)
    model = read_model(runtime / "models")
    plan = json.loads((runtime / "plan.json").read_text())
    if plan.get("baseline") != BASELINE or plan.get("openings_sha256") != OPENINGS_SHA:
        raise ValueError("wrong current D37 plan")
    files = sorted(downloads.glob("lane-*.tar.gz"))
    if [f.name for f in files] != [f"lane-{i:02d}.tar.gz" for i in range(20)]:
        raise ValueError("not exactly 20 lane artifacts")
    rows = output / "games"
    rows.mkdir(parents=True, exist_ok=True)
    seen = set()
    counts = Counter()
    total_plies = 0
    longmoves = 0
    for lane, archive in enumerate(files):
        with tarfile.open(archive, "r:gz") as tar:
            members = sorted([m for m in tar if m.isfile()], key=lambda m: m.name)
            if [m.name for m in members] != ["game-0.json", "game-1.json"]:
                raise ValueError("lane missing a complete 2-game pair")
            for side, member in enumerate(members):
                game = json.load(tar.extractfile(member))
                expected = plan["families"][lane]["openings"][side]
                if (game.get("lane"), game.get("game_side"), game.get("parent_id"),
                    game.get("models")) != (lane, side, expected["parent_id"], [BASELINE, BASELINE]):
                    raise ValueError("game metadata or model changed")
                if game["moves"][:len(expected["prefix"])] != expected["prefix"]:
                    raise ValueError("opening prefix changed")
                key = game_key(game)
                if key in seen:
                    raise ValueError("duplicate game history")
                seen.add(key)
                validate_game(game, model, training=True)
                total_plies += len(game["moves"])
                longmoves += sum(bool(x.get("longthink")) for x in game["telemetry"])
                counts[game["status"]] += 1
                atomic_json(rows / f"lane-{lane:02d}-game-{side}.json", game)
    if len(seen) != 40:
        raise ValueError("expected 40 verified games")
    report = {
        "schema": "fukaji-d41-d37-20runners-selfplay-report-v1",
        "baseline": BASELINE, "weights_sha256": WEIGHTS_SHA,
        "planned_jobs": 20, "validated_games": 40, "statuses": dict(counts),
        "total_plies": total_plies, "longthink_moves": longmoves,
        "trained": False, "optimizer_present": False,
        "model_adopted": False,
        "next": "restore D37 exact Adam2106 and train on verified games, then compare old/new",
    }
    atomic_json(output / "report.json", report)
    atomic_json(output / "plan.json", plan)
    publication = output / "publication"
    publication.mkdir(exist_ok=True)
    for entry in [output / "report.json", output / "plan.json", *sorted(rows.glob("*.json"))]:
        dest = publication / (entry.name if entry.parent == output else "game-" + entry.name)
        dest.write_bytes(entry.read_bytes())
    package = output / "D41-D37-20runner-selfplay-40games.tar.gz"
    core.pack_tar(publication, package)
    atomic_json(output / "receipt.json", {
        "sha256": digest(package.read_bytes()), "bytes": package.stat().st_size,
        "games": 40, "trained": False,
    })
    print(json.dumps(report))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--bundle", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    b = sub.add_parser("segment")
    b.add_argument("--runtime", type=Path, required=True)
    b.add_argument("--lane", type=int, required=True)
    b.add_argument("--side", type=int, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--search-limit", type=int, default=36)
    c = sub.add_parser("worker")
    c.add_argument("--runtime-bundle", type=Path, required=True)
    c.add_argument("--lane", type=int, required=True)
    c.add_argument("--output", type=Path, required=True)
    d = sub.add_parser("collect")
    d.add_argument("--runtime-bundle", type=Path, required=True)
    d.add_argument("--downloads", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True)
    z = p.parse_args()
    if z.command == "prepare":
        prepare(z.bundle, z.output)
    elif z.command == "segment":
        segment(z.runtime, z.lane, z.side, z.output, z.search_limit)
    elif z.command == "worker":
        worker(z.runtime_bundle, z.lane, z.output)
    elif z.command == "collect":
        collect(z.runtime_bundle, z.downloads, z.output)


if __name__ == "__main__":
    main()
