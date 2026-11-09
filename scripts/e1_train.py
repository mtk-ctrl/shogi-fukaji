"""E1 (深路=NNUE主軸): D66のNNUEを、過去の全自己対局の全局面で学習し直す。

- 探索・評価の人手ヒューリスティクスは一切使わない。使うのは自己対局棋譜の結果と、NNUE自身の予測(TD-λ)だけ。
- 全局面(per_game制限なし)を使い、評価用に5%の棋譜を学習から除いて損失を前後比較する。
- 正式版(formal/)は変更しない。出力は研究用snapshotと推論専用npz。
"""
from pathlib import Path
import argparse, hashlib, json, multiprocessing as mp, sys, tarfile, time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fukaji.online_learning import (atomic_json, digest, load_snapshot, save_snapshot, make_samples,
                                    fit, game_key, logits_for, probability, parameters)

BASE = "c29b5ba1ed9264ba298ffdef2ab0deb71f37b327a32c3a1f8505061f67401ac3"
_PARENT = None
_PROTECTED = frozenset()


def _one(args):
    game, seed = args
    try:
        ix, y, st = make_samples([game], _PARENT, seed=seed, per_game=100000, outcome_weight=.75,
                                 td_lambda=.8, protected=_PROTECTED)
        return ix, y, st["resolved"]
    except Exception as e:  # 古い棋譜で検証に通らないものは数えて除外する
        return None, None, repr(e)[:80]


def pad(rows, inputs):
    width = max(r.shape[1] for r in rows)
    out = []
    for r in rows:
        if r.shape[1] < width:
            r = np.concatenate([r, np.full((len(r), width - r.shape[1]), inputs, np.int32)], axis=1)
        out.append(r)
    return np.concatenate(out)


def heldout_loss(model, ix, y):
    params = parameters(model)
    tot = 0.
    for i in range(0, len(y), 4096):
        lg = logits_for(params, ix[i:i+4096])[2]
        tot += float(np.sum(np.logaddexp(0, lg) - y[i:i+4096] * lg))
    return tot / len(y)


def main():
    global _PARENT, _PROTECTED
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--baseline-trained", type=Path, required=True)
    p.add_argument("--archives", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=.0002)
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    formal = json.loads(Path("formal/model.json").read_text())
    if formal["research_snapshot_sha256"] != BASE or formal["optimizer_step"] != 11864:
        raise SystemExit("formal model changed; refuse")
    a.out.mkdir(parents=True, exist_ok=True)
    md = a.out / "models"; md.mkdir(exist_ok=True)
    with tarfile.open(a.baseline_trained) as t:
        for ext in ("npz", "json"):
            (md / f"{BASE}.{ext}").write_bytes(t.extractfile(f"{BASE}.{ext}").read())
    parent, opt, _ = load_snapshot(md, BASE)
    if opt is None or opt["step"] != 11864:
        raise SystemExit("Adam state missing")
    with tarfile.open(a.state) as t:
        meta = json.load(t.extractfile("d17-state/state.json"))
        boot = t.extractfile("d17-state/bootstrap-train.jsonl").read()
    _PROTECTED = frozenset(meta["protected_parents"])
    _PARENT = parent
    games = [json.loads(x) for x in boot.splitlines() if x.strip()]
    n_boot = len(games)
    per = {}
    for f in sorted(a.archives.glob("*-1000games.tar.gz")):
        with tarfile.open(f) as t:
            names = sorted(n for n in t.getnames() if n.startswith("games/game-") and n.endswith(".json"))
            gs = [json.load(t.extractfile(n)) for n in names]
        per[f.name] = len(gs)
        games += gs
    print("loaded", n_boot, "bootstrap +", per, flush=True)
    seen, uniq = set(), []
    for g in games:
        k = game_key(g)
        if k not in seen:
            seen.add(k); uniq.append(g)
    games = uniq
    hold = [g for g in games if int(game_key(g)[:8], 16) % 20 == 0]
    train_games = [g for g in games if int(game_key(g)[:8], 16) % 20 != 0]
    print("unique", len(games), "train", len(train_games), "holdout", len(hold), flush=True)

    def build(gl, tag):
        t0 = time.time()
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = pool.map(_one, [(g, a.seed + i) for i, g in enumerate(gl)], chunksize=64)
        rows = [r[0] for r in res if r[0] is not None]
        labs = [r[1] for r in res if r[0] is not None]
        bad = sum(1 for r in res if r[0] is None)
        resolved = sum(r[2] for r in res if r[0] is not None)
        ix = pad(rows, parent.inputs); y = np.concatenate(labs)
        print(f"{tag}: games={len(gl)} used={len(rows)} resolved={resolved} rejected={bad} "
              f"positions={len(y)} {time.time()-t0:.0f}s", flush=True)
        return ix, y, {"games": len(gl), "used": len(rows), "resolved": resolved, "rejected": bad, "positions": int(len(y))}

    hx, hy, hs = build(hold, "holdout")
    tx, ty, ts = build(train_games, "train")
    before = heldout_loss(parent, hx, hy)
    t0 = time.time()
    cand, opt2, fs = fit(parent, opt, tx, ty, epochs=a.epochs, lr=a.lr, seed=a.seed)
    after = heldout_loss(cand, hx, hy)
    print(f"fit {time.time()-t0:.0f}s heldout loss {before:.5f} -> {after:.5f}", flush=True)
    key = save_snapshot(md, cand, opt2, {"parent": BASE, "purpose": "E1 all-positions retrain from D66",
                                         "epochs": a.epochs, "learning_rate": a.lr, "training": fs})
    w, b, v, o = parameters(cand)
    np.savez(a.out / "e1-weights.npz", w=w, b=b, v=v, o=o)
    wsha = hashlib.sha256((a.out / "e1-weights.npz").read_bytes()).hexdigest()
    report = {"schema": "e1-allpositions-v1", "parent": BASE, "snapshot": key, "weights_file_sha256": wsha,
              "archives": per, "bootstrap_games": n_boot, "unique_games": len(games),
              "train": ts, "holdout": hs, "heldout_loss_before": before, "heldout_loss_after": after,
              "epochs": a.epochs, "lr": a.lr, "optimizer_step_after": opt2["step"],
              "fit": fs, "strength_proven": False, "formal_model_changed": False}
    atomic_json(a.out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "fit"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
