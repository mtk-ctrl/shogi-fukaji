"""G1 大量データ学習: g1_gen.py の自己対局データ(数百万局面)で、親(現行正式NNUE)から学習する。
ラベル = (1-w)*sigmoid(探索cp/scale) + w*対局結果(手番側)。雲路の指し手・評価値は不使用(評価は常に深路NNUE)。
feature-set: base(現行の絶対特徴2548) / zone(玉周辺±2升, +1300) / king(玉相対17x17, +15028)。拡張直後の評価は親と一致(新規行ゼロ)。
棋譜の5%を検証用に除き、検証ラベル損失が下がることを確認(gate)。棋力の判定は親との対局(hybrid-match)で行う。
"""
from pathlib import Path
import argparse, gzip, hashlib, json, multiprocessing as mp, sys, time
import numpy as np
import scipy.sparse as sp
import shogi
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fukaji.online_learning import atomic_json, load_snapshot, save_snapshot, parameters, expand_king_relative, expand_king_zone
from fukaji.nnue_research import SmallNNUE, HIDDEN, active_indices, king_zone_indices, king_relative_indices

WIDTH = {"base": 40, "zone": 116, "king": 116}
INPUTS = {}
_CFG = None


def featurize(chunk):
    fs, scale, ow, min_ply, stride_seed = _CFG
    pad = INPUTS[fs]
    xs, ys, hold = [], [], []
    for gi, line in chunk:
        g = json.loads(line)
        if g["r"] == "illegal":
            continue
        h = int(hashlib.sha256(" ".join(g["m"]).encode()).hexdigest()[:8], 16) % 20 == 0
        board = shogi.Board()
        for ply, tok in enumerate(g["m"]):
            s = g["s"][ply]
            if s is not None and ply >= max(g["k"], min_ply):
                ix = active_indices(board, board.turn)
                if fs == "zone":
                    ix = ix + king_zone_indices(board, board.turn)
                elif fs == "king":
                    ix = ix + king_relative_indices(board, board.turn)
                row = np.full(WIDTH[fs], pad, np.int16); row[:len(ix)] = ix
                z = .5 if g["w"] is None else float(g["w"] == int(board.turn))
                ps = 1 / (1 + np.exp(-float(np.clip(s / scale, -8, 8))))
                xs.append(row); ys.append((1 - ow) * ps + ow * z); hold.append(h)
            board.push(shogi.Move.from_usi(tok))
    if not xs:
        return np.zeros((0, WIDTH[fs]), np.int16), np.zeros(0, np.float32), np.zeros(0, bool)
    return np.stack(xs), np.asarray(ys, np.float32), np.asarray(hold, bool)


def onehot(ix, inputs):
    n, w = ix.shape
    return sp.csr_matrix((np.ones(n * w, np.float32), ix.reshape(-1).astype(np.int32), np.arange(0, n * w + 1, w)), shape=(n, inputs + 1))


def forward(params, X):
    w, b, v, o = params
    pre = X[:, :-1] @ w + b
    hidden = np.clip(pre, 0, 1)
    return pre, hidden, hidden @ v + o[0]


def loss_of(params, ix, y, inputs):
    tot = 0.
    for i in range(0, len(y), 8192):
        lg = forward(params, onehot(ix[i:i+8192], inputs))[2]
        tot += float(np.sum(np.logaddexp(0, lg) - y[i:i+8192] * lg))
    return tot / max(1, len(y))


def fit(params, opt, ix, y, hx, hy, inputs, *, epochs, lr, batch, seed):
    """fukaji.online_learning.fit と同じ更新式(Adam, 勾配ノルム上限1, 減衰1e-5)。疎行列で高速化し、バッチを大きくできる。"""
    rng = np.random.default_rng(seed)
    history = [{"epoch": 0, "heldout": loss_of(params, hx, hy, inputs)}]
    t0 = time.time()
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        for n, start in enumerate(range(0, len(order), batch)):
            sel = np.sort(order[start:start+batch])
            X = onehot(ix[sel], inputs); t = y[sel]
            pre, hidden, logits = forward(params, X)
            error = ((1 / (1 + np.exp(-logits)) - t) / len(t)).astype(np.float32)
            dh = (error[:, None] * params[2][None, :] * ((pre > 0) & (pre < 1))).astype(np.float32)
            grads = [np.asarray(X.T @ dh)[:-1], dh.sum(axis=0), hidden.T @ error, np.array([error.sum()], np.float32)]
            norm = float(np.sqrt(sum(np.sum(g.astype(float)**2) for g in grads)))
            if not np.isfinite(norm):
                raise ValueError("nonfinite gradient")
            opt["step"] += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                g = g / max(1., norm)
                opt["m"][i] = .9 * opt["m"][i] + .1 * g
                opt["s"][i] = .999 * opt["s"][i] + .001 * g * g
                p -= lr * ((opt["m"][i] / (1 - .9**opt["step"])) / (np.sqrt(opt["s"][i] / (1 - .999**opt["step"])) + 1e-8) + 1e-5 * p)
            if n % 2000 == 0:
                print(f"epoch {epoch+1} step {n}/{len(order)//batch} elapsed {time.time()-t0:.0f}s", flush=True)
        history.append({"epoch": epoch + 1, "heldout": loss_of(params, hx, hy, inputs)})
        print(history[-1], flush=True)
    return history


def main():
    global _CFG
    p = argparse.ArgumentParser()
    p.add_argument("--parent-models", type=Path, required=True)
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=.0002)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--outcome-weight", type=float, default=.2)
    p.add_argument("--feature-set", choices=["base", "zone", "king"], default="base")
    p.add_argument("--scale", type=float, default=600.)
    p.add_argument("--min-ply", type=int, default=8)
    p.add_argument("--max-games", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261114)
    a = p.parse_args()
    formal = json.loads(Path("formal/model.json").read_text())
    BASE = formal["research_snapshot_sha256"]
    a.out.mkdir(parents=True, exist_ok=True)
    md = a.out / "models"; md.mkdir(exist_ok=True)
    for ext in ("npz", "json"):
        (md / f"{BASE}.{ext}").write_bytes((a.parent_models / f"{BASE}.{ext}").read_bytes())
    base, base_opt, _ = load_snapshot(md, BASE)
    if base_opt is None or base_opt["step"] != formal["optimizer_step"]:
        raise SystemExit("Adam state missing or step differs from formal")
    parent, opt = (base, base_opt) if a.feature_set == "base" else (expand_king_zone if a.feature_set == "zone" else expand_king_relative)(base, base_opt)
    for fs, m in (("base", 2548), ("zone", 3848), ("king", 17576)):
        INPUTS[fs] = m
    assert parent.inputs == INPUTS[a.feature_set]
    lines = []
    for d in a.data:
        for f in sorted(d.rglob("games-*.jsonl.gz")):
            with gzip.open(f, "rt") as fh:
                lines += fh.read().splitlines()
    if a.max_games:
        lines = lines[:a.max_games]
    print("games", len(lines), flush=True)
    _CFG = (a.feature_set, a.scale, a.outcome_weight, a.min_ply, a.seed)
    items = list(enumerate(lines))
    chunks = [items[i:i+200] for i in range(0, len(items), 200)]
    with mp.Pool() as pool:
        parts = pool.map(featurize, chunks, chunksize=1)
    ix = np.concatenate([x for x, _, _ in parts]); y = np.concatenate([t for _, t, _ in parts]); hold = np.concatenate([h for _, _, h in parts])
    del parts
    hx, hy = ix[hold], y[hold]; ix, y = ix[~hold], y[~hold]
    print(f"train positions {len(y)} hold {len(hy)} mean label {float(y.mean()):.3f}", flush=True)
    if len(y) < int(__import__("os").environ.get("G1_MIN_TRAIN","20000")) or len(hy) < 100:
        raise SystemExit("not enough positions")
    params = parameters(parent)
    opt2 = {"step": opt["step"], "m": [x.copy() for x in opt["m"]], "s": [x.copy() for x in opt["s"]]}
    history = fit(params, opt2, ix, y, hx, hy, parent.inputs, epochs=a.epochs, lr=a.lr, batch=a.batch, seed=a.seed)
    before, after = history[0]["heldout"], history[-1]["heldout"]
    gate = after < before
    print(f"heldout label loss {before:.5f} -> {after:.5f} gate {'pass' if gate else 'fail'}", flush=True)
    cand = SmallNNUE(params[0], params[1], params[2], params[3][0])
    key = save_snapshot(md, cand, opt2, {"parent": BASE, "purpose": "G1 mass self-play data", "epochs": a.epochs,
                                         "learning_rate": a.lr, "training": {"positions": int(len(y)), "optimizer_step": opt2["step"], "strength_proven": False}})
    w, b, v, o = parameters(cand)
    np.savez(a.out / "cand-weights.npz", w=w, b=b, v=v, o=o)
    report = {"schema": "g1-train-v1", "parent": BASE, "snapshot": key,
              "weights_file_sha256": hashlib.sha256((a.out / "cand-weights.npz").read_bytes()).hexdigest(),
              "games": len(lines), "train_positions": int(len(y)), "heldout_positions": int(len(hy)),
              "outcome_weight": a.outcome_weight, "feature_set": a.feature_set, "inputs": int(parent.inputs),
              "heldout_history": history, "heldout_before": before, "heldout_after": after, "gate": "pass" if gate else "fail",
              "epochs": a.epochs, "lr": a.lr, "batch": a.batch, "optimizer_step_after": opt2["step"],
              "strength_proven": False, "formal_model_changed": False}
    atomic_json(a.out / "report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
