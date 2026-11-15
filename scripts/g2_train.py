"""G2 研究: 評価の形を変える(両視点入力＋2層目)。データ・ラベル・親は G1(g1_train.py) と同じ。
現行: 評価 = o + v・clip(W x自 + b)                      … 手番側から見た特徴だけ、1層
G2  : 評価 = 現行の式 + w3・clip(L2 [clip(W x自+b), clip(W x相+b)] + b2)
  - x相 は相手側から見た同じ局面の特徴。W は自・相で共有(行は増えない)。
  - 2層目(L2: 2H→h2、w3: h2→1)は駒どうしの組み合わせを表すための追加。w3 を 0 で始めるので、学習開始時の評価は親と完全に一致する。
教師は G1 と同じ(探索値と対局結果の混合、左右反転あり)。雲路の評価値・指し手は不使用。
出力: cand-weights.npz (w,b,v,o,l2,b2,w3) / v2-state.npz (Adam状態つき) / report.json (schema g2-train-v1)。正式snapshot形式には入れない(研究)。
"""
from pathlib import Path
import argparse, gzip, hashlib, json, multiprocessing as mp, os, sys, time
import numpy as np
import shogi
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_train as g1
from fukaji.online_learning import atomic_json, load_snapshot, parameters
from fukaji.nnue_research import active_indices, king_zone_indices

W_ = 116
PAD = 3848
_CFG = None


def featurize(chunk):
    ow, min_ply = _CFG
    xu, xt, ys, hold = [], [], [], []
    for gi, (line, scale) in chunk:
        g = json.loads(line)
        if g["r"] == "illegal":
            continue
        h = int(hashlib.sha256(" ".join(g["m"]).encode()).hexdigest()[:8], 16) % 20 == 0
        board = shogi.Board()
        for ply, tok in enumerate(g["m"]):
            s = g["s"][ply]
            if s is not None and ply >= max(g["k"], min_ply):
                for dst, persp in ((xu, board.turn), (xt, board.turn ^ 1)):
                    ix = active_indices(board, persp) + king_zone_indices(board, persp)
                    row = np.full(W_, PAD, np.int16); row[:len(ix)] = ix
                    dst.append(row)
                z = .5 if g["w"] is None else float(g["w"] == int(board.turn))
                ps = 1 / (1 + np.exp(-float(np.clip(s / scale, -8, 8))))
                ys.append((1 - ow) * ps + ow * z); hold.append(h)
            board.push(shogi.Move.from_usi(tok))
    if not ys:
        e = np.zeros((0, W_), np.int16)
        return e, e, np.zeros(0, np.float32), np.zeros(0, bool)
    return np.stack(xu), np.stack(xt), np.asarray(ys, np.float32), np.asarray(hold, bool)


def forward(params, Xu, Xt):
    w, b, v, o, l2, b2, w3 = params
    pu = Xu[:, :-1] @ w + b; hu = np.clip(pu, 0, 1)
    pt = Xt[:, :-1] @ w + b; ht = np.clip(pt, 0, 1)
    c = np.concatenate((hu, ht), axis=1)
    p2 = c @ l2 + b2; h2 = np.clip(p2, 0, 1)
    return pu, hu, pt, ht, c, p2, h2, hu @ v + o[0] + h2 @ w3


def loss_of(params, xu, xt, y):
    tot = 0.
    for i in range(0, len(y), 8192):
        lg = forward(params, g1.onehot(xu[i:i+8192], PAD), g1.onehot(xt[i:i+8192], PAD))[-1]
        tot += float(np.sum(np.logaddexp(0, lg) - y[i:i+8192] * lg))
    return tot / max(1, len(y))


def fit(params, opt, xu, xt, y, hxu, hxt, hy, *, epochs, lr, batch, seed, perm):
    """更新式は g1_train.fit と同じ(Adam, 勾配ノルム上限1, 減衰1e-5)。"""
    rng = np.random.default_rng(seed)
    history = [{"epoch": 0, "heldout": loss_of(params, hxu, hxt, hy)}]
    t0 = time.time()
    for epoch in range(epochs):
        order = rng.permutation(len(y) * (2 if perm is not None else 1))
        for n, start in enumerate(range(0, len(order), batch)):
            sel = np.sort(order[start:start+batch])
            ru, rt, t = xu[sel % len(y)], xt[sel % len(y)], y[sel % len(y)]
            if perm is not None:
                flip = sel >= len(y); ru[flip] = perm[ru[flip]]; rt[flip] = perm[rt[flip]]
            Xu, Xt = g1.onehot(ru, PAD), g1.onehot(rt, PAD)
            w, b, v, o, l2, b2, w3 = params
            pu, hu, pt, ht, c, p2, h2, logits = forward(params, Xu, Xt)
            e = ((1 / (1 + np.exp(-logits)) - t) / len(t)).astype(np.float32)
            dp2 = (e[:, None] * w3[None, :] * ((p2 > 0) & (p2 < 1))).astype(np.float32)
            dc = dp2 @ l2.T
            H = w.shape[1]
            dpu = ((e[:, None] * v[None, :] + dc[:, :H]) * ((pu > 0) & (pu < 1))).astype(np.float32)
            dpt = (dc[:, H:] * ((pt > 0) & (pt < 1))).astype(np.float32)
            grads = [np.asarray(Xu.T @ dpu)[:-1] + np.asarray(Xt.T @ dpt)[:-1], dpu.sum(axis=0) + dpt.sum(axis=0), hu.T @ e,
                     np.array([e.sum()], np.float32), c.T @ dp2, dp2.sum(axis=0), h2.T @ e]
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
        history.append({"epoch": epoch + 1, "heldout": loss_of(params, hxu, hxt, hy)})
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
    p.add_argument("--h2", type=int, default=32)
    p.add_argument("--min-ply", type=int, default=8)
    p.add_argument("--max-games", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--no-mirror", action="store_true")
    a = p.parse_args()
    formal = json.loads(Path("formal/model.json").read_text()); hyb = formal["hybrid"]
    BASE = hyb["research_snapshot_sha256"]
    base, base_opt, _ = load_snapshot(a.parent_models, BASE)
    if base_opt is None or base_opt["step"] != hyb["optimizer_step"] or base.inputs != PAD:
        raise SystemExit("parent must be the formal hybrid NNUE (zone, with Adam state)")
    g1.INPUTS.update(base=2548, zone=3848, king=17576)
    a.out.mkdir(parents=True, exist_ok=True)
    lines, scales = [], {}
    for d in a.data:
        for f in sorted(d.rglob("games-*.jsonl.gz")):
            sm = f.with_name(f.name.replace("games-", "summary-").replace(".jsonl.gz", ".json"))
            sc = float(json.loads(sm.read_text()).get("nnue_scale", 600.)) if sm.exists() else 600.
            scales[sc] = scales.get(sc, 0) + 1
            with gzip.open(f, "rt") as fh:
                lines += [(x, sc) for x in fh.read().splitlines()]
    if a.max_games:
        lines = lines[:a.max_games]
    print("games", len(lines), "label scales (files)", scales, flush=True)
    _CFG = (a.outcome_weight, a.min_ply)
    items = list(enumerate(lines))
    chunks = [items[i:i+200] for i in range(0, len(items), 200)]
    with mp.Pool() as pool:
        parts = pool.map(featurize, chunks, chunksize=1)
    hold = np.concatenate([q[3] for q in parts]); y = np.concatenate([q[2] for q in parts])
    xu = np.concatenate([q[0] for q in parts]); parts = [(None, q[1]) for q in parts]
    xt = np.concatenate([q[1] for q in parts]); del parts
    hxu, hxt, hy = xu[hold], xt[hold], y[hold]; xu, xt, y = xu[~hold], xt[~hold], y[~hold]
    print(f"train positions {len(y)} hold {len(hy)} mean label {float(y.mean()):.3f}", flush=True)
    if len(y) < int(os.environ.get("G1_MIN_TRAIN", "20000")) or len(hy) < 100:
        raise SystemExit("not enough positions")
    w, b, v, o = parameters(base)
    H = w.shape[1]
    rng = np.random.default_rng(a.seed + 11)
    new = [rng.normal(0, 0.05, (2 * H, a.h2)).astype(w.dtype), np.full(a.h2, 0.3, w.dtype), np.zeros(a.h2, w.dtype)]
    params = [w, b, v, o] + new
    opt = {"step": base_opt["step"], "m": [x.copy() for x in base_opt["m"]] + [np.zeros_like(x) for x in new],
           "s": [x.copy() for x in base_opt["s"]] + [np.zeros_like(x) for x in new]}
    parent_loss = g1.loss_of([w, b, v, o], hxu, hy, PAD)
    history = fit(params, opt, xu, xt, y, hxu, hxt, hy, epochs=a.epochs, lr=a.lr, batch=a.batch, seed=a.seed,
                  perm=None if a.no_mirror else g1.mirror_perm("zone"))
    before, after = history[0]["heldout"], history[-1]["heldout"]
    if abs(before - parent_loss) > 1e-6:
        raise SystemExit(f"initial network is not identical to the parent: {before} vs {parent_loss}")
    gate = after < before
    print(f"heldout label loss {before:.5f} -> {after:.5f} gate {'pass' if gate else 'fail'}", flush=True)
    names = ("w", "b", "v", "o", "l2", "b2", "w3")
    out = {k: np.asarray(x, np.float32) for k, x in zip(names, params)}
    np.savez(a.out / "cand-weights.npz", **out)
    np.savez_compressed(a.out / "v2-state.npz", **out, step=opt["step"],
                        **{f"m{i}": x for i, x in enumerate(opt["m"])}, **{f"s{i}": x for i, x in enumerate(opt["s"])})
    report = {"schema": "g2-train-v1", "architecture": f"dual-perspective {PAD}x{H} shared + skip + [{2*H}->{a.h2}->1]",
              "parent": BASE, "snapshot": "v2:" + hashlib.sha256((a.out / "v2-state.npz").read_bytes()).hexdigest(),
              "weights_file_sha256": hashlib.sha256((a.out / "cand-weights.npz").read_bytes()).hexdigest(),
              "games": len(lines), "train_positions": int(len(y)), "heldout_positions": int(len(hy)),
              "outcome_weight": a.outcome_weight, "label_scales": {str(k): n for k, n in scales.items()}, "mirror": not a.no_mirror,
              "inputs": PAD, "hidden": int(H), "h2": a.h2, "heldout_history": history, "heldout_before": before, "heldout_after": after,
              "gate": "pass" if gate else "fail", "epochs": a.epochs, "lr": a.lr, "batch": a.batch,
              "optimizer_step_after": opt["step"], "strength_proven": False, "formal_model_changed": False}
    atomic_json(a.out / "report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
