"""H: E5(32隠れ)を機能保存のまま広げ(新ニューロンの出力重み0)、E5の探索評価値ラベル(蒸留)で学習する。研究専用。
雲路の指し手・評価値は不使用。ゲート＝評価用ラベル損失が親(=広げた直後、親と同一関数)より下がること。最終判定は親E5との対局。
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fukaji.online_learning import atomic_json, load_snapshot, fit, validate_game, parameters
from fukaji.nnue_research import SmallNNUE
from scripts.e1_train import pad, heldout_loss
from scripts.e2_data import load_games
from scripts.h_widen_lib import use_hidden


def widen(parent, opt, H, rng):
    h0 = parent.hidden_weights.shape[1]
    extra = H - h0
    w = np.concatenate([parent.hidden_weights, rng.normal(0, .02, (parent.inputs, extra)).astype(np.float32)], axis=1)
    b = np.concatenate([parent.hidden_bias, np.full(extra, .1)])
    v = np.concatenate([parent.output_weights, np.zeros(extra, np.float32)])
    cand = SmallNNUE(w, b, v, parent.output_bias)
    def grow(a, axis0=False):
        if a.ndim == 2:
            return np.concatenate([a, np.zeros((a.shape[0], extra), a.dtype)], axis=1)
        if a.shape == (1,):
            return a.copy()
        return np.concatenate([a, np.zeros(extra, a.dtype)])
    o = {"step": opt["step"], "m": [grow(x) for x in opt["m"]], "s": [grow(x) for x in opt["s"]]}
    return cand, o


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--parent-models", type=Path, required=True)
    p.add_argument("--parts", type=Path, required=True)
    p.add_argument("--archives", type=Path, nargs="+", required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=.0001)
    p.add_argument("--outcome-weight", type=float, default=.2)
    p.add_argument("--seed", type=int, default=20261112)
    a = p.parse_args()
    formal = json.loads(Path("formal/model.json").read_text())
    BASE = formal["research_snapshot_sha256"]
    a.out.mkdir(parents=True, exist_ok=True)
    md = a.out / "parent-models"; md.mkdir(exist_ok=True)
    for ext in ("npz", "json"):
        (md / f"{BASE}.{ext}").write_bytes((a.parent_models / f"{BASE}.{ext}").read_bytes())
    parent, opt, _ = load_snapshot(md, BASE)
    if opt is None or opt["step"] != formal["optimizer_step"]:
        raise SystemExit("Adam state missing or step differs from formal")
    labels = {}
    for f in sorted(a.labels.rglob("labels-*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line); labels.setdefault(r["key"], {})[r["ply"]] = r["score"]
    items = [x for x in load_games(a.parts, a.archives, a.state) if x["key"] in labels]
    rows_t, y_t, rows_h, y_h, bad = [], [], [], [], 0
    for x in items:
        try:
            ix, probs, z, _ = validate_game(x["game"], parent, training=True)
        except Exception:
            bad += 1; continue
        hold = int(x["key"][:8], 16) % 20 == 0
        for ply, score in labels[x["key"]].items():
            if ply >= len(ix):
                continue
            ps = 1 / (1 + np.exp(-float(np.clip(score / 1000., -8, 8))))
            y = (1 - a.outcome_weight) * ps + a.outcome_weight * z[ply]
            (rows_h if hold else rows_t).append(np.asarray(ix[ply], np.int32)[None, :]); (y_h if hold else y_t).append(y)
    tx = pad(rows_t, parent.inputs); ty = np.asarray(y_t, np.float32)
    hx = pad(rows_h, parent.inputs); hy = np.asarray(y_h, np.float32)
    print(f"train positions {len(ty)} hold {len(hy)} rejected games {bad}", flush=True)
    if len(ty) < 20000 or len(hy) < 500:
        raise SystemExit("not enough labeled positions")
    before32 = heldout_loss(parent, hx, hy)
    use_hidden(a.hidden)  # 以降は広いネットワーク
    wide, wopt = widen(parent, opt, a.hidden, np.random.default_rng(a.seed))
    before = heldout_loss(wide, hx, hy)
    if abs(before - before32) > 1e-4:
        raise SystemExit(f"widening is not function-preserving: {before32} vs {before}")
    cand, opt2, fs = fit(wide, wopt, tx, ty, epochs=a.epochs, lr=a.lr, seed=a.seed)
    after = heldout_loss(cand, hx, hy)
    gate = after < before
    print(f"heldout label loss {before:.5f} -> {after:.5f} gate {'pass' if gate else 'fail'}", flush=True)
    w, b, v, o = parameters(cand)
    np.savez(a.out / "cand-weights.npz", w=w, b=b, v=v, o=o)
    report = {"schema": "h-wide-distill-v1", "parent": BASE, "hidden": a.hidden,
              "weights_file_sha256": hashlib.sha256((a.out / "cand-weights.npz").read_bytes()).hexdigest(),
              "train_positions": int(len(ty)), "heldout_positions": int(len(hy)),
              "outcome_weight": a.outcome_weight, "heldout_before": before, "heldout_after": after,
              "gate": "pass" if gate else "fail", "epochs": a.epochs, "lr": a.lr,
              "optimizer_step_after": opt2["step"], "strength_proven": False, "formal_model_changed": False}
    atomic_json(a.out / "report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
