"""E11(玉相対特徴): 親E10の重みを変えずに玉相対特徴(+15028入力)へ拡張(新規行ゼロ)し、ハイブリッド探索ラベルで学習する。
元E2: 正式探索(固定深さ4)の評価値を教師にして現行NNUEを控えめに更新する(蒸留)。
ラベル = (1-w)*sigmoid(search_logit) + w*対局結果。雲路の指し手・評価値は不使用。
評価用に5%の棋譜を除き、ラベルに対する損失が下がることを確認(gate)。最終判定はE3との対局。
"""
from pathlib import Path
import argparse, hashlib, json, sys, tarfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fukaji.online_learning import atomic_json, load_snapshot, save_snapshot, fit, validate_game, parameters, expand_king_relative, expand_king_zone
from scripts.e1_train import pad, heldout_loss
from scripts.e2_data import load_games


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--parent-models", type=Path, required=True)
    p.add_argument("--parts", type=Path, required=True)
    p.add_argument("--archives", type=Path, nargs="+", required=True)
    p.add_argument("--labels", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=.00005)
    p.add_argument("--outcome-weight", type=float, default=.2)
    p.add_argument("--feature-set", choices=["king", "zone"], default="king")
    p.add_argument("--seed", type=int, default=20261111)
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
    parent, opt = (expand_king_zone if a.feature_set == "zone" else expand_king_relative)(base, base_opt)   # 拡張直後の評価は親E10と一致(新規行ゼロ)
    labels = {}
    for d in a.labels:  # 後に並べたラベルが同じ局面を上書き(新しい教師を後ろへ)
        for f in sorted(d.rglob("labels-*.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line); labels.setdefault(r["key"], {})[r["ply"]] = r["score"]
    items = [x for x in load_games(a.parts, a.archives, a.state) if x["key"] in labels]
    print("games with labels", len(items), "labels", sum(len(v) for v in labels.values()), flush=True)
    rows_t, y_t, rows_h, y_h, bad = [], [], [], [], 0
    stats = {"search_mean": 0., "n": 0}
    for x in items:
        try:
            ix, probs, z, _ = validate_game(x["game"], base, training=True, feature_model=parent)
        except Exception:
            bad += 1; continue
        hold = int(x["key"][:8], 16) % 20 == 0
        for ply, score in labels[x["key"]].items():
            if ply >= len(ix):
                continue
            ps = 1 / (1 + np.exp(-float(np.clip(score / 1000., -8, 8))))
            y = (1 - a.outcome_weight) * ps + a.outcome_weight * z[ply]
            (rows_h if hold else rows_t).append(np.asarray(ix[ply], np.int32)[None, :]); (y_h if hold else y_t).append(y)
            stats["search_mean"] += ps; stats["n"] += 1
    tx = pad(rows_t, parent.inputs); ty = np.asarray(y_t, np.float32)
    hx = pad(rows_h, parent.inputs); hy = np.asarray(y_h, np.float32)
    print(f"train positions {len(ty)} hold {len(hy)} rejected games {bad} mean search p {stats['search_mean']/max(1,stats['n']):.3f}", flush=True)
    if len(ty) < 20000 or len(hy) < 500:
        raise SystemExit("not enough labeled positions")
    before = heldout_loss(parent, hx, hy)
    cand, opt2, fs = fit(parent, opt, tx, ty, epochs=a.epochs, lr=a.lr, seed=a.seed)
    after = heldout_loss(cand, hx, hy)
    gate = after < before
    print(f"heldout label loss {before:.5f} -> {after:.5f} gate {'pass' if gate else 'fail'}", flush=True)
    key = save_snapshot(md, cand, opt2, {"parent": BASE, "purpose": "E11 king-relative expansion",
                                         "epochs": a.epochs, "learning_rate": a.lr, "training": fs})
    w, b, v, o = parameters(cand)
    np.savez(a.out / "cand-weights.npz", w=w, b=b, v=v, o=o)
    report = {"schema": "e11-king-relative-v1", "parent": BASE, "snapshot": key,
              "weights_file_sha256": hashlib.sha256((a.out / "cand-weights.npz").read_bytes()).hexdigest(),
              "train_positions": int(len(ty)), "heldout_positions": int(len(hy)),
              "outcome_weight": a.outcome_weight, "heldout_before": before, "heldout_after": after,
              "gate": "pass" if gate else "fail", "epochs": a.epochs, "lr": a.lr,
              "optimizer_step_after": opt2["step"], "strength_proven": False, "formal_model_changed": False, "inputs": int(parent.inputs), "feature_set": a.feature_set}
    atomic_json(a.out / "report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
