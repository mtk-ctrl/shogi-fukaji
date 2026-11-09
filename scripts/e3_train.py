"""E3 (深路=NNUE主軸): 雲路を練習相手にした対局の「深路の手番の局面」で、D66のNNUEを控えめに更新する。

- 雲路の指し手・評価値・知識は教師にしない。使うのは対局の勝敗と、深路NNUE自身の予測(TD-λ)だけ。
- 雲路側の手番の局面は使わない。深路の手番(ply6以降)の局面だけ。
- 偏り防止のため、現行世代(D66/D68)の自己対局の局面を同数混ぜる。
- 評価用に5%の対局を除き、(a)自己対局の損失が悪化しない (b)雲路戦の損失が下がる の両方を満たす場合だけ gate=pass。
"""
from pathlib import Path
import argparse, hashlib, json, sys, tarfile
import numpy as np
import shogi
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fukaji.online_learning import (atomic_json, load_snapshot, save_snapshot, make_samples, fit, game_key,
                                    validate_game, targets, parameters)
from scripts.e1_train import pad, heldout_loss

BASE = "c29b5ba1ed9264ba298ffdef2ab0deb71f37b327a32c3a1f8505061f67401ac3"


def convert(g):
    """usi_match の対局を学習用の対局形式へ。深路の側(黒/白)も返す。"""
    if g["reason"] != "checkmate" or g["winner"] not in ("FUKAJI", "KUMOJI"):
        return None
    fk_black = g["black"] == "FUKAJI"
    fk_turn = shogi.BLACK if fk_black else shogi.WHITE
    winner = fk_turn if g["winner"] == "FUKAJI" else 1 - fk_turn
    return {"source": "selfplay", "role": "train", "start_sfen": shogi.STARTING_SFEN, "moves": g["moves"],
            "winner": int(winner), "status": "checkmate", "parent_id": "kumoji-sparring",
            "learning_start_ply": 6}, int(fk_turn)


def kumoji_samples(items, parent):
    rows, labs, bad = [], [], 0
    for g, fk in items:
        try:
            ix, probs, z, resolved = validate_game(g, parent, training=True)
        except Exception:
            bad += 1; continue
        y = targets(probs, z, outcome_weight=.75, td_lambda=.8)
        sel = [i for i in range(6, len(ix)) if (i % 2 == 0) == (fk == shogi.BLACK)]
        if not sel:
            continue
        w = max(len(ix[i]) for i in sel)
        for i in sel:
            r = np.full(w, parent.inputs, np.int32); r[:len(ix[i])] = ix[i]
            rows.append(r); labs.append(y[i])
    ix = pad([r[None, :] for r in rows], parent.inputs)
    return ix, np.asarray(labs, np.float32), bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--baseline-trained", type=Path, required=True)
    p.add_argument("--selfplay", type=Path, nargs="+", required=True)
    p.add_argument("--parts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=.0001)
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--selfplay-tol", type=float, default=.003, help="自己対局の評価用損失の許容悪化幅(ゲート)")
    p.add_argument("--kumoji-share", type=float, default=0.0, help="学習混合に占める雲路戦局面の割合(0=全局面を使う)")
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
        protected = frozenset(json.load(t.extractfile("d17-state/state.json"))["protected_parents"])
    # 雲路戦
    kg = []
    for f in sorted(a.parts.rglob("shard-*.json")):
        kg += json.loads(f.read_text())["games"]
    conv = [c for c in map(convert, kg) if c]
    seen, uniq = set(), []
    for g, fk in conv:
        k = game_key(g)
        if k not in seen:
            seen.add(k); uniq.append((g, fk))
    hold_k = [c for c in uniq if int(game_key(c[0])[:8], 16) % 20 == 0]
    train_k = [c for c in uniq if int(game_key(c[0])[:8], 16) % 20 != 0]
    kx, ky, kbad = kumoji_samples(train_k, parent)
    hkx, hky, hkbad = kumoji_samples(hold_k, parent)
    # 現行世代の自己対局
    sp = []
    for f in a.selfplay:
        with tarfile.open(f) as t:
            names = sorted(n for n in t.getnames() if n.startswith("games/game-") and n.endswith(".json"))
            sp += [json.load(t.extractfile(n)) for n in names]
    hold_s = [g for g in sp if int(game_key(g)[:8], 16) % 20 == 0]
    train_s = [g for g in sp if int(game_key(g)[:8], 16) % 20 != 0]
    per_game = max(8, int(round(len(ky) * 2 / max(1, len(train_s)))))
    sx, sy, _ = make_samples(train_s, parent, seed=a.seed, per_game=per_game, outcome_weight=.75,
                             td_lambda=.8, protected=protected)
    hsx, hsy, _ = make_samples(hold_s, parent, seed=a.seed + 1, per_game=64, outcome_weight=.75,
                               td_lambda=.8, protected=protected)
    print(f"kumoji games={len(kg)} usable={len(uniq)} train={len(train_k)} hold={len(hold_k)} "
          f"rejected={kbad}/{hkbad} positions train={len(ky)} hold={len(hky)}; "
          f"selfplay train={len(sy)} hold={len(hsy)}", flush=True)
    if len(ky) < 5000 or len(hky) < 200:
        raise SystemExit("not enough 雲路 positions")
    if a.kumoji_share > 0:
        keep = int(len(sy) * a.kumoji_share / (1 - a.kumoji_share))
        if keep < len(ky):
            pick = np.sort(np.random.default_rng(a.seed + 2).choice(len(ky), keep, replace=False))
            kx, ky = kx[pick], ky[pick]
    ix = pad([kx, sx], parent.inputs); y = np.concatenate([ky, sy])
    before = {"kumoji": heldout_loss(parent, hkx, hky), "selfplay": heldout_loss(parent, hsx, hsy)}
    cand, opt2, fs = fit(parent, opt, ix, y, epochs=a.epochs, lr=a.lr, seed=a.seed)
    after = {"kumoji": heldout_loss(cand, hkx, hky), "selfplay": heldout_loss(cand, hsx, hsy)}
    gate = after["kumoji"] < before["kumoji"] and after["selfplay"] <= before["selfplay"] + a.selfplay_tol
    print("heldout before", before, "after", after, "gate", "pass" if gate else "fail", flush=True)
    key = save_snapshot(md, cand, opt2, {"parent": BASE, "purpose": "E3 kumoji-sparring hard positions + selfplay mix",
                                         "epochs": a.epochs, "learning_rate": a.lr, "training": fs})
    w, b, v, o = parameters(cand)
    np.savez(a.out / "e3-weights.npz", w=w, b=b, v=v, o=o)
    report = {"schema": "e3-kumoji-sparring-v1", "parent": BASE, "snapshot": key,
              "weights_file_sha256": hashlib.sha256((a.out / "e3-weights.npz").read_bytes()).hexdigest(),
              "kumoji_games": len(kg), "kumoji_usable": len(uniq),
              "kumoji_positions_used": int(len(ky)), "kumoji_share": a.kumoji_share, "selfplay_positions_train": int(len(sy)),
              "heldout_before": before, "heldout_after": after, "gate": "pass" if gate else "fail", "selfplay_tol": a.selfplay_tol,
              "epochs": a.epochs, "lr": a.lr, "optimizer_step_after": opt2["step"],
              "strength_proven": False, "formal_model_changed": False}
    atomic_json(a.out / "report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
