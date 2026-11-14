"""E11: 親E10。大量自己対局データ(G1, 約513万局面)＋玉周辺特徴(入力2548→3848)で学習したNNUEを、ユーザー承認(2026-11-14)のもと
正式エンジン(ハイブリッド＝雲路の探索＋深路NNUE)の重みとして採用する。
- 玉周辺特徴はハイブリッドの評価器だけが計算できる。旧エンジン(深路の自前探索 v0.4.1)は入力2548固定のため E10 のまま据え置く。
  そのため formal/model.json は最上位(旧エンジン用: E10)を変えず、hybrid節に E11 の重み・snapshot・検証を追加する。
- 事前基準: 同一探索・100局で、対E10 +12以上、対G1a(同データ・現行特徴の候補) +12以上、対雲路がE10の16勝を上回る。
実行はCI(e11-adopt.yml)のみ。実runのartifactから学習報告・勝敗・重みSHA・Adam状態を再検証してから書き換える。
"""
from __future__ import annotations
import hashlib, io, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fukaji.online_learning import load_snapshot
from fukaji.formal_model import load as formal_load

OLD = "c7eb865f9800bc75c7067c37d69a750ae9b32a824d1ddeb21f5a3bc9ff60a72f"          # E10 snapshot
OLD_WEIGHTS = "5ddc922e992e3c71815cd3c707f7a8ce1d654090a607fb1a02d477493df868b2"  # models/fukaji-e10.nnue.npz
OLD_STEP = 29173
CPP = "166b0819116364956e94ab47e54eb1891a87f841f8d17d0c59f1adff3f62ecc5"
MODEL_PATH = "models/fukaji-e11.nnue.npz"
TAG, ASSET = "fukaji-current-e11-hybrid-zone-20261114", "E11-trained-state-Adam.tar.gz"
DECISION = "docs/decisions/2026-11-14_E11_大量自己対局データと玉周辺特徴のNNUEを正式採用.md"
digest = lambda raw: hashlib.sha256(raw).hexdigest()


def write_json(path, data):
    p = ROOT / path if not Path(path).is_absolute() else Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def once(path, old, new):
    p = ROOT / path; t = p.read_text()
    if t.count(old) != 1:
        raise ValueError(f"doc conflict {path}: {old[:60]}")
    p.write_text(t.replace(old, new))


def section(path, start, end, new):
    p = ROOT / path; t = p.read_text()
    if t.count(start) != 1:
        raise ValueError("section ambiguous " + path)
    a = t.index(start); b = t.index(end, a)
    p.write_text(t[:a] + new + t[b:])


def append(path, new):
    p = ROOT / path; p.write_text(p.read_text().rstrip() + "\n\n" + new.strip() + "\n")


def result(path, margin=None, more_than=None):
    r = json.loads(Path(path).read_text())
    ok = r["games"] == 100 and r["replay_verified"] is True and r["illegal_pairs"] == [] and r["labels"] == ["HYBRID", "OPP"]
    if margin is not None:
        ok = ok and r["fukaji_wins"] - r["kumoji_wins"] >= margin
    if more_than is not None:
        ok = ok and r["fukaji_wins"] > more_than
    if not ok:
        raise ValueError(f"result check failed {path}: {r}")
    return r


def main(trained_dir, res_e10, res_g1a, res_kumoji, out, commit):
    trained = Path(trained_dir)
    rep = json.loads((trained / "report.json").read_text())
    if (rep["schema"] != "g1-train-v1" or rep["parent"] != OLD or rep["gate"] != "pass" or rep["feature_set"] != "zone" or
            rep["inputs"] != 3848 or rep["formal_model_changed"] is not False or rep["optimizer_step_after"] <= OLD_STEP):
        raise ValueError("E11 training report mismatch")
    re10, rg1a, rk = result(res_e10, margin=12), result(res_g1a, margin=12), result(res_kumoji, more_than=16)
    formal = json.loads((ROOT / "formal/model.json").read_text())
    for k, v in (("version", "FUKAJI v0.4.1"), ("search_implementation_sha256", CPP), ("weights_sha256", OLD_WEIGHTS),
                 ("research_snapshot_sha256", OLD), ("optimizer_step", OLD_STEP)):
        if formal.get(k) != v:
            raise ValueError("formal current E10 drift " + k)
    if "weights_file" in formal["hybrid"]:
        raise ValueError("hybrid weights already set")
    contract_path = "research/d17-fast-self-learning/formal-contract.json"
    contract = json.loads((ROOT / contract_path).read_text())
    for path, expected in contract.items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ValueError("frozen formal contract changed " + path)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    new = rep["snapshot"]
    learned, opt, _ = load_snapshot(trained / "models", new)
    if opt is None or opt["step"] != rep["optimizer_step_after"]:
        raise ValueError("E11 complete Adam state could not be restored")
    if learned.inputs != 3848 or learned.hidden_weights.shape != (3848, 128):
        raise ValueError("architecture is not base2548+zone1300 x 128")
    raw_src = (trained / "cand-weights.npz").read_bytes()
    if digest(raw_src) != rep["weights_file_sha256"]:
        raise ValueError("weights file digest mismatch with report")
    with np.load(io.BytesIO(raw_src), allow_pickle=False) as z:
        if set(z.files) != {"w", "b", "v", "o"}:
            raise ValueError("NNUE inference shape contract")
        np.testing.assert_array_equal(z["w"], learned.hidden_weights)
        np.testing.assert_array_equal(z["v"], learned.output_weights)
        np.testing.assert_array_equal(z["b"], learned.hidden_bias)
        for n in "wbv":
            if not np.isfinite(z[n]).all():
                raise ValueError("non-finite weights")
    pick = lambda r: {k: r[k] for k in ("games", "fukaji_wins", "kumoji_wins", "draws_or_unresolved")}
    receipt = {"schema": "e11-verified-adoption-v1", "former_snapshot_sha256": OLD, "current_snapshot_sha256": new,
               "optimizer_step": rep["optimizer_step_after"], "weights_path": MODEL_PATH, "architecture": [3848, 128, 1],
               "legacy_engine_weights_unchanged": OLD_WEIGHTS,
               "hybrid_vs_e10": pick(re10), "hybrid_vs_g1a_same_data_base_features": pick(rg1a), "hybrid_vs_kumoji": pick(rk),
               "train_positions": rep["train_positions"], "games": rep["games"], "outcome_weight": rep["outcome_weight"],
               "lr": rep["lr"], "epochs": rep["epochs"], "batch": rep["batch"],
               "heldout_before": rep["heldout_before"], "heldout_after": rep["heldout_after"]}
    write_json(out / "receipt.json", receipt)
    if not commit:
        print("DRY RUN OK", json.dumps(receipt, ensure_ascii=False)); return
    dest = ROOT / MODEL_PATH
    if dest.exists():
        raise ValueError("E11 inference already exists")
    buf = io.BytesIO()
    np.savez_compressed(buf, w=learned.hidden_weights, b=learned.hidden_bias, v=learned.output_weights,
                        o=np.array([learned.output_bias], dtype=np.float32))
    raw = buf.getvalue(); dest.write_bytes(raw); wsha = digest(raw); receipt["weights_sha256"] = wsha
    w = (f"E11 ハイブリッド(雲路の探索＋深路NNUE, 200ms・同一探索・各100局)で対E10 {re10['fukaji_wins']}-{re10['kumoji_wins']}、"
         f"対雲路 {rk['fukaji_wins']}-{rk['kumoji_wins']}(E10は16-84)、同じデータで現行特徴のまま学習した候補に {rg1a['fukaji_wins']}-{rg1a['kumoji_wins']}")
    nf = json.loads(json.dumps(formal))
    nf["hybrid"].update({
        "weights_file": MODEL_PATH, "weights_sha256": wsha, "architecture": [3848, 128, 1],
        "feature_set": "base 2548 + king zone 5x5 (1300): fukaji.nnue_research.king_zone_indices / research/hybrid/nnue_eval.h",
        "research_snapshot_sha256": new, "research_parent_sha256": OLD, "optimizer_step": rep["optimizer_step_after"],
        "nnue_approved_on": "2026-11-14", "nnue_approved_by": f"owner-approved E11 (2026-11-14) {w}",
        "state_release_tag": TAG, "state_release_asset": ASSET,
        "training_report_sha256": digest((trained / "report.json").read_bytes()),
        "nnue_validation": {"hybrid_vs_e10": pick(re10), "hybrid_vs_g1a_same_data_base_features": pick(rg1a), "hybrid_vs_kumoji": pick(rk),
                            "training": f"G1: hybrid(E10) self-play at 50ms/move, {rep['games']} games, {rep['train_positions']} positions; "
                                        f"label = search value mixed {rep['outcome_weight']} with game outcome; 1 epoch"},
        "legacy_note": "The legacy engine (run-fukaji-legacy.sh, own search v0.4.1, inputs 2548) cannot evaluate the king-zone rows and stays on the top-level E10 weights.",
        "validation": f"hybrid(E11) vs hybrid(E10) {re10['fukaji_wins']}-{re10['kumoji_wins']}, vs KUMOJI {rk['fukaji_wins']}-{rk['kumoji_wins']} (100 games each, 200ms, same options); "
                      + formal["hybrid"]["validation"]})
    for k in formal:
        if k != "hybrid" and nf[k] != formal[k]:
            raise ValueError("legacy formal field changed " + k)
    write_json("formal/model.json", nf)
    loaded, spec = formal_load()      # 旧エンジンは従来どおりE10を読める
    if spec["weights_sha256"] != OLD_WEIGHTS or loaded.inputs != 2548:
        raise ValueError("legacy formal load changed")
    step = rep["optimizer_step_after"]
    dec = DECISION.split("/")[-1]
    (ROOT / DECISION).write_text(
        f"# E11：大量自己対局データ＋玉周辺特徴のNNUEを正式採用（2026-11-14）\n\n"
        f"ユーザー承認(2026-11-14「正式採用してください」)。親E10(Adam{OLD_STEP})から、ハイブリッド(E10)の短時間自己対局(1手50ms、{rep['games']}局)で作った"
        f"約{rep['train_positions']}局面で学習。入力に玉周辺±2升の特徴1300行を追加(2548→3848、追加行はゼロ初期化で開始時の評価は親と一致)。"
        f"教師は探索値と対局結果({rep['outcome_weight']}混合)。雲路の評価値・指し手は不使用。\n\n"
        f"- {w}。\n"
        f"- 検証ラベル損失 {rep['heldout_before']:.4f}→{rep['heldout_after']:.4f}。\n"
        f"- **採用範囲は正式エンジン(ハイブリッド)**。玉周辺特徴はハイブリッドの評価器(research/hybrid/nnue_eval.h)だけが計算できるため、"
        f"旧エンジン(run-fukaji-legacy.sh、自前探索v0.4.1、入力2548固定)はE10のまま。formal/model.jsonは最上位(旧エンジン用)を変えず、hybrid節にE11を記録した。\n"
        f"- 正式snapshot {new}、推論 {MODEL_PATH} SHA256 {wsha}、Adam{step}。完全stateは Release `{TAG}` の `{ASSET}`。\n"
        f"- 次の学習(g1-train.yml)はこのE11を親にする。戻す場合は formal/model.json の hybrid節から weights_file 以下のE11項目を除く(E10に戻る)。\n"
        f"- 経緯と比較した候補の一覧: [G1の結果](2026-11-14_G1大量データ学習_対E10は89-11_対雲路は改善なし.md)。\n")
    spec_doc = "docs/現行エンジン仕様.md"
    once(spec_doc, "**現行のNNUE重みはE10（c7eb865f、Adam29173、隠れ層128。E9からハイブリッド探索ラベルで再学習）**",
         f"**正式エンジン(ハイブリッド)のNNUE重みはE11（{new[:8]}、Adam{step}、入力3848＝現行2548＋玉周辺1300、隠れ層128。E10から大量自己対局データで学習）。"
         f"旧エンジン(legacy)の重みはE10（c7eb865f、Adam29173）のまま**")
    once(spec_doc, "次のNNUE学習は現行E10（Adam29173）のGitHub保存済み完全stateから開始し、更新前snapshot E10と更新後",
         f"次のNNUE学習は現行E11（Adam{step}、Release {TAG}）のGitHub保存済み完全stateから開始し、更新前snapshot E11と更新後")
    append(spec_doc, f"## E11：大量自己対局データ＋玉周辺特徴のNNUEを正式エンジン(ハイブリッド)に採用（2026-11-14）\n\n"
           f"- snapshot {new}、推論 `{MODEL_PATH}` SHA256 {wsha}、Adam{step}、入力3848・隠れ層128。{w}。[判断](decisions/{dec})。\n"
           f"- 上の表(研究snapshot・推論用ファイル・SHA)は旧エンジン(legacy、E10)の値。正式エンジンの重みは formal/model.json の hybrid節を参照。")
    common = (f"正式E11（{new}、Adam{step}、入力3848＝現行2548＋玉周辺1300。E10から大量自己対局データ約{rep['train_positions']}局面で学習）。{w}。"
              f"[E11採用判断](decisions/{dec})。採用範囲は正式エンジン(ハイブリッド)。旧エンジン(legacy、USI v0.4.1・PVS＋詰み探索)はE10のまま不変。")
    section("docs/00_開発ルーター.md", "## 現行モデル", "## 現在の研究・保存再開点", "## 現行モデル\n\n" + common + "\n\n")
    append("docs/00_開発ルーター.md", "## 最新NNUE（E11、2026-11-14）\n\n" + common +
           "\n\n強化の進め方は「生成(g1-gen.yml)→学習(g1-train.yml)→対局(hybrid-match.yml)」の周回。")
    section("docs/ROADMAP.md", "## 現行モデル", "## 現在の研究", "## 現行モデル\n\n" + common + "\n\n")
    append("docs/ROADMAP.md", "## E11：大量自己対局データ＋玉周辺特徴のNNUE正式採用（2026-11-14）\n\n" + common)
    append("AGENTS.md", f"## 最新正式NNUE（E11、2026-11-14）\n\n正式E11／Adam{step}、snapshot {new}。{w}。"
           f"正式エンジン(ハイブリッド)の重みは formal/model.json の hybrid節(`{MODEL_PATH}`)。旧エンジン(legacy)はE10のまま。"
           f"次の学習は g1-gen.yml → g1-train.yml(親E11) → hybrid-match.yml。")
    mt = "docs/モデル名対応表.md"
    once(mt, "現行の正式モデルは **E10**", "現行の正式モデルは **E11**（正式エンジン＝ハイブリッド。旧エンジンはE10）")
    once(mt, "| **E10（現行）** |", "| E10（旧エンジンの現行） |")
    row = (f"| **E11（現行）** | E10のハイブリッド自己対局(50ms)約{rep['train_positions']}局面で学習。入力に玉周辺特徴を追加(3848) | E10 | {new[:8]} | {step} | {wsha[:8]} | "
           f"ハイブリッド下 対E10 {re10['fukaji_wins']}-{re10['kumoji_wins']}、対雲路 {rk['fukaji_wins']}-{rk['kumoji_wins']}。[記録](decisions/{dec}) |\n")
    p = ROOT / mt; t = p.read_text(); i = t.index("\n通算確認")
    p.write_text(t[:i].rstrip("\n") + "\n" + row + t[i:])
    for p in ("formal/model.json", spec_doc):
        contract[p] = digest((ROOT / p).read_bytes())
    write_json(contract_path, contract)
    assert digest((ROOT / "native/nnue_search_ffi.cpp").read_bytes()) == CPP
    assert digest((ROOT / "models/fukaji-e10.nnue.npz").read_bytes()) == OLD_WEIGHTS
    receipt["contract_sha256"] = digest((ROOT / contract_path).read_bytes())
    write_json(out / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--trained", type=Path, required=True)
    a.add_argument("--result-e10", type=Path, required=True)
    a.add_argument("--result-g1a", type=Path, required=True)
    a.add_argument("--result-kumoji", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--commit", action="store_true")
    x = a.parse_args()
    main(x.trained, x.result_e10, x.result_g1a, x.result_kumoji, x.out, x.commit)
