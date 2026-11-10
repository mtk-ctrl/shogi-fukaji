"""E6: 隠れ層を32→128に広げたNNUE(E5を機能保存で拡張→E5の探索評価値で蒸留)を、ユーザー承認(2026-11-10 21:05)のもと正式採用する。
実行はCI(e6-adopt.yml)のみ。学習をCI内で再現し、監査(h-wide run 42770108054)と重みSHAが一致することを確認してから書き換える。
構造変更：native/nnue_search_ffi.cpp の kNHidden、nnue_research.HIDDEN、formal_model の構造検査、契約の固定値を更新する。探索の中身は不変。
"""
from __future__ import annotations
import hashlib, io, json, re, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.h_widen_lib import use_hidden

OLD = "638795c217155c86392affbd8857d07bfc6d58f1e36f1fd4fd5ccf69a5556d23"
OLD_WEIGHTS = "ac8680f0d7f756ff9f7be3f8ccfce0f39979aa204f3e0afeac4a6dcf83c935d9"
OLD_STEP = 18074
CPP_OLD = "c9340af265368547c37cbecc311d8d3982900885b433d1b4ccdc27fbdd7d041a"
EXPECT_CAND = "1247914ad2540172e4b1395b1f707280da196d7ec454b35fb5bc383b90b1e441"
H = 128
MODEL_PATH = "models/fukaji-e6.nnue.npz"
DECISION = "docs/decisions/2026-11-10_E6_隠れ層128NNUEを正式採用.md"
digest = lambda raw: hashlib.sha256(raw).hexdigest()


def write_json(path, data):
    p = ROOT / path; p.parent.mkdir(parents=True, exist_ok=True)
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


def main(trained, result200, out, commit):
    trained = Path(trained)
    rep = json.loads((trained / "report.json").read_text())
    if (rep["schema"] != "h-wide-distill-v1" or rep["parent"] != OLD or rep["hidden"] != H or rep["gate"] != "pass" or
            rep["formal_model_changed"] is not False or rep["optimizer_step_after"] <= OLD_STEP):
        raise ValueError("E6 training report mismatch")
    if rep["weights_file_sha256"] != EXPECT_CAND:
        raise ValueError("reproduced weights differ from the audited candidate")
    r = json.loads(Path(result200).read_text())
    if not (r["games"] == 200 and r["replay_verified"] is True and r["illegal_pairs"] == [] and
            r["labels"] == ["PREV", "NEW"] and r["new_wins"] - r["prev_wins"] >= 12):
        raise ValueError(f"audit result check failed: {r}")
    formal = json.loads((ROOT / "formal/model.json").read_text())
    for k, v in (("version", "FUKAJI v0.4.1"), ("search_backend_variant", "d48_pvs_mate"),
                 ("search_implementation_sha256", CPP_OLD), ("weights_sha256", OLD_WEIGHTS),
                 ("research_snapshot_sha256", OLD), ("optimizer_step", OLD_STEP), ("architecture", [2548, 32, 1])):
        if formal.get(k) != v:
            raise ValueError("formal current E5 drift " + k)
    cpp_path = ROOT / "native/nnue_search_ffi.cpp"
    if digest(cpp_path.read_bytes()) != CPP_OLD:
        raise ValueError("official PVS/mate source differs from expectation")
    contract_path = "research/d17-fast-self-learning/formal-contract.json"
    contract = json.loads((ROOT / contract_path).read_text())
    for path, expected in contract.items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ValueError("frozen formal contract changed " + path)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    use_hidden(H)
    from fukaji.online_learning import load_snapshot
    new = rep["snapshot"]
    learned, opt, _ = load_snapshot(trained / "models", new)
    if opt is None or opt["step"] != rep["optimizer_step_after"]:
        raise ValueError("E6 complete Adam state could not be restored")
    if learned.inputs != 2548 or learned.hidden_weights.shape != (2548, H):
        raise ValueError("architecture mismatch")
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
    receipt = {"schema": "e6-verified-adoption-v1", "former_snapshot_sha256": OLD, "current_snapshot_sha256": new,
               "optimizer_step": rep["optimizer_step_after"], "hidden": H, "weights_path": MODEL_PATH,
               "former_cpp_sha256": CPP_OLD, "usi_version": "FUKAJI v0.4.1",
               "audit200": {k: r[k] for k in ("games", "prev_wins", "new_wins", "draws_or_unresolved", "pair_sweeps_fukaji", "pair_sweeps_kumoji")},
               "train_positions": rep["train_positions"], "outcome_weight": rep["outcome_weight"], "lr": rep["lr"], "epochs": rep["epochs"],
               "heldout_before": rep["heldout_before"], "heldout_after": rep["heldout_after"],
               "reproduced_weights_sha256": rep["weights_file_sha256"]}
    write_json(str(out / "receipt.json"), receipt)
    if not commit:
        print("DRY RUN OK", json.dumps(receipt, ensure_ascii=False)); return
    dest = ROOT / MODEL_PATH
    if dest.exists():
        raise ValueError("E6 inference already exists")
    buf = io.BytesIO()
    np.savez_compressed(buf, w=learned.hidden_weights, b=learned.hidden_bias, v=learned.output_weights,
                        o=np.array([learned.output_bias], dtype=np.float32))
    raw = buf.getvalue(); dest.write_bytes(raw); wsha = digest(raw)
    # ---- 構造変更：ソースの書き換え（隠れ層の幅のみ）
    cpp = cpp_path.read_text()
    if cpp.count("kNHidden=32") != 1:
        raise ValueError("kNHidden not unique")
    cpp_path.write_text(cpp.replace("kNHidden=32", f"kNHidden={H}"))
    new_cpp = digest(cpp_path.read_bytes())
    once("fukaji/nnue_research.py", "HIDDEN = 32", f"HIDDEN = {H}")
    once("fukaji/nnue_research.py", "32-unit clipped-ReLU test architecture", f"{H}-unit clipped-ReLU architecture (hidden width = HIDDEN)")
    once("fukaji/formal_model.py", "from .nnue_research import SmallNNUE,INPUTS", "from .nnue_research import SmallNNUE,INPUTS,HIDDEN")
    once("fukaji/formal_model.py", "spec['architecture']!=[INPUTS,32,1]", "spec['architecture']!=[INPUTS,HIDDEN,1]")
    once("tests/test_formal_usi.py", "architecture=[2548,32,1]", f"architecture=[2548,{H},1]")
    once("tests/test_nnue_research.py", "self.assertEqual(HIDDEN, 32)", f"self.assertEqual(HIDDEN, {H})")
    # 幅が広がると有限差分の数値誤差(float32)が増える。解析勾配との許容差を3e-5→1e-4に緩める(理論式は不変)
    once("tests/test_online_learning.py", "self.assertAlmostEqual(float(analytic), float(numeric), delta=3e-5)",
         "self.assertAlmostEqual(float(analytic), float(numeric), delta=1e-4)")
    w200 = f"E6(隠れ層{H}) {r['new_wins']}-{r['prev_wins']}(対E5 200局)"
    nf = dict(formal)
    nf.update({
        "approved_on": "2026-11-10",
        "approved_by": f"owner-approved E6 hidden{H} (2026-11-10 21:05) {w200}",
        "architecture": [2548, H, 1], "search_implementation_sha256": new_cpp,
        "research_snapshot_sha256": new, "research_parent_sha256": OLD,
        "weights_file": MODEL_PATH, "weights_sha256": wsha, "optimizer_step": rep["optimizer_step_after"],
        "training_release_sha256": digest((trained / "report.json").read_bytes()),
        "training_source_sha256": rep["weights_file_sha256"],
        "comparison_release_sha256": digest(Path(result200).read_bytes()),
        "model_distribution": f"GitHub {MODEL_PATH}; complete Adam{rep['optimizer_step_after']} in the E6 trained Release.",
        "validation": {
            "opponent": "E5/Adam18074 (previous formal, hidden 32), paired unseen openings, same search, 200ms flat",
            "audit200": receipt["audit200"],
            "direct_roundrobin_longthink": "128 vs 256: 103-94; 128 vs 512: 107-91; 256 vs 512: 105-94 (200ms + 1s x10/side, 200 games each)",
            "strength_conclusion": f"{w200}; 事前基準(+12)を満たす。幅256/512は長考ありの直接対決で128を上回らず、128を採用",
            "search_backend_variant": "d48_pvs_mate", "quiescence": 0,
            "training": "function-preserving widening of E5 (new units output 0), distillation of the formal fixed-depth-4 search value mixed 0.2 with game outcome; positions (~15万) from E5-vs-KUMOJI games and selfplay",
        }})
    for k in ("version", "backend", "quiescence", "default_move_time_ms", "default_search_depth",
              "search_backend_variant", "search_adoption"):
        if nf[k] != formal[k]:
            raise ValueError("unauthorized search change " + k)
    write_json("formal/model.json", nf)
    (ROOT / DECISION).write_text(
        f"# E6：隠れ層{H}のNNUEを正式採用（2026-11-10）\n\n"
        f"ユーザー承認（2026-11-10 21:05）。E5(隠れ層32, Adam{OLD_STEP})を機能保存のまま{H}に広げ、E5の正式探索(固定深さ4)の評価値で蒸留"
        f"（E5と雲路の練習局面・自己対局の約{rep['train_positions']}局面、結果ラベル混合{rep['outcome_weight']}、雲路の指し手・評価値は不使用）。\n\n"
        f"- 対E5 200局：{r['new_wins']}-{r['prev_wins']}（差+{r['new_wins']-r['prev_wins']}、長考なし200ms）。幅の総当たり（長考あり）は128が256・512と同等以上で、128を採用。\n"
        f"- 正式snapshot {new}、推論SHA256 {wsha}、Adam{rep['optimizer_step_after']}。構造 2548→{H}→1。探索は不変（USI v0.4.1、PVS＋詰み探索）。\n"
        f"- 構造変更のためC++の kNHidden（SHA {CPP_OLD[:8]}→{new_cpp[:8]}）と契約の固定値を更新。旧E5（隠れ層32）の重みは models/fukaji-e5.nnue.npz に保全。旧モデルを動かす場合は、この採用前のcommitに戻す。\n")
    spec_doc = "docs/現行エンジン仕様.md"
    once(spec_doc, "**現行のNNUE重みはE5（638795c2、Adam18074。E4から蒸留で更新）**",
         f"**現行のNNUE重みはE6（{new[:8]}、Adam{rep['optimizer_step_after']}、隠れ層{H}。E5から幅を広げて蒸留）**")
    once(spec_doc, f"| 研究snapshot SHA256 | `{OLD}` |", f"| 研究snapshot SHA256 | `{new}` |")
    once(spec_doc, "| 推論用ファイル | `models/fukaji-e5.nnue.npz` |", f"| 推論用ファイル | `{MODEL_PATH}` |")
    once(spec_doc, f"| 推論用SHA256 | `{OLD_WEIGHTS}` |", f"| 推論用SHA256 | `{wsha}` |")
    once(spec_doc, "| 構造 | 絶対特徴2548 → 32 → 1 |", f"| 構造 | 絶対特徴2548 → {H} → 1 |")
    once(spec_doc, "次のNNUE学習は現行E5（Adam18074）", f"次のNNUE学習は現行E6（Adam{rep['optimizer_step_after']}）")
    once(spec_doc, "更新前snapshot E5と更新後", "更新前snapshot E6と更新後")
    append(spec_doc, f"## E6：隠れ層{H}のNNUEを正式採用（2026-11-10）\n\n- snapshot {new}、推論SHA256 {wsha}、Adam{rep['optimizer_step_after']}、構造2548→{H}→1。{w200}。"
           f"[判断](decisions/{DECISION.split('/')[-1]})。旧E5（隠れ層32）は models/fukaji-e5.nnue.npz。")
    common = (f"正式E6（{new}、Adam{rep['optimizer_step_after']}、隠れ層{H}。E5から幅を広げて蒸留）。{w200}。"
              f"[E6採用判断](decisions/{DECISION.split('/')[-1]})。USI v0.4.1・PVS＋詰み探索不変。")
    section("docs/00_開発ルーター.md", "## 現行モデル", "## 現在の研究・保存再開点", "## 現行モデル\n\n" + common + "\n\n")
    append("docs/00_開発ルーター.md", "## 最新NNUE（E6、2026-11-10）\n\n" + common)
    section("docs/ROADMAP.md", "## 現行モデル", "## 現在の研究", "## 現行モデル\n\n" + common + "\n\n")
    append("docs/ROADMAP.md", "## E6：隠れ層128NNUE正式採用（2026-11-10）\n\n" + common)
    append("AGENTS.md", f"## 最新正式NNUE（E6、2026-11-10）\n\n正式E6／隠れ層{H}／Adam{rep['optimizer_step_after']}、snapshot {new}。{w200}。構造変更あり（kNHidden・契約更新）。旧E5以前はmodels/に保全（隠れ層32）。")
    mt = "docs/モデル名対応表.md"
    once(mt, "現行の正式モデルは **E5**", "現行の正式モデルは **E6**")
    once(mt, "| **E5（現行）** |", "| E5 |")
    row = (f"| **E6（現行）** | E5を機能保存のまま隠れ層{H}に広げ、E5の探索評価値で蒸留（約15.3万局面）。構造変更 | E5 | {new[:8]} | {rep['optimizer_step_after']} | {wsha[:8]} | "
           f"対E5 200局 {r['new_wins']}-{r['prev_wins']}。幅の総当たり（長考あり）で128≧256≧512。[記録](decisions/{DECISION.split('/')[-1]}) |\n")
    p = ROOT / mt; t = p.read_text(); i = t.index("\n通算確認")
    p.write_text(t[:i].rstrip("\n") + "\n" + row + t[i:])
    for path in ("native/nnue_search_ffi.cpp", "fukaji/formal_model.py", "formal/model.json", spec_doc):
        contract[path] = digest((ROOT / path).read_bytes())
    write_json(contract_path, contract)
    chk = subprocess.run([sys.executable, "scripts/check_fukaji_contracts.py"], cwd=ROOT, capture_output=True, text=True)
    if chk.returncode:
        raise ValueError("post-edit contract check failed: " + chk.stdout[-800:] + chk.stderr[-800:])
    receipt["former_cpp_sha256"] = CPP_OLD; receipt["current_cpp_sha256"] = new_cpp
    receipt["weights_sha256"] = wsha; receipt["contract_sha256"] = digest((ROOT / contract_path).read_bytes())
    write_json(str(out / "receipt.json"), receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--trained", type=Path, required=True)
    a.add_argument("--result200", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--commit", action="store_true")
    x = a.parse_args()
    main(x.trained, x.result200, x.out, x.commit)
