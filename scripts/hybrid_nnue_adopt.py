"""正式エンジン(ハイブリッド)のNNUEを次の版へ更新する共通の採用手続き(E12以降、周回ごとに使う)。
- 親は formal/model.json の hybrid節にある現行NNUE。学習報告(g1-train-v1)の parent がそれと一致すること。
- 事前基準: 同一探索・100局(F30以降は108局)で、対現行正式 +12以上、対雲路が --kumoji-min-wins 以上。
- 旧エンジン(最上位の項目、E10)は変更しない。hybrid節・文書・契約ハッシュだけを書き換える。
実行はCI(hybrid-nnue-adopt.yml)のみ。実runのartifactから再検証してから書き換える。
"""
from __future__ import annotations
import argparse, hashlib, io, json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fukaji.online_learning import load_snapshot
from fukaji.formal_model import load as formal_load
digest = lambda raw: hashlib.sha256(raw).hexdigest()
CONTRACT = "research/d17-fast-self-learning/formal-contract.json"


def write_json(path, data):
    p = Path(path) if Path(path).is_absolute() else ROOT / path; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def resub(path, pattern, new):
    p = ROOT / path; t = p.read_text()
    if len(re.findall(pattern, t)) != 1:
        raise ValueError(f"doc conflict {path}: {pattern}")
    p.write_text(re.sub(pattern, lambda _: new, t))


def section(path, start, end, new):
    p = ROOT / path; t = p.read_text()
    if t.count(start) != 1:
        raise ValueError("section ambiguous " + path)
    a = t.index(start); b = t.index(end, a)
    p.write_text(t[:a] + new + t[b:])


def append(path, new):
    p = ROOT / path; p.write_text(p.read_text().rstrip() + "\n\n" + new.strip() + "\n")


def result(path, margin=None, min_wins=None):
    r = json.loads(Path(path).read_text())
    ok = r["games"] in (100, 108) and r["replay_verified"] is True and r["illegal_pairs"] == [] and r["labels"] == ["HYBRID", "OPP"]
    if margin is not None:
        ok = ok and r["fukaji_wins"] - r["kumoji_wins"] >= margin
    if min_wins is not None:
        ok = ok and r["fukaji_wins"] >= min_wins
    if not ok:
        raise ValueError(f"result check failed {path}: {r}")
    return r


def main(a):
    name, date = a.name, a.date
    if not re.fullmatch(r"E\d+", name) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError("name/date format")
    formal = json.loads((ROOT / "formal/model.json").read_text()); h = formal["hybrid"]
    prev = re.fullmatch(r"models/fukaji-(e\d+)\.nnue\.npz", h["weights_file"]).group(1).upper()
    if int(name[1:]) != int(prev[1:]) + 1:
        raise ValueError(f"{name} must follow {prev}")
    old, old_step = h["research_snapshot_sha256"], h["optimizer_step"]
    if digest((ROOT / h["weights_file"]).read_bytes()) != h["weights_sha256"]:
        raise ValueError("current hybrid weights drift")
    contract = json.loads((ROOT / CONTRACT).read_text())
    for path, expected in contract.items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ValueError("frozen formal contract changed " + path)
    trained = Path(a.trained)
    rep = json.loads((trained / "report.json").read_text())
    arch = h["architecture"]
    rp = result(a.result_prev, margin=12)
    # 2026-11-18: 損失ゲートは強さの粗い目安。自動採用では、損失の悪化が小さく(0.001未満)、対現行が+30以上の大差なら、ゲート失敗でも採用できる(判断記録 2026-11-18_周回14)。
    loss_small = rep.get("heldout_after", 9) - rep.get("heldout_before", 0) < 0.001
    gate_ok = rep["gate"] == "pass" or (a.auto and loss_small and rp["fukaji_wins"] - rp["kumoji_wins"] >= 30)
    if (rep["schema"] != "g1-train-v1" or rep["parent"] != old or not gate_ok or rep["inputs"] != arch[0] or
            rep["formal_model_changed"] is not False or rep["optimizer_step_after"] <= old_step):
        raise ValueError(f"{name} training report mismatch")
    rk = result(a.result_kumoji, min_wins=a.kumoji_min_wins) if a.result_kumoji else None  # 2026-11-17: 対雲路は省略可
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    new, step = rep["snapshot"], rep["optimizer_step_after"]
    learned, opt, _ = load_snapshot(trained / "models", new)
    if opt is None or opt["step"] != step:
        raise ValueError("complete Adam state could not be restored")
    if learned.hidden_weights.shape != (arch[0], arch[1]):
        raise ValueError("architecture changed; use a dedicated adoption for structure changes")
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
    low = name.lower()
    model_path = f"models/fukaji-{low}.nnue.npz"
    tag, asset = f"fukaji-current-{low}-hybrid-zone-{date.replace('-', '')}", f"{name}-trained-state-Adam.tar.gz"
    pick = lambda r: None if r is None else {k: r[k] for k in ("games", "fukaji_wins", "kumoji_wins", "draws_or_unresolved")}
    receipt = {"schema": "hybrid-nnue-adoption-v1", "name": name, "previous": prev, "former_snapshot_sha256": old,
               "current_snapshot_sha256": new, "optimizer_step": step, "weights_path": model_path, "architecture": arch,
               "release_tag": tag, "release_asset": asset, "hybrid_vs_previous": pick(rp), "hybrid_vs_kumoji": pick(rk),
               "train_positions": rep["train_positions"], "games": rep["games"], "outcome_weight": rep["outcome_weight"],
               "heldout_before": rep["heldout_before"], "heldout_after": rep["heldout_after"]}
    write_json(out / "receipt.json", receipt)
    if not a.commit:
        print("DRY RUN OK", json.dumps(receipt, ensure_ascii=False)); return
    dest = ROOT / model_path
    if dest.exists():
        raise ValueError("inference file already exists")
    buf = io.BytesIO()
    np.savez_compressed(buf, w=learned.hidden_weights, b=learned.hidden_bias, v=learned.output_weights,
                        o=np.array([learned.output_bias], dtype=np.float32))
    raw = buf.getvalue(); dest.write_bytes(raw); wsha = digest(raw); receipt["weights_sha256"] = wsha
    draws = lambda r: f"-{r['draws_or_unresolved']}" if r["draws_or_unresolved"] else ""
    kv = "対雲路は省略" if rk is None else f"対雲路 {rk['fukaji_wins']}-{rk['kumoji_wins']}{draws(rk)}"
    kvs = "vs KUMOJI skipped" if rk is None else f"vs KUMOJI {rk['fukaji_wins']}-{rk['kumoji_wins']}"
    w = (f"{name} ハイブリッド(200ms・同一探索・各{rp['games']}局)で対{prev} {rp['fukaji_wins']}-{rp['kumoji_wins']}{draws(rp)}、{kv}")
    approval = (f"auto-adopted {name} ({date}; owner rule 2026-11-17: +12 or more vs current and 0 illegal)" if a.auto
                else f"owner-approved {name} ({date})")
    approval_ja = (f"自動採用({date}。2026-11-17本人決定の基準：対現行+12以上かつ違法手0)" if a.auto else f"ユーザー承認({date})")
    nf = json.loads(json.dumps(formal))
    nf["hybrid"].update({
        "weights_file": model_path, "weights_sha256": wsha, "research_snapshot_sha256": new, "research_parent_sha256": old,
        "optimizer_step": step, "nnue_approved_on": date, "nnue_approved_by": f"{approval} {w}",
        "state_release_tag": tag, "state_release_asset": asset,
        "training_report_sha256": digest((trained / "report.json").read_bytes()),
        "nnue_validation": {"hybrid_vs_previous": dict(pick(rp), previous=prev), "hybrid_vs_kumoji": pick(rk),
                            "training": f"hybrid({prev}) self-play at 50ms/move, {rep['games']} games, {rep['train_positions']} positions; "
                                        f"label = search value mixed {rep['outcome_weight']} with game outcome; {rep['epochs']} epoch"},
        "validation": f"hybrid({name}) vs hybrid({prev}) {rp['fukaji_wins']}-{rp['kumoji_wins']}, {kvs} "
                      f"({rp['games']} games each, 200ms, same options); " + h["validation"]})
    for k in formal:
        if k != "hybrid" and nf[k] != formal[k]:
            raise ValueError("legacy formal field changed " + k)
    write_json("formal/model.json", nf)
    loaded, spec = formal_load()
    if spec["weights_sha256"] != formal["weights_sha256"] or loaded.inputs != 2548:
        raise ValueError("legacy formal load changed")
    decision = f"docs/decisions/{date}_{name}_自己対局の周回でNNUEを更新.md"
    dec = decision.split("/")[-1]
    (ROOT / decision).write_text(
        f"# {name}：自己対局の周回でNNUEを更新（正式採用、{date}）\n\n"
        f"{approval_ja}。親{prev}(Adam{old_step})から、ハイブリッド({prev})の短時間自己対局(1手50ms、{rep['games']}局)で作り直した"
        f"約{rep['train_positions']}局面で学習(玉周辺特徴つき入力{arch[0]}・隠れ層{arch[1]}、勝敗{rep['outcome_weight']}混合、{rep['epochs']}エポック)。"
        f"教師は探索値と対局結果だけで、雲路の評価値・指し手は不使用。\n\n"
        f"- {w}。\n"
        f"- 検証ラベル損失 {rep['heldout_before']:.4f}→{rep['heldout_after']:.4f}。\n"
        f"- 採用範囲は正式エンジン(ハイブリッド)。\n"
        f"- 正式snapshot {new}、推論 {model_path} SHA256 {wsha}、Adam{step}。完全stateは Release `{tag}` の `{asset}`。\n"
        f"- 次の周回(g1-gen.yml → g1-train.yml → hybrid-match.yml)はこの{name}を親にする(各workflowの既定値 formal が自動で指す)。\n")
    spec_doc = "docs/現行エンジン仕様.md"
    resub(spec_doc, r"正式エンジン\(ハイブリッド\)のNNUE重みはE\d+（[^）]*）",
          f"正式エンジン(ハイブリッド)のNNUE重みは{name}（{new[:8]}、Adam{step}、入力{arch[0]}＝現行2548＋玉周辺1300、隠れ層{arch[1]}。{prev}から自己対局の周回で学習）")
    resub(spec_doc, r"次のNNUE学習は現行E\d+（[^）]*）", f"次のNNUE学習は現行{name}（Adam{step}、Release {tag}）")
    append(spec_doc, f"## {name}：自己対局の周回でNNUEを更新（{date}）\n\n"
           f"- snapshot {new}、推論 `{model_path}` SHA256 {wsha}、Adam{step}。{w}。[判断](decisions/{dec})。")
    common = (f"正式{name}（{new}、Adam{step}、入力{arch[0]}＝現行2548＋玉周辺1300。{prev}の自己対局約{rep['train_positions']}局面で学習）。{w}。"
              f"[{name}採用判断](decisions/{dec})。採用範囲は正式エンジン(ハイブリッド)。")
    section("docs/00_開発ルーター.md", "## 現行モデル", "## 現在の研究・保存再開点", "## 現行モデル\n\n" + common + "\n\n")
    append("docs/00_開発ルーター.md", f"## 最新NNUE（{name}、{date}）\n\n" + common +
           "\n\n強化の進め方は「生成(g1-gen.yml)→学習(g1-train.yml)→対局(hybrid-match.yml)→採用(hybrid-nnue-adopt.yml)」の周回。")
    section("docs/ROADMAP.md", "## 現行モデル", "## 現在の研究", "## 現行モデル\n\n" + common + "\n\n")
    append("docs/ROADMAP.md", f"## {name}：自己対局の周回でNNUEを更新（{date}）\n\n" + common)
    append("AGENTS.md", f"## 最新正式NNUE（{name}、{date}）\n\n正式{name}／Adam{step}、snapshot {new}。{w}。"
           f"正式エンジン(ハイブリッド)の重みは formal/model.json の hybrid節(`{model_path}`)。")
    mt = "docs/モデル名対応表.md"
    resub(mt, r"現行の正式モデルは \*\*E\d+\*\*", f"現行の正式モデルは **{name}**")
    resub(mt, re.escape(f"| **{prev}（現行）** |"), f"| {prev} |")
    row = (f"| **{name}（現行）** | {prev}のハイブリッド自己対局(50ms)約{rep['train_positions']}局面で学習(玉周辺特徴つき) | {prev} | {new[:8]} | {step} | {wsha[:8]} | "
           f"ハイブリッド下 対{prev} {rp['fukaji_wins']}-{rp['kumoji_wins']}、{kv}。[記録](decisions/{dec}) |\n")
    p = ROOT / mt; t = p.read_text(); i = t.index("\n通算確認")
    p.write_text(t[:i].rstrip("\n") + "\n" + row + t[i:])
    for p in ("formal/model.json", spec_doc):
        contract[p] = digest((ROOT / p).read_bytes())
    write_json(CONTRACT, contract)
    receipt["decision"] = decision
    write_json(out / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--name", required=True)
    a.add_argument("--date", required=True)
    a.add_argument("--trained", type=Path, required=True)
    a.add_argument("--result-prev", type=Path, required=True)
    a.add_argument("--result-kumoji", type=Path, default=None)
    a.add_argument("--kumoji-min-wins", type=int, default=0)
    a.add_argument("--auto", action="store_true", help="2026-11-17本人決定の自動採用(承認待ちなし)として記録する")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--commit", action="store_true")
    main(a.parse_args())
