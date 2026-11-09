"""E2: 現行E3の正式探索(固定深さ4)の評価値を教師にした蒸留で得たNNUEを、ユーザー承認(2026-11-09)のもと正式採用する。
実行はCI(e2-adopt.yml)のみ。結果は不変のartifact(実run)から読み、勝敗・重みSHA・Adam状態を再検証してから書き換える。
NNUE重み以外(探索・USI v0.4.1・C++)は変更しない。
"""
from __future__ import annotations
import hashlib, io, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fukaji.online_learning import load_snapshot
from fukaji.formal_model import load as formal_load

OLD = "193bc0e2ce961284958660f8709473551b3e31a42e352a5b88a76ebce41d27ac"
OLD_WEIGHTS = "95e9add33a60f57ade2d1eb6f44ee091ad7392452bff9e7e34b492dc51cd0355"
OLD_STEP = 12562
CPP = "c9340af265368547c37cbecc311d8d3982900885b433d1b4ccdc27fbdd7d041a"
MODEL_PATH = "models/fukaji-e2.nnue.npz"
DECISION = "docs/decisions/2026-11-09_E2_探索評価値蒸留NNUEを正式採用.md"
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


def check_result(path, games, margin):
    r = json.loads(Path(path).read_text())
    d66, e3 = r["prev_wins"], r["new_wins"]
    ok = (r["games"] == games and r["replay_verified"] is True and r["illegal_pairs"] == [] and
          r["labels"] == ["PREV", "NEW"] and e3 - d66 >= margin)
    if not ok:
        raise ValueError(f"result check failed {path}: {r}")
    return r


def main(trained_dirs, result200, result400, out, commit):
    trained = [Path(x) for x in trained_dirs]
    reports = [json.loads((d / "report.json").read_text()) for d in trained]
    rep = reports[0]
    for r in reports:  # 200局・400局の両監査で使った重みが同一であること
        if (r["weights_file_sha256"] != rep["weights_file_sha256"] or r["snapshot"] != rep["snapshot"]):
            raise ValueError("audited weights differ between runs")
    if (rep["schema"] != "e2-search-distill-v1" or rep["parent"] != OLD or rep["gate"] != "pass" or
            rep["formal_model_changed"] is not False or rep["optimizer_step_after"] <= OLD_STEP):
        raise ValueError("E2 training report mismatch")
    r200 = check_result(result200, 200, 12)
    r400 = check_result(result400, 400, 24)
    formal = json.loads((ROOT / "formal/model.json").read_text())
    for k, v in (("version", "FUKAJI v0.4.1"), ("search_backend_variant", "d48_pvs_mate"),
                 ("search_implementation_sha256", CPP), ("weights_sha256", OLD_WEIGHTS),
                 ("research_snapshot_sha256", OLD), ("optimizer_step", OLD_STEP)):
        if formal.get(k) != v:
            raise ValueError("formal current E3 drift " + k)
    if digest((ROOT / "native/nnue_search_ffi.cpp").read_bytes()) != CPP:
        raise ValueError("official PVS/mate changed")
    contract_path = "research/d17-fast-self-learning/formal-contract.json"
    contract = json.loads((ROOT / contract_path).read_text())
    for path, expected in contract.items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ValueError("frozen formal contract changed " + path)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    new = rep["snapshot"]
    learned, opt, _ = load_snapshot(trained[0] / "models", new)
    if opt is None or opt["step"] != rep["optimizer_step_after"]:
        raise ValueError("E2 complete Adam state could not be restored")
    if learned.inputs != 2548 or learned.hidden_weights.shape != (2548, 32):
        raise ValueError("architecture changed")
    raw_src = (trained[0] / "cand-weights.npz").read_bytes()
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
    wsha = digest(raw_src)
    receipt = {"schema": "e2-verified-adoption-v1", "former_snapshot_sha256": OLD, "current_snapshot_sha256": new,
               "optimizer_step": rep["optimizer_step_after"], "weights_path": MODEL_PATH, "weights_sha256": wsha,
               "unchanged_cpp_sha256": CPP, "usi_version": "FUKAJI v0.4.1",
               "audit200": {k: r200[k] for k in ("games", "prev_wins", "new_wins", "draws_or_unresolved", "pair_sweeps_fukaji", "pair_sweeps_kumoji")},
               "audit400": {k: r400[k] for k in ("games", "prev_wins", "new_wins", "draws_or_unresolved", "pair_sweeps_fukaji", "pair_sweeps_kumoji")},
               "train_positions": rep["train_positions"], "outcome_weight": rep["outcome_weight"], "lr": rep["lr"], "epochs": rep["epochs"],
               "heldout_before": rep["heldout_before"], "heldout_after": rep["heldout_after"],
               "weights_unchanged_across_audits": True}
    write_json(str(out / "receipt.json"), receipt)
    if not commit:
        print("DRY RUN OK", json.dumps(receipt, ensure_ascii=False)); return
    dest = ROOT / MODEL_PATH
    if dest.exists():
        raise ValueError("E2 inference already exists")
    buf = io.BytesIO()
    np.savez_compressed(buf, w=learned.hidden_weights, b=learned.hidden_bias, v=learned.output_weights,
                        o=np.array([learned.output_bias], dtype=np.float32))
    raw = buf.getvalue(); dest.write_bytes(raw); wsha = digest(raw); receipt["weights_sha256"] = wsha
    w200 = f"E2 {r200['new_wins']}-{r200['prev_wins']}(200局)・E2 {r400['new_wins']}-{r400['prev_wins']}(確認400局)"
    nf = dict(formal)
    nf.update({
        "approved_on": "2026-11-09",
        "approved_by": f"owner-approved E2 (2026-11-09) {w200}",
        "research_snapshot_sha256": new, "research_parent_sha256": OLD,
        "weights_file": MODEL_PATH, "weights_sha256": wsha, "optimizer_step": rep["optimizer_step_after"],
        "training_release_sha256": digest((trained[0] / "report.json").read_bytes()),
        "training_source_sha256": rep["weights_file_sha256"],
        "comparison_release_sha256": digest(Path(result400).read_bytes()),
        "model_distribution": f"GitHub {MODEL_PATH}; complete Adam{rep['optimizer_step_after']} in the E2 trained Release.",
        "validation": {
            "opponent": "E3/Adam12562 (previous formal), paired unseen openings, same search, 200ms",
            "audit200": receipt["audit200"], "audit400_confirmation": receipt["audit400"],
            "strength_conclusion": f"{w200}; 事前基準(+12/+24)を満たす",
            "search_backend_variant": "d48_pvs_mate", "quiescence": 0,
            "training": "distillation of the formal fixed-depth-4 search value (labels) mixed 0.2 with game outcome; positions from E3-vs-KUMOJI games and selfplay",
        }})
    for k in ("version", "architecture", "backend", "quiescence", "default_move_time_ms", "default_search_depth",
              "search_backend_variant", "search_implementation_sha256", "search_adoption"):
        if nf[k] != formal[k]:
            raise ValueError("unauthorized search change " + k)
    write_json("formal/model.json", nf)
    loaded, spec = formal_load()
    if spec["weights_sha256"] != wsha:
        raise ValueError("formal load failure")
    np.testing.assert_array_equal(loaded.hidden_weights, learned.hidden_weights)
    (ROOT / DECISION).write_text(
        f"# E2：探索評価値の蒸留NNUEを正式採用（2026-11-09）\n\n"
        f"ユーザー承認（2026-11-09 22:53）。E3(Adam{OLD_STEP})から、現行E3の正式探索(固定深さ4)の評価値を教師にした蒸留"
        f"（雲路戦棋譜とE3自己対局の約{rep['train_positions']}局面、対局結果を{rep['outcome_weight']}混合、雲路の指し手・評価値は不使用）で更新。\n\n"
        f"- {w200}。差+{r200['new_wins']-r200['prev_wins']} / +{r400['new_wins']-r400['prev_wins']}。\n"
        f"- 正式snapshot {new}、推論SHA256 {wsha}、Adam{rep['optimizer_step_after']}。NNUE以外は不変（USI v0.4.1、PVS＋詰み探索）。\n"
        f"- 旧E3重みは models/fukaji-e3.nnue.npz、旧D66は models/fukaji-d66.nnue.npz に保全。戻す場合はformal/model.jsonを前版に戻す。\n")
    spec_doc = "docs/現行エンジン仕様.md"
    once(spec_doc, "**現行のNNUE重みはE3（193bc0e2、Adam12562。D66から雲路練習局面で更新）**",
         f"**現行のNNUE重みはE2（{new[:8]}、Adam{rep['optimizer_step_after']}。E3から探索評価値の蒸留で更新）**")
    once(spec_doc, f"| 研究snapshot SHA256 | `{OLD}` |", f"| 研究snapshot SHA256 | `{new}` |")
    once(spec_doc, "| 推論用ファイル | `models/fukaji-e3.nnue.npz` |", f"| 推論用ファイル | `{MODEL_PATH}` |")
    once(spec_doc, f"| 推論用SHA256 | `{OLD_WEIGHTS}` |", f"| 推論用SHA256 | `{wsha}` |")
    once(spec_doc, "次のNNUE学習は現行E3（Adam12562）", f"次のNNUE学習は現行E2（Adam{rep['optimizer_step_after']}）")
    once(spec_doc, "更新前snapshot E3と更新後", "更新前snapshot E2と更新後")
    append(spec_doc, f"## E2：探索評価値蒸留NNUEを正式採用（2026-11-09）\n\n- snapshot {new}、推論SHA256 {wsha}、Adam{rep['optimizer_step_after']}。{w200}。"
           f"[判断](decisions/{DECISION.split('/')[-1]})。旧E3は models/fukaji-e3.nnue.npz。")
    common = (f"正式E2（{new}、Adam{rep['optimizer_step_after']}、E3から探索評価値の蒸留で更新）。{w200}。"
              f"[E2採用判断](decisions/{DECISION.split('/')[-1]})。USI v0.4.1・PVS＋詰み探索不変。")
    section("docs/00_開発ルーター.md", "## 現行モデル", "## 現在の研究・保存再開点", "## 現行モデル\n\n" + common + "\n\n")
    append("docs/00_開発ルーター.md", "## 最新NNUE（E2、2026-11-09）\n\n" + common)
    section("docs/ROADMAP.md", "## 現行モデル", "## 現在の研究", "## 現行モデル\n\n" + common + "\n\n")
    append("docs/ROADMAP.md", "## E2：探索評価値蒸留NNUE正式採用（2026-11-09）\n\n" + common)
    append("AGENTS.md", f"## 最新正式NNUE（E2、2026-11-09）\n\n正式E2／Adam{rep['optimizer_step_after']}、snapshot {new}。{w200}。NNUEのみ更新、旧E3・旧D66を保全。")
    for p in ("formal/model.json", spec_doc):
        contract[p] = digest((ROOT / p).read_bytes())
    write_json(contract_path, contract)
    formal_load()
    assert digest((ROOT / "native/nnue_search_ffi.cpp").read_bytes()) == CPP
    assert digest((ROOT / "models/fukaji-e3.nnue.npz").read_bytes()) == OLD_WEIGHTS
    receipt["contract_sha256"] = digest((ROOT / contract_path).read_bytes())
    write_json(str(out / "receipt.json"), receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--trained", type=Path, nargs="+", required=True)
    a.add_argument("--result200", type=Path, required=True)
    a.add_argument("--result400", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--commit", action="store_true")
    x = a.parse_args()
    main(x.trained, x.result200, x.result400, x.out, x.commit)
