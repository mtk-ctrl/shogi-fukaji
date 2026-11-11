"""E9: 現行E8の正式探索(固定深さ5)の評価値を教師にした蒸留(約5.7万局面)で得たNNUEを、ユーザー承認(2026-11-11)のもと正式採用する。
事前基準(400局で+24)に1届かない(+23)が、600局合算+46・held-out損失通過を踏まえユーザーが採用を指示した。その事実を記録に明記する。
実行はCI(e9-adopt.yml)のみ。結果は不変のartifact(実run)から読み、勝敗・重みSHA・Adam状態を再検証してから書き換える。
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

OLD = "18e6860405caf40c610c21c1b65851505372d6cd1734f720d3f71a271c746d30"
OLD_WEIGHTS = "3d1e2fb506b3aa6b47c82f8984fda00df461374f1c2a6a94d6eb32912f2e0ade"
OLD_STEP = 26573
CPP = "166b0819116364956e94ab47e54eb1891a87f841f8d17d0c59f1adff3f62ecc5"
MODEL_PATH = "models/fukaji-e9.nnue.npz"
DECISION = "docs/decisions/2026-11-11_E9_E8親の深さ5蒸留NNUEを正式採用.md"
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
        raise ValueError("E9 training report mismatch")
    ra = check_result(result200, 200, 12)   # 最初の200局(基準+12)
    rb = check_result(result400, 400, 1)    # 追試400局(基準+24には+23で1届かず。ユーザー判断で採用)
    r200 = dict(rb); r200["new_wins"] = ra["new_wins"] + rb["new_wins"]; r200["prev_wins"] = ra["prev_wins"] + rb["prev_wins"]
    r200["games"] = 600; r200["draws_or_unresolved"] = ra["draws_or_unresolved"] + rb["draws_or_unresolved"]
    for k in ("pair_sweeps_fukaji", "pair_sweeps_kumoji"):
        r200[k] = ra[k] + rb[k]
    formal = json.loads((ROOT / "formal/model.json").read_text())
    for k, v in (("version", "FUKAJI v0.4.1"), ("search_backend_variant", "d48_pvs_mate"),
                 ("search_implementation_sha256", CPP), ("weights_sha256", OLD_WEIGHTS),
                 ("research_snapshot_sha256", OLD), ("optimizer_step", OLD_STEP)):
        if formal.get(k) != v:
            raise ValueError("formal current E8 drift " + k)
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
        raise ValueError("E9 complete Adam state could not be restored")
    if learned.inputs != 2548 or learned.hidden_weights.shape != (2548, 128):
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
    receipt = {"schema": "e9-verified-adoption-v1", "former_snapshot_sha256": OLD, "current_snapshot_sha256": new,
               "optimizer_step": rep["optimizer_step_after"], "weights_path": MODEL_PATH, "weights_sha256": wsha,
               "unchanged_cpp_sha256": CPP, "usi_version": "FUKAJI v0.4.1",
               "audit_first200": {k: ra[k] for k in ("games", "prev_wins", "new_wins", "draws_or_unresolved")},
               "audit_repeat400": {k: rb[k] for k in ("games", "prev_wins", "new_wins", "draws_or_unresolved")},
               "audit200": {k: r200[k] for k in ("games", "prev_wins", "new_wins", "draws_or_unresolved", "pair_sweeps_fukaji", "pair_sweeps_kumoji")},
               "train_positions": rep["train_positions"], "outcome_weight": rep["outcome_weight"], "lr": rep["lr"], "epochs": rep["epochs"],
               "heldout_before": rep["heldout_before"], "heldout_after": rep["heldout_after"],
               "weights_unchanged_across_audits": True}
    write_json(str(out / "receipt.json"), receipt)
    if not commit:
        print("DRY RUN OK", json.dumps(receipt, ensure_ascii=False)); return
    dest = ROOT / MODEL_PATH
    if dest.exists():
        raise ValueError("E9 inference already exists")
    buf = io.BytesIO()
    np.savez_compressed(buf, w=learned.hidden_weights, b=learned.hidden_bias, v=learned.output_weights,
                        o=np.array([learned.output_bias], dtype=np.float32))
    raw = buf.getvalue(); dest.write_bytes(raw); wsha = digest(raw); receipt["weights_sha256"] = wsha
    w200 = f"E9 {r200['new_wins']}-{r200['prev_wins']}(対E8 合算600局: 最初の200局{ra['new_wins']}-{ra['prev_wins']}＋別シード400局{rb['new_wins']}-{rb['prev_wins']}・200ms)"
    nf = dict(formal)
    nf.update({
        "approved_on": "2026-11-11",
        "approved_by": f"owner-approved E9 (2026-11-11) {w200}",
        "research_snapshot_sha256": new, "research_parent_sha256": OLD,
        "weights_file": MODEL_PATH, "weights_sha256": wsha, "optimizer_step": rep["optimizer_step_after"],
        "training_release_sha256": digest((trained[0] / "report.json").read_bytes()),
        "training_source_sha256": rep["weights_file_sha256"],
        "comparison_release_sha256": digest(Path(result200).read_bytes()),
        "model_distribution": f"GitHub {MODEL_PATH}; complete Adam{rep['optimizer_step_after']} in the E9 trained Release.",
        "validation": {
            "opponent": "E8/Adam26573 (previous formal, hidden 128), paired unseen openings, same search, 200ms",
            "audit200": receipt["audit200"],
            "strength_conclusion": f"{w200}; 事前基準(400局で+24)に+23で1届かないが、合算+46・held-out損失通過・ユーザー指示により採用",
            "search_backend_variant": "d48_pvs_mate", "quiescence": 0,
            "training": "distillation of the formal fixed-depth-5 search value (labels) mixed 0.2 with game outcome; positions (~5.7万, 1 variant) from E8-vs-KUMOJI games and selfplay, relabeled by the E8 formal search",
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
        f"# E9：E8親の深さ5蒸留NNUEを正式採用（2026-11-11）\n\n"
        f"ユーザー承認。E8(Adam{OLD_STEP})から、現行E8の正式探索(固定深さ5)の評価値を教師にした蒸留"
        f"（E8対雲路の練習棋譜と自己対局の約{rep['train_positions']}局面、対局結果を{rep['outcome_weight']}混合、雲路の指し手・評価値は不使用）で更新。\n\n"
        f"- {w200}。差+{r200['new_wins']-r200['prev_wins']}。\n"
        f"- 事前基準は『200局+12→別シード400局+24』。最初の200局は+{ra['new_wins']-ra['prev_wins']}、追試400局は+{rb['new_wins']-rb['prev_wins']}で、追試は基準に1届かなかった。"
        f"それでもユーザーが採用を指示した（小さいが一貫した正の傾向）。差は小さいため、次の強化で再確認される。\n"
        f"- held-out label loss {rep['heldout_before']:.4f}→{rep['heldout_after']:.4f}（通過）。深さ6は240分上限に収まらず中断（採用に無関係）。\n"
        f"- 正式snapshot {new}、推論SHA256 {wsha}、Adam{rep['optimizer_step_after']}。NNUE以外は不変（USI v0.4.1、PVS＋詰み探索）。\n"
        f"- 旧E8重みは models/fukaji-e8.nnue.npz、旧E7以前も models/ に保全。戻す場合はformal/model.jsonを前版に戻す。\n")
    spec_doc = "docs/現行エンジン仕様.md"
    once(spec_doc, "**現行のNNUE重みはE8（18e68604、Adam26573、隠れ層128。E7から蒸留で更新）**",
         f"**現行のNNUE重みはE9（{new[:8]}、Adam{rep['optimizer_step_after']}、隠れ層128。E8から蒸留で更新）**")
    once(spec_doc, f"| 研究snapshot SHA256 | `{OLD}` |", f"| 研究snapshot SHA256 | `{new}` |")
    once(spec_doc, "| 推論用ファイル | `models/fukaji-e8.nnue.npz` |", f"| 推論用ファイル | `{MODEL_PATH}` |")
    once(spec_doc, f"| 推論用SHA256 | `{OLD_WEIGHTS}` |", f"| 推論用SHA256 | `{wsha}` |")
    once(spec_doc, "次のNNUE学習は現行E8（Adam26573）", f"次のNNUE学習は現行E9（Adam{rep['optimizer_step_after']}）")
    once(spec_doc, "更新前snapshot E8と更新後", "更新前snapshot E9と更新後")
    append(spec_doc, f"## E9：E8親の深さ5蒸留NNUEを正式採用（2026-11-11）\n\n- snapshot {new}、推論SHA256 {wsha}、Adam{rep['optimizer_step_after']}。{w200}。"
           f"[判断](decisions/{DECISION.split('/')[-1]})。旧E8は models/fukaji-e8.nnue.npz。")
    common = (f"正式E9（{new}、Adam{rep['optimizer_step_after']}、E8から蒸留で更新）。{w200}。"
              f"[E9採用判断](decisions/{DECISION.split('/')[-1]})。USI v0.4.1・PVS＋詰み探索不変。")
    section("docs/00_開発ルーター.md", "## 現行モデル", "## 現在の研究・保存再開点", "## 現行モデル\n\n" + common + "\n\n")
    append("docs/00_開発ルーター.md", "## 最新NNUE（E9、2026-11-11）\n\n" + common)
    section("docs/ROADMAP.md", "## 現行モデル", "## 現在の研究", "## 現行モデル\n\n" + common + "\n\n")
    append("docs/ROADMAP.md", "## E9：E8親の深さ5蒸留NNUE正式採用（2026-11-11）\n\n" + common)
    append("AGENTS.md", f"## 最新正式NNUE（E9、2026-11-11）\n\n正式E9／Adam{rep['optimizer_step_after']}、snapshot {new}。{w200}。NNUEのみ更新、旧E8・旧E7・旧E2・旧E3・旧D66を保全。")
    mt = "docs/モデル名対応表.md"
    once(mt, "現行の正式モデルは **E8**", "現行の正式モデルは **E9**")
    once(mt, "| **E8（現行）** |", "| E8 |")
    row = (f"| **E9（現行）** | E8と雲路の練習局面を、E8の探索評価値(深さ5)で蒸留（約5.7万局面） | E8 | {new[:8]} | {rep['optimizer_step_after']} | {wsha[:8]} | "
           f"対E8 600局 {r200['new_wins']}-{r200['prev_wins']}（追試は基準に1届かず、ユーザー判断で採用）。[記録](decisions/{DECISION.split('/')[-1]}) |\n")
    p = ROOT / mt; t = p.read_text(); i = t.index("\n通算確認")
    p.write_text(t[:i].rstrip("\n") + "\n" + row + t[i:])
    for p in ("formal/model.json", spec_doc):
        contract[p] = digest((ROOT / p).read_bytes())
    write_json(contract_path, contract)
    formal_load()
    assert digest((ROOT / "native/nnue_search_ffi.cpp").read_bytes()) == CPP
    assert digest((ROOT / "models/fukaji-e8.nnue.npz").read_bytes()) == OLD_WEIGHTS
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
