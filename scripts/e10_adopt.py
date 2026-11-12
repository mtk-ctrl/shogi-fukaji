"""E10: 親E9。ハイブリッド探索(雲路の探索＋深路NNUE)の探索評価値(1局面2秒)を教師に再学習したNNUEを、ユーザー承認(2026-11-13)のもと正式採用する。
事前基準: 対E9ハイブリッド100局で+12以上。実績 83-17、対雲路最新 16-84(E9ハイブリッドは4-96)。
注意: 強さ確認はハイブリッド探索下。正式エンジンの探索(USI v0.4.1・PVS＋詰み)は変更しない。旧探索での強さは別途確認する。
実行はCI(e10-adopt.yml)のみ。実runのartifactから学習報告・勝敗・重みSHA・Adam状態を再検証してから書き換える。
"""
from __future__ import annotations
import hashlib, io, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fukaji.online_learning import load_snapshot
from fukaji.formal_model import load as formal_load

OLD = "c9302b9552d90fdfe88ac38ca54970fa3eceeaf67c024368f13d12b87af2b050"
OLD_WEIGHTS = "c5b12ff0bf4b0bd3fa0551d1ba1fd5d17cce311b6dd88cadd7106534c0f01173"
OLD_STEP = 27469
CPP = "166b0819116364956e94ab47e54eb1891a87f841f8d17d0c59f1adff3f62ecc5"
MODEL_PATH = "models/fukaji-e10.nnue.npz"
DECISION = "docs/decisions/2026-11-13_E10_ハイブリッド探索ラベル再学習NNUEを正式採用.md"
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


def check_result(path, games, margin, a_name):
    r = json.loads(Path(path).read_text())
    ok = (r["games"] == games and r["replay_verified"] is True and r["illegal_pairs"] == [] and
          r["labels"] == ["HYBRID", "OPP"] and r["fukaji_wins"] - r["kumoji_wins"] >= margin)
    if not ok:
        raise ValueError(f"result check failed {path}: {r}")
    return r


def main(trained_dir, result_e9, result_kumoji, out, commit):
    trained = Path(trained_dir)
    rep = json.loads((trained / "report.json").read_text())
    if (rep["schema"] != "e2-search-distill-v1" or rep["parent"] != OLD or rep["gate"] != "pass" or
            rep["formal_model_changed"] is not False or rep["optimizer_step_after"] <= OLD_STEP):
        raise ValueError("E10 training report mismatch")
    ra = check_result(result_e9, 100, 12, "vsE9")
    rk = json.loads(Path(result_kumoji).read_text())
    if not (rk["games"] == 100 and rk["replay_verified"] is True and rk["illegal_pairs"] == [] and rk["labels"] == ["HYBRID", "OPP"]):
        raise ValueError("kumoji result check failed")
    formal = json.loads((ROOT / "formal/model.json").read_text())
    for k, v in (("version", "FUKAJI v0.4.1"), ("search_backend_variant", "d48_pvs_mate"),
                 ("search_implementation_sha256", CPP), ("weights_sha256", OLD_WEIGHTS),
                 ("research_snapshot_sha256", OLD), ("optimizer_step", OLD_STEP)):
        if formal.get(k) != v:
            raise ValueError("formal current E9 drift " + k)
    if digest((ROOT / "native/nnue_search_ffi.cpp").read_bytes()) != CPP:
        raise ValueError("official PVS/mate changed")
    contract_path = "research/d17-fast-self-learning/formal-contract.json"
    contract = json.loads((ROOT / contract_path).read_text())
    for path, expected in contract.items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ValueError("frozen formal contract changed " + path)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    new = rep["snapshot"]
    learned, opt, _ = load_snapshot(trained / "models", new)
    if opt is None or opt["step"] != rep["optimizer_step_after"]:
        raise ValueError("E10 complete Adam state could not be restored")
    if learned.inputs != 2548 or learned.hidden_weights.shape != (2548, 128):
        raise ValueError("architecture changed")
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
    wsha = digest(raw_src)
    pick = lambda r: {k: r[k] for k in ("games", "fukaji_wins", "kumoji_wins", "draws_or_unresolved")}
    receipt = {"schema": "e10-verified-adoption-v1", "former_snapshot_sha256": OLD, "current_snapshot_sha256": new,
               "optimizer_step": rep["optimizer_step_after"], "weights_path": MODEL_PATH, "weights_sha256": wsha,
               "unchanged_cpp_sha256": CPP, "usi_version": "FUKAJI v0.4.1",
               "hybrid_vs_e9_hybrid": pick(ra), "hybrid_vs_kumoji": pick(rk),
               "train_positions": rep["train_positions"], "outcome_weight": rep["outcome_weight"], "lr": rep["lr"], "epochs": rep["epochs"],
               "heldout_before": rep["heldout_before"], "heldout_after": rep["heldout_after"]}
    write_json(str(out / "receipt.json"), receipt)
    if not commit:
        print("DRY RUN OK", json.dumps(receipt, ensure_ascii=False)); return
    dest = ROOT / MODEL_PATH
    if dest.exists():
        raise ValueError("E10 inference already exists")
    buf = io.BytesIO()
    np.savez_compressed(buf, w=learned.hidden_weights, b=learned.hidden_bias, v=learned.output_weights,
                        o=np.array([learned.output_bias], dtype=np.float32))
    raw = buf.getvalue(); dest.write_bytes(raw); wsha = digest(raw); receipt["weights_sha256"] = wsha
    w = (f"E10 ハイブリッド探索下(雲路の探索＋深路NNUE, 1手2秒相当・同一探索)で対E9 {ra['fukaji_wins']}-{ra['kumoji_wins']}(100局)、"
         f"対雲路最新 {rk['fukaji_wins']}-{rk['kumoji_wins']}(100局。E9は4-96)")
    nf = dict(formal)
    nf.update({
        "approved_on": "2026-11-13",
        "approved_by": f"owner-approved E10 (2026-11-13) {w}",
        "research_snapshot_sha256": new, "research_parent_sha256": OLD,
        "weights_file": MODEL_PATH, "weights_sha256": wsha, "optimizer_step": rep["optimizer_step_after"],
        "training_release_sha256": digest((trained / "report.json").read_bytes()),
        "training_source_sha256": rep["weights_file_sha256"],
        "comparison_release_sha256": digest(Path(result_e9).read_bytes()),
        "model_distribution": f"GitHub {MODEL_PATH}; complete Adam{rep['optimizer_step_after']} in the E10 trained Release.",
        "validation": {
            "opponent": "E9 (previous formal, hidden 128), hybrid search (KUMOJI search + FUKAJI NNUE), paired unseen openings, same search",
            "hybrid_vs_e9": receipt["hybrid_vs_e9_hybrid"], "hybrid_vs_kumoji": receipt["hybrid_vs_kumoji"],
            "strength_conclusion": f"{w}。正式探索(USI v0.4.1)での強さは未確認(別途確認)",
            "search_backend_variant": "d48_pvs_mate", "quiescence": 0,
            "training": "distillation of the hybrid search value (1 position = 2000ms, KUMOJI search + E9 NNUE eval) mixed 0.2 with game outcome; ~11.5万 positions from past games",
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
    dec = DECISION.split("/")[-1]
    (ROOT / DECISION).write_text(
        f"# E10：ハイブリッド探索ラベルで再学習したNNUEを正式採用（2026-11-13）\n\n"
        f"ユーザー承認(2026-11-13「正式採用してください」)。E9(Adam{OLD_STEP})から、ハイブリッド探索（雲路の探索＋深路NNUE評価、1局面2秒）の評価値を教師に"
        f"約{rep['train_positions']}局面で再学習（対局結果{rep['outcome_weight']}混合、雲路の評価値・指し手は不使用）。\n\n"
        f"- {w}。事前基準は対E9ハイブリッド+12以上で、+{ra['fukaji_wins']-ra['kumoji_wins']}。\n"
        f"- held-out label loss {rep['heldout_before']:.4f}→{rep['heldout_after']:.4f}（通過）。\n"
        f"- 重要な制約: 強さ確認はハイブリッド探索下。正式エンジン(USI v0.4.1・PVS＋詰み探索)は不変で、旧探索での強さは採用後に確認する。悪化していれば models/fukaji-e9.nnue.npz へ戻す。\n"
        f"- 正式snapshot {new}、推論SHA256 {wsha}、Adam{rep['optimizer_step_after']}。\n"
        f"- 旧E9重みは models/fukaji-e9.nnue.npz、E8以前も models/ に保全。戻す場合はformal/model.jsonを前版に戻す。\n")
    spec_doc = "docs/現行エンジン仕様.md"
    once(spec_doc, "**現行のNNUE重みはE9（c9302b95、Adam27469、隠れ層128。E8から蒸留で更新）**",
         f"**現行のNNUE重みはE10（{new[:8]}、Adam{rep['optimizer_step_after']}、隠れ層128。E9からハイブリッド探索ラベルで再学習）**")
    once(spec_doc, f"| 研究snapshot SHA256 | `{OLD}` |", f"| 研究snapshot SHA256 | `{new}` |")
    once(spec_doc, "| 推論用ファイル | `models/fukaji-e9.nnue.npz` |", f"| 推論用ファイル | `{MODEL_PATH}` |")
    once(spec_doc, f"| 推論用SHA256 | `{OLD_WEIGHTS}` |", f"| 推論用SHA256 | `{wsha}` |")
    once(spec_doc, "次のNNUE学習は現行E9（Adam27469）", f"次のNNUE学習は現行E10（Adam{rep['optimizer_step_after']}）")
    once(spec_doc, "更新前snapshot E9と更新後", "更新前snapshot E10と更新後")
    append(spec_doc, f"## E10：ハイブリッド探索ラベル再学習NNUEを正式採用（2026-11-13）\n\n- snapshot {new}、推論SHA256 {wsha}、Adam{rep['optimizer_step_after']}。{w}。"
           f"[判断](decisions/{dec})。旧E9は models/fukaji-e9.nnue.npz。")
    common = (f"正式E10（{new}、Adam{rep['optimizer_step_after']}、E9からハイブリッド探索ラベルで再学習）。{w}。"
              f"[E10採用判断](decisions/{dec})。USI v0.4.1・PVS＋詰み探索不変(旧探索での強さは採用後に確認)。")
    section("docs/00_開発ルーター.md", "## 現行モデル", "## 現在の研究・保存再開点", "## 現行モデル\n\n" + common + "\n\n")
    append("docs/00_開発ルーター.md", "## 最新NNUE（E10、2026-11-13）\n\n" + common)
    section("docs/ROADMAP.md", "## 現行モデル", "## 現在の研究", "## 現行モデル\n\n" + common + "\n\n")
    append("docs/ROADMAP.md", "## E10：ハイブリッド探索ラベル再学習NNUE正式採用（2026-11-13）\n\n" + common)
    append("AGENTS.md", f"## 最新正式NNUE（E10、2026-11-13）\n\n正式E10／Adam{rep['optimizer_step_after']}、snapshot {new}。{w}。NNUEのみ更新、旧E9以前を保全。")
    mt = "docs/モデル名対応表.md"
    once(mt, "現行の正式モデルは **E9**", "現行の正式モデルは **E10**")
    once(mt, "| **E9（現行）** |", "| E9 |")
    row = (f"| **E10（現行）** | E9と過去棋譜の局面を、ハイブリッド探索(雲路探索＋深路NNUE, 2秒)の評価値で再学習（約{rep['train_positions']}局面） | E9 | {new[:8]} | {rep['optimizer_step_after']} | {wsha[:8]} | "
           f"ハイブリッド下 対E9 {ra['fukaji_wins']}-{ra['kumoji_wins']}、対雲路 {rk['fukaji_wins']}-{rk['kumoji_wins']}。[記録](decisions/{dec}) |\n")
    p = ROOT / mt; t = p.read_text(); i = t.index("\n通算確認")
    p.write_text(t[:i].rstrip("\n") + "\n" + row + t[i:])
    for p in ("formal/model.json", spec_doc):
        contract[p] = digest((ROOT / p).read_bytes())
    write_json(contract_path, contract)
    formal_load()
    assert digest((ROOT / "native/nnue_search_ffi.cpp").read_bytes()) == CPP
    assert digest((ROOT / "models/fukaji-e9.nnue.npz").read_bytes()) == OLD_WEIGHTS
    receipt["contract_sha256"] = digest((ROOT / contract_path).read_bytes())
    write_json(str(out / "receipt.json"), receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--trained", type=Path, required=True)
    a.add_argument("--result-e9", type=Path, required=True)
    a.add_argument("--result-kumoji", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--commit", action="store_true")
    x = a.parse_args()
    main(x.trained, x.result_e9, x.result_kumoji, x.out, x.commit)
