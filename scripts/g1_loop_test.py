#!/usr/bin/env python3
"""g1_loop.py の判定の試験(ネットワーク不要)。python scripts/g1_loop_test.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import g1_loop as L

F = {"snapshot": "aaaa1111", "asset": "E27-trained-state-Adam.tar.gz"}


def M(w, l, ill=0):
    return {"w": w, "l": l, "illegal": ill}


def ev(kind, state, run="1", result="success", tag=None):
    return {"event": kind, "run_id": run, "tag": state["tag"] if tag is None else tag, "result": result}


def started():
    s = L.new_state("aaaa1111")
    s, a = L.step(s, {"event": "start", "run_id": "", "tag": "", "result": "success"}, {"formal": F})
    assert a[0]["workflow"] == "g1-gen.yml" and a[0]["inputs"]["weights"] == "formal" and s["stage"] == "gen"
    assert a[0]["inputs"]["loop_tag"] == s["tag"]
    return s


# --- 判定
assert L.judge([M(61, 46)]) == "adopt"                # +15
assert L.judge([M(59, 49)]) == "retest"               # +10
assert L.judge([M(55, 53)]) == "reject"
assert L.judge([M(70, 30, 1)]) == "reject"            # 違法手
assert L.judge([M(59, 49), M(59, 48)]) == "adopt"     # 両方+6以上(E27の実例)
assert L.judge([M(59, 49), M(50, 58)]) == "reject"    # +10と-8、合算+2
assert L.judge([M(66, 42), M(46, 62)]) == "reject"    # +24と-16、合算+8
assert L.judge([M(80, 28), M(50, 58)]) == "adopt"     # +52と-8、合算+44

# --- 流れ: 生成2回 → 学習 → 対局+追試 → 採用 → 次の生成
s = started()
s, a = L.step(s, ev("gen", s, "101"), {"formal": F})
assert a[0]["workflow"] == "g1-gen.yml" and s["gens"] == [101]            # 1回分では学習しない
s, a = L.step(s, ev("gen", s, "102"), {"formal": F})
assert a[0]["workflow"] == "g1-train.yml" and a[0]["inputs"]["data_runs"] == "101 102"   # 空白区切り
s, a = L.step(s, ev("train", s, "201"), {"formal": F, "report": {"gate": "pass", "mean_label": 0.499, "heldout_before": 0.5545, "heldout_after": 0.5544}})
assert a[0]["workflow"] == "hybrid-match.yml" and a[0]["inputs"]["weights"] == "run:201" and a[0]["inputs"]["opp_weights"] == "formal"
s, a = L.step(s, ev("match", s, "301"), {"formal": F, "match": {"w": 59, "l": 49, "illegal": 0}})
assert a[0]["workflow"] == "hybrid-match.yml"                              # +10なので追試
seed1 = s["matches"] and a[0]["inputs"]["seed"]
s, a = L.step(s, ev("match", s, "302"), {"formal": F, "match": {"w": 59, "l": 48, "illegal": 0, "draw": 1}})
i = a[0]["inputs"]
assert a[0]["workflow"] == "hybrid-nnue-adopt.yml" and i["name"] == "E28" and i["run_prev"] == "301" and i["run_prev2"] == "302"
assert i["run_train"] == "201" and i["auto"] == "true" and i["run_kumoji"] == "none" and i["commit"] == "true"
new_formal = {"snapshot": "bbbb2222", "asset": "E28-trained-state-Adam.tar.gz"}
s, a = L.step(s, ev("adopt", s, "401"), {"formal": new_formal})
assert a[0]["workflow"] == "g1-gen.yml" and s["parent"] == "bbbb2222" and s["gens"] == [] and s["round"] == 2

# --- 学習のゲート失敗: 対局せず、生成を足して学習し直す(古いデータは残す)
s = started()
for r in ("101", "102"):
    s, a = L.step(s, ev("gen", s, r), {"formal": F})
s, a = L.step(s, ev("train", s, "201"), {"formal": F, "report": {"gate": "fail", "mean_label": 0.499}})
assert a[0]["workflow"] == "g1-gen.yml" and s["gens"] == [101, 102]
s, a = L.step(s, ev("gen", s, "103"), {"formal": F})
assert a[0]["workflow"] == "g1-train.yml" and a[0]["inputs"]["data_runs"] == "101 102 103"
for r in ("104",):
    s, a = L.step(s, ev("train", s, "202"), {"formal": F, "report": {"gate": "fail"}})
    s, a = L.step(s, ev("gen", s, r), {"formal": F})
assert a[0]["inputs"]["data_runs"] == "102 103 104"                       # 最新3回分まで

# --- ラベルの偏り: 対局しない
s = started()
for r in ("101", "102"):
    s, a = L.step(s, ev("gen", s, r), {"formal": F})
s, a = L.step(s, ev("train", s, "201"), {"formal": F, "report": {"gate": "pass", "mean_label": 0.73}})
assert a[0]["workflow"] == "g1-gen.yml"

# --- 不採用: データを捨てて新しい生成から
s = started()
for r in ("101", "102"):
    s, a = L.step(s, ev("gen", s, r), {"formal": F})
s, a = L.step(s, ev("train", s, "201"), {"formal": F, "report": {"gate": "pass", "mean_label": 0.5}})
s, a = L.step(s, ev("match", s, "301"), {"formal": F, "match": {"w": 41, "l": 66, "illegal": 0}})
assert a[0]["workflow"] == "g1-gen.yml" and s["gens"] == [] and s["round"] == 2

# --- 失敗の再起動と停止
s = started()
for k in range(L.MAX_RETRY):
    s, a = L.step(s, ev("gen", s, "9", result="failure"), {"formal": F})
    assert a and a[0]["workflow"] == "g1-gen.yml" and s["active"]
s, a = L.step(s, ev("gen", s, "9", result="failure"), {"formal": F})
assert not a and not s["active"] and s["stage"] == "stopped"

# --- 待っていない通知・停止中・親の変更
s = started()
s2, a = L.step(dict(s, history=[]), ev("gen", s, "5", tag="手動"), {"formal": F})
assert not a and s2["stage"] == "gen"
s2, a = L.step(dict(s, history=[]), ev("train", s, "5"), {"formal": F})            # 段階が違う
assert not a
s2, a = L.step(dict(s, history=[]), ev("gen", s, "5"), {"formal": {"snapshot": "cccc3333", "asset": "E28-x"}})
assert a[0]["workflow"] == "g1-gen.yml" and s2["parent"] == "cccc3333" and s2["gens"] == []   # 手動採用で親が替わった
s2, a = L.step(dict(s, history=[]), {"event": "stop", "run_id": "", "tag": "", "result": "success", "note": "試験"}, {"formal": F})
s3, a = L.step(s2, ev("gen", s, "5"), {"formal": F})
assert not a

# --- 引き継ぎ(boot): 走っている学習をそのまま待つ
s = L.new_state("aaaa1111", 34)
s.update(active=True, stage="train", tag="boot", gens=[11, 12])
s, a = L.step(s, ev("train", s, "777", tag="boot"), {"formal": F, "report": {"gate": "pass", "mean_label": 0.499, "heldout_before": 0.5, "heldout_after": 0.49}})
assert a[0]["workflow"] == "hybrid-match.yml" and a[0]["inputs"]["weights"] == "run:777"

# --- タグが毎回変わり、シードが重ならない
s = started(); tags = {s["tag"]}
seeds = set()
for r in range(5):
    s, a = L.step(s, ev("gen", s, str(100 + r)), {"formal": F})
    assert a[0]["inputs"]["loop_tag"] not in tags; tags.add(a[0]["inputs"]["loop_tag"])
    if "seed" in a[0]["inputs"]:
        assert a[0]["inputs"]["seed"] not in seeds; seeds.add(a[0]["inputs"]["seed"])
    if a[0]["workflow"] == "g1-train.yml":
        s, a = L.step(s, ev("train", s, "2"), {"formal": F, "report": {"gate": "fail"}})
        tags.add(a[0]["inputs"]["loop_tag"])
# --- ゲート失敗でも損失の悪化が微差なら、ゲートを通ったものと同じ基準で対局・採用する(2026-11-20 本人指示)
s = started()
for r in ("101", "102"):
    s, a = L.step(s, ev("gen", s, r), {"formal": F})
s, a = L.step(s, ev("train", s, "201"), {"formal": F, "report": {"gate": "fail", "mean_label": 0.499, "heldout_before": 0.55295, "heldout_after": 0.55323}})
assert a[0]["workflow"] == "hybrid-match.yml" and s["gate_fail"]
s, a = L.step(s, ev("match", s, "301"), {"formal": F, "match": {"w": 61, "l": 46, "illegal": 0}})
assert a[0]["workflow"] == "hybrid-nnue-adopt.yml"                               # +15で採用(単独+30は要らない)
s = started()
for r in ("101", "102"):
    s, a = L.step(s, ev("gen", s, r), {"formal": F})
s, a = L.step(s, ev("train", s, "201"), {"formal": F, "report": {"gate": "fail", "heldout_before": 0.55, "heldout_after": 0.56}})
assert a[0]["workflow"] == "g1-gen.yml"                                          # 悪化が大きい→対局せず
print("g1_loop_test: すべて合格")
