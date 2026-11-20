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


def tags(state):
    return list(state["trains"])


def rep(before=0.5500, after=0.5499, gate="pass", ml=0.499):
    return {"report": {"gate": gate, "mean_label": ml, "heldout_before": before, "heldout_after": after}}


def to_train(s, mode_gens):
    for r in mode_gens:
        s, a = L.step(s, ev("gen", s, r), {"formal": F})
    return s, a


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


# --- fast: 生成2回 → 3変種を並列で学習 → 検証損失が小さい順に対局 → 追試 → 採用 → 次はdeepの生成
s = started()
assert s["mode"] == "fast" and L.MODES["fast"]["movetime"] == "50"
s, a = L.step(s, ev("gen", s, "101"), {"formal": F})
assert a[0]["workflow"] == "g1-gen.yml" and s["gens"] == [101]            # 1回分では学習しない
s, a = L.step(s, ev("gen", s, "102"), {"formal": F})
assert [x["workflow"] for x in a] == ["g1-train.yml"] * 3 and [x["inputs"]["lr"] for x in a] == ["0.00001", "0.00002", "0.0001"]
assert all(x["inputs"]["data_runs"] == "101 102" for x in a) and len({x["inputs"]["loop_tag"] for x in a}) == 3
t1, t2, t3 = tags(s)
s, a = L.step(s, ev("train", s, "201", tag=t1), {"formal": F, **rep(0.5500, 0.5502)})     # 悪化+0.0002
assert not a
s, a = L.step(s, ev("train", s, "202", tag=t2), {"formal": F, **rep(0.5500, 0.5498)})     # 改善
assert not a
s, a = L.step(s, ev("train", s, "203", tag=t3), {"formal": F, **rep(0.5500, 0.5560, gate="fail")})   # 悪化が大きい→対象外
assert a[0]["workflow"] == "hybrid-match.yml" and a[0]["inputs"]["weights"] == "run:202"   # 損失が小さい変種から
assert [c["name"] for c in s["queue"]] == ["lr1e-5"] and s["cand"]["name"] == "lr2e-5"
s, a = L.step(s, ev("match", s, "301"), {"formal": F, "match": {"w": 59, "l": 49, "illegal": 0}})
assert a[0]["workflow"] == "hybrid-match.yml"                              # +10なので追試
s, a = L.step(s, ev("match", s, "302"), {"formal": F, "match": {"w": 59, "l": 48, "illegal": 0, "draw": 1}})
i = a[0]["inputs"]
assert a[0]["workflow"] == "hybrid-nnue-adopt.yml" and i["name"] == "E28" and i["run_train"] == "202" and i["run_prev"] == "301" and i["run_prev2"] == "302"
new_formal = {"snapshot": "bbbb2222", "asset": "E28-trained-state-Adam.tar.gz"}
s, a = L.step(s, ev("adopt", s, "401"), {"formal": new_formal})
assert a[0]["workflow"] == "g1-gen.yml" and s["parent"] == "bbbb2222" and s["gens"] == [] and s["round"] == 2
assert s["mode"] == "deep" and a[0]["inputs"]["movetime_ms"] == "150" and s["queue"] == [] and s["trains"] == {}

# --- deep: 生成3回分がそろってから、2変種を学習
for r in ("501", "502"):
    s, a = L.step(s, ev("gen", s, r), {"formal": new_formal})
    assert a[0]["workflow"] == "g1-gen.yml"
s, a = L.step(s, ev("gen", s, "503"), {"formal": new_formal})
assert len(a) == 2 and a[0]["inputs"]["data_runs"] == "501 502 503" and [x["inputs"]["lr"] for x in a] == ["0.00005", "0.00002"]

# --- 1つ目の変種が不採用なら、同じ周回の次の変種を対局。全部不採用ならデータを捨てて次の周回(fast)へ
s = started()
s, a = to_train(s, ["101", "102"])
t1, t2, t3 = tags(s)
for tg, run, aft in ((t1, "201", 0.5499), (t2, "202", 0.5498), (t3, "203", 0.5497)):
    s, a = L.step(s, ev("train", s, run, tag=tg), {"formal": F, **rep(0.5500, aft)})
assert s["cand"]["name"] == "lr1e-4" and len(s["queue"]) == 2                 # 最も損失が小さい変種から
s, a = L.step(s, ev("match", s, "301"), {"formal": F, "match": {"w": 41, "l": 66, "illegal": 0}})
assert a[0]["workflow"] == "hybrid-match.yml" and a[0]["inputs"]["weights"] == "run:202"
s, a = L.step(s, ev("match", s, "302"), {"formal": F, "match": {"w": 55, "l": 52, "illegal": 0}})
assert a[0]["inputs"]["weights"] == "run:201"
s, a = L.step(s, ev("match", s, "303"), {"formal": F, "match": {"w": 50, "l": 58, "illegal": 0}})
assert a[0]["workflow"] == "g1-gen.yml" and s["gens"] == [] and s["round"] == 2 and s["mode"] == "deep"

# --- すべての変種が対象外(悪化が大きい・偏り): 対局せず、生成を足して(データは残して)学習し直す
s = started()
s, a = to_train(s, ["101", "102"])
for tg, run in zip(tags(s), ("201", "202", "203")):
    s, a = L.step(s, ev("train", s, run, tag=tg), {"formal": F, **rep(0.55, 0.56, gate="fail")})
assert a[0]["workflow"] == "g1-gen.yml" and s["gens"] == [101, 102] and s["mode"] == "fast"
s, a = L.step(s, ev("gen", s, "103"), {"formal": F})
assert a[0]["inputs"]["data_runs"] == "101 102 103"

s = started()
s, a = to_train(s, ["101", "102"])
for tg, run in zip(tags(s), ("201", "202", "203")):
    s, a = L.step(s, ev("train", s, run, tag=tg), {"formal": F, **rep(ml=0.73)})
assert a[0]["workflow"] == "g1-gen.yml"                                       # 偏りは対局しない

# --- 変種1本の失敗は、その1本だけ再起動する。3回失敗したら、その変種は見送って残りで進める
s = started()
s, a = to_train(s, ["101", "102"])
t1, t2, t3 = tags(s)
s, a = L.step(s, ev("train", s, "9", result="failure", tag=t1), {"formal": F})
assert len(a) == 1 and a[0]["inputs"]["lr"] == "0.00001" and len(s["trains"]) == 3 and t1 not in s["trains"]
t1n = [k for k in s["trains"] if k not in (t2, t3)][0]
for k in range(L.MAX_RETRY):
    s, a = L.step(s, ev("train", s, "9", result="failure", tag=t1n), {"formal": F})
    t1n = [k for k in s["trains"] if k not in (t2, t3)][0] if a else t1n
assert s["trains"][t1n]["failed"] and not a                                    # 見送り。他の変種を待つ
s, a = L.step(s, ev("train", s, "202", tag=t2), {"formal": F, **rep(0.55, 0.5499)})
s, a = L.step(s, ev("train", s, "203", tag=t3), {"formal": F, **rep(0.55, 0.5498)})
assert a[0]["workflow"] == "hybrid-match.yml" and len(s["queue"]) == 1

# --- ゲート失敗でも微差なら、ゲートを通ったものと同じ基準で採用できる(+15)
s = started()
s, a = to_train(s, ["101", "102"])
for tg, run in zip(tags(s), ("201", "202", "203")):
    s, a = L.step(s, ev("train", s, run, tag=tg), {"formal": F, **rep(0.55295, 0.55323, gate="fail")})
s, a = L.step(s, ev("match", s, "301"), {"formal": F, "match": {"w": 61, "l": 46, "illegal": 0}})
assert a[0]["workflow"] == "hybrid-nnue-adopt.yml"

# --- 失敗の再起動と停止(生成)
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

# --- 古い状態ファイル(mode・trains などが無い)でも動く
old = {"schema": "g1-loop-v1", "active": True, "round": 38, "parent": "aaaa1111", "stage": "gen", "tag": "r38-gen-17", "gens": [], "train_run": None,
       "matches": [], "match_runs": [], "retries": 0, "seq": 17, "last_dispatch": None, "updated": "", "reason": "", "history": []}
old = L.normalize(old)
old, a = L.step(old, ev("gen", old, "1"), {"formal": F})
assert a[0]["workflow"] == "g1-gen.yml" and old["mode"] == "fast"
old, a = L.step(old, ev("gen", old, "2"), {"formal": F})
assert len(a) == 3

# --- resume: 学習中なら未完了の変種だけ再起動
s = started()
s, a = to_train(s, ["101", "102"])
t1, t2, t3 = tags(s)
s, a = L.step(s, ev("train", s, "201", tag=t1), {"formal": F, **rep()})
s, a = L.step(s, {"event": "resume", "run_id": "", "tag": "", "result": "success"}, {"formal": F})
assert len(a) == 2 and a[0]["workflow"] == "g1-train.yml"

# --- タグが毎回変わり、シードが重ならない
s = started(); seen = {s["tag"]}; seeds = set()
for r in range(6):
    s, a = L.step(s, ev("gen", s, str(100 + r)), {"formal": F})
    for x in a:
        assert x["inputs"]["loop_tag"] not in seen; seen.add(x["inputs"]["loop_tag"])
        if "seed" in x["inputs"]:
            assert x["inputs"]["seed"] not in seeds; seeds.add(x["inputs"]["seed"])
    if a[0]["workflow"] == "g1-train.yml":
        for tg in tags(s):
            s, a = L.step(s, ev("train", s, "2", tag=tg), {"formal": F, **rep(0.55, 0.56, gate="fail")})
        seen.add(a[0]["inputs"]["loop_tag"])
print("g1_loop_test: すべて合格")
