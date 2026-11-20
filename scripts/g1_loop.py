#!/usr/bin/env python3
"""G1周回の自動連鎖(2026-11-20、本人指示「高速に回数を重ねることが重要」)。

生成(g1-gen) → 学習(g1-train) → 対局(hybrid-match、必要なら追試) → 採用(hybrid-nnue-adopt) → 次の生成、を
人の確認待ちなしでつなぐ。各workflowの最後の job が、終わったことを g1-loop.yml に知らせ、このスクリプトが次を起動する。

判断の基準は、これまで人が手でやっていたものと同じ:
  - 生成は MIN_GENS 回分そろってから学習する(1回分だけの学習は周回31・32で続けて不採用・ゲート失敗だった)。
  - 学習のゲートが failなら対局せず、生成を足して(最新 MAX_GENS 回分まで)学習し直す。平均ラベルが偏っていたら(自己対局だけなら起きない)同じ。
  - 対局(108局): +12以上かつ違法手0 → 採用。+6以上 → 別シードで追試し、両方+6以上または合算+24以上で採用。
  - 採用できなければ使ったデータを捨てて、親はそのまま新しい生成から。
  - 採用したら親が替わるので、旧世代の生成は捨てて新しい親で生成し直す。
状態は loop-state ブランチの state.json に持つ(mainを動かさない。採用workflowが main の先頭を確認するため)。
Floodgateの記録は混ぜない(2026-11-20 本人方針。負け局300局たまってから別枠で判断)。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = os.environ.get("GITHUB_REPOSITORY", "mtk-ctrl/shogi-fukaji")
MIN_GENS = 2          # 学習に要る生成の回数(1回約530万局面)
MAX_GENS = 3          # 学習に使う生成の最大回数(最新から)
PASS_DIFF = 12        # 1本目でこれ以上なら採用
RETEST_DIFF = 6       # これ以上なら追試
COMBINED_DIFF = 24    # 2本の合算でこれ以上なら採用
LABEL_RANGE = (0.40, 0.70)   # 平均ラベルがこの外なら偏りとみなして対局しない
LOSS_SMALL = 0.001            # 検証損失の悪化がこれ未満の微差なら、損失ではなく対局の勝敗で決める(2026-11-20 本人指示。ゲートを通ったものと同じ基準)
MAX_RETRY = 2         # 失敗したときの再起動の最大回数
PAIRS = "54"          # 108局
GEN_MINUTES = "60"
GEN_MOVETIME = "50"
HISTORY_KEEP = 80


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jst_date() -> str:
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)).strftime("%Y-%m-%d")


# ------------------------------------------------------------------ 状態
def new_state(parent: str, round_no: int = 1) -> dict:
    return {"schema": "g1-loop-v1", "active": False, "round": round_no, "parent": parent, "stage": "idle", "tag": "",
            "gens": [], "train_run": None, "matches": [], "match_runs": [], "retries": 0, "seq": 0,
            "last_dispatch": None, "updated": now_iso(), "reason": "", "history": []}


def log(state: dict, note: str) -> None:
    state["history"].append({"t": now_iso(), "round": state["round"], "stage": state["stage"], "note": note})
    del state["history"][:-HISTORY_KEEP]


def next_tag(state: dict, stage: str) -> str:
    state["seq"] += 1
    return f"r{state['round']}-{stage}-{state['seq']}"


def seed_for(state: dict, kind: str) -> str:
    base = {"gen": 2026112100, "match": 202611300000}[kind]
    return str(base + state["seq"] * (1 if kind == "gen" else 10))


# ------------------------------------------------------------------ 起動する中身
def gen_action(state: dict) -> dict:
    tag = next_tag(state, "gen")
    state["stage"], state["tag"] = "gen", tag
    return {"workflow": "g1-gen.yml", "inputs": {"weights": "formal", "minutes": GEN_MINUTES, "movetime_ms": GEN_MOVETIME,
                                                 "opponent": "self", "seed": seed_for(state, "gen"), "loop_tag": tag}}


def train_action(state: dict) -> dict:
    tag = next_tag(state, "train")
    state["stage"], state["tag"] = "train", tag
    runs = " ".join(str(r) for r in state["gens"][-MAX_GENS:])
    return {"workflow": "g1-train.yml", "inputs": {"data_runs": runs, "feature_set": "band", "lr": "0.00005", "loop_tag": tag}}


def match_action(state: dict) -> dict:
    tag = next_tag(state, "match")
    state["stage"], state["tag"] = "match", tag
    return {"workflow": "hybrid-match.yml",
            "inputs": {"weights": f"run:{state['train_run']}", "opponent": "hybrid", "opp_weights": "formal", "scale": "formal",
                       "incremental": "0", "opp_scale": "same", "pairs": PAIRS, "seed": seed_for(state, "match"), "loop_tag": tag}}


def adopt_action(state: dict, name: str, prev: int, prev2: int | None) -> dict:
    tag = next_tag(state, "adopt")
    state["stage"], state["tag"] = "adopt", tag
    return {"workflow": "hybrid-nnue-adopt.yml",
            "inputs": {"name": name, "date": jst_date(), "commit": "true", "run_train": str(state["train_run"]), "run_prev": str(prev),
                       "run_prev2": str(prev2) if prev2 else "none", "run_kumoji": "none", "auto": "true", "new_inputs": "0",
                       "loop_tag": tag}}


# ------------------------------------------------------------------ 判定(純粋関数)
def judge(matches: list[dict]) -> str:
    """matches=[{run,w,l,illegal}]。戻り値: adopt / retest / reject。"""
    if any(m["illegal"] for m in matches):
        return "reject"
    diffs = [m["w"] - m["l"] for m in matches]
    if len(matches) == 1:
        if diffs[0] >= PASS_DIFF:
            return "adopt"
        return "retest" if diffs[0] >= RETEST_DIFF else "reject"
    if all(d >= RETEST_DIFF for d in diffs) or sum(diffs) >= COMBINED_DIFF:
        return "adopt"
    return "reject"


def next_name(formal: dict) -> str:
    m = re.match(r"E(\d+)-", formal["asset"])
    return f"E{int(m.group(1)) + 1}" if m else "E?"


def step(state: dict, ev: dict, info: dict) -> tuple[dict, list[dict]]:
    """1つの完了通知で状態を進める。ev={event,run_id,tag,result}。info={formal:{snapshot,asset}, report:{}, match:{}}。"""
    actions: list[dict] = []
    kind = ev["event"]
    formal = info.get("formal") or {}
    if kind == "start":
        state["active"], state["reason"], state["retries"] = True, "", 0
        state["parent"] = formal.get("snapshot", state["parent"])
        state["gens"], state["matches"], state["match_runs"], state["train_run"] = [], [], [], None
        log(state, "開始")
        return state, [gen_action(state)]
    if kind == "stop":
        state["active"], state["stage"], state["reason"] = False, "stopped", ev.get("note", "手動停止")
        log(state, "停止: " + state["reason"])
        return state, []
    if not state["active"]:
        log(state, f"停止中のため{kind}を無視")
        return state, []
    if kind == "resume":   # 通知が欠けたとき、いまのstageをやり直す
        state["retries"] = 0
        return state, [relaunch(state)]
    # 通知が今待っている run のものか
    if ev["tag"] != state["tag"] and not (ev["tag"] == "boot" and state["tag"] == "boot"):
        log(state, f"待っていない通知を無視({kind} tag={ev['tag']} 待ち={state['tag']})")
        return state, []
    want = {"gen": "gen", "train": "train", "match": "match", "adopt": "adopt"}[state["stage"]]
    if kind != want:
        log(state, f"段階が違う通知を無視({kind}, 待ち={want})")
        return state, []
    if ev["result"] != "success":
        state["retries"] += 1
        if state["retries"] > MAX_RETRY:
            state["active"], state["reason"] = False, f"{kind}が{MAX_RETRY + 1}回失敗(run {ev['run_id']}, {ev['result']})"
            state["stage"] = "stopped"
            log(state, "停止: " + state["reason"])
            return state, []
        log(state, f"{kind}が失敗({ev['result']}, run {ev['run_id']})。再起動{state['retries']}回目")
        return state, [relaunch(state)]
    state["retries"] = 0
    snapshot = formal.get("snapshot")
    # 手動の採用などで親が替わっていたら、旧世代のデータは捨てて新しい親から
    if snapshot and state["parent"] and snapshot != state["parent"] and kind != "adopt":
        log(state, f"正式の親が{state['parent'][:8]}から{snapshot[:8]}に替わった。旧世代のデータを捨てて生成からやり直す")
        state.update(parent=snapshot, gens=[], matches=[], match_runs=[], train_run=None)
        return state, [gen_action(state)]

    if kind == "gen":
        state["gens"].append(int(ev["run_id"]))
        state["gens"] = state["gens"][-MAX_GENS:]
        if len(state["gens"]) < MIN_GENS:
            log(state, f"生成完了(run {ev['run_id']})。{len(state['gens'])}/{MIN_GENS}回分。もう1回生成")
            return state, [gen_action(state)]
        log(state, f"生成完了(run {ev['run_id']})。{len(state['gens'])}回分で学習へ")
        return state, [train_action(state)]

    if kind == "train":
        rep = info.get("report") or {}
        state["train_run"] = int(ev["run_id"])
        ml = rep.get("mean_label")
        biased = ml is not None and not (LABEL_RANGE[0] <= ml <= LABEL_RANGE[1])
        hb, ha = rep.get("heldout_before"), rep.get("heldout_after")
        gate_ok = rep.get("gate") == "pass"
        small = (not gate_ok) and hb is not None and ha is not None and (ha - hb) < LOSS_SMALL
        if biased or (not gate_ok and not small):
            why = f"平均ラベル{ml:.3f}が偏っている" if biased else "ゲート失敗(損失の悪化が大きい)"
            log(state, f"学習(run {ev['run_id']}): {why}。対局せず、生成を足して学習し直す")
            return state, [gen_action(state)]
        state["gate_fail"] = not gate_ok
        state["matches"], state["match_runs"] = [], []
        loss = f"検証損失{hb:.5f}→{ha:.5f}" if hb is not None and ha is not None else "検証損失の記録なし"
        log(state, f"学習(run {ev['run_id']}): ゲート{'pass' if gate_ok else '失敗(悪化が微差なので勝敗で決める)'}、{loss}。対局へ")
        return state, [match_action(state)]

    if kind == "match":
        m = info["match"]
        state["matches"].append({"run": int(ev["run_id"]), "w": m["w"], "l": m["l"], "illegal": m["illegal"], "draw": m.get("draw", 0)})
        verdict = judge(state["matches"])
        ms = " / ".join(f"{x['w']}-{x['l']}" + (f"-{x['draw']}" if x.get("draw") else "") for x in state["matches"])
        if verdict == "adopt":
            name = next_name(formal)
            log(state, f"対局 {ms}、違法手{sum(x['illegal'] for x in state['matches'])}。基準を満たした。{name}として採用")
            prev2 = state["matches"][1]["run"] if len(state["matches"]) > 1 else None
            return state, [adopt_action(state, name, state["matches"][0]["run"], prev2)]
        if verdict == "retest":
            log(state, f"対局 {ms}。基準に届かないが+6以上のため追試")
            return state, [match_action(state)]
        log(state, f"対局 {ms}、違法手{sum(x['illegal'] for x in state['matches'])}。不採用。データを捨てて新しい生成から")
        state.update(gens=[], matches=[], match_runs=[], train_run=None, gate_fail=False, round=state["round"] + 1)
        return state, [gen_action(state)]

    if kind == "adopt":
        state["parent"] = snapshot or state["parent"]
        log(state, f"採用完了。新しい親は{state['parent'][:8]}")
        state.update(gens=[], matches=[], match_runs=[], train_run=None, gate_fail=False, round=state["round"] + 1)
        return state, [gen_action(state)]
    raise ValueError(kind)


def relaunch(state: dict) -> dict:
    """失敗・欠落のとき、いまのstageの起動をやり直す(タグだけ新しくする)。"""
    stage = state["stage"]
    if stage == "gen":
        return gen_action(state)
    if stage == "train":
        return train_action(state)
    if stage == "match":
        return match_action(state)
    raise RuntimeError(f"relaunch不可: {stage}")   # adoptは自動で再起動しない(main書き換えのため)


# ------------------------------------------------------------------ 実世界との接続(gh)
def gh(*args: str, check: bool = True) -> str:
    p = subprocess.run(["gh", *args], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:4])} 失敗: {p.stderr.strip()[:300]}")
    return p.stdout


def read_formal() -> dict:
    raw = gh("api", f"repos/{REPO}/contents/formal/model.json?ref=main", "--jq", ".content")
    import base64
    m = json.loads(base64.b64decode(raw))
    h = m["hybrid"]
    return {"snapshot": h["research_snapshot_sha256"], "asset": h["state_release_asset"]}


def read_report(run_id: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        gh("run", "download", str(run_id), "-n", f"e2-trained-{run_id}", "-D", d, "-R", REPO)
        return json.loads((Path(d) / "report.json").read_text(encoding="utf-8"))


def read_match(run_id: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        gh("run", "download", str(run_id), "-n", f"hybrid-result-{run_id}", "-D", d, "-R", REPO)
        r = json.loads((Path(d) / "result.json").read_text(encoding="utf-8"))
    return {"w": int(r["fukaji_wins"]), "l": int(r["kumoji_wins"]), "illegal": len(r.get("illegal_pairs") or []),
            "draw": int(r.get("draws_or_unresolved", 0))}


def dispatch(a: dict, dry: bool) -> None:
    args = ["api", "-X", "POST", f"repos/{REPO}/actions/workflows/{a['workflow']}/dispatches", "-f", "ref=main"]
    for k, v in a["inputs"].items():
        args += ["-f", f"inputs[{k}]={v}"]
    if dry:
        print("DRY dispatch", a["workflow"], json.dumps(a["inputs"], ensure_ascii=False))
        return
    gh(*args)
    print("dispatched", a["workflow"], a["inputs"].get("loop_tag"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--event", required=True, choices=["gen", "train", "match", "adopt", "start", "stop", "resume", "status"])
    ap.add_argument("--run-id", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--result", default="success")
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sp = Path(a.state)
    formal = read_formal()
    state = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else new_state(formal["snapshot"])
    if a.event == "status":
        print(json.dumps({k: v for k, v in state.items() if k != "history"}, ensure_ascii=False, indent=1))
        print("\n".join(f"{h['t']} r{h['round']} {h['note']}" for h in state["history"][-12:]))
        return 0
    ev = {"event": a.event, "run_id": a.run_id, "tag": a.tag, "result": a.result, "note": a.note}
    info: dict = {"formal": formal}
    waiting = state["active"] and a.tag == state["tag"] and a.result == "success"
    if waiting and a.event == "train":
        info["report"] = read_report(a.run_id)
    if waiting and a.event == "match":
        info["match"] = read_match(a.run_id)
    state, actions = step(state, ev, info)
    state["updated"] = now_iso()
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for act in actions:
        dispatch(act, a.dry_run)
    print(f"stage={state['stage']} tag={state['tag']} active={state['active']} round={state['round']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
