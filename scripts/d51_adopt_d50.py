"""D51: automatically adopt owner-authorized D50 NNUE after verified paired200 win.

The source/USI search backend remains unchanged (D49 PVS+exact mate).
This is a one-shot guarded preparation script; CI must pass and commit atomically.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fukaji.online_learning import load_snapshot
from fukaji.formal_model import load as formal_load

OLD="941598f2a9ec926d5f67dadf03582ccef718a56a843070dc1f576cc7eeca44e8"
NEW="641f36677b8d93c3237cfc4f0b6329a6beee55cff692302f529b3899ea74afd0"
OLD_WEIGHTS="9faf66a9d1fee9d282e35835628377ab7951bb94335a5d0498384bf88f4d2c26"
CPP="c9340af265368547c37cbecc311d8d3982900885b433d1b4ccdc27fbdd7d041a"
GAMES_SHA="92a27e9ea5094157e1213086966bfae9d3921c0537e1506b3563279bc63faf76"
TRAIN_SHA="a6df97b104302a4b7c9f570c289af0d001155eea44f3a7c9bb0f21184a2b3688"
AUDIT_SHA="f6acc785531eab92e199dfb5df4d4a64f5e55246c08ec0962e24b6123fb81324"
RUN=42343709475
MODEL_PATH="models/fukaji-d50.nnue.npz"
DECISION="docs/decisions/2026-11-07_D51_D50学習後重みを現行採用.md"

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read_archive(path,expected):
    if sha(path.read_bytes())!=expected:
        raise ValueError("archive sha mismatch: "+str(path))
    return tarfile.open(path,"r:gz")

def member(t,name):
    f=t.extractfile(name)
    if f is None:raise ValueError("missing "+name)
    return f.read()

def write_json(path,object):
    p=ROOT/path
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(object,ensure_ascii=False,indent=2)+"\n")

def modify(path,before,after):
    p=ROOT/path
    s=p.read_text()
    if s.count(before)!=1:raise ValueError("unexpected document: "+path+" marker "+before[:70]+" count="+str(s.count(before)))
    p.write_text(s.replace(before,after))

def append(path,block):
    p=ROOT/path
    p.write_text(p.read_text().rstrip()+"\n\n"+block.strip()+"\n")

def change_section(path,start,end,content):
    p=ROOT/path
    s=p.read_text()
    if s.count(start)!=1:raise ValueError("section start not unique "+path)
    a=s.index(start)
    b=s.index(end,a)
    p.write_text(s[:a]+content+s[b:])

def prepare(trained,audit,out):
    request=json.loads((ROOT/"research/d51-adopt/START.json").read_text())
    expected={
        "enabled":True,"request_id":"d51-d50-20261107-auto-adopt-on-audit-win",
        "source_run":RUN,"former":OLD,"next":NEW,
        "games_trained":1000,"audit_games":200,
        "after_wins":159,"before_wins":41,"draws":0,"unresolved":0,
        "optimizer_before":3794,"optimizer_after":4702,
        "train_sha256":TRAIN_SHA,"audit_sha256":AUDIT_SHA,
        "search_sha256":CPP,"version":"FUKAJI v0.4.1",
        "authority":"F00/F20 preauthorized direct current NNUE continuation when stronger"
    }
    if request!=expected:raise ValueError("not exact D51 adoption request")
    manifest_path=ROOT/"formal/model.json"
    spec=json.loads(manifest_path.read_text())
    assert (spec["version"]=="FUKAJI v0.4.1" and
            spec["search_backend_variant"]=="d48_pvs_mate" and
            spec["search_implementation_sha256"]==CPP and
            spec["weights_sha256"]==OLD_WEIGHTS and
            spec["research_snapshot_sha256"]==OLD and
            spec["optimizer_step"]==3794 and
            spec["weights_file"]=="models/fukaji-d45.nnue.npz"),"old formal model changed"
    if sha((ROOT/"native/nnue_search_ffi.cpp").read_bytes())!=CPP:
        raise ValueError("D49 official native search changed")
    if sha((ROOT/spec["weights_file"]).read_bytes())!=OLD_WEIGHTS:
        raise ValueError("existing D45 official weights changed")
    contract_path="research/d17-fast-self-learning/formal-contract.json"
    contract=json.loads((ROOT/contract_path).read_text())
    for name,value in contract.items():
        if sha((ROOT/name).read_bytes())!=value:
            raise ValueError("original formal contract drift: "+name)
    out.mkdir(parents=True,exist_ok=True)
    model_dir=out/"models"
    model_dir.mkdir(exist_ok=True)
    with read_archive(trained,TRAIN_SHA) as t:
        report=json.loads(member(t,"report.json"))
        requirements={
            "schema":"d50-learned1000-v1","baseline":OLD,"trained_sha256":NEW,
            "optimizer_step_before":3794,"optimizer_step_after":4702,
            "new_games":1000,"resolved_new_games":1000,
            "old_bootstrap_replay_games":665,"prior_d45_replay_games":300,
            "source_sha256":GAMES_SHA,"formal_search_variant":"d48_pvs_mate",
            "strength_proven":False,"formal_model_changed":False
        }
        for key,val in requirements.items():
            if report.get(key)!=val:raise ValueError("D50 training mismatch "+key)
        if report.get("fit",{}).get("positions")!=58077:
            raise ValueError("unexpected D50 training position count")
        for ext in ("npz","json"):
            (model_dir/(NEW+"."+ext)).write_bytes(member(t,NEW+"."+ext))
    model,optimizer,_=load_snapshot(model_dir,NEW)
    if optimizer is None or optimizer["step"]!=4702:
        raise ValueError("full D50 optimizer not loaded")
    if model.inputs!=2548 or model.hidden_weights.shape!=(2548,32):
        raise ValueError("D50 NNUE architecture drift")

    with read_archive(audit,AUDIT_SHA) as t:
        result=json.loads(member(t,"report.json"))
        plan=json.loads(member(t,"plan.json"))
        requirements={
            "schema":"d50-paired200-result-v1","before":OLD,"after":NEW,
            "games":200,"opening_pairs":100,"after_wins":159,"before_wins":41,
            "draws":0,"unresolved":0,"difference":118,
            "predeclared_verdict":"after_better","validated_plies":20331,
            "formal_changed":False
        }
        for key,val in requirements.items():
            if result.get(key)!=val:raise ValueError("D50 audit result mismatch "+key)
        if (plan["schema"]!="d50-paired200-v1" or
            plan["before"]!=OLD or plan["after"]!=NEW or
            plan["pairs"]!=100 or plan["games"]!=200 or
            len(plan["families"])!=100 or
            len({x["parent_id"] for x in plan["families"]})!=100):
            raise ValueError("D50 frozen audit plan drift")
        decision=plan.get("decision",{})
        if decision.get("margin_win_count")!=12 or decision.get("max_unresolved")!=10:
            raise ValueError("D50 predefined admission standard moved")
        conditions=plan.get("conditions",{})
        for key,val in {"backend":"cpp_nnue","normal_ms":200,"long_ms":1000,
                        "max_long_per_side":10,"quiescence":0}.items():
            if conditions.get(key)!=val:raise ValueError("D50 clock or engine changed: "+key)
        hashes=result.get("game_sha256",{})
        if len(hashes)!=200:raise ValueError("D50 audit games missing")
        for name,expected_sha in hashes.items():
            if sha(member(t,"games/"+name))!=expected_sha:
                raise ValueError("D50 game SHA mismatch "+name)

    dest=ROOT/MODEL_PATH
    if dest.exists():raise ValueError("refusing to overwrite D50 official model")
    buf=io.BytesIO()
    np.savez_compressed(buf,w=model.hidden_weights,b=model.hidden_bias,
                        v=model.output_weights,
                        o=np.array([model.output_bias],dtype=np.float32))
    raw=buf.getvalue()
    dest.write_bytes(raw)
    weights_sha=sha(raw)
    with np.load(io.BytesIO(raw),allow_pickle=False) as arr:
        if set(arr.files)!={"w","b","v","o"}:raise ValueError("unapproved NNUE tensor")
        for name,value in (("w",model.hidden_weights),("b",model.hidden_bias),
                           ("v",model.output_weights)):
            np.testing.assert_array_equal(arr[name],value)
            if not np.isfinite(arr[name]).all():raise ValueError("nonfinite NNUE weight")
        assert float(arr["o"][0])==float(model.output_bias)

    after=dict(spec)
    after.update({
       "approved_on":"2026-11-07",
       "approved_by":"owner-delegated F00/F20 current NNUE update after D50 159-41 paired200",
       "research_snapshot_sha256":NEW,
       "research_parent_sha256":OLD,
       "weights_file":MODEL_PATH,
       "weights_sha256":weights_sha,
       "optimizer_step":4702,
       "training_source_sha256":GAMES_SHA,
       "training_release_sha256":TRAIN_SHA,
       "comparison_release_sha256":AUDIT_SHA,
       "model_distribution":"GitHub models/fukaji-d50.nnue.npz plus D51 current release; D50 full optimizer and state are in training release.",
       "validation":{
           "opponent":"D45 before D50 selfplay 1000 (941598f2); 100 new paired openings with both colors",
           "wins":159,"losses":41,"draws":0,"unresolved":0,
           "opening_families":100,"games":200,
           "planned_opening_families":100,"planned_games":200,
           "normal_move_time_ms":200,"longthink_ms":1000,
           "longthink_max_per_side":10,"quiescence":0,
           "search_backend_variant":"d48_pvs_mate",
           "strength_conclusion":"D50 after 1000 PVS+mate selfplay games wins 159-41 over D45 in matched v0.4.1 paired200; predefined margin passed"
        }
    })
    for key in ("version","architecture","backend","quiescence","default_move_time_ms",
                "default_search_depth","search_backend_variant","search_implementation_sha256","search_adoption"):
        if after[key]!=spec[key]:raise ValueError("not authorized search / USI change: "+key)
    write_json("formal/model.json",after)
    loaded,loaded_spec=formal_load()
    if loaded_spec["weights_sha256"]!=weights_sha:raise ValueError("new formal loader failed")
    np.testing.assert_array_equal(loaded.hidden_weights,model.hidden_weights)
    np.testing.assert_array_equal(loaded.output_weights,model.output_weights)

    decision_path=ROOT/DECISION
    if decision_path.exists():raise ValueError("decision already exists")
    decision_path.write_text(f"""# D51：D50の自己学習後NNUEを現行モデルに採用（2026-11-07）

## 委任と結論
F00/F20の既存本人指示：「現行NNUEを直接学習し、同条件200局で改善すれば重みだけ正式化」に従い、D50の1000局強化と独立200局比較が採用条件を満たしたため、D50重みを現行へ昇格。本人はD50第3周の1000局自己学習を2026-11-07に明示指示。**探索コード、USI v0.4.1、詰みのルール、NNUE構造は変更しない**。

## 比較実測
- [D50 Actions実run](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/{RUN}) success、全125ジョブ（100訓練＋20比較＋5準備集約等）。1000局決着1000、未決着0。新100独立序盤×先後入替え200局を全20ランナーで実施し**新D50 159勝・旧D45 41勝、引分/未決着とも0**。20,331手合法再生。事前基準勝数差12以上・未決着10以下を満たす（差+118）。200ms＋長考1000ms最大10回/陣営、双方同じPVS＋1/3/5手詰み探索。
- 新NNUE snapshot SHA256 \`{NEW}\`、optimizer **Adam3794→4702**。学習58,077局面、2epoch。旧bootstrap665と旧D45train300を許可済みreplayとして利用。独立モデル系列は作らず更新前D45 snapshotを保全。
- [完全NNUE＋optimizer Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d50-d45-trained-{RUN}) SHA256 \`{TRAIN_SHA}\`。[1000局全原棋譜](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d50-d45-1000-selfplay-{RUN}) SHA256 \`{GAMES_SHA}\`。[200局全原棋譜・比較](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d50-d45-before-after-200games-{RUN}) SHA256 \`{AUDIT_SHA}\`。いずれも元D50 ActionsでRelease再取得SHAを照合。
- 正式推論用 \`{MODEL_PATH}\` SHA256 \`{weights_sha}\`。旧D45推論用 \`models/fukaji-d45.nnue.npz\` と完全optimizerは不変更で復元可能。

## 認可範囲・検査
本人委任の範囲はNNUE重み更新まで。PVS＋詰み探索C++ SHA256 \`{CPP}\` とFUKAJI v0.4.1は不変。D51 Actionsで契約整合、回帰、実USI、1手詰み、隔離配布物、Release資産のSHA再取得を通して初めて現行化完了とする。CI失敗なら旧D45が現行のまま。学習後の159勝41敗はこの評価分布での強さの証拠であって未知の相手や時間制御で同じ勝率を保証しない。
""")

    specdoc="docs/現行エンジン仕様.md"
    modify(specdoc,
       "更新：2026-11-07。**現行のNNUE重みはD45（941598f2、Adam3794）**。D43から第2周の1000局自己対局で追加学習し、200ms＋長考1秒10回の200局直接比較で**155勝44敗・未決着1局**。D49でPVS＋詰み探索を正式採用し、USI versionはFUKAJI v0.4.1。NNUE重み不変。",
       "更新：2026-11-07。**現行のNNUE重みはD50（641f3667、Adam4702）**。現行D45からPVS＋詰み探索で第3周1000局を自己対局し、標準200局比較で**159勝41敗・引分/未決着0局**。USI version FUKAJI v0.4.1・D49正式PVS＋詰み探索は不変。")
    modify(specdoc,f"\`{OLD}\`",f"\`{NEW}\`")
    modify(specdoc,"\`models/fukaji-d45.nnue.npz\`",f"\`{MODEL_PATH}\`")
    modify(specdoc,f"\`{OLD_WEIGHTS}\`",f"\`{weights_sha}\`")
    change_section(specdoc,"## 現行モデルの採用判断と強さの不確実性",
                   "## 復元・研究との境界",
                   f"""## 現行モデルの採用判断と強さの不確実性

[D51正式採用判断](decisions/2026-11-07_D51_D50学習後重みを現行採用.md)。旧現行D45/Adam3794で新規自己対局1000局（全1000決着）を実施してD50/Adam4702へ追加学習。新規100序盤×先後交替200局、双方正式v0.4.1（PVS＋詰み1/3/5手＋被1手詰みガード）、200ms/1秒長考10回で**新D50が159勝41敗・引分0・未決着0**。20,331手合法検証。学習前後の相手・時計が限定された比較であり未知の分布での勝率保証ではない。原棋譜は[D50比較Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d50-d45-before-after-200games-{RUN})で不変保全。

""")
    modify(specdoc,"次のNNUE自己学習は現行D45（Adam3794）のGitHub保存済み完全stateから開始し、更新前snapshotと更新後を同条件で対局比較する。",
       "次のNNUE自己学習は現行D50（Adam4702）のGitHub保存済み完全stateから開始し、更新前snapshot D50と更新後を同条件で対局比較する。")
    modify(specdoc,"失敗・悪化時は更新前D45へ戻す。","失敗・悪化時は更新前D50へ戻す。")
    append(specdoc,f"""## D51：正式D50 NNUEを採用し探索は不変

- 現行NNUE snapshot \`{NEW}\`、推論 \`{MODEL_PATH}\` SHA256 \`{weights_sha}\`、Adam4702。追加学習の完全stateはD50学習Release。正式NNUE差替え以外のUSI v0.4.1、PVS＋詰み探索、時計・機能・仕様は変更しない。
- [D51正式採用記録](decisions/2026-11-07_D51_D50学習後重みを現行採用.md)、CIの正式配布・実USI・SHA読戻しの検証結果を最終根拠にする。旧D45 v0.4.1配布を復元用に保持。""")
    # Update authoritative current section; historical D49 notes remain as history.
    change_section("docs/00_開発ルーター.md","## 現行モデル","## 現在の研究・保存再開点",
        f"""## 現行モデル

**現行NNUE D50（\`{NEW}\`、Adam4702）**。D45から新規1000局自己対局・直接学習し、同一正式PVS＋詰み探索を用いる200局標準比較でD50が159勝41敗、未決着0で採用。**正式 USI FUKAJI v0.4.1、探索はD49採用仕様のまま**。[D51採用判断](decisions/2026-11-07_D51_D50学習後重みを現行採用.md)、[D50完全学習state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d50-d45-trained-{RUN})、[現行仕様](現行エンジン仕様.md)、[manifest](../formal/model.json)を正本とする。次周はD50/Adam4702が唯一の学習親。旧D45・D43は不変snapshotと復元専用。

""")
    append("docs/00_開発ルーター.md",
        """## 最新NNUE・探索の関係（D51、2026-11-07）

最新**D50/Adam4702 NNUE**（2026-11-07追加1000局、200局比較159勝41敗）、正式**FUKAJI v0.4.1 PVS＋詰み探索**。上のD49におけるD45維持は当時の歴史的状態。次周はD50完全optimizerからのみ継続。[D51判断](decisions/2026-11-07_D51_D50学習後重みを現行採用.md)。""")
    append("AGENTS.md",
       f"""## 最新現行NNUE（D51、2026-11-07）

**現行はD50/Adam4702、snapshot \`{NEW}\`。** 現行NNUE D45（Adam3794）から自己対局1000局＋比較200局で159勝41敗となり、F00/F20の本人委任に従ってNNUE重みのみ正式反映。**FUKAJI v0.4.1、D49のPVS＋1/3/5手詰み探索はそのまま**。上部のD45/旧版表示はその時点の歴史的記述。次周は[D50完全学習state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d50-d45-trained-{RUN})のAdam4702を唯一の学習親とする。[判断](docs/decisions/2026-11-07_D51_D50学習後重みを現行採用.md)。""")
    change_section("docs/ROADMAP.md","## 現行モデル","## 現在の研究",
        f"""## 現行モデル

**現行D50 NNUE（\`{NEW}\`、Adam4702）、USI FUKAJI v0.4.1・PVS＋詰み探索は継続。** 旧D45/Adam3794の1000局自己対局から追加学習し、新規100序盤×先後200局でD50が**159勝41敗、未決着0**。[D51採用判断](decisions/2026-11-07_D51_D50学習後重みを現行採用.md)／[D50実行](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/{RUN})／[D50完全学習state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d50-d45-trained-{RUN})／[manifest](../formal/model.json)／[現行仕様](現行エンジン仕様.md)。次周の学習親は**D50/Adam4702のみ**。旧D45・D43は復元用に保持。

""")
    append("docs/ROADMAP.md",
        f"""## D51：D50の正式採用（2026-11-07）

- [D50学習・200局直接比較](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/{RUN})：1000局全決着、58,077学習局面、Adam3794→4702、**D50 159勝D45 41勝、未決着0**。事前基準通過。
- 正式NNUEを \`{NEW}\` へ更新（推論SHA \`{weights_sha}\`）。D49のPVS＋詰み探索とUSI v0.4.1は維持。[採用判断](decisions/2026-11-07_D51_D50学習後重みを現行採用.md)。契約・回帰・実USI・隔離bundle・Release SHAの通過をD51 Actionsで必須とし、正式配布の読戻しを検証する。""")
    # Only change hash locks that actually need to change.
    for name in ("formal/model.json",specdoc):
        contract[name]=sha((ROOT/name).read_bytes())
    write_json(contract_path,contract)
    loaded2,loaded_spec2=formal_load()
    assert loaded_spec2["weights_sha256"]==weights_sha
    assert sha((ROOT/"native/nnue_search_ffi.cpp").read_bytes())==CPP
    assert sha((ROOT/"models/fukaji-d45.nnue.npz").read_bytes())==OLD_WEIGHTS
    receipt={
        "schema":"fukaji-d51-nnue-adopt-prepare-v1",
        "current_snapshot_sha256":NEW,"former_snapshot_sha256":OLD,
        "optimizer_step":4702,"weights_path":MODEL_PATH,
        "weights_sha256":weights_sha,"unchanged_cpp_sha256":CPP,
        "usi_version":spec["version"],"train_archive_sha256":TRAIN_SHA,
        "audit_archive_sha256":AUDIT_SHA,"after_wins":159,"before_wins":41,
        "draws":0,"unresolved":0,"training_games":1000,
        "audit_games":200,"validated_plies":20331,
        "contract_sha256":sha((ROOT/contract_path).read_bytes())
    }
    write_json(out.relative_to(ROOT)/"receipt.json",receipt)
    print(json.dumps(receipt))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--trained",type=Path,required=True)
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    prepare(a.trained,a.audit,a.out)
