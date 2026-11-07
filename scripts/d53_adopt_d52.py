"""D53: adopt owner-authorized D52 NNUE after complete paired200 win.

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

OLD="641f36677b8d93c3237cfc4f0b6329a6beee55cff692302f529b3899ea74afd0"
NEW="55ccc97db6c2d4ce9ff7c75f5e81d543fb6f12d169639c3ea6d24c89e95342ae"
OLD_WEIGHTS="fbabd6b98de1d03ff8755733bb6a5a5daac7a0fd1f2f2c11ff2b21c26836c599"
CPP="c9340af265368547c37cbecc311d8d3982900885b433d1b4ccdc27fbdd7d041a"
GAMES_SHA="d8adb89bff7d3cd2dc83c994f6664ce73ef7798e6326cc6a893a2ef2ca1d2134"
TRAIN_SHA="65acb35edd7e19a29b86c9cd321e245ad98cfa78f471f53e5e083f3ee473f793"
AUDIT_SHA="5f05369e2ad7edec6aa4098a9393889c989697e83a607cdc062967049630a542"
RUN=42401026786
MODEL_PATH="models/fukaji-d52.nnue.npz"
DECISION="docs/decisions/2026-11-08_D53_D52学習後重みを現行採用.md"

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
    request=json.loads((ROOT/"research/d53-adopt/START.json").read_text())
    expected={
        "enabled":True,"request_id":"d53-d52-20261108-auto-adopt-on-audit-win",
        "source_run":RUN,"former":OLD,"next":NEW,
        "games_trained":1000,"audit_games":200,
        "after_wins":156,"before_wins":44,"draws":0,"unresolved":0,
        "optimizer_before":4702,"optimizer_after":5606,
        "train_sha256":TRAIN_SHA,"audit_sha256":AUDIT_SHA,
        "search_sha256":CPP,"version":"FUKAJI v0.4.1",
        "authority":"F00/F20 preauthorized direct current NNUE continuation when stronger"
    }
    if request!=expected:raise ValueError("not exact D53 adoption request")
    manifest_path=ROOT/"formal/model.json"
    spec=json.loads(manifest_path.read_text())
    assert (spec["version"]=="FUKAJI v0.4.1" and
            spec["search_backend_variant"]=="d48_pvs_mate" and
            spec["search_implementation_sha256"]==CPP and
            spec["weights_sha256"]==OLD_WEIGHTS and
            spec["research_snapshot_sha256"]==OLD and
            spec["optimizer_step"]==4702 and
            spec["weights_file"]=="models/fukaji-d50.nnue.npz"),"old formal model changed"
    if sha((ROOT/"native/nnue_search_ffi.cpp").read_bytes())!=CPP:
        raise ValueError("D49 official native search changed")
    if sha((ROOT/spec["weights_file"]).read_bytes())!=OLD_WEIGHTS:
        raise ValueError("existing D50 official weights changed")
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
            "schema":"d52-learned1000-v1","baseline":OLD,"trained_sha256":NEW,
            "optimizer_step_before":4702,"optimizer_step_after":5606,
            "new_games":1000,"resolved_new_games":1000,
            "old_bootstrap_replay_games":665,"prior_d50_replay_games":300,
            "source_sha256":GAMES_SHA,"formal_search_variant":"d48_pvs_mate",
            "strength_proven":False,"formal_model_changed":False
        }
        for key,val in requirements.items():
            if report.get(key)!=val:raise ValueError("D52 training mismatch "+key)
        if report.get("fit",{}).get("positions")!=57755:
            raise ValueError("unexpected D50 training position count")
        for ext in ("npz","json"):
            (model_dir/(NEW+"."+ext)).write_bytes(member(t,NEW+"."+ext))
    model,optimizer,_=load_snapshot(model_dir,NEW)
    if optimizer is None or optimizer["step"]!=5606:
        raise ValueError("full D52 optimizer not loaded")
    if model.inputs!=2548 or model.hidden_weights.shape!=(2548,32):
        raise ValueError("D52 NNUE architecture drift")

    with read_archive(audit,AUDIT_SHA) as t:
        result=json.loads(member(t,"report.json"))
        plan=json.loads(member(t,"plan.json"))
        requirements={
            "schema":"d52-paired200-result-v1","before":OLD,"after":NEW,
            "games":200,"opening_pairs":100,"after_wins":156,"before_wins":44,
            "draws":0,"unresolved":0,"difference":112,
            "predeclared_verdict":"after_better","validated_plies":19104,
            "formal_changed":False
        }
        for key,val in requirements.items():
            if result.get(key)!=val:raise ValueError("D52 audit result mismatch "+key)
        if (plan["schema"]!="d52-paired200-v1" or
            plan["before"]!=OLD or plan["after"]!=NEW or
            plan["pairs"]!=100 or plan["games"]!=200 or
            len(plan["families"])!=100 or
            len({x["parent_id"] for x in plan["families"]})!=100):
            raise ValueError("D52 frozen audit plan drift")
        decision=plan.get("decision",{})
        if decision.get("margin_win_count")!=12 or decision.get("max_unresolved")!=10:
            raise ValueError("D52 predefined admission standard moved")
        conditions=plan.get("conditions",{})
        for key,val in {"backend":"cpp_nnue","normal_ms":200,"long_ms":1000,
                        "max_long_per_side":10,"quiescence":0}.items():
            if conditions.get(key)!=val:raise ValueError("D52 clock or engine changed: "+key)
        hashes=result.get("game_sha256",{})
        if len(hashes)!=200:raise ValueError("D52 audit games missing")
        for name,expected_sha in hashes.items():
            if sha(member(t,"games/"+name))!=expected_sha:
                raise ValueError("D52 game SHA mismatch "+name)

    dest=ROOT/MODEL_PATH
    if dest.exists():raise ValueError("refusing to overwrite D52 official model")
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
       "approved_on":"2026-11-08",
       "approved_by":"owner-delegated F00/F20 current NNUE update after D52 156-44 paired200",
       "research_snapshot_sha256":NEW,
       "research_parent_sha256":OLD,
       "weights_file":MODEL_PATH,
       "weights_sha256":weights_sha,
       "optimizer_step":5606,
       "training_source_sha256":GAMES_SHA,
       "training_release_sha256":TRAIN_SHA,
       "comparison_release_sha256":AUDIT_SHA,
       "model_distribution":"GitHub models/fukaji-d52.nnue.npz plus D53 current release; D52 full optimizer and state are in training release.",
       "validation":{
           "opponent":"D50 before D52 selfplay 1000 (641f3667); 100 new paired openings with both colors",
           "wins":156,"losses":44,"draws":0,"unresolved":0,
           "opening_families":100,"games":200,
           "planned_opening_families":100,"planned_games":200,
           "normal_move_time_ms":200,"longthink_ms":1000,
           "longthink_max_per_side":10,"quiescence":0,
           "search_backend_variant":"d48_pvs_mate",
           "strength_conclusion":"D52 after 1000 PVS+mate selfplay games wins 156-44 over D50 in matched v0.4.1 paired200; predefined margin passed"
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
    if decision_path.exists():raise ValueError("D53 decision already exists")
    decision_path.write_text(f"""# D53：学習後D52のNNUEを現行採用（2026-11-08）

## 判断と試験の出所
F00/F20にある「現行モデルの学習前後を同条件で比較し、棋力向上を確認できればNNUE重みだけ正式化」という本人の委任に基づく。学習前D50から新1000局を生成・合法性検査し、NNUEをAdam4702からAdam5606まで追加学習した。
[元の1000局学習・200局比較](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/{RUN}) は全24job success。新100独立序盤、先後交替200局、同一v0.4.1 PVS＋1/3/5手詰みと相手1手詰み防止、通常200ms＋長考1000msを各陣営10回以内。D52 156勝、D50 44勝、引分0・未決着0、19,104手合法再生。勝数差112、事前基準12以上かつ未決着10以下を通過。未知の相手で78%勝率を保証する意味ではない。

## 保存・再現
- D52完全学習済みNNUE・Adam5606 snapshot: {NEW}。57,755学習局面。親は旧正式D50 {OLD}。
- [新規1000局Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d52-d50-1000-selfplay-{RUN}) SHA256 {GAMES_SHA}。
- [完全NNUEとoptimizer Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d52-d50-trained-{RUN}) SHA256 {TRAIN_SHA}。
- [新旧200局比較Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d52-d50-before-after-200games-{RUN}) SHA256 {AUDIT_SHA}。
- 正式用NNUE {MODEL_PATH} SHA256 {weights_sha}。旧正式D50 NNUEと配布物は保全。
- USI FUKAJI v0.4.1、PVSと詰み探索のC++ SHA256 {CPP} は一切変更しない。正式採用はD53 Actionsで契約・回帰・実USI・隔離配布・Release SHA再取得がすべて成功して初めて完了とする。エラー時はD50維持。
""")

    specdoc="docs/現行エンジン仕様.md"
    modify(specdoc,"**現行のNNUE重みはD50（641f3667、Adam4702）**","**現行のNNUE重みはD52（55ccc97d、Adam5606）**")
    modify(specdoc,"現行D45からPVS＋詰み探索で第3周1000局を自己対局し、標準200局比較で**159勝41敗・引分/未決着0局**。",
                  "現行D50からPVS＋詰み探索で第4周1000局を自己対局し、標準200局比較で**156勝44敗・引分/未決着0局**。")
    modify(specdoc,f"{chr(96)}{OLD}{chr(96)}",f"{chr(96)}{NEW}{chr(96)}")
    modify(specdoc,"models/fukaji-d50.nnue.npz","models/fukaji-d52.nnue.npz")
    modify(specdoc,f"{chr(96)}{OLD_WEIGHTS}{chr(96)}",f"{chr(96)}{weights_sha}{chr(96)}")
    change_section(specdoc,"## 現行モデルの採用判断と強さの不確実性","## 復元・研究との境界",
        f"""## 現行モデルの採用判断と強さの不確実性

[D53正式採用判断](decisions/2026-11-08_D53_D52学習後重みを現行採用.md)。D50/Adam4702を起点に自己対局1000局すべて決着、Adam5606へ直接学習。新規100序盤×先後交替200局、双方同じUSI v0.4.1（PVS＋厳密な1/3/5手詰み＋相手1手詰みガード）、同じ時計で **D52が156勝、D50が44勝、引分/未決着0**、19,104手合法確認。事前勝数差12の採用基準に対して112勝差。未知の時間制御や他エンジンで同じ勝率を保証しない。[200局全棋譜](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d52-d50-before-after-200games-{RUN})。

""")
    modify(specdoc,"次のNNUE自己学習は現行D50（Adam4702）のGitHub保存済み完全stateから開始し、更新前snapshot D50と更新後を同条件で対局比較する。",
        "次のNNUE自己学習は現行D52（Adam5606）のGitHub保存済み完全stateから開始し、更新前snapshot D52と更新後を同条件で対局比較する。")
    modify(specdoc,"失敗・悪化時は更新前D50へ戻す。","失敗・悪化時は更新前D52へ戻す。")
    append(specdoc,f"""## D53：正式D52 NNUE採用と探索維持（2026-11-08）

- 正式NNUE snapshot {NEW}、推論 {MODEL_PATH} SHA256 {weights_sha}、optimizer Adam5606。追加学習の完全stateはD52 Release。旧D50は復元可能。
- [D53正式採用記録](decisions/2026-11-08_D53_D52学習後重みを現行採用.md)。USI v0.4.1、PVS＋詰み探索のコードは未変更。""")

    change_section("docs/00_開発ルーター.md","## 現行モデル","## 現在の研究・保存再開点",
        f"""## 現行モデル

**現行NNUE D52（{NEW}、Adam5606）**。D50正式NNUEから新1000局の対局と追加学習、同じPVS＋詰み探索で200局D52 156勝D50 44勝、未決着0。正式 USI FUKAJI v0.4.1、探索はD49のまま。[D53判断](decisions/2026-11-08_D53_D52学習後重みを現行採用.md)、[D52完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d52-d50-trained-{RUN})、[現行仕様](現行エンジン仕様.md)、[manifest](../formal/model.json)。次周学習親は正式D52/Adam5606。旧D50・D45は復元用に保持。

""")
    append("docs/00_開発ルーター.md",f"""## 最新NNUE・探索の関係（D53、2026-11-08）

正式 D52/Adam5606 NNUE。200局でD52が156勝D50が44勝。USI FUKAJI v0.4.1・D49 PVS＋1/3/5手詰み探索は維持。[D53判断](decisions/2026-11-08_D53_D52学習後重みを現行採用.md)。""")
    append("AGENTS.md",f"""## 最新正式NNUE（D53、2026-11-08）

**正式D52/Adam5606、snapshot {NEW}。** D50/Adam4702から自己対局1000局→学習、比較200局でD52 156勝44敗、引分0・未決着0。F00/F20に従いNNUE重みのみ更新。FUKAJI v0.4.1と正式PVS＋詰み探索は不変。次周唯一の親は[D52完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d52-d50-trained-{RUN})。旧D50系の記述はその時点の履歴。[D53判断](docs/decisions/2026-11-08_D53_D52学習後重みを現行採用.md)。""")
    change_section("docs/ROADMAP.md","## 現行モデル","## 現在の研究",
        f"""## 現行モデル

**正式D52 NNUE（{NEW}、Adam5606）、FUKAJI v0.4.1・PVS＋詰み探索維持**。前正式D50/Adam4702から1000局自己対局で追加学習、新100序盤×先後入替200局で **D52 156勝、D50 44勝、引分/未決着0**、事前基準通過。[D53採用判断](decisions/2026-11-08_D53_D52学習後重みを現行採用.md)／[D52実験](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/{RUN})／[完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d52-d50-trained-{RUN})／[正式manifest](../formal/model.json)／[現行仕様](現行エンジン仕様.md)。今後はD52/Adam5606から継続。旧D50/Adam4702は不変で保護。

""")
    append("docs/ROADMAP.md",f"""## D53：D52 NNUE正式採用（2026-11-08）

- [実行run](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/{RUN})：1000局全局決着、57,755局面・Adam4702→5606、200局比較で新D52が156勝44敗、未決着0、19,104手合法検証。学習・比較の全3 ReleaseのSHA再取得成功。
- 正式NNUE {NEW}、推論SHA {weights_sha}。PVS＋詰み探索・USI v0.4.1不変。契約・回帰・実USI・隔離配布を別D53 CIで検証。[判断](decisions/2026-11-08_D53_D52学習後重みを現行採用.md)。""")
    for name in ("formal/model.json",specdoc):
        contract[name]=sha((ROOT/name).read_bytes())
    write_json(contract_path,contract)
    loaded2,loaded_spec2=formal_load()
    assert loaded_spec2["weights_sha256"]==weights_sha
    assert sha((ROOT/"native/nnue_search_ffi.cpp").read_bytes())==CPP
    assert sha((ROOT/"models/fukaji-d50.nnue.npz").read_bytes())==OLD_WEIGHTS
    receipt={
        "schema":"fukaji-d53-nnue-adopt-prepare-v1",
        "current_snapshot_sha256":NEW,"former_snapshot_sha256":OLD,
        "optimizer_step":5606,"weights_path":MODEL_PATH,
        "weights_sha256":weights_sha,"unchanged_cpp_sha256":CPP,
        "usi_version":spec["version"],"train_archive_sha256":TRAIN_SHA,
        "audit_archive_sha256":AUDIT_SHA,"after_wins":156,"before_wins":44,
        "draws":0,"unresolved":0,"training_games":1000,
        "audit_games":200,"validated_plies":19104,
        "contract_sha256":sha((ROOT/contract_path).read_bytes())
    }
    write_json(str(out/"receipt.json"),receipt)
    print(json.dumps(receipt))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--trained",type=Path,required=True)
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    prepare(a.trained,a.audit,a.out)
