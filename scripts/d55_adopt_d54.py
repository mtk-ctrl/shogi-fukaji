"""D55: adopt owner-authorized D54 NNUE after complete paired200 win.

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

OLD="55ccc97db6c2d4ce9ff7c75f5e81d543fb6f12d169639c3ea6d24c89e95342ae"
NEW="1cb833c18be57d4881388bb6947f17493cca4974faac4023a4550ea47463f3f5"
OLD_WEIGHTS="c0bfe9640e8fc8d6750d263f367581d33c428c7c05cea99f384afaeabfbb569f"
CPP="c9340af265368547c37cbecc311d8d3982900885b433d1b4ccdc27fbdd7d041a"
GAMES_SHA="6eb7db54ccea6fa9da9a70f9d7aef25709dd13f97ba9b94f0d81f764f54c44ac"
TRAIN_SHA="19a0f06427d68e4b98f9b4af67722f909937b272134837af79315a25c7e401a1"
AUDIT_SHA="2506e8a17f9c3d775f774adefda5d7da1cf1355d4b1f0449fe41dc6c25c846bf"
RUN=42429671383
TRAIN_RUN=42426372939
MODEL_PATH="models/fukaji-d54.nnue.npz"
DECISION="docs/decisions/2026-11-08_D55_D54学習後重みを現行採用.md"

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
    request=json.loads((ROOT/"research/d55-adopt/START.json").read_text())
    expected={
        "enabled":True,"request_id":"d55-d54-20261108-auto-adopt-on-audit-win",
        "source_run":RUN,"former":OLD,"next":NEW,
        "games_trained":1000,"audit_games":200,
        "after_wins":137,"before_wins":62,"draws":0,"unresolved":1,
        "optimizer_before":5606,"optimizer_after":6508,
        "train_sha256":TRAIN_SHA,"audit_sha256":AUDIT_SHA,
        "search_sha256":CPP,"version":"FUKAJI v0.4.1",
        "authority":"F00/F20 preauthorized direct current NNUE continuation when stronger"
    }
    if request!=expected:raise ValueError("not exact D55 adoption request")
    manifest_path=ROOT/"formal/model.json"
    spec=json.loads(manifest_path.read_text())
    assert (spec["version"]=="FUKAJI v0.4.1" and
            spec["search_backend_variant"]=="d48_pvs_mate" and
            spec["search_implementation_sha256"]==CPP and
            spec["weights_sha256"]==OLD_WEIGHTS and
            spec["research_snapshot_sha256"]==OLD and
            spec["optimizer_step"]==5606 and
            spec["weights_file"]=="models/fukaji-d52.nnue.npz"),"old formal model changed"
    if sha((ROOT/"native/nnue_search_ffi.cpp").read_bytes())!=CPP:
        raise ValueError("D49 official native search changed")
    if sha((ROOT/spec["weights_file"]).read_bytes())!=OLD_WEIGHTS:
        raise ValueError("existing D52 official weights changed")
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
            "schema":"d54-learned1000-v1","baseline":OLD,"trained_sha256":NEW,
            "optimizer_step_before":5606,"optimizer_step_after":6508,
            "new_games":1000,"resolved_new_games":1000,
            "old_bootstrap_replay_games":665,"prior_d52_replay_games":300,
            "source_sha256":GAMES_SHA,"formal_search_variant":"d48_pvs_mate",
            "strength_proven":False,"formal_model_changed":False
        }
        for key,val in requirements.items():
            if report.get(key)!=val:raise ValueError("D54 training mismatch "+key)
        if report.get("fit",{}).get("positions")!=57615:
            raise ValueError("unexpected D54 training position count")
        for ext in ("npz","json"):
            (model_dir/(NEW+"."+ext)).write_bytes(member(t,NEW+"."+ext))
    model,optimizer,_=load_snapshot(model_dir,NEW)
    if optimizer is None or optimizer["step"]!=6508:
        raise ValueError("full D54 optimizer not loaded")
    if model.inputs!=2548 or model.hidden_weights.shape!=(2548,32):
        raise ValueError("D54 NNUE architecture drift")

    with read_archive(audit,AUDIT_SHA) as t:
        result=json.loads(member(t,"report.json"))
        plan=json.loads(member(t,"plan.json"))
        requirements={
            "schema":"d54-paired200-result-v1","before":OLD,"after":NEW,
            "games":200,"opening_pairs":100,"after_wins":137,"before_wins":62,
            "draws":0,"unresolved":1,"difference":75,
            "predeclared_verdict":"after_better","validated_plies":20297,
            "formal_changed":False
        }
        for key,val in requirements.items():
            if result.get(key)!=val:raise ValueError("D54 audit result mismatch "+key)
        if (plan["schema"]!="d54-paired200-v1" or
            plan["before"]!=OLD or plan["after"]!=NEW or
            plan["pairs"]!=100 or plan["games"]!=200 or
            len(plan["families"])!=100 or
            len({x["parent_id"] for x in plan["families"]})!=100):
            raise ValueError("D54 frozen audit plan drift")
        decision=plan.get("decision",{})
        if decision.get("margin_win_count")!=12 or decision.get("max_unresolved")!=10:
            raise ValueError("D54 predefined admission standard moved")
        conditions=plan.get("conditions",{})
        for key,val in {"backend":"cpp_nnue","normal_ms":200,"long_ms":1000,
                        "max_long_per_side":10,"quiescence":0}.items():
            if conditions.get(key)!=val:raise ValueError("D54 clock or engine changed: "+key)
        hashes=result.get("game_sha256",{})
        if len(hashes)!=200:raise ValueError("D54 audit games missing")
        for name,expected_sha in hashes.items():
            if sha(member(t,"games/"+name))!=expected_sha:
                raise ValueError("D54 game SHA mismatch "+name)

    dest=ROOT/MODEL_PATH
    if dest.exists():raise ValueError("refusing to overwrite D54 official model")
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
       "approved_by":"owner-delegated F00/F20 current NNUE update after D54 137-62 paired200, one unresolved",
       "research_snapshot_sha256":NEW,
       "research_parent_sha256":OLD,
       "weights_file":MODEL_PATH,
       "weights_sha256":weights_sha,
       "optimizer_step":6508,
       "training_source_sha256":GAMES_SHA,
       "training_release_sha256":TRAIN_SHA,
       "comparison_release_sha256":AUDIT_SHA,
       "model_distribution":"GitHub models/fukaji-d54.nnue.npz plus D55 current release; D54 full optimizer and state are in training release.",
       "validation":{
           "opponent":"D52 before D54 selfplay 1000 (55ccc97d); 100 new paired openings with both colors",
           "wins":137,"losses":62,"draws":0,"unresolved":1,
           "opening_families":100,"games":200,
           "planned_opening_families":100,"planned_games":200,
           "normal_move_time_ms":200,"longthink_ms":1000,
           "longthink_max_per_side":10,"quiescence":0,
           "search_backend_variant":"d48_pvs_mate",
           "strength_conclusion":"D54 after 1000 PVS+mate selfplay games wins 137-62 over D52 with 1 unresolved in matched v0.4.1 paired200; predefined margin passed"
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
    if decision_path.exists():raise ValueError("D55 decision already exists")
    decision_path.write_text(f"""# D55：学習後D54 NNUEを正式採用（2026-11-08）

## 判断と試験結果
F00/F20本人委任に基づき、正式D52/Adam5606の完全stateから新1,000局を生成・全局合法検証し、Adam6508へ学習。新100独立序盤×先後交替200局ではD54 137勝、D52 62勝、引分0、未決着1、20,297手合法再生。差75は事前基準12以上で未決着1は許容10以下。未知の相手で同じ勝率は保証しない。比較はFUKAJI v0.4.1、同一PVS＋1/3/5手詰み、200ms＋1秒長考各陣営10回以内。

## 正本・出所
- [学習run](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/{TRAIN_RUN})、[比較run](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/{RUN})。
- 1000局Release: fukaji-d54-d52-1000-selfplay-{TRAIN_RUN}、SHA256 {GAMES_SHA}。
- 完全state Release: fukaji-d54-d52-trained-{TRAIN_RUN}、SHA256 {TRAIN_SHA}。
- 比較200局 Release: fukaji-d54-d52-before-after-200games-{RUN}、SHA256 {AUDIT_SHA}。
- 新正式モデル {MODEL_PATH} SHA256 {weights_sha}。旧D52 NNUEは保全。
- PVS＋詰み探索のSHA256 {CPP}とUSI v0.4.1は変更なし。正式採用の完了判定はCIの全回帰・実USI・隔離配布とRelease再取得SHA成功が条件。
""")
    specdoc="docs/現行エンジン仕様.md"
    modify(specdoc,"**現行のNNUE重みはD52（55ccc97d、Adam5606）**","**現行のNNUE重みはD54（1cb833c1、Adam6508）**")
    modify(specdoc,"現行D50からPVS＋詰み探索で第4周1000局を自己対局し、標準200局比較で**156勝44敗・引分/未決着0局**。",
        "現行D52からPVS＋詰み探索で第5周1000局を自己対局し、標準200局比較で**137勝62敗・引分0／未決着1局**。")
    modify(specdoc,f"| 研究snapshot SHA256 | {chr(96)}{OLD}{chr(96)} |",f"| 研究snapshot SHA256 | {chr(96)}{NEW}{chr(96)} |")
    modify(specdoc,"| 推論用ファイル | "+chr(96)+"models/fukaji-d52.nnue.npz"+chr(96)+" |","| 推論用ファイル | "+chr(96)+"models/fukaji-d54.nnue.npz"+chr(96)+" |")
    modify(specdoc,f"| 推論用SHA256 | {chr(96)}{OLD_WEIGHTS}{chr(96)} |",f"| 推論用SHA256 | {chr(96)}{weights_sha}{chr(96)} |")
    change_section(specdoc,"## 現行モデルの採用判断と強さの不確実性","## 復元・研究との境界",
        f"""## 現行モデルの採用判断と強さの不確実性

[D55正式採用判断](decisions/2026-11-08_D55_D54学習後重みを現行採用.md)。D52/Adam5606から新規1000局の完全自己対局とNNUE追加学習でAdam6508。未見100序盤×先後200局の直接比較でD54 137勝、D52 62勝、引分0・未決着1、20,297手合法検証。事前差12勝以上に対して差75勝、未決着許容10以内。同じv0.4.1 PVS＋詰み探索、200ms＋最大10回の1秒長考。未知相手への勝率保証ではない。[全棋譜](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d54-d52-before-after-200games-{RUN})。

""")
    modify(specdoc,"次のNNUE自己学習は現行D52（Adam5606）のGitHub保存済み完全stateから開始し、更新前snapshot D52と更新後を同条件で対局比較する。",
        "次のNNUE自己学習は現行D54（Adam6508）のGitHub保存済み完全stateから開始し、更新前snapshot D54と更新後を同条件で対局比較する。")
    modify(specdoc,"失敗・悪化時は更新前D52へ戻す。","失敗・悪化時は更新前D54へ戻す。")
    append(specdoc,f"""## D55：正式D54 NNUE採用と探索維持（2026-11-08）

- 正式NNUE snapshot {NEW}、推論 {MODEL_PATH} SHA256 {weights_sha}、Adam6508。完全stateはD54 Release、D52は復元可能。
- [D55採用記録](decisions/2026-11-08_D55_D54学習後重みを現行採用.md)。USI v0.4.1・PVS＋詰み探索は不変。""")
    change_section("docs/00_開発ルーター.md","## 現行モデル","## 現在の研究・保存再開点",
        f"""## 現行モデル

**現行NNUE D54（{NEW}、Adam6508）**。D52から新1000局学習、200局比較でD54 137勝・D52 62勝・引分0・未決着1。USI FUKAJI v0.4.1、PVS＋詰み探索はD49維持。[D55判断](decisions/2026-11-08_D55_D54学習後重みを現行採用.md)、[D54完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d54-d52-trained-{TRAIN_RUN})、[仕様](現行エンジン仕様.md)、[manifest](../formal/model.json)。次周はD54/Adam6508を唯一の親とする。

""")
    append("docs/00_開発ルーター.md",f"""## 最新NNUE（D55、2026-11-08）

正式D54/Adam6508。比較200局でD54 137勝、D52 62勝、未決着1。PVS＋詰み探索とFUKAJI v0.4.1は不変更。[判断](decisions/2026-11-08_D55_D54学習後重みを現行採用.md)。""")
    append("AGENTS.md",f"""## 最新正式NNUE（D55、2026-11-08）

**正式D54/Adam6508、snapshot {NEW}。** 旧正式D52/Adam5606から1000局→学習、200局で137勝62敗、未決着1。NNUE重みのみ更新、PVS＋詰み探索・v0.4.1不変。次周唯一の親は[D54完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d54-d52-trained-{TRAIN_RUN})。旧モデル説明はその当時の履歴。[D55判断](docs/decisions/2026-11-08_D55_D54学習後重みを現行採用.md)。""")
    change_section("docs/ROADMAP.md","## 現行モデル","## 現在の研究",
        f"""## 現行モデル

**正式D54 NNUE（{NEW}、Adam6508）**。旧正式D52/Adam5606から新1000局を学習、200局比較でD54 137勝、D52 62勝、引分0・未決着1。USI FUKAJI v0.4.1・PVS＋詰み探索は変更なし。[D55判断](decisions/2026-11-08_D55_D54学習後重みを現行採用.md)、[全200局](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d54-d52-before-after-200games-{RUN})、[完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d54-d52-trained-{TRAIN_RUN})、[manifest](../formal/model.json)。次周学習の唯一の親は正式D54/Adam6508、旧D52保護。

""")
    append("docs/ROADMAP.md",f"""## D55：D54 NNUE正式採用（2026-11-08）

- 1000局全局決着、57615局面でAdam5606→6508、別の200局比較では新D54 137勝、D52 62勝、未決着1。合計20,297手合法検証。3件のReleaseのSHA再取得確認。
- 正式NNUE {NEW}、推論SHA {weights_sha}。USI v0.4.1・PVS＋詰み探索維持。[判断](decisions/2026-11-08_D55_D54学習後重みを現行採用.md)。""")
    for name in ("formal/model.json",specdoc):
        contract[name]=sha((ROOT/name).read_bytes())
    write_json(contract_path,contract)
    loaded2,loaded_spec2=formal_load()
    assert loaded_spec2["weights_sha256"]==weights_sha
    assert sha((ROOT/"native/nnue_search_ffi.cpp").read_bytes())==CPP
    assert sha((ROOT/"models/fukaji-d52.nnue.npz").read_bytes())==OLD_WEIGHTS
    receipt={
        "schema":"fukaji-d55-nnue-adopt-prepare-v1",
        "current_snapshot_sha256":NEW,"former_snapshot_sha256":OLD,
        "optimizer_step":6508,"weights_path":MODEL_PATH,
        "weights_sha256":weights_sha,"unchanged_cpp_sha256":CPP,
        "usi_version":spec["version"],"train_archive_sha256":TRAIN_SHA,
        "audit_archive_sha256":AUDIT_SHA,"after_wins":137,"before_wins":62,
        "draws":0,"unresolved":1,"training_games":1000,
        "audit_games":200,"validated_plies":20297,
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
