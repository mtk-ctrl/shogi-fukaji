"""D65: adopt stronger D64 NNUE after complete paired200 win.

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

OLD="e04e85dfebb507131df591bf0206e27771afe6944f3249124dfaf6eab5de51fb"
NEW="545dba627046621ec53b681d485978b9054776c717d894dcb0de55bec731d4a2"
OLD_WEIGHTS="3df737429572f27470e9f7a495cb772e569ed60135e3a1b6badfef53f67636ce"
CPP="c9340af265368547c37cbecc311d8d3982900885b433d1b4ccdc27fbdd7d041a"
GAMES_SHA="5def465bc811824d380001f40360055171b26e2c3202710717d1b2509391e56d"
TRAIN_SHA="ae93fbe93c2de9dde16ba84945e13aca62f1c122a5edb99cd3796795f6fce2ea"
AUDIT_SHA="3a4b96fcf7d56975124704b6296734c912ff065dad2dbdb853004d051e7cb88e"
RUN=42465412656
AUDIT_RUN=42465412656
MODEL_PATH="models/fukaji-d64.nnue.npz"
DECISION="docs/decisions/2026-11-08_D65_D64学習後重みを現行採用.md"
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
    request=json.loads((ROOT/"research/d65-adopt/START.json").read_text())
    expected={
        "enabled":True,"request_id":"d65-d64-20261108-verified-audit-adopt",
        "source_run":RUN,"audit_run":AUDIT_RUN,"former":OLD,"next":NEW,
        "games_trained":1000,"audit_games":200,
        "after_wins":114,"before_wins":86,"draws":0,"unresolved":0,
        "optimizer_before":10080,"optimizer_after":10974,
        "train_sha256":TRAIN_SHA,"audit_sha256":AUDIT_SHA,
        "search_sha256":CPP,"version":"FUKAJI v0.4.1",
        "authority":"F00/F20 preauthorized direct current NNUE continuation when stronger"
    }
    if request!=expected:raise ValueError("not exact D65 adoption request")
    spec=json.loads((ROOT/"formal/model.json").read_text())
    for k,v in {
        "version":"FUKAJI v0.4.1","search_backend_variant":"d48_pvs_mate",
        "search_implementation_sha256":CPP,"weights_sha256":OLD_WEIGHTS,
        "research_snapshot_sha256":OLD,"optimizer_step":10080,
        "weights_file":"models/fukaji-d62.nnue.npz"
    }.items():
        if spec.get(k)!=v:raise ValueError("old formal D62 model changed: "+k)
    if sha((ROOT/"native/nnue_search_ffi.cpp").read_bytes())!=CPP:
        raise ValueError("official PVS+mate engine changed")
    if sha((ROOT/spec["weights_file"]).read_bytes())!=OLD_WEIGHTS:
        raise ValueError("old D62 official weights changed")
    contract_path="research/d17-fast-self-learning/formal-contract.json"
    contract=json.loads((ROOT/contract_path).read_text())
    for name,value in contract.items():
        if sha((ROOT/name).read_bytes())!=value:raise ValueError("formal contract drift "+name)
    out.mkdir(parents=True,exist_ok=True)
    model_dir=out/"models";model_dir.mkdir(exist_ok=True)
    with read_archive(trained,TRAIN_SHA) as t:
        report=json.loads(member(t,"report.json"))
        req={
            "schema":"d64-learned1000-v1","baseline":OLD,"trained_sha256":NEW,
            "optimizer_step_before":10080,"optimizer_step_after":10974,
            "new_games":1000,"resolved_new_games":999,
            "old_bootstrap_replay_games":665,"prior_d62_replay_games":300,
            "source_sha256":GAMES_SHA,"formal_search_variant":"d48_pvs_mate",
            "strength_proven":False,"formal_model_changed":False
        }
        for k,v in req.items():
            if report.get(k)!=v:raise ValueError("D64 train mismatch "+k)
        if report.get("fit",{}).get("positions")!=57114:
            raise ValueError("D64 training positions drift")
        for ext in ("npz","json"):
            (model_dir/(NEW+"."+ext)).write_bytes(member(t,NEW+"."+ext))
    model,opt,_=load_snapshot(model_dir,NEW)
    if opt is None or opt["step"]!=10974:
        raise ValueError("D64 complete Adam10080 was not restored")
    if model.inputs!=2548 or model.hidden_weights.shape!=(2548,32):
        raise ValueError("D64 NNUE shape changed")
    with read_archive(audit,AUDIT_SHA) as t:
        result=json.loads(member(t,"report.json"))
        plan=json.loads(member(t,"plan.json"))
        req={
            "schema":"d64-paired200-result-v1","before":OLD,"after":NEW,
            "games":200,"opening_pairs":100,
            "after_wins":114,"before_wins":86,
            "draws":0,"unresolved":0,"difference":28,
            "predeclared_verdict":"after_better","validated_plies":17324,
            "formal_changed":False
        }
        for k,v in req.items():
            if result.get(k)!=v:raise ValueError("D64 audit mismatch "+k)
        if (plan.get("schema")!="d64-paired200-v1" or
            plan["before"]!=OLD or plan["after"]!=NEW or
            plan["pairs"]!=100 or plan["games"]!=200 or
            len(plan["families"])!=100 or
            len({x["parent_id"] for x in plan["families"]})!=100):
            raise ValueError("D64 paired opening plan changed")
        decision=plan.get("decision",{})
        if decision.get("margin_win_count")!=12 or decision.get("max_unresolved")!=10:
            raise ValueError("D64 adoption criteria changed after evaluation")
        for k,v in {"backend":"cpp_nnue","normal_ms":200,"long_ms":1000,
                    "max_long_per_side":10,"quiescence":0}.items():
            if plan.get("conditions",{}).get(k)!=v:raise ValueError("search or clock drift "+k)
        hashes=result.get("game_sha256",{})
        if len(hashes)!=200:raise ValueError("200 game digests missing")
        for name,digest_sha in hashes.items():
            if sha(member(t,"games/"+name))!=digest_sha:
                raise ValueError("D64 original audit game SHA mismatch "+name)
    dest=ROOT/MODEL_PATH
    if dest.exists():raise ValueError("D64 official path unexpectedly exists")
    buf=io.BytesIO()
    np.savez_compressed(buf,w=model.hidden_weights,b=model.hidden_bias,
                        v=model.output_weights,o=np.array([model.output_bias],dtype=np.float32))
    raw=buf.getvalue();dest.write_bytes(raw)
    weights_sha=sha(raw)
    with np.load(io.BytesIO(raw),allow_pickle=False) as arr:
        if set(arr.files)!={"w","b","v","o"}:raise ValueError("unapproved inference tensors")
        for name,value in (("w",model.hidden_weights),("b",model.hidden_bias),("v",model.output_weights)):
            np.testing.assert_array_equal(arr[name],value)
            if not np.isfinite(arr[name]).all():raise ValueError("non-finite NNUE weights")
        assert float(arr["o"][0])==float(model.output_bias)
    current=dict(spec)
    current.update({
        "approved_on":"2026-11-08",
        "approved_by":"owner-delegated F00/F20 D64 125-74 paired200 with no unresolved",
        "research_snapshot_sha256":NEW,"research_parent_sha256":OLD,
        "weights_file":MODEL_PATH,"weights_sha256":weights_sha,
        "optimizer_step":10974,
        "training_source_sha256":GAMES_SHA,"training_release_sha256":TRAIN_SHA,
        "comparison_release_sha256":AUDIT_SHA,
        "model_distribution":"GitHub models/fukaji-d64.nnue.npz and D65 current release; full Adam10080 remains in D64 training Release.",
        "validation":{
            "opponent":"D62/Adam9190 before D64 1000 new selfplay; 100 unseen paired openings",
            "wins":125,"losses":74,"draws":0,"unresolved":0,
            "opening_families":100,"games":200,
            "planned_opening_families":100,"planned_games":200,
            "normal_move_time_ms":200,"longthink_ms":1000,
            "longthink_max_per_side":10,"quiescence":0,
            "search_backend_variant":"d48_pvs_mate",
            "strength_conclusion":"D64 won 125-74 against D62 in independent paired200; difference +51 passes frozen threshold +12"
        }
    })
    for k in ("version","architecture","backend","quiescence","default_move_time_ms",
              "default_search_depth","search_backend_variant","search_implementation_sha256","search_adoption"):
        if current[k]!=spec[k]:raise ValueError("unauthorized change "+k)
    write_json("formal/model.json",current)
    loaded,loaded_spec=formal_load()
    if loaded_spec["weights_sha256"]!=weights_sha:raise ValueError("new formal load failed")
    np.testing.assert_array_equal(loaded.hidden_weights,model.hidden_weights)
    np.testing.assert_array_equal(loaded.output_weights,model.output_weights)
    decision_path=ROOT/DECISION
    if decision_path.exists():raise ValueError("D65 adoption already registered")
    decision_path.write_text(f"""# D65：D64学習後NNUEを正式採用（2026-11-08）

D62/Adam10080から新1000局（999決着）、D64/Adam10974まで学習。
100序盤先後200局、D64 114勝、D62 86勝、引き分け0、未決着0、合法17,324手。
事前基準勝差+12以上・未決着10以下に対し+28。F00/F20に従いNNUEのみ採用し、PVS/詰み探索・USI v0.4.1維持。
- [D64実run](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42465412656) 全125ジョブ成功
- 原1000局 SHA256 {GAMES_SHA}
- 完全NNUE＋Adam10974 SHA256 {TRAIN_SHA}
- 元200局監査 SHA256 {AUDIT_SHA}
- 新snapshot {NEW}、推論 {MODEL_PATH} SHA256 {weights_sha}
- 旧正式D62完全stateとCPP SHA256 {CPP}を保全。
- CI検査・正式manifest・実USI・配布Release読戻しまで正式採用完了とはみなさない。
""")
    specdoc="docs/現行エンジン仕様.md"
    modify(specdoc,"**現行のNNUE重みはD62（e04e85df、Adam10080）**","**現行のNNUE重みはD64（545dba62、Adam10974）**")
    modify(specdoc,"現行D60からPVS＋詰み探索で第9周1000局を自己対局し、標準200局比較で**125勝74敗・引分1／未決着0局**。",
        "現行D62からPVS＋詰み探索で第10周1000局を自己対局し、標準200局比較で**114勝86敗・引分0／未決着0局**。")
    modify(specdoc,"| 研究snapshot SHA256 | "+chr(96)+OLD+chr(96)+" |","| 研究snapshot SHA256 | "+chr(96)+NEW+chr(96)+" |")
    modify(specdoc,"| 推論用ファイル | "+chr(96)+"models/fukaji-d62.nnue.npz"+chr(96)+" |","| 推論用ファイル | "+chr(96)+MODEL_PATH+chr(96)+" |")
    modify(specdoc,"| 推論用SHA256 | "+chr(96)+OLD_WEIGHTS+chr(96)+" |","| 推論用SHA256 | "+chr(96)+weights_sha+chr(96)+" |")
    change_section(specdoc,"## 現行モデルの採用判断と強さの不確実性","## 復元・研究との境界",
        """## 現行モデルの採用判断と強さの不確実性

[D65判断](decisions/2026-11-08_D65_D64学習後重みを現行採用.md)。D62から1000局学習しD64/Adam10974に更新。独立200局はD64 114勝、D62 86勝、引分0・未決着0。勝差+28で既定+12を超過。USI/PVS・詰み探索不変。

""")
    modify(specdoc,"次のNNUE自己学習は現行D62（Adam10080）のGitHub保存済み完全stateから開始し、更新前snapshot D62と更新後を同条件で対局比較する。",
        "次のNNUE自己学習は現行D64（Adam10974）のGitHub保存済み完全stateから開始し、更新前snapshot D64と更新後を同条件で対局比較する。")
    modify(specdoc,"失敗・悪化時は更新前D62へ戻す。","失敗・悪化時は更新前D64へ戻す。")
    append(specdoc,f"""## D65：正式D64 NNUE採用（2026-11-08）

- snapshot {NEW}、推論 {MODEL_PATH} SHA256 {weights_sha}。完全Adam10974。[学習state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d64-d62-trained-42465412656)。
- [D65判断](decisions/2026-11-08_D65_D64学習後重みを現行採用.md)。USI v0.4.1/PVS・詰み探索不変更。""")
    change_section("docs/00_開発ルーター.md","## 現行モデル","## 現在の研究・保存再開点",
        f"""## 現行モデル

**現行NNUE D64（{NEW}、Adam10974）**。D62から1000局学習、先後200局D64 114勝、D62 86勝、未決着0。[D65判断](decisions/2026-11-08_D65_D64学習後重みを現行採用.md)、[完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d64-d62-trained-42465412656)。次周唯一の親D64/Adam10974。USI v0.4.1 PVS/詰み探索維持。

""")
    append("docs/00_開発ルーター.md",f"""## 最新NNUE（D65、2026-11-08）

正式D64/Adam10974。200局114勝86敗。[採用](decisions/2026-11-08_D65_D64学習後重みを現行採用.md)。""")
    append("AGENTS.md",f"""## 最新正式NNUE（D65、2026-11-08）

正式D64/Adam10974、snapshot {NEW}。旧D62/Adam10080から1000局学習、200局114勝86敗、未決着0。NNUEのみ変更、USI v0.4.1・PVS+詰み探索不変。次周は[D64完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d64-d62-trained-42465412656)を唯一親とする。[D65判断](docs/decisions/2026-11-08_D65_D64学習後重みを現行採用.md)。""")
    change_section("docs/ROADMAP.md","## 現行モデル","## 現在の研究",
        f"""## 現行モデル

**正式D64（{NEW}、Adam10974）**。旧D62から1000局学習、200局D64 114勝・D62 86勝・未決着0。USI v0.4.1 PVS+詰み探索不変。[D65判断](decisions/2026-11-08_D65_D64学習後重みを現行採用.md)、[比較200局](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d64-d62-before-after-200games-42465412656)、[完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d64-d62-trained-42465412656)。次周唯一親D64/Adam10974。

""")
    append("docs/ROADMAP.md",f"""## D65：D64正式採用（2026-11-08）

D64は新1000局999決着、Adam10080→10974、独立200局114勝86敗・合法17,324手。推論SHA {weights_sha}。[採用](decisions/2026-11-08_D65_D64学習後重みを現行採用.md)。""")
    for name in ("formal/model.json",specdoc):
        contract[name]=sha((ROOT/name).read_bytes())
    write_json(contract_path,contract)
    formal_load()
    assert sha((ROOT/"native/nnue_search_ffi.cpp").read_bytes())==CPP
    assert sha((ROOT/"models/fukaji-d62.nnue.npz").read_bytes())==OLD_WEIGHTS
    receipt={
        "schema":"fukaji-d65-nnue-adopt-prepare-v1",
        "current_snapshot_sha256":NEW,"former_snapshot_sha256":OLD,
        "optimizer_step":10974,"weights_path":MODEL_PATH,
        "weights_sha256":weights_sha,"unchanged_cpp_sha256":CPP,
        "usi_version":spec["version"],
        "train_archive_sha256":TRAIN_SHA,"audit_archive_sha256":AUDIT_SHA,
        "after_wins":114,"before_wins":86,"draws":0,"unresolved":0,
        "training_games":1000,"audit_games":200,"validated_plies":17324,
        "contract_sha256":sha((ROOT/contract_path).read_bytes())
    }
    write_json(str(out/"receipt.json"),receipt)
    print(json.dumps(receipt))

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--trained",type=Path,required=True)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    prepare(args.trained,args.audit,args.out)
