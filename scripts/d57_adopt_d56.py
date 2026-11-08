"""D57: adopt owner-authorized D54 NNUE after complete paired200 win.

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

OLD="1cb833c18be57d4881388bb6947f17493cca4974faac4023a4550ea47463f3f5"
NEW="89562fd4fed576f4a883dba465add35c532cc2c7ccdae9efd24d685270246363"
OLD_WEIGHTS="5c2e011225e914d95b36afaecf56fd267db273a325de47bf371eaa93e8dc36a2"
CPP="c9340af265368547c37cbecc311d8d3982900885b433d1b4ccdc27fbdd7d041a"
GAMES_SHA="94a6cd1fae005ad3bf31d50cdd32a75413b946269a0b4875b9ee9d86dde2c21c"
TRAIN_SHA="4aa33cbc6991cb69008bb3b586ca089e36ed3a79f2c80340fc53dd8198c8bb87"
AUDIT_SHA="df4806113e4d1b07218172bc6464233783a6c10bec1c636a0d0c9bab9ef500dd"
RUN=42433340061
TRAIN_RUN=42433340061
MODEL_PATH="models/fukaji-d56.nnue.npz"
DECISION="docs/decisions/2026-11-08_D57_D56学習後重みを現行採用.md"

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
    request=json.loads((ROOT/"research/d57-adopt/START.json").read_text())
    expected={
        "enabled":True,"request_id":"d57-d56-20261108-verified-audit-adopt",
        "source_run":RUN,"former":OLD,"next":NEW,
        "games_trained":1000,"audit_games":200,
        "after_wins":135,"before_wins":65,"draws":0,"unresolved":0,
        "optimizer_before":6508,"optimizer_after":7402,
        "train_sha256":TRAIN_SHA,"audit_sha256":AUDIT_SHA,
        "search_sha256":CPP,"version":"FUKAJI v0.4.1",
        "authority":"F00/F20 preauthorized direct current NNUE continuation when stronger"
    }
    if request!=expected:raise ValueError("not exact D57 adoption request")
    manifest_path=ROOT/"formal/model.json"
    spec=json.loads(manifest_path.read_text())
    for k,v in {
        "version":"FUKAJI v0.4.1","search_backend_variant":"d48_pvs_mate",
        "search_implementation_sha256":CPP,"weights_sha256":OLD_WEIGHTS,
        "research_snapshot_sha256":OLD,"optimizer_step":6508,
        "weights_file":"models/fukaji-d54.nnue.npz"
    }.items():
        if spec.get(k)!=v:raise ValueError("old formal D54 model changed: "+k)
    if sha((ROOT/"native/nnue_search_ffi.cpp").read_bytes())!=CPP:
        raise ValueError("official PVS+mate engine changed")
    if sha((ROOT/spec["weights_file"]).read_bytes())!=OLD_WEIGHTS:
        raise ValueError("D54 official old weights changed")
    contract_path="research/d17-fast-self-learning/formal-contract.json"
    contract=json.loads((ROOT/contract_path).read_text())
    for name,value in contract.items():
        if sha((ROOT/name).read_bytes())!=value:raise ValueError("old formal contract drift "+name)
    out.mkdir(parents=True,exist_ok=True)
    model_dir=out/"models";model_dir.mkdir(exist_ok=True)
    with read_archive(trained,TRAIN_SHA) as t:
        report=json.loads(member(t,"report.json"))
        req={
            "schema":"d56-learned1000-v1","baseline":OLD,"trained_sha256":NEW,
            "optimizer_step_before":6508,"optimizer_step_after":7402,
            "new_games":1000,"resolved_new_games":997,
            "old_bootstrap_replay_games":665,"prior_d52_replay_games":300,
            "source_sha256":GAMES_SHA,"formal_search_variant":"d48_pvs_mate",
            "strength_proven":False,"formal_model_changed":False
        }
        for k,v in req.items():
            if report.get(k)!=v:raise ValueError("D56 training mismatch "+k)
        if report.get("fit",{}).get("positions")!=57210:
            raise ValueError("D56 training positions unexpectedly changed")
        for ext in ("npz","json"):
            (model_dir/(NEW+"."+ext)).write_bytes(member(t,NEW+"."+ext))
    model,optimizer,_=load_snapshot(model_dir,NEW)
    if optimizer is None or optimizer["step"]!=7402:
        raise ValueError("D56 full optimizer cannot be restored")
    if model.inputs!=2548 or model.hidden_weights.shape!=(2548,32):
        raise ValueError("D56 NNUE architecture drift")
    with read_archive(audit,AUDIT_SHA) as t:
        result=json.loads(member(t,"report.json"))
        plan=json.loads(member(t,"plan.json"))
        req={
            "schema":"d56-paired200-result-v1","before":OLD,"after":NEW,
            "games":200,"opening_pairs":100,"after_wins":135,"before_wins":65,
            "draws":0,"unresolved":0,"difference":70,
            "predeclared_verdict":"after_better","validated_plies":19009,
            "formal_changed":False
        }
        for k,v in req.items():
            if result.get(k)!=v:raise ValueError("D56 audit mismatch "+k)
        if (plan.get("schema")!="d56-paired200-v1" or
            plan["before"]!=OLD or plan["after"]!=NEW or
            plan["pairs"]!=100 or plan["games"]!=200 or
            len(plan["families"])!=100 or
            len({x["parent_id"] for x in plan["families"]})!=100):
            raise ValueError("D56 frozen independent paired200 plan drift")
        decision=plan.get("decision",{})
        if decision.get("margin_win_count")!=12 or decision.get("max_unresolved")!=10:
            raise ValueError("D56 predefined admission criteria changed")
        for k,v in {"backend":"cpp_nnue","normal_ms":200,"long_ms":1000,
                    "max_long_per_side":10,"quiescence":0}.items():
            if plan.get("conditions",{}).get(k)!=v:raise ValueError("search or clock drift "+k)
        hashes=result.get("game_sha256",{})
        if len(hashes)!=200:raise ValueError("D56 200 game SHA set incomplete")
        for name,digest_sha in hashes.items():
            if sha(member(t,"games/"+name))!=digest_sha:
                raise ValueError("D56 original audit game SHA mismatch "+name)
    dest=ROOT/MODEL_PATH
    if dest.exists():raise ValueError("refusing to overwrite an existing D56 official model")
    buf=io.BytesIO()
    np.savez_compressed(buf,w=model.hidden_weights,b=model.hidden_bias,
                        v=model.output_weights,o=np.array([model.output_bias],dtype=np.float32))
    raw=buf.getvalue();dest.write_bytes(raw)
    weights_sha=sha(raw)
    with np.load(io.BytesIO(raw),allow_pickle=False) as arr:
        if set(arr.files)!={"w","b","v","o"}:raise ValueError("unapproved NNUE tensors")
        for name,value in (("w",model.hidden_weights),("b",model.hidden_bias),("v",model.output_weights)):
            np.testing.assert_array_equal(arr[name],value)
            if not np.isfinite(arr[name]).all():raise ValueError("nonfinite NNUE tensor")
        assert float(arr["o"][0])==float(model.output_bias)
    after=dict(spec)
    after.update({
        "approved_on":"2026-11-08",
        "approved_by":"owner-delegated F00/F20 current NNUE update after D56 wins 135-65, unresolved 0",
        "research_snapshot_sha256":NEW,
        "research_parent_sha256":OLD,
        "weights_file":MODEL_PATH,
        "weights_sha256":weights_sha,
        "optimizer_step":7402,
        "training_source_sha256":GAMES_SHA,
        "training_release_sha256":TRAIN_SHA,
        "comparison_release_sha256":AUDIT_SHA,
        "model_distribution":"GitHub models/fukaji-d56.nnue.npz and D57 current release; D56 full Adam7402 in frozen training Release.",
        "validation":{
            "opponent":"D54/Adam6508 before D56 selfplay1000; 100 independent opening pairs",
            "wins":135,"losses":65,"draws":0,"unresolved":0,
            "opening_families":100,"games":200,
            "planned_opening_families":100,"planned_games":200,
            "normal_move_time_ms":200,"longthink_ms":1000,
            "longthink_max_per_side":10,"quiescence":0,
            "search_backend_variant":"d48_pvs_mate",
            "strength_conclusion":"D56 won 135-65 over D54 with no unresolved; predefined margin+12 passed"
        }
    })
    for k in ("version","architecture","backend","quiescence","default_move_time_ms",
              "default_search_depth","search_backend_variant","search_implementation_sha256","search_adoption"):
        if after[k]!=spec[k]:raise ValueError("unauthorized search or USI change "+k)
    write_json("formal/model.json",after)
    loaded,loaded_spec=formal_load()
    if loaded_spec["weights_sha256"]!=weights_sha:raise ValueError("new formal loader failed")
    np.testing.assert_array_equal(loaded.hidden_weights,model.hidden_weights)
    np.testing.assert_array_equal(loaded.output_weights,model.output_weights)
    decision_path=ROOT/DECISION
    if decision_path.exists():raise ValueError("D57 decision already exists")
    decision_path.write_text(f"""# D57：D56学習後NNUEを正式採用（2026-11-08）

## 採用判断
F00/F20委任と本人指示。D54/Adam6508から新1000局を生成、997決着・3未決着、Adam7402へ学習。独立先後200局でD56 135勝・D54 65勝・引分0・未決着0、合法19,009手。事前基準の勝数差12以上・未決着10以下を通過。PVS＋1/3/5手詰みとUSI v0.4.1は不変。外部対局相手への勝率は別件。

## 出所・復旧
- D56 run: https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42433340061
- 新1000局 SHA256 {GAMES_SHA}
- 完全state＋Adam7402 SHA256 {TRAIN_SHA}
- 独立200局 SHA256 {AUDIT_SHA}
- 正式研究snapshot {NEW}
- 新正式重み {MODEL_PATH}, SHA256 {weights_sha}
- 旧D54推論と完全stateは保持。詰み探索 SHA256 {CPP} は不変。
- D57 CIの全回帰、実USI、隔離配布、Release再取得SHA確認、main反映までは正式採用完了としない。
""")
    specdoc="docs/現行エンジン仕様.md"
    modify(specdoc,"**現行のNNUE重みはD54（1cb833c1、Adam6508）**","**現行のNNUE重みはD56（89562fd4、Adam7402）**")
    modify(specdoc,"現行D52からPVS＋詰み探索で第5周1000局を自己対局し、標準200局比較で**137勝62敗・引分0／未決着1局**。",
           "現行D54からPVS＋詰み探索で第6周1000局を自己対局し、標準200局比較で**135勝65敗・引分0／未決着0局**。")
    modify(specdoc,"| 研究snapshot SHA256 | "+chr(96)+OLD+chr(96)+" |","| 研究snapshot SHA256 | "+chr(96)+NEW+chr(96)+" |")
    modify(specdoc,"| 推論用ファイル | "+chr(96)+"models/fukaji-d54.nnue.npz"+chr(96)+" |",
           "| 推論用ファイル | "+chr(96)+MODEL_PATH+chr(96)+" |")
    modify(specdoc,"| 推論用SHA256 | "+chr(96)+OLD_WEIGHTS+chr(96)+" |",
           "| 推論用SHA256 | "+chr(96)+weights_sha+chr(96)+" |")
    change_section(specdoc,"## 現行モデルの採用判断と強さの不確実性","## 復元・研究との境界",
        """## 現行モデルの採用判断と強さの不確実性

[D57採用](decisions/2026-11-08_D57_D56学習後重みを現行採用.md)。新規1000局からNNUEをAdam6508→7402へ学習。独立先後200局でD56 135勝・旧D54 65勝、未決着0、合法19,009手。差70は基準12以上。v0.4.1とPVS＋詰み探索不変。未知相手への勝率は保証しない。

""")
    modify(specdoc,"次のNNUE自己学習は現行D54（Adam6508）のGitHub保存済み完全stateから開始し、更新前snapshot D54と更新後を同条件で対局比較する。",
           "次のNNUE自己学習は現行D56（Adam7402）のGitHub保存済み完全stateから開始し、更新前snapshot D56と更新後を同条件で対局比較する。")
    modify(specdoc,"失敗・悪化時は更新前D54へ戻す。","失敗・悪化時は更新前D56へ戻す。")
    append(specdoc,f"""## D57：正式D56 NNUE採用と探索維持（2026-11-08）

- snapshot {NEW}、推論 {MODEL_PATH} SHA256 {weights_sha}、Adam7402。完全stateはD56 Release、旧D54復元可能。
- [正式採用記録](decisions/2026-11-08_D57_D56学習後重みを現行採用.md)。USI v0.4.1とPVS＋詰み探索は不変。""")
    change_section("docs/00_開発ルーター.md","## 現行モデル","## 現在の研究・保存再開点",
        f"""## 現行モデル

**現行NNUE D56（{NEW}、Adam7402）**。D54から1000局学習、200局でD56 135勝・D54 65勝・未決着0。USI v0.4.1、PVS＋詰み探索維持。[D57判断](decisions/2026-11-08_D57_D56学習後重みを現行採用.md)、[D56完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d56-d54-trained-42433340061)。次周はD56/Adam7402だけを親にする。

""")
    append("docs/00_開発ルーター.md",f"""## 最新NNUE（D57、2026-11-08）

正式D56/Adam7402。200局でD56 135勝D54 65勝、未決着0。v0.4.1とPVS＋詰み探索不変。[判断](decisions/2026-11-08_D57_D56学習後重みを現行採用.md)。""")
    append("AGENTS.md",f"""## 最新正式NNUE（D57、2026-11-08）

正式D56/Adam7402、snapshot {NEW}。旧D54から1000局学習、200局135勝65敗・未決着0。NNUE重みのみ更新、USI v0.4.1とPVS＋詰み探索不変。次周の親は[D56完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d56-d54-trained-42433340061)だけ。[判断](docs/decisions/2026-11-08_D57_D56学習後重みを現行採用.md)。""")
    change_section("docs/ROADMAP.md","## 現行モデル","## 現在の研究",
        f"""## 現行モデル

**正式D56（{NEW}、Adam7402）**。旧D54から新1000局学習、200局比較D56 135勝・D54 65勝・未決着0。USI v0.4.1、PVS＋詰み探索不変。[D57判断](decisions/2026-11-08_D57_D56学習後重みを現行採用.md)、[200局Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d56-d54-before-after-200games-42433340061)、[完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d56-d54-trained-42433340061)。次周唯一の親はD56/Adam7402。

""")
    append("docs/ROADMAP.md",f"""## D57正式採用（2026-11-08）

- D56 1000局中997決着、Adam6508→7402、独立200局135勝65敗・未決着0。正式NNUE {NEW}、推論SHA {weights_sha}。v0.4.1とPVS＋詰み探索不変。
- [判断記録](decisions/2026-11-08_D57_D56学習後重みを現行採用.md)。""")
    for name in ("formal/model.json",specdoc):
        contract[name]=sha((ROOT/name).read_bytes())
    write_json(contract_path,contract)
    formal_load()
    assert sha((ROOT/"native/nnue_search_ffi.cpp").read_bytes())==CPP
    assert sha((ROOT/"models/fukaji-d54.nnue.npz").read_bytes())==OLD_WEIGHTS
    receipt={
        "schema":"fukaji-d57-nnue-adopt-prepare-v1",
        "current_snapshot_sha256":NEW,"former_snapshot_sha256":OLD,
        "optimizer_step":7402,"weights_path":MODEL_PATH,
        "weights_sha256":weights_sha,"unchanged_cpp_sha256":CPP,
        "usi_version":spec["version"],"train_archive_sha256":TRAIN_SHA,
        "audit_archive_sha256":AUDIT_SHA,"after_wins":135,"before_wins":65,
        "draws":0,"unresolved":0,"training_games":1000,
        "audit_games":200,"validated_plies":19009,
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
