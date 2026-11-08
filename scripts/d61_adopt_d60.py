"""D61: adopt stronger D60 NNUE after complete paired200 win.

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

OLD="d190c1a1b8d5a44b3fe4791588306fb493f1b9534cca268fcbb656ba8604265c"
NEW="1f4ef9f32dfff7a94ad3ddb1c57f64f123ca8b6f6caa422cb9c86ace06c57654"
OLD_WEIGHTS="d856765813508a29cb2f35e2aceba8b67dc7797798eb114ae23d826424baf30d"
CPP="c9340af265368547c37cbecc311d8d3982900885b433d1b4ccdc27fbdd7d041a"
GAMES_SHA="1db4cdb8ba016479b67ab5c46ec111031bbdda11892433c30c5c12fe306e3b7e"
TRAIN_SHA="406c67f6d03c79f7a38e49e1e9c9ab35b21c68ac6d07e1029f4d45746cf9f07e"
AUDIT_SHA="6b9f2c432481d19caded577858d118fb99a028142d25d6dd239f1c75bb16ed11"
RUN=42447988443
AUDIT_RUN=42455966713
MODEL_PATH="models/fukaji-d60.nnue.npz"
DECISION="docs/decisions/2026-11-08_D61_D60学習後重みを現行採用.md"
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
    request=json.loads((ROOT/"research/d61-adopt/START.json").read_text())
    expected={
        "enabled":True,"request_id":"d61-d60-20261108-verified-audit-adopt",
        "source_run":RUN,"audit_run":AUDIT_RUN,"former":OLD,"next":NEW,
        "games_trained":1000,"audit_games":200,
        "after_wins":123,"before_wins":77,"draws":0,"unresolved":0,
        "optimizer_before":8296,"optimizer_after":9190,
        "train_sha256":TRAIN_SHA,"audit_sha256":AUDIT_SHA,
        "search_sha256":CPP,"version":"FUKAJI v0.4.1",
        "authority":"F00/F20 preauthorized direct current NNUE continuation when stronger"
    }
    if request!=expected:raise ValueError("not exact D61 adoption request")
    spec=json.loads((ROOT/"formal/model.json").read_text())
    for k,v in {
        "version":"FUKAJI v0.4.1","search_backend_variant":"d48_pvs_mate",
        "search_implementation_sha256":CPP,"weights_sha256":OLD_WEIGHTS,
        "research_snapshot_sha256":OLD,"optimizer_step":8296,
        "weights_file":"models/fukaji-d58.nnue.npz"
    }.items():
        if spec.get(k)!=v:raise ValueError("old formal D58 model changed: "+k)
    if sha((ROOT/"native/nnue_search_ffi.cpp").read_bytes())!=CPP:
        raise ValueError("official PVS+mate engine changed")
    if sha((ROOT/spec["weights_file"]).read_bytes())!=OLD_WEIGHTS:
        raise ValueError("old D58 official weights changed")
    contract_path="research/d17-fast-self-learning/formal-contract.json"
    contract=json.loads((ROOT/contract_path).read_text())
    for name,value in contract.items():
        if sha((ROOT/name).read_bytes())!=value:raise ValueError("formal contract drift "+name)
    out.mkdir(parents=True,exist_ok=True)
    model_dir=out/"models";model_dir.mkdir(exist_ok=True)
    with read_archive(trained,TRAIN_SHA) as t:
        report=json.loads(member(t,"report.json"))
        req={
            "schema":"d60-learned1000-v1","baseline":OLD,"trained_sha256":NEW,
            "optimizer_step_before":8296,"optimizer_step_after":9190,
            "new_games":1000,"resolved_new_games":997,
            "old_bootstrap_replay_games":665,"prior_d58_replay_games":300,
            "source_sha256":GAMES_SHA,"formal_search_variant":"d48_pvs_mate",
            "strength_proven":False,"formal_model_changed":False
        }
        for k,v in req.items():
            if report.get(k)!=v:raise ValueError("D60 train mismatch "+k)
        if report.get("fit",{}).get("positions")!=57112:
            raise ValueError("D60 training positions drift")
        for ext in ("npz","json"):
            (model_dir/(NEW+"."+ext)).write_bytes(member(t,NEW+"."+ext))
    model,opt,_=load_snapshot(model_dir,NEW)
    if opt is None or opt["step"]!=9190:
        raise ValueError("D60 complete Adam9190 was not restored")
    if model.inputs!=2548 or model.hidden_weights.shape!=(2548,32):
        raise ValueError("D60 NNUE shape changed")
    with read_archive(audit,AUDIT_SHA) as t:
        result=json.loads(member(t,"report.json"))
        plan=json.loads(member(t,"plan.json"))
        req={
            "schema":"d60-paired200-result-v1","before":OLD,"after":NEW,
            "games":200,"opening_pairs":100,
            "after_wins":123,"before_wins":77,
            "draws":0,"unresolved":0,"difference":46,
            "predeclared_verdict":"after_better","validated_plies":18815,
            "formal_changed":False
        }
        for k,v in req.items():
            if result.get(k)!=v:raise ValueError("D60 audit mismatch "+k)
        if (plan.get("schema")!="d60-paired200-v1" or
            plan["before"]!=OLD or plan["after"]!=NEW or
            plan["pairs"]!=100 or plan["games"]!=200 or
            len(plan["families"])!=100 or
            len({x["parent_id"] for x in plan["families"]})!=100):
            raise ValueError("D60 paired opening plan changed")
        decision=plan.get("decision",{})
        if decision.get("margin_win_count")!=12 or decision.get("max_unresolved")!=10:
            raise ValueError("D60 adoption criteria changed after evaluation")
        for k,v in {"backend":"cpp_nnue","normal_ms":200,"long_ms":1000,
                    "max_long_per_side":10,"quiescence":0}.items():
            if plan.get("conditions",{}).get(k)!=v:raise ValueError("search or clock drift "+k)
        hashes=result.get("game_sha256",{})
        if len(hashes)!=200:raise ValueError("200 game digests missing")
        for name,digest_sha in hashes.items():
            if sha(member(t,"games/"+name))!=digest_sha:
                raise ValueError("D60 original audit game SHA mismatch "+name)
    dest=ROOT/MODEL_PATH
    if dest.exists():raise ValueError("D60 official path unexpectedly exists")
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
        "approved_by":"owner-delegated F00/F20 D60 123-77 paired200 with no unresolved",
        "research_snapshot_sha256":NEW,"research_parent_sha256":OLD,
        "weights_file":MODEL_PATH,"weights_sha256":weights_sha,
        "optimizer_step":9190,
        "training_source_sha256":GAMES_SHA,"training_release_sha256":TRAIN_SHA,
        "comparison_release_sha256":AUDIT_SHA,
        "model_distribution":"GitHub models/fukaji-d60.nnue.npz and D61 current release; full Adam9190 remains in D60 training Release.",
        "validation":{
            "opponent":"D58/Adam8296 before D60 1000 new selfplay; 100 unseen paired openings",
            "wins":123,"losses":77,"draws":0,"unresolved":0,
            "opening_families":100,"games":200,
            "planned_opening_families":100,"planned_games":200,
            "normal_move_time_ms":200,"longthink_ms":1000,
            "longthink_max_per_side":10,"quiescence":0,
            "search_backend_variant":"d48_pvs_mate",
            "strength_conclusion":"D60 won 123-77 against D58 in independent paired200; difference +46 passes frozen threshold +12"
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
    if decision_path.exists():raise ValueError("D61 adoption already registered")
    decision_path.write_text(f"""# D61：D60学習後NNUEを正式採用（2026-11-08）

## 採否
D58/Adam8296を親として新自己対局1000局（997決着）、Adam9190まで学習。
先後交替100組・200局、D60 123勝D58 77勝、引分0・未決着0、18,815合法手。
勝差+46、既定+12以上、未決着0は許容10以下。F00/F20本人委任によりD60 NNUEのみ現行採用。
外部相手への強さは未検証。PVS＋1/3/5手詰み／USI v0.4.1は不変更。

## 保存・戻し
- D60 1000局+完全学習：[run 42447988443](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42447988443)の100独立自己対局ジョブ・集約・学習成功（後続の監査準備失敗は別工程）\n- D60独立200局： [run 42455966713](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42455966713)の20評価ジョブ成功、最終集約のみ[run 42456661212](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42456661212)でSHA再読戻し・保存成功
- 原1000局 SHA256 {GAMES_SHA}
- D60 NNUE＋完全Adam9190 Release SHA256 {TRAIN_SHA}
- D60独立200局 Release SHA256 {AUDIT_SHA}
- 新研究snapshot {NEW}
- 新推論 {MODEL_PATH} SHA256 {weights_sha}
- 旧D58学習前NNUE＋Adam8296保存。探索実装SHA256 {CPP}維持。
- この準備記録だけでは正式採用CI、main反映・配布の完了を意味しない。D61 Actionsのテスト・実USI・SHA読戻しを確認する。
""")
    specdoc="docs/現行エンジン仕様.md"
    modify(specdoc,"**現行のNNUE重みはD58（d190c1a1、Adam8296）**",
           "**現行のNNUE重みはD60（1f4ef9f3、Adam9190）**")
    modify(specdoc,"現行D56からPVS＋詰み探索で第7周1000局を自己対局し、標準200局比較で**121勝79敗・引分0／未決着0局**。",
           "現行D58からPVS＋詰み探索で第8周1000局を自己対局し、標準200局比較で**123勝77敗・引分0／未決着0局**。")
    modify(specdoc,"| 研究snapshot SHA256 | "+chr(96)+OLD+chr(96)+" |",
           "| 研究snapshot SHA256 | "+chr(96)+NEW+chr(96)+" |")
    modify(specdoc,"| 推論用ファイル | "+chr(96)+"models/fukaji-d58.nnue.npz"+chr(96)+" |",
           "| 推論用ファイル | "+chr(96)+MODEL_PATH+chr(96)+" |")
    modify(specdoc,"| 推論用SHA256 | "+chr(96)+OLD_WEIGHTS+chr(96)+" |",
           "| 推論用SHA256 | "+chr(96)+weights_sha+chr(96)+" |")
    change_section(specdoc,"## 現行モデルの採用判断と強さの不確実性","## 復元・研究との境界",
        """## 現行モデルの採用判断と強さの不確実性

[D61採用判断](decisions/2026-11-08_D61_D60学習後重みを現行採用.md)。D58/Adam8296から新1000局でD60/Adam9190へ学習。未見100序盤×先後200局でD60 123勝、D58 77勝、引分0・未決着0、合法18,815手。事前差+12に対し+46勝。v0.4.1 PVS＋詰み探索は不変更。未知相手への勝率は保証しない。

""")
    modify(specdoc,"次のNNUE自己学習は現行D58（Adam8296）のGitHub保存済み完全stateから開始し、更新前snapshot D58と更新後を同条件で対局比較する。",
           "次のNNUE自己学習は現行D60（Adam9190）のGitHub保存済み完全stateから開始し、更新前snapshot D60と更新後を同条件で対局比較する。")
    modify(specdoc,"失敗・悪化時は更新前D58へ戻す。","失敗・悪化時は更新前D60へ戻す。")
    append(specdoc,f"""## D61：正式D60 NNUE採用（2026-11-08）

- snapshot {NEW}、推論 {MODEL_PATH} SHA256 {weights_sha}、完全Adam9190。D60 Releaseより復元可能。
- [D61採用判断](decisions/2026-11-08_D61_D60学習後重みを現行採用.md)。USI v0.4.1とPVS＋詰み探索不変更。""")
    change_section("docs/00_開発ルーター.md","## 現行モデル","## 現在の研究・保存再開点",
        f"""## 現行モデル

**現行NNUE D60（{NEW}、Adam9190）**。D58から1000局学習、200局でD60 123勝・D58 77勝・未決着0。USI v0.4.1 PVS＋詰み探索維持。[D61判断](decisions/2026-11-08_D61_D60学習後重みを現行採用.md)、[完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d60-d58-trained-42447988443)。次周はD60/Adam9190だけを親にする。

""")
    append("docs/00_開発ルーター.md",f"""## 最新NNUE（D61、2026-11-08）

正式D60/Adam9190。200局比較123勝77敗、未決着0。v0.4.1・PVS＋詰み探索は不変更。[D61判断](decisions/2026-11-08_D61_D60学習後重みを現行採用.md)。""")
    append("AGENTS.md",f"""## 最新正式NNUE（D61、2026-11-08）

正式D60/Adam9190、snapshot {NEW}。旧D58/Adam8296から1000局学習、先後200局123勝77敗、未決着0。NNUEのみ変更、USI v0.4.1とPVS＋詰み探索不変更。次周は[D60完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d60-d58-trained-42447988443)を唯一の親とする。[D61判断](docs/decisions/2026-11-08_D61_D60学習後重みを現行採用.md)。""")
    change_section("docs/ROADMAP.md","## 現行モデル","## 現在の研究",
        f"""## 現行モデル

**正式D60（{NEW}、Adam9190）**。旧D58から新1000局学習、200局比較でD60 123勝・D58 77勝・未決着0。USI v0.4.1、PVS＋詰み探索不変更。[D61判断](decisions/2026-11-08_D61_D60学習後重みを現行採用.md)、[独立200局](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d60-d58-before-after-200games-42455966713)、[完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d60-d58-trained-42447988443)。次周唯一の親はD60/Adam9190。

""")
    append("docs/ROADMAP.md",f"""## D61：D60正式採用（2026-11-08）

D60は新1000局997決着、Adam8296→9190、直接200局123勝77敗・未決着0・合法18,815手。正式推論SHA {weights_sha}。[採用](decisions/2026-11-08_D61_D60学習後重みを現行採用.md)。""")
    for name in ("formal/model.json",specdoc):
        contract[name]=sha((ROOT/name).read_bytes())
    write_json(contract_path,contract)
    formal_load()
    assert sha((ROOT/"native/nnue_search_ffi.cpp").read_bytes())==CPP
    assert sha((ROOT/"models/fukaji-d58.nnue.npz").read_bytes())==OLD_WEIGHTS
    receipt={
        "schema":"fukaji-d61-nnue-adopt-prepare-v1",
        "current_snapshot_sha256":NEW,"former_snapshot_sha256":OLD,
        "optimizer_step":9190,"weights_path":MODEL_PATH,
        "weights_sha256":weights_sha,"unchanged_cpp_sha256":CPP,
        "usi_version":spec["version"],
        "train_archive_sha256":TRAIN_SHA,"audit_archive_sha256":AUDIT_SHA,
        "after_wins":123,"before_wins":77,"draws":0,"unresolved":0,
        "training_games":1000,"audit_games":200,"validated_plies":18815,
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
