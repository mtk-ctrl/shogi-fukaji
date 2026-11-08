"""D59: adopt stronger D58 NNUE after complete paired200 win.

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

OLD="89562fd4fed556f4a883dba465add35c532cc2c7ccdae9efd24d685270246363"
NEW="d190c1a1b8d5a44b3fe4791588306fb493f1b9534cca268fcbb656ba8604265c"
OLD_WEIGHTS="bcc1da5a3756c9a70fd864add32571a5b289a47d3614f794eed4fcf9802d5d24"
CPP="c9340af265368547c37cbecc311d8d3982900885b433d1b4ccdc27fbdd7d041a"
GAMES_SHA="a03be0ea93e62a1cefa2b40c6de2096a23f620690fb58603e346edd7062a86d1"
TRAIN_SHA="d8fb217fb21d80b3fe9423f4b9f6f6ca39ef3349850babfb7b7d4ab5c0bda778"
AUDIT_SHA="524d291fe5f9a7fe01ac02167992fa2c6992f936c3bd75787d1f652d29d8bc2b"
RUN=42439684313
MODEL_PATH="models/fukaji-d58.nnue.npz"
DECISION="docs/decisions/2026-11-08_D59_D58学習後重みを現行採用.md"
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
    request=json.loads((ROOT/"research/d59-adopt/START.json").read_text())
    expected={
        "enabled":True,"request_id":"d59-d58-20261108-verified-audit-adopt",
        "source_run":RUN,"former":OLD,"next":NEW,
        "games_trained":1000,"audit_games":200,
        "after_wins":121,"before_wins":79,"draws":0,"unresolved":0,
        "optimizer_before":7402,"optimizer_after":8296,
        "train_sha256":TRAIN_SHA,"audit_sha256":AUDIT_SHA,
        "search_sha256":CPP,"version":"FUKAJI v0.4.1",
        "authority":"F00/F20 preauthorized direct current NNUE continuation when stronger"
    }
    if request!=expected:raise ValueError("not exact D59 adoption request")
    spec=json.loads((ROOT/"formal/model.json").read_text())
    for k,v in {
        "version":"FUKAJI v0.4.1","search_backend_variant":"d48_pvs_mate",
        "search_implementation_sha256":CPP,"weights_sha256":OLD_WEIGHTS,
        "research_snapshot_sha256":OLD,"optimizer_step":7402,
        "weights_file":"models/fukaji-d56.nnue.npz"
    }.items():
        if spec.get(k)!=v:raise ValueError("old formal D56 model changed: "+k)
    if sha((ROOT/"native/nnue_search_ffi.cpp").read_bytes())!=CPP:
        raise ValueError("official PVS+mate engine changed")
    if sha((ROOT/spec["weights_file"]).read_bytes())!=OLD_WEIGHTS:
        raise ValueError("old D56 official weights changed")
    contract_path="research/d17-fast-self-learning/formal-contract.json"
    contract=json.loads((ROOT/contract_path).read_text())
    for name,value in contract.items():
        if sha((ROOT/name).read_bytes())!=value:raise ValueError("formal contract drift "+name)
    out.mkdir(parents=True,exist_ok=True)
    model_dir=out/"models";model_dir.mkdir(exist_ok=True)
    with read_archive(trained,TRAIN_SHA) as t:
        report=json.loads(member(t,"report.json"))
        req={
            "schema":"d58-learned1000-v1","baseline":OLD,"trained_sha256":NEW,
            "optimizer_step_before":7402,"optimizer_step_after":8296,
            "new_games":1000,"resolved_new_games":999,
            "old_bootstrap_replay_games":665,"prior_d56_replay_games":300,
            "source_sha256":GAMES_SHA,"formal_search_variant":"d48_pvs_mate",
            "strength_proven":False,"formal_model_changed":False
        }
        for k,v in req.items():
            if report.get(k)!=v:raise ValueError("D58 train mismatch "+k)
        if report.get("fit",{}).get("positions")!=57200:
            raise ValueError("D58 training positions drift")
        for ext in ("npz","json"):
            (model_dir/(NEW+"."+ext)).write_bytes(member(t,NEW+"."+ext))
    model,opt,_=load_snapshot(model_dir,NEW)
    if opt is None or opt["step"]!=8296:
        raise ValueError("D58 complete Adam8296 was not restored")
    if model.inputs!=2548 or model.hidden_weights.shape!=(2548,32):
        raise ValueError("D58 NNUE shape changed")
    with read_archive(audit,AUDIT_SHA) as t:
        result=json.loads(member(t,"report.json"))
        plan=json.loads(member(t,"plan.json"))
        req={
            "schema":"d58-paired200-result-v1","before":OLD,"after":NEW,
            "games":200,"opening_pairs":100,
            "after_wins":121,"before_wins":79,
            "draws":0,"unresolved":0,"difference":42,
            "predeclared_verdict":"after_better","validated_plies":18739,
            "formal_changed":False
        }
        for k,v in req.items():
            if result.get(k)!=v:raise ValueError("D58 audit mismatch "+k)
        if (plan.get("schema")!="d58-paired200-v1" or
            plan["before"]!=OLD or plan["after"]!=NEW or
            plan["pairs"]!=100 or plan["games"]!=200 or
            len(plan["families"])!=100 or
            len({x["parent_id"] for x in plan["families"]})!=100):
            raise ValueError("D58 paired opening plan changed")
        decision=plan.get("decision",{})
        if decision.get("margin_win_count")!=12 or decision.get("max_unresolved")!=10:
            raise ValueError("D58 adoption criteria changed after evaluation")
        for k,v in {"backend":"cpp_nnue","normal_ms":200,"long_ms":1000,
                    "max_long_per_side":10,"quiescence":0}.items():
            if plan.get("conditions",{}).get(k)!=v:raise ValueError("search or clock drift "+k)
        hashes=result.get("game_sha256",{})
        if len(hashes)!=200:raise ValueError("200 game digests missing")
        for name,digest_sha in hashes.items():
            if sha(member(t,"games/"+name))!=digest_sha:
                raise ValueError("D58 original audit game SHA mismatch "+name)
    dest=ROOT/MODEL_PATH
    if dest.exists():raise ValueError("D58 official path unexpectedly exists")
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
        "approved_by":"owner-delegated F00/F20 D58 121-79 paired200 with no unresolved",
        "research_snapshot_sha256":NEW,"research_parent_sha256":OLD,
        "weights_file":MODEL_PATH,"weights_sha256":weights_sha,
        "optimizer_step":8296,
        "training_source_sha256":GAMES_SHA,"training_release_sha256":TRAIN_SHA,
        "comparison_release_sha256":AUDIT_SHA,
        "model_distribution":"GitHub models/fukaji-d58.nnue.npz and D59 current release; full Adam8296 remains in D58 training Release.",
        "validation":{
            "opponent":"D56/Adam7402 before D58 1000 new selfplay; 100 unseen paired openings",
            "wins":121,"losses":79,"draws":0,"unresolved":0,
            "opening_families":100,"games":200,
            "planned_opening_families":100,"planned_games":200,
            "normal_move_time_ms":200,"longthink_ms":1000,
            "longthink_max_per_side":10,"quiescence":0,
            "search_backend_variant":"d48_pvs_mate",
            "strength_conclusion":"D58 won 121-79 against D56 in independent paired200; difference +42 passes frozen threshold +12"
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
    if decision_path.exists():raise ValueError("D59 adoption already registered")
    decision_path.write_text(f"""# D59：D58学習後NNUEを正式採用（2026-11-08）

## 採否
D56/Adam7402を親として新自己対局1000局（999決着）、Adam8296まで学習。
先後交替100組・200局、D58 121勝D56 79勝、引分0・未決着0、18,739合法手。
勝差+42、既定+12以上、未決着0は許容10以下。F00/F20本人委任によりD58 NNUEのみ現行採用。
外部相手への強さは未検証。PVS＋1/3/5手詰み／USI v0.4.1は不変更。

## 保存・戻し
- D58 [Actions run 42439684313](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42439684313)全125job成功
- 原1000局 SHA256 {GAMES_SHA}
- D58 NNUE＋完全Adam8296 Release SHA256 {TRAIN_SHA}
- D58独立200局 Release SHA256 {AUDIT_SHA}
- 新研究snapshot {NEW}
- 新推論 {MODEL_PATH} SHA256 {weights_sha}
- 旧D56学習前NNUE＋Adam7402保存。探索実装SHA256 {CPP}維持。
- この準備記録だけでは正式採用CI、main反映・配布の完了を意味しない。D59 Actionsのテスト・実USI・SHA読戻しを確認する。
""")
    specdoc="docs/現行エンジン仕様.md"
    modify(specdoc,"**現行のNNUE重みはD56（89562fd4、Adam7402）**",
           "**現行のNNUE重みはD58（d190c1a1、Adam8296）**")
    modify(specdoc,"現行D54からPVS＋詰み探索で第6周1000局を自己対局し、標準200局比較で**135勝65敗・引分0／未決着0局**。",
           "現行D56からPVS＋詰み探索で第7周1000局を自己対局し、標準200局比較で**121勝79敗・引分0／未決着0局**。")
    modify(specdoc,"| 研究snapshot SHA256 | "+chr(96)+OLD+chr(96)+" |",
           "| 研究snapshot SHA256 | "+chr(96)+NEW+chr(96)+" |")
    modify(specdoc,"| 推論用ファイル | "+chr(96)+"models/fukaji-d56.nnue.npz"+chr(96)+" |",
           "| 推論用ファイル | "+chr(96)+MODEL_PATH+chr(96)+" |")
    modify(specdoc,"| 推論用SHA256 | "+chr(96)+OLD_WEIGHTS+chr(96)+" |",
           "| 推論用SHA256 | "+chr(96)+weights_sha+chr(96)+" |")
    change_section(specdoc,"## 現行モデルの採用判断と強さの不確実性","## 復元・研究との境界",
        """## 現行モデルの採用判断と強さの不確実性

[D59採用判断](decisions/2026-11-08_D59_D58学習後重みを現行採用.md)。D56/Adam7402から新1000局でD58/Adam8296へ学習。未見100序盤×先後200局でD58 121勝、D56 79勝、引分0・未決着0、合法18,739手。事前差+12に対し+42勝。v0.4.1 PVS＋詰み探索は不変更。未知相手への勝率は保証しない。

""")
    modify(specdoc,"次のNNUE自己学習は現行D56（Adam7402）のGitHub保存済み完全stateから開始し、更新前snapshot D56と更新後を同条件で対局比較する。",
           "次のNNUE自己学習は現行D58（Adam8296）のGitHub保存済み完全stateから開始し、更新前snapshot D58と更新後を同条件で対局比較する。")
    modify(specdoc,"失敗・悪化時は更新前D56へ戻す。","失敗・悪化時は更新前D58へ戻す。")
    append(specdoc,f"""## D59：正式D58 NNUE採用（2026-11-08）

- snapshot {NEW}、推論 {MODEL_PATH} SHA256 {weights_sha}、完全Adam8296。D58 Releaseより復元可能。
- [D59採用判断](decisions/2026-11-08_D59_D58学習後重みを現行採用.md)。USI v0.4.1とPVS＋詰み探索不変更。""")
    change_section("docs/00_開発ルーター.md","## 現行モデル","## 現在の研究・保存再開点",
        f"""## 現行モデル

**現行NNUE D58（{NEW}、Adam8296）**。D56から1000局学習、200局でD58 121勝・D56 79勝・未決着0。USI v0.4.1 PVS＋詰み探索維持。[D59判断](decisions/2026-11-08_D59_D58学習後重みを現行採用.md)、[完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d58-d56-trained-42439684313)。次周はD58/Adam8296だけを親にする。

""")
    append("docs/00_開発ルーター.md",f"""## 最新NNUE（D59、2026-11-08）

正式D58/Adam8296。200局比較121勝79敗、未決着0。v0.4.1・PVS＋詰み探索は不変更。[D59判断](decisions/2026-11-08_D59_D58学習後重みを現行採用.md)。""")
    append("AGENTS.md",f"""## 最新正式NNUE（D59、2026-11-08）

正式D58/Adam8296、snapshot {NEW}。旧D56/Adam7402から1000局学習、先後200局121勝79敗、未決着0。NNUEのみ変更、USI v0.4.1とPVS＋詰み探索不変更。次周は[D58完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d58-d56-trained-42439684313)を唯一の親とする。[D59判断](docs/decisions/2026-11-08_D59_D58学習後重みを現行採用.md)。""")
    change_section("docs/ROADMAP.md","## 現行モデル","## 現在の研究",
        f"""## 現行モデル

**正式D58（{NEW}、Adam8296）**。旧D56から新1000局学習、200局比較でD58 121勝・D56 79勝・未決着0。USI v0.4.1、PVS＋詰み探索不変更。[D59判断](decisions/2026-11-08_D59_D58学習後重みを現行採用.md)、[独立200局](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d58-d56-before-after-200games-42439684313)、[完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d58-d56-trained-42439684313)。次周唯一の親はD58/Adam8296。

""")
    append("docs/ROADMAP.md",f"""## D59：D58正式採用（2026-11-08）

D58は新1000局999決着、Adam7402→8296、直接200局121勝79敗・未決着0・合法18,739手。正式推論SHA {weights_sha}。[採用](decisions/2026-11-08_D59_D58学習後重みを現行採用.md)。""")
    for name in ("formal/model.json",specdoc):
        contract[name]=sha((ROOT/name).read_bytes())
    write_json(contract_path,contract)
    formal_load()
    assert sha((ROOT/"native/nnue_search_ffi.cpp").read_bytes())==CPP
    assert sha((ROOT/"models/fukaji-d56.nnue.npz").read_bytes())==OLD_WEIGHTS
    receipt={
        "schema":"fukaji-d59-nnue-adopt-prepare-v1",
        "current_snapshot_sha256":NEW,"former_snapshot_sha256":OLD,
        "optimizer_step":8296,"weights_path":MODEL_PATH,
        "weights_sha256":weights_sha,"unchanged_cpp_sha256":CPP,
        "usi_version":spec["version"],
        "train_archive_sha256":TRAIN_SHA,"audit_archive_sha256":AUDIT_SHA,
        "after_wins":121,"before_wins":79,"draws":0,"unresolved":0,
        "training_games":1000,"audit_games":200,"validated_plies":18739,
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
