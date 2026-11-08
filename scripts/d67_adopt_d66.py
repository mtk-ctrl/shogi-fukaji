"""D67: event-driven, finite NNUE-only promotion of D66 after SHA-verified 1000+200.
Executed only after main D66 workflow succeeds; outcome is read from immutable
Release data rather than guessed and prefilled in an LLM timer prompt.
"""
from __future__ import annotations
import hashlib, io, json, tarfile
from pathlib import Path
import numpy as np
from fukaji.online_learning import load_snapshot
from fukaji.formal_model import load as formal_load

ROOT=Path(__file__).resolve().parents[1]
OLD="545dba627046621ec53b681d485978b9054776c717d894dcb0de55bec731d4a2"
OLD_WEIGHTS="fd5f832e0ded9519c23ad7469eee7a79f70d78f41bc0dab87c5f7bc57c8d478a"
CPP="c9340af265368547c37cbecc311d8d3982900885b433d1b4ccdc27fbdd7d041a"
MODEL_PATH="models/fukaji-d66.nnue.npz"
DECISION="docs/decisions/2026-11-08_D67_D66学習後NNUEを正式採用.md"

def digest(raw): return hashlib.sha256(raw).hexdigest()
def read_tgz(path, expected):
    if digest(Path(path).read_bytes()) != expected: raise ValueError("release archive SHA changed: "+str(path))
    return tarfile.open(path,"r:gz")
def member(tf, name):
    item=tf.extractfile(name)
    if item is None: raise ValueError("missing archive member "+name)
    return item.read()
def write_json(path, data):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
def edit_one(path, old, new):
    p=ROOT/path;text=p.read_text()
    if text.count(old)!=1: raise ValueError("documentation conflict "+path+": "+old[:65])
    p.write_text(text.replace(old,new))
def replace_section(path, start, end, new):
    p=ROOT/path;text=p.read_text()
    if text.count(start)!=1:raise ValueError("section ambiguous "+path)
    a=text.index(start);b=text.index(end,a)
    p.write_text(text[:a]+new+text[b:])
def append(path, new):
    p=ROOT/path;p.write_text(p.read_text().rstrip()+"\n\n"+new.strip()+"\n")

def prepare(trained, audit, out):
    q=json.loads((ROOT/"research/d67-adopt/START.json").read_text())
    permitted={"enabled","request_id","source_run","former","next","optimizer_before","optimizer_after",
               "train_sha256","audit_sha256","games_sha256","after_wins","before_wins",
               "draws","unresolved","validated_plies","positions"}
    if set(q)!=permitted or q["enabled"] is not True or q["request_id"]!="d67-d66-after-success" or q["former"]!=OLD:
        raise ValueError("unauthorized D67 adoption request")
    if type(q["source_run"]) is not int or q["source_run"]<=0:
        raise ValueError("no source run ID")
    for field in ("next","train_sha256","audit_sha256","games_sha256"):
        if not isinstance(q[field],str) or len(q[field])!=64 or any(c not in "0123456789abcdef" for c in q[field]):
            raise ValueError("bad SHA "+field)
    for field in ("optimizer_before","optimizer_after","after_wins","before_wins","draws",
                  "unresolved","validated_plies","positions"):
        if type(q[field]) is not int or q[field]<0:raise ValueError("invalid result value "+field)
    if (q["optimizer_before"]!=10974 or not 10974<q["optimizer_after"]<=20000 or
        q["after_wins"]+q["before_wins"]+q["draws"]+q["unresolved"]!=200 or
        q["after_wins"]-q["before_wins"]<12 or q["unresolved"]>10 or
        q["positions"]<1000 or q["validated_plies"]<200):
        raise ValueError("D66 failed predeclared adoption gate")
    new=q["next"]
    formal=json.loads((ROOT/"formal/model.json").read_text())
    for key,value in (
       ("version","FUKAJI v0.4.1"),("search_backend_variant","d48_pvs_mate"),
       ("search_implementation_sha256",CPP),("weights_sha256",OLD_WEIGHTS),
       ("research_snapshot_sha256",OLD),("optimizer_step",10974),
       ("weights_file","models/fukaji-d64.nnue.npz")):
        if formal.get(key)!=value:raise ValueError("formal current D64 drift "+key)
    if digest((ROOT/"native/nnue_search_ffi.cpp").read_bytes())!=CPP:
        raise ValueError("official PVS/mate changed")
    if digest((ROOT/formal["weights_file"]).read_bytes())!=OLD_WEIGHTS:
        raise ValueError("prior official NNUE bytes changed")
    contract_path="research/d17-fast-self-learning/formal-contract.json"
    contract=json.loads((ROOT/contract_path).read_text())
    for path,expected in contract.items():
        if digest((ROOT/path).read_bytes())!=expected:
            raise ValueError("frozen formal contract changed "+path)
    out.mkdir(parents=True,exist_ok=True)
    model_dir=out/"models";model_dir.mkdir(exist_ok=True)
    with read_tgz(trained,q["train_sha256"]) as t:
        train=json.loads(member(t,"report.json"))
        known={"schema":"d66-learned1000-v1","baseline":OLD,"trained_sha256":new,
            "optimizer_step_before":10974,"optimizer_step_after":q["optimizer_after"],
            "new_games":1000,"old_bootstrap_replay_games":665,"prior_d64_replay_games":300,
            "source_sha256":q["games_sha256"],"formal_search_variant":"d48_pvs_mate",
            "strength_proven":False,"formal_model_changed":False}
        for key,value in known.items():
            if train.get(key)!=value:raise ValueError("D66 trained state mismatch "+key)
        if train.get("resolved_new_games",0)<800 or train.get("fit",{}).get("positions")!=q["positions"]:
            raise ValueError("training report mismatch")
        for ext in ("npz","json"):
            (model_dir/(new+"."+ext)).write_bytes(member(t,new+"."+ext))
    learned,opt,_=load_snapshot(model_dir,new)
    if opt is None or opt["step"]!=q["optimizer_after"]:
        raise ValueError("D66 complete Adam state could not be restored")
    if learned.inputs!=2548 or learned.hidden_weights.shape!=(2548,32):
        raise ValueError("D66 NNUE architecture changed")
    with read_tgz(audit,q["audit_sha256"]) as t:
        score=json.loads(member(t,"report.json"))
        plan=json.loads(member(t,"plan.json"))
        fixed={"schema":"d66-paired200-result-v1","before":OLD,"after":new,
            "games":200,"opening_pairs":100,
            "after_wins":q["after_wins"],"before_wins":q["before_wins"],
            "draws":q["draws"],"unresolved":q["unresolved"],
            "difference":q["after_wins"]-q["before_wins"],
            "validated_plies":q["validated_plies"],
            "predeclared_verdict":"after_better","formal_changed":False}
        for key,value in fixed.items():
            if score.get(key)!=value:raise ValueError("D66 audit result mismatch "+key)
        if (plan.get("schema")!="d66-paired200-v1" or
            plan.get("before")!=OLD or plan.get("after")!=new or
            plan.get("pairs")!=100 or plan.get("games")!=200 or
            len(plan.get("families",[]))!=100 or
            len({x["parent_id"] for x in plan["families"]})!=100):
            raise ValueError("independent paired plan changed")
        decision=plan.get("decision",{})
        if decision.get("margin_win_count")!=12 or decision.get("max_unresolved")!=10:
            raise ValueError("frozen D66 comparison rules changed")
        for key,value in (("backend","cpp_nnue"),("normal_ms",200),
                          ("long_ms",1000),("max_long_per_side",10),("quiescence",0)):
            if plan.get("conditions",{}).get(key)!=value:
                raise ValueError("clock/search changed "+key)
        checks=score.get("game_sha256",{})
        if not isinstance(checks,dict) or len(checks)!=200:raise ValueError("missing original200 digests")
        for name,oldsha in checks.items():
            if name.startswith("/") or ".." in name or len(oldsha)!=64:
                raise ValueError("unsafe audit game path")
            if digest(member(t,"games/"+name))!=oldsha:
                raise ValueError("altered audit game "+name)
    dest=ROOT/MODEL_PATH
    if dest.exists():raise ValueError("D66 formal inference already exists")
    buf=io.BytesIO()
    np.savez_compressed(buf,w=learned.hidden_weights,b=learned.hidden_bias,
                         v=learned.output_weights,o=np.array([learned.output_bias],dtype=np.float32))
    raw=buf.getvalue();dest.write_bytes(raw)
    weights_sha=digest(raw)
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:
        if set(z.files)!={"w","b","v","o"}:raise ValueError("NNUE inference shape contract")
        for name,expected in (("w",learned.hidden_weights),("b",learned.hidden_bias),("v",learned.output_weights)):
            np.testing.assert_array_equal(z[name],expected)
            if not np.isfinite(z[name]).all():raise ValueError("non-finite model weights")
        assert float(z["o"][0])==float(learned.output_bias)
    newformal=dict(formal)
    newformal.update({
      "approved_on":"2026-11-08",
      "approved_by":f"owner-delegated D67 D66 wins {q['after_wins']}-{q['before_wins']} on 200 paired",
      "research_snapshot_sha256":new,"research_parent_sha256":OLD,
      "weights_file":MODEL_PATH,"weights_sha256":weights_sha,
      "optimizer_step":q["optimizer_after"],
      "training_source_sha256":q["games_sha256"],"training_release_sha256":q["train_sha256"],
      "comparison_release_sha256":q["audit_sha256"],
      "model_distribution":f"GitHub {MODEL_PATH} and D67 current release; complete Adam{q['optimizer_after']} in D66 trained Release.",
      "validation":{
        "opponent":"D64/Adam10974 before D66 selfplay, 100 unseen paired openings",
        "wins":q["after_wins"],"losses":q["before_wins"],
        "draws":q["draws"],"unresolved":q["unresolved"],
        "opening_families":100,"games":200,"planned_opening_families":100,
        "planned_games":200,"normal_move_time_ms":200,"longthink_ms":1000,
        "longthink_max_per_side":10,"quiescence":0,"search_backend_variant":"d48_pvs_mate",
        "strength_conclusion":f"D66 {q['after_wins']}-{q['before_wins']} difference +{q['after_wins']-q['before_wins']} passes frozen +12"
      }})
    for key in ("version","architecture","backend","quiescence","default_move_time_ms",
                "default_search_depth","search_backend_variant","search_implementation_sha256","search_adoption"):
        if newformal[key]!=formal[key]:raise ValueError("unauthorized search change "+key)
    write_json("formal/model.json",newformal)
    loaded,loaded_spec=formal_load()
    if loaded_spec["weights_sha256"]!=weights_sha:raise ValueError("formal model load failure")
    np.testing.assert_array_equal(loaded.hidden_weights,learned.hidden_weights)
    np.testing.assert_array_equal(loaded.output_weights,learned.output_weights)
    decision_path=ROOT/DECISION
    if decision_path.exists():raise ValueError("D67 adoption duplicate")
    mark=f"D66 {q['after_wins']}勝・D64 {q['before_wins']}勝・引分{q['draws']}・未決着{q['unresolved']}"
    decision_path.write_text(
       f"# D67：D66 NNUE正式採用（2026-11-08）\n\n"
       f"新1000局からD66 Adam{q['optimizer_after']}に学習し、独立200局で{mark}。"
       f"事前勝差+12に対し+{q['after_wins']-q['before_wins']}。"
       f"合法手数{q['validated_plies']}。NNUE以外の探索とUSI v0.4.1は変更しない。\n\n"
       f"- [D66実run](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/{q['source_run']})\n"
       f"- 原1000局 SHA256 {q['games_sha256']}\n"
       f"- 全Adam学習 SHA256 {q['train_sha256']}\n"
       f"- 200局監査 SHA256 {q['audit_sha256']}\n"
       f"- 正式snapshot {new}、推論SHA256 {weights_sha}\n")
    specdoc="docs/現行エンジン仕様.md"
    edit_one(specdoc,"**現行のNNUE重みはD64（545dba62、Adam10974）**",
             f"**現行のNNUE重みはD66（{new[:8]}、Adam{q['optimizer_after']}）**")
    edit_one(specdoc,"| 研究snapshot SHA256 | "+chr(96)+OLD+chr(96)+" |",
             "| 研究snapshot SHA256 | "+chr(96)+new+chr(96)+" |")
    edit_one(specdoc,"| 推論用ファイル | "+chr(96)+"models/fukaji-d64.nnue.npz"+chr(96)+" |",
             "| 推論用ファイル | "+chr(96)+MODEL_PATH+chr(96)+" |")
    edit_one(specdoc,"| 推論用SHA256 | "+chr(96)+OLD_WEIGHTS+chr(96)+" |",
             "| 推論用SHA256 | "+chr(96)+weights_sha+chr(96)+" |")
    edit_one(specdoc,"次のNNUE自己学習は現行D64（Adam10974）のGitHub保存済み完全stateから開始し、更新前snapshot D64と更新後を同条件で対局比較する。",
             f"次のNNUE自己学習は現行D66（Adam{q['optimizer_after']}）のGitHub保存済み完全stateから開始し、更新前snapshot D66と更新後を同条件で対局比較する。")
    edit_one(specdoc,"失敗・悪化時は更新前D64へ戻す。","失敗・悪化時は更新前D66へ戻す。")
    replace_section(specdoc,"## 現行モデルの採用判断と強さの不確実性","## 復元・研究との境界",
       f"## 現行モデルの採用判断と強さの不確実性\n\n"
       f"[D67採用判断](decisions/2026-11-08_D67_D66学習後NNUEを正式採用.md)。{mark}。"
       f"合法手数{q['validated_plies']}。勝差+{q['after_wins']-q['before_wins']}。"
       f"USI v0.4.1とPVS・詰み探索は変更しない。\n\n")
    append(specdoc,f"## D67：正式D66 NNUE採用（2026-11-08）\n\n"
       f"- snapshot {new}、推論SHA256 {weights_sha}、Adam{q['optimizer_after']}。"
       f"[完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d66-d64-trained-{q['source_run']})。")
    common=(f"正式D66（{new}、Adam{q['optimizer_after']}）。{mark}。"
            f"[D67判断](decisions/2026-11-08_D67_D66学習後NNUEを正式採用.md)。"
            f"次周唯一親は[D66完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d66-d64-trained-{q['source_run']})。"
            f"USI v0.4.1・PVS＋詰み探索不変。")
    replace_section("docs/00_開発ルーター.md","## 現行モデル","## 現在の研究・保存再開点",
                    "## 現行モデル\n\n"+common+"\n\n")
    append("docs/00_開発ルーター.md","## 最新NNUE（D67、2026-11-08）\n\n"+common)
    append("AGENTS.md",f"## 最新正式NNUE（D67、2026-11-08）\n\n"
       f"正式D66／Adam{q['optimizer_after']}、snapshot {new}。{mark}。"
       f"学習前D64を保全、NNUEのみ採用。[完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d66-d64-trained-{q['source_run']})。")
    replace_section("docs/ROADMAP.md","## 現行モデル","## 現在の研究",
                    "## 現行モデル\n\n"+common+"\n\n")
    append("docs/ROADMAP.md","## D67：D66正式採用（2026-11-08）\n\n"+common)
    for path in ("formal/model.json",specdoc):contract[path]=digest((ROOT/path).read_bytes())
    write_json(contract_path,contract)
    formal_load()
    assert digest((ROOT/"native/nnue_search_ffi.cpp").read_bytes())==CPP
    assert digest((ROOT/"models/fukaji-d64.nnue.npz").read_bytes())==OLD_WEIGHTS
    receipt={"schema":"d67-verified-auto-adoption-v1",
        "source_run":q["source_run"],"former_snapshot_sha256":OLD,
        "current_snapshot_sha256":new,"optimizer_step":q["optimizer_after"],
        "weights_path":MODEL_PATH,"weights_sha256":weights_sha,
        "unchanged_cpp_sha256":CPP,"usi_version":"FUKAJI v0.4.1",
        "train_archive_sha256":q["train_sha256"],
        "games_archive_sha256":q["games_sha256"],
        "audit_archive_sha256":q["audit_sha256"],
        "after_wins":q["after_wins"],"before_wins":q["before_wins"],
        "draws":q["draws"],"unresolved":q["unresolved"],
        "training_games":1000,"audit_games":200,
        "validated_plies":q["validated_plies"],
        "contract_sha256":digest((ROOT/contract_path).read_bytes())}
    write_json(str(out/"receipt.json"),receipt)
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--trained",type=Path,required=True)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    prepare(args.trained,args.audit,args.out)
