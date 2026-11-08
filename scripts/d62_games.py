"""D62 1000-game training data transport: 5 waves x 20 real Actions jobs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import d41_public_20 as core
from scripts import d62_native_game as native_game
from scripts import d43_games as history
from scripts import d42_d37_vs_d41_paired100 as d42
from fukaji.online_cycle import fresh_prefix
from fukaji.online_learning import atomic_json,load_snapshot,validate_game,digest,game_key

BASE="1f4ef9f32dfff7a94ad3ddb1c57f64f123ca8b6f6caa422cb9c86ace06c57654"
STATE_SHA="41c8f56b76cd5d7aa87bff9131d553b304fa4134bf7448c05d3309458f66760e"
D41_SHA="8708db460d4d2bcad949d91ab881d9aeb804abcb8e28808dbbb70bda5c21bd9f"
D42_SHA="5b2cb8495ae7fcdbe2b9181c3d5b3aedb87b2fd72226b1e4de76abb2d3d1f570"
SEED=202611083220
D43_TRAIN_SHA="955a1785940f9f2aac946739af1b8637e5ef53d06297e6ad7d32b75a5a94a236"
D43_SELFPLAY_SHA="fe0a4eaec246a7e2e32c911c4327232eef83f8a87897e7a3a96538220ced6bbd"
D43_AUDIT_SHA="c08ccab3f5e9ef90c288e8ccfc5f3951eb188dfc30876af775c2ba3cfb03f510"
D45_TRAIN_SHA="d44349470b4068224ce1383f29d1afdf941c39931f6141c278927976da52c820"
D45_SELFPLAY_SHA="be22c4e92eff0b684f06502dcdf9be52071ab60beab5a4a91edd8605f4d873c4"
D45_AUDIT_SHA="6531747f10d015bec7817186f6fbed149aa6ac1822d94fbcce7858a7ceae211b"
D47_AUDIT_SHA="7acd4abaed27f1598520228987067c459f677c2985544e8cf532d0eb7381e5a7"
D48_AUDIT_SHA="427ebf9a8671d56b3f3b0fef24a62a2242e293a559e4307027747d30eb3a2f65"
D50_TRAIN_SHA="a6df97b104302a4b7c9f570c289af0d001155eea44f3a7c9bb0f21184a2b3688"
D50_SELFPLAY_SHA="92a27e9ea5094157e1213086966bfae9d3921c0537e1506b3563279bc63faf76"
D50_AUDIT_SHA="f6acc785531eab92e199dfb5df4d4a64f5e55246c08ec0962e24b6123fb81324"
D52_TRAIN_SHA="65acb35edd7e19a29b86c9cd321e245ad98cfa78f471f53e5e083f3ee473f793"
D52_SELFPLAY_SHA="d8adb89bff7d3cd2dc83c994f6664ce73ef7798e6326cc6a893a2ef2ca1d2134"
D52_AUDIT_SHA="5f05369e2ad7edec6aa4098a9393889c989697e83a607cdc062967049630a542"
D54_TRAIN_SHA="19a0f06427d68e4b98f9b4af67722f909937b272134837af79315a25c7e401a1"
D54_SELFPLAY_SHA="6eb7db54ccea6fa9da9a70f9d7aef25709dd13f97ba9b94f0d81f764f54c44ac"
D54_AUDIT_SHA="2506e8a17f9c3d775f774adefda5d7da1cf1355d4b1f0449fe41dc6c25c846bf"
D56_TRAIN_SHA="4aa33cbc6991cb69008bb3b586ca089e36ed3a79f2c80340fc53dd8198c8bb87"
D56_SELFPLAY_SHA="94a6cd1fae005ad3bf31d50cdd32a75413b946269a0b4875b9ee9d86dde2c21c"
D56_AUDIT_SHA="df4806113e4d1b07218172bc6464233783a6c10bec1c636a0d0c9bab9ef500dd"
D58_TRAIN_SHA="d8fb217fb21d80b3fe9423f4b9f6f6ca39ef3349850babfb7b7d4ab5c0bda778"
D58_SELFPLAY_SHA="a03be0ea93e62a1cefa2b40c6de2096a23f620690fb58603e346edd7062a86d1"
D58_AUDIT_SHA="524d291fe5f9a7fe01ac02167992fa2c6992f936c3bd75787d1f652d29d8bc2b"
D60_TRAIN_SHA="406c67f6d03c79f7a38e49e1e9c9ab35b21c68ac6d07e1029f4d45746cf9f07e"
D60_SELFPLAY_SHA="1db4cdb8ba016479b67ab5c46ec111031bbdda11892433c30c5c12fe306e3b7e"
D60_AUDIT_SHA="6b9f2c432481d19caded577858d118fb99a028142d25d6dd239f1c75bb16ed11"
SEARCH_SHA="c9340af265368547c37cbecc311d8d3982900885b433d1b4ccdc27fbdd7d041a"
ROUNDS=5
LANES=20
PER_LANE=10
TOTAL=1000

def checksum(p,sha):
    if digest(Path(p).read_bytes())!=sha:raise ValueError("SHA mismatch: "+str(p))
def older_d40(state):
    # Same frozen, verified D40 parent reconstruction used by D42.
    pilots=["a672b8bb21b18419bbf97aa8a03c23b043aa463e289ecedad0a4a80727ae7251",
    "66b8bb3cfdccac6997a85ec4d8f96000ffa35a0be450d933e15ad9c1805ccdbd",
    "6bf701d0a96a844b3bd9f70f3acaeef476ee018f5eb2ea14f3684bd3b6f87f82",
    "175454137ddca8b4ccd1433e5dc76a1c3248a41aef226e4b75f5ced4dc90bec5"]
    used=set(state["excluded_openings"])|set(pilots)
    ids=[fresh_prefix(20261104223027+i*1019,used)[1] for i in range(50)]
    canonical=json.dumps(ids,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    if digest(canonical)!="3cafea69849c65e2cf0c570a67d6a8af58263b6070e842dd732754baec57abf7":
        raise ValueError("D40 frozen parents fail digest")
    return set(ids)|set(pilots)

def prepare(state,d41_source,d42_source,d43_trained,d43_selfplay,d43_audit,d45_trained,d45_selfplay,d45_audit,d47_audit,d48_audit,d50_trained,d50_selfplay,d50_audit,d52_trained,d52_selfplay,d52_audit,d54_trained,d54_selfplay,d54_audit,d56_trained,d56_selfplay,d56_audit,d58_trained,d58_selfplay,d58_audit,d60_trained,d60_selfplay,d60_audit,out):
    for path,sha in ((state,STATE_SHA),(d41_source,D41_SHA),(d42_source,D42_SHA),
                     (d43_trained,D43_TRAIN_SHA),(d43_selfplay,D43_SELFPLAY_SHA),
                     (d43_audit,D43_AUDIT_SHA),(d45_trained,D45_TRAIN_SHA),
                     (d45_selfplay,D45_SELFPLAY_SHA),(d45_audit,D45_AUDIT_SHA),
                     (d47_audit,D47_AUDIT_SHA),(d48_audit,D48_AUDIT_SHA),
                     (d50_trained,D50_TRAIN_SHA),(d50_selfplay,D50_SELFPLAY_SHA),(d50_audit,D50_AUDIT_SHA),
                     (d52_trained,D52_TRAIN_SHA),(d52_selfplay,D52_SELFPLAY_SHA),(d52_audit,D52_AUDIT_SHA),
                     (d54_trained,D54_TRAIN_SHA),(d54_selfplay,D54_SELFPLAY_SHA),(d54_audit,D54_AUDIT_SHA),
                     (d56_trained,D56_TRAIN_SHA),(d56_selfplay,D56_SELFPLAY_SHA),(d56_audit,D56_AUDIT_SHA),
                     (d58_trained,D58_TRAIN_SHA),(d58_selfplay,D58_SELFPLAY_SHA),(d58_audit,D58_AUDIT_SHA),
                     (d60_trained,D60_TRAIN_SHA),(d60_selfplay,D60_SELFPLAY_SHA),(d60_audit,D60_AUDIT_SHA)):
        checksum(path,sha)
    manifest=json.loads(Path("formal/model.json").read_text())
    if (manifest["research_snapshot_sha256"]!=BASE or manifest["optimizer_step"]!=9190 or
        manifest.get("search_backend_variant")!="d48_pvs_mate" or
        manifest.get("search_implementation_sha256")!=SEARCH_SHA or
        manifest.get("version")!="FUKAJI v0.4.1"):
        raise ValueError("D62 requires adopted D60 v0.4.1 + exact adopted search") 
    if manifest["weights_sha256"]!="__D61_FORMAL_WEIGHT_SHA__":
        raise ValueError("D60 official inference weights changed")
    if manifest["research_snapshot_sha256"]!=BASE or manifest["optimizer_step"]!=9190:
        raise ValueError("not current D60/Adam9190; do not start")
    with tarfile.open(state,"r:gz") as t:
        meta=json.load(t.extractfile("d17-state/state.json"))
        if meta["research_latest"]!=core.D37:raise ValueError("D37 historical splits changed")
    models=out/"runtime"/"models";models.mkdir(parents=True,exist_ok=True)
    with tarfile.open(d60_trained,"r:gz") as t:
        report=json.load(t.extractfile("report.json"))
        if (report["schema"]!="d60-learned1000-v1" or
            report["baseline"]!="d190c1a1b8d5a44b3fe4791588306fb493f1b9534cca268fcbb656ba8604265c" or report["trained_sha256"]!=BASE or
            report["optimizer_step_before"]!=8296 or report["optimizer_step_after"]!=9190 or
            report["source_sha256"]!=D60_SELFPLAY_SHA):
            raise ValueError("D60 full trained parent state failed verification")
        for ext in ("npz","json"):
            name=f"{BASE}.{ext}"
            (models/name).write_bytes(t.extractfile(name).read())
    model,opt,_=load_snapshot(models,BASE)
    if opt is None or opt["step"]!=9190:
        raise ValueError("D60 Adam9190 was not restored")
    with tarfile.open(d43_selfplay,"r:gz") as t:
        previous=json.load(t.extractfile("plan.json"))
        prev_report=json.load(t.extractfile("report.json"))
        if (previous["schema"]!="d43-d37-1000-v1" or len(previous["families"])!=1000 or
            prev_report["games"]!=1000 or prev_report["model"]!=core.D37):
            raise ValueError("first-round train history wrong")
    with tarfile.open(d43_audit,"r:gz") as t:
        prior_audit=json.load(t.extractfile("plan.json"))
        prior_result=json.load(t.extractfile("report.json"))
        if (prior_audit["schema"]!="d43-paired200-v1" or
            len(prior_audit["families"])!=100 or
            prior_result["before"]!=core.D37 or prior_result["after"]!="bf8c26f9f8f6adff8a87c46ab04fcf5cd76a7915be241da11b99bb4ae2ac8b9f" or
            prior_result["after_wins"]!=150 or prior_result["before_wins"]!=50):
            raise ValueError("D43 old held-out comparison mismatch")
    with tarfile.open(d42_source,"r:gz") as t:
        d42_plan=json.load(t.extractfile("plan.json"))
        if d42_plan["schema"]!="fukaji-d42-before-after-20lane-paired100-v1":
            raise ValueError("D42 role boundary changed")
    train_old=d42.read_d41_parent_ids(d41_source)
    protected=set(meta["protected_parents"])|set(meta["excluded_openings"])|older_d40(meta)|train_old
    protected.update(x["parent_id"] for x in d42_plan["families"])
    protected.update(x["parent_id"] for x in previous["families"])
    protected.update(x["parent_id"] for x in prior_audit["families"])
    # D45 prior train/audit and D47/D48 search benchmarks are protected.
    for file,label,count,schema in (
       (d45_selfplay,"D45 train",1000,"d45-d43-1000-v1"),
       (d45_audit,"D45 audit",100,"d45-paired200-v1"),
       (d47_audit,"D47 audit",150,"d47-pvs-paired300-v1"),
       (d48_audit,"D48 audit",50,"d48-pvs-mate-paired100-v1"),
       (d50_selfplay,"D50 train",1000,"d50-d45-1000-v1"),
       (d50_audit,"D50 audit",100,"d50-paired200-v1"),
        (d52_selfplay,"D52 train",1000,"d52-d50-1000-v1"),
        (d52_audit,"D52 audit",100,"d52-paired200-v1"),
         (d54_selfplay,"D54 train",1000,"d54-d52-1000-v1"),
         (d54_audit,"D54 audit",100,"d54-paired200-v1"),
         (d56_selfplay,"D56 train",1000,"d56-d54-1000-v1"),
         (d56_audit,"D56 audit",100,"d56-paired200-v1"),
         (d58_selfplay,"D58 train",1000,"d58-d56-1000-v1"),
         (d58_audit,"D58 audit",100,"d58-paired200-v1"),
         (d60_selfplay,"D60 train",1000,"d60-d58-1000-v1"),
         (d60_audit,"D60 audit",100,"d60-paired200-v1")):
        with tarfile.open(file,"r:gz") as z:
            item=next((m for m in z.getmembers() if m.name.lstrip("./")=="plan.json"),None)
            if item is None:raise ValueError("missing protected plan: "+label)
            frozen=json.load(z.extractfile(item))
        if frozen["schema"]!=schema or len(frozen["families"])!=count:
            raise ValueError("protected plan schema/count mismatch: "+label)
        ids=[f["parent_id"] for f in frozen["families"]]
        if len(set(ids))!=count:raise ValueError("duplicate "+label)
        protected.update(ids)
    excluded=set(protected)
    families=[]
    for i in range(TOTAL):
        prefix,parent=fresh_prefix(SEED+i*1019,excluded)
        if parent in protected or len(prefix)!=6:
            raise ValueError("reused training/audit parent or invalid opening")
        families.append({"index":i,"prefix":prefix,"parent_id":parent,
                         "lane":(i//PER_LANE)%LANES,"round":i//(PER_LANE*LANES)})
        excluded.add(parent)
    if len({x["parent_id"] for x in families})!=TOTAL:
        raise ValueError("1000 independent second-round openings missing")
    plan={"schema":"d62-d60-1000-v1","baseline":BASE,"adam_step":9190,
          "search_variant":"d48_pvs_mate","search_sha256":SEARCH_SHA,
          "formal_usi":"FUKAJI v0.4.1",
          "seed":SEED,"games":TOTAL,"rounds":ROUNDS,"lanes":LANES,
          "per_lane_per_round":PER_LANE,"protected_count":len(protected),
          "d60_current_training_sha256":D60_TRAIN_SHA,
          "d60_prior_training_games_sha256":D60_SELFPLAY_SHA,
          "d60_prior_audit_sha256":D60_AUDIT_SHA,
          "d58_historical_training_sha256":D58_TRAIN_SHA,
          "d58_prior_training_games_sha256":D58_SELFPLAY_SHA,
          "d58_prior_audit_sha256":D58_AUDIT_SHA,
          "d56_historical_training_sha256":D56_TRAIN_SHA,
          "d56_prior_training_games_sha256":D56_SELFPLAY_SHA,
          "d56_prior_audit_sha256":D56_AUDIT_SHA,
          "d54_historical_training_sha256":D54_TRAIN_SHA,
          "d54_prior_training_games_sha256":D54_SELFPLAY_SHA,
          "d54_prior_audit_sha256":D54_AUDIT_SHA,
          "d52_historical_training_sha256":D52_TRAIN_SHA,
           "d52_prior_training_games_sha256":D52_SELFPLAY_SHA,
           "d52_prior_audit_sha256":D52_AUDIT_SHA,
           "d50_historical_training_sha256":D50_TRAIN_SHA,
          "d50_historical_training_games_sha256":D50_SELFPLAY_SHA,
          "d50_historical_audit_sha256":D50_AUDIT_SHA,
          "d45_historical_training_sha256":D45_TRAIN_SHA,
          "d45_prior_training_games_sha256":D45_SELFPLAY_SHA,
          "d45_prior_audit_sha256":D45_AUDIT_SHA,
          "d43_prior_training_games_sha256":D43_SELFPLAY_SHA,
          "d43_prior_audit_sha256":D43_AUDIT_SHA,
          "normal_ms":200,"long_ms":1000,"max_long_per_side":10,
          "max_plies":512,"cpp_nnue":True,"exploration_first_24":.05,
          "families":families}
    atomic_json(out/"runtime"/"master.json",plan)
    core.pack_tar(out/"runtime",out/"runtime.tar.gz")
    atomic_json(out/"receipt.json",{"plan_sha256":digest((out/"runtime"/"master.json").read_bytes()),
        "planned_games":TOTAL,"input_trained_sha256":D60_TRAIN_SHA,
        "excluded_count":len(protected)})
    print(json.dumps({"prepared":TOTAL,"rounds":ROUNDS,"lanes":LANES,
                      "excluded":len(protected),"baseline":BASE[:12]}))

def segment(work,index,output):
    master=json.loads((work/"master.json").read_text())
    if not 0<=index<TOTAL:raise ValueError("bad game index")
    item=master["families"][index]
    lane=item["lane"]
    slot={"prefix":item["prefix"],"parent_id":item["parent_id"]}
    # Reuse the already proven D41 game/clock/checkpoint implementation.
    native_game.SEED=SEED+index*31
    atomic_json(work/"plan.json",{"baseline":BASE,
             "families":[{"openings":[slot,slot]} for _ in range(LANES)]})
    native_game.segment(work,lane,0,output,36)

def worker(runtime,round_number,lane,out):
    if not(0<=round_number<ROUNDS and 0<=lane<LANES):raise ValueError("invalid shard")
    work=out/"runtime";core.safe_extract(runtime,work)
    master=json.loads((work/"master.json").read_text())
    model,opt,_=load_snapshot(work/"models",BASE)
    if opt["step"]!=9190:raise ValueError("D60 Adam9190 optimizer not inherited")
    games=out/"games";games.mkdir(parents=True,exist_ok=True)
    for i in range(round_number*200+lane*PER_LANE,round_number*200+(lane+1)*PER_LANE):
        dest=games/f"temporary-{i:04d}"
        final=games/f"game-{i:04d}.json"
        if final.exists():continue
        dest.mkdir(exist_ok=True)
        for trial in range(22):
            if (dest/"game-0.json").exists():break
            cmd=[sys.executable,__file__,"segment","--work",str(work),
                 "--index",str(i),"--out",str(dest)]
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=150)
            if p.returncode:raise RuntimeError(p.stderr[-1800:])
        if not(dest/"game-0.json").exists():raise RuntimeError("unfinished game "+str(i))
        obj=json.loads((dest/"game-0.json").read_text())
        obj.update({"game_index":i,"round":round_number,"lane":lane})
        if obj["parent_id"]!=master["families"][i]["parent_id"]:
            raise ValueError("wrong parent")
        validate_game(obj,model,training=True)
        atomic_json(final,obj)
        (dest/"game-0.json").unlink()
        # Delete temporary only after the immutable complete game is recorded.
        (dest/"game-0.partial.json").unlink(missing_ok=True)
        dest.rmdir()
    pkg=out/f"r{round_number}-lane-{lane:02d}.tar.gz"
    export=out/"complete";export.mkdir(exist_ok=True)
    for p in games.glob("game-*.json"):
        (export/p.name).write_bytes(p.read_bytes())
    core.pack_tar(export,pkg)
    print(json.dumps({"round":round_number,"lane":lane,"games":PER_LANE,
                      "sha256":digest(pkg.read_bytes())}))

def collect(runtime,downloads,out):
    work=out/"runtime";core.safe_extract(runtime,work)
    master=json.loads((work/"master.json").read_text())
    model,opt,_=load_snapshot(work/"models",BASE)
    expected={f"r{r}-lane-{lane:02d}.tar.gz" for r in range(ROUNDS) for lane in range(LANES)}
    present={p.name for p in downloads.glob("r*-lane-*.tar.gz")}
    if present!=expected:raise ValueError("missing worker archives: "+str(sorted(expected-present)[:6]))
    publication=out/"publication";(publication/"games").mkdir(parents=True,exist_ok=True)
    seen=set();parents=set();counter=Counter();hashes={}
    for r in range(ROUNDS):
      for lane in range(LANES):
        archive=downloads/f"r{r}-lane-{lane:02d}.tar.gz"
        names={f"game-{r*200+lane*10+j:04d}.json" for j in range(PER_LANE)}
        with tarfile.open(archive,"r:gz") as t:
          members={m.name for m in t.getmembers() if m.isfile()}
          if members!=names:raise ValueError("incorrect shard contents")
          for name in sorted(names):
            raw=t.extractfile(name).read();g=json.loads(raw)
            i=g["game_index"];item=master["families"][i]
            if (g["lane"]!=lane or g["round"]!=r or g["models"]!=[BASE,BASE] or
                g["parent_id"]!=item["parent_id"] or g["moves"][:6]!=item["prefix"] or
                g["role"]!="train"):raise ValueError("provenance mismatch")
            validate_game(g,model,training=True)
            if g.get("conditions",{}).get("formal_search")!="d48_pvs_mate":
                raise ValueError("old search used in D58 game")
            key=game_key(g)
            if key in seen or g["parent_id"] in parents:raise ValueError("duplicate")
            seen.add(key);parents.add(g["parent_id"])
            if len(g["telemetry"])!=len(g["moves"])-6:raise ValueError("telemetry missing")
            used=[0,0]
            for telemetry in g["telemetry"]:
                if telemetry.get("backend")!="cpp_nnue" or telemetry.get("budget_ms") not in (200,1000) or "research_mate_nodes" not in telemetry:
                    raise ValueError("changed NNUE backend or move time")
                if telemetry.get("longthink"):used[int(telemetry["side"])]+=1
            if used!=g["conditions"]["longthink_used_per_side"] or max(used)>10:
                raise ValueError("longthink exceeded or trace changed")
            counter[g["status"]]+=1;counter["plies"]+=len(g["moves"])
            hashes[name]=digest(raw)
            (publication/"games"/name).write_bytes(raw)
    if len(seen)!=TOTAL:raise ValueError("not exactly 1000 new complete games")
    report={"schema":"d62-selfplay1000-result-v1","model":BASE,"games":TOTAL,
            "formal_search_variant":"d48_pvs_mate",
            "formal_search_sha256":SEARCH_SHA,
            "resolved":sum(counter[k] for k in ("checkmate","perpetual_check","repetition_draw")),
            "unresolved":TOTAL-sum(counter[k] for k in ("checkmate","perpetual_check","repetition_draw")),
            "status":dict(counter),"plan_sha256":digest((work/"master.json").read_bytes()),
            "game_sha256":hashes,"formal_changed":False}
    atomic_json(publication/"report.json",report)
    (publication/"plan.json").write_bytes((work/"master.json").read_bytes())
    pkg=out/"D62-D60-1000games.tar.gz"
    core.pack_tar(publication,pkg)
    sha=digest(pkg.read_bytes())
    atomic_json(out/"receipt.json",{"sha256":sha,"games":TOTAL,"size":pkg.stat().st_size})
    print(json.dumps({"games":TOTAL,"resolved":report["resolved"],
                      "unresolved":report["unresolved"],"source_sha256":sha}))
    
def main():
    a=argparse.ArgumentParser();sub=a.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("prepare")
    for key in ("state","d41","d42","d43-trained","d43-selfplay","d43-audit","d45-trained","d45-selfplay","d45-audit","d47-audit","d48-audit","d50-trained","d50-selfplay","d50-audit","d52-trained","d52-selfplay","d52-audit","d54-trained","d54-selfplay","d54-audit","d56-trained","d56-selfplay","d56-audit","d58-trained","d58-selfplay","d58-audit","d60-trained","d60-selfplay","d60-audit","out"):p.add_argument("--"+key,required=True,type=Path)
    p=sub.add_parser("worker")
    for key in ("runtime","out"):p.add_argument("--"+key,required=True,type=Path)
    for key in ("round","lane"):p.add_argument("--"+key,required=True,type=int)
    p=sub.add_parser("segment")
    for key in ("work","out"):p.add_argument("--"+key,required=True,type=Path)
    p.add_argument("--index",required=True,type=int)
    p=sub.add_parser("collect")
    for key in ("runtime","downloads","out"):p.add_argument("--"+key,required=True,type=Path)
    x=a.parse_args()
    if x.cmd=="prepare":prepare(x.state,x.d41,x.d42,x.d43_trained,x.d43_selfplay,x.d43_audit,x.d45_trained,x.d45_selfplay,x.d45_audit,x.d47_audit,x.d48_audit,x.d50_trained,x.d50_selfplay,x.d50_audit,x.d52_trained,x.d52_selfplay,x.d52_audit,x.d54_trained,x.d54_selfplay,x.d54_audit,x.d56_trained,x.d56_selfplay,x.d56_audit,x.d58_trained,x.d58_selfplay,x.d58_audit,x.d60_trained,x.d60_selfplay,x.d60_audit,x.out)
    elif x.cmd=="worker":worker(x.runtime,x.round,x.lane,x.out)
    elif x.cmd=="segment":segment(x.work,x.index,x.out)
    else:collect(x.runtime,x.downloads,x.out)
if __name__=="__main__":main()
