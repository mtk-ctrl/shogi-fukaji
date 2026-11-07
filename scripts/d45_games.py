"""D45 second 1000-game training data transport: 5 waves x 20 real Actions jobs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import d41_public_20 as core
from scripts import d45_native_game as native_game
from scripts import d43_games as history
from scripts import d42_d37_vs_d41_paired100 as d42
from fukaji.online_cycle import fresh_prefix
from fukaji.online_learning import atomic_json,load_snapshot,validate_game,digest,game_key

BASE="bf8c26f9f8f6adff8a87c46ab04fcf5cd76a7915be241da11b99bb4ae2ac8b9f"
STATE_SHA="41c8f56b76cd5d7aa87bff9131d553b304fa4134bf7448c05d3309458f66760e"
D41_SHA="8708db460d4d2bcad949d91ab881d9aeb804abcb8e28808dbbb70bda5c21bd9f"
D42_SHA="5b2cb8495ae7fcdbe2b9181c3d5b3aedb87b2fd72226b1e4de76abb2d3d1f570"
SEED=202611071050
D43_TRAIN_SHA="955a1785940f9f2aac946739af1b8637e5ef53d06297e6ad7d32b75a5a94a236"
D43_SELFPLAY_SHA="fe0a4eaec246a7e2e32c911c4327232eef83f8a87897e7a3a96538220ced6bbd"
D43_AUDIT_SHA="c08ccab3f5e9ef90c288e8ccfc5f3951eb188dfc30876af775c2ba3cfb03f510"
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

def prepare(state,d41_source,d42_source,d43_trained,d43_selfplay,d43_audit,out):
    for path,sha in ((state,STATE_SHA),(d41_source,D41_SHA),(d42_source,D42_SHA),
                     (d43_trained,D43_TRAIN_SHA),(d43_selfplay,D43_SELFPLAY_SHA),
                     (d43_audit,D43_AUDIT_SHA)):
        checksum(path,sha)
    manifest=json.loads(Path("formal/model.json").read_text())
    if manifest["research_snapshot_sha256"]!=BASE or manifest["optimizer_step"]!=2882:
        raise ValueError("not current adopted D43/Adam2882; do not start another baseline")
    with tarfile.open(state,"r:gz") as t:
        meta=json.load(t.extractfile("d17-state/state.json"))
        if meta["research_latest"]!=core.D37:raise ValueError("D37 historical splits changed")
    models=out/"runtime"/"models";models.mkdir(parents=True,exist_ok=True)
    with tarfile.open(d43_trained,"r:gz") as t:
        report=json.load(t.extractfile("report.json"))
        if (report["schema"]!="d43-learned1000-v1" or
            report["baseline"]!=core.D37 or report["trained_sha256"]!=BASE or
            report["optimizer_step_before"]!=2106 or report["optimizer_step_after"]!=2882 or
            report["source_sha256"]!=D43_SELFPLAY_SHA):
            raise ValueError("trained D43 is not the verified current continuation")
        for ext in ("npz","json"):
            name=f"{BASE}.{ext}"
            (models/name).write_bytes(t.extractfile(name).read())
    model,opt,_=load_snapshot(models,BASE)
    if opt is None or opt["step"]!=2882:
        raise ValueError("D43 Adam2882 was not restored")
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
            prior_result["before"]!=core.D37 or prior_result["after"]!=BASE or
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
    plan={"schema":"d45-d43-1000-v1","baseline":BASE,"adam_step":2882,
          "seed":SEED,"games":TOTAL,"rounds":ROUNDS,"lanes":LANES,
          "per_lane_per_round":PER_LANE,"protected_count":len(protected),
          "d43_current_training_sha256":D43_TRAIN_SHA,
          "d43_prior_training_games_sha256":D43_SELFPLAY_SHA,
          "d43_prior_audit_sha256":D43_AUDIT_SHA,
          "normal_ms":200,"long_ms":1000,"max_long_per_side":10,
          "max_plies":512,"cpp_nnue":True,"exploration_first_24":.05,
          "families":families}
    atomic_json(out/"runtime"/"master.json",plan)
    core.pack_tar(out/"runtime",out/"runtime.tar.gz")
    atomic_json(out/"receipt.json",{"plan_sha256":digest((out/"runtime"/"master.json").read_bytes()),
        "planned_games":TOTAL,"input_trained_sha256":D43_TRAIN_SHA,
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
    if opt["step"]!=2882:raise ValueError("optimizer not inherited")
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
            key=game_key(g)
            if key in seen or g["parent_id"] in parents:raise ValueError("duplicate")
            seen.add(key);parents.add(g["parent_id"])
            if len(g["telemetry"])!=len(g["moves"])-6:raise ValueError("telemetry missing")
            used=[0,0]
            for telemetry in g["telemetry"]:
                if telemetry.get("backend")!="cpp_nnue" or telemetry.get("budget_ms") not in (200,1000):
                    raise ValueError("changed NNUE backend or move time")
                if telemetry.get("longthink"):used[int(telemetry["side"])]+=1
            if used!=g["conditions"]["longthink_used_per_side"] or max(used)>10:
                raise ValueError("longthink exceeded or trace changed")
            counter[g["status"]]+=1;counter["plies"]+=len(g["moves"])
            hashes[name]=digest(raw)
            (publication/"games"/name).write_bytes(raw)
    if len(seen)!=TOTAL:raise ValueError("not exactly 1000 new complete games")
    report={"schema":"d45-selfplay1000-result-v1","model":BASE,"games":TOTAL,
            "resolved":sum(counter[k] for k in ("checkmate","perpetual_check","repetition_draw")),
            "unresolved":TOTAL-sum(counter[k] for k in ("checkmate","perpetual_check","repetition_draw")),
            "status":dict(counter),"plan_sha256":digest((work/"master.json").read_bytes()),
            "game_sha256":hashes,"formal_changed":False}
    atomic_json(publication/"report.json",report)
    (publication/"plan.json").write_bytes((work/"master.json").read_bytes())
    pkg=out/"D45-D43-1000games.tar.gz"
    core.pack_tar(publication,pkg)
    sha=digest(pkg.read_bytes())
    atomic_json(out/"receipt.json",{"sha256":sha,"games":TOTAL,"size":pkg.stat().st_size})
    print(json.dumps({"games":TOTAL,"resolved":report["resolved"],
                      "unresolved":report["unresolved"],"source_sha256":sha}))
    
def main():
    a=argparse.ArgumentParser();sub=a.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("prepare")
    for key in ("state","d41","d42","d43-trained","d43-selfplay","d43-audit","out"):p.add_argument("--"+key,required=True,type=Path)
    p=sub.add_parser("worker")
    for key in ("runtime","out"):p.add_argument("--"+key,required=True,type=Path)
    for key in ("round","lane"):p.add_argument("--"+key,required=True,type=int)
    p=sub.add_parser("segment")
    for key in ("work","out"):p.add_argument("--"+key,required=True,type=Path)
    p.add_argument("--index",required=True,type=int)
    p=sub.add_parser("collect")
    for key in ("runtime","downloads","out"):p.add_argument("--"+key,required=True,type=Path)
    x=a.parse_args()
    if x.cmd=="prepare":prepare(x.state,x.d41,x.d42,x.d43_trained,x.d43_selfplay,x.d43_audit,x.out)
    elif x.cmd=="worker":worker(x.runtime,x.round,x.lane,x.out)
    elif x.cmd=="segment":segment(x.work,x.index,x.out)
    else:collect(x.runtime,x.downloads,x.out)
if __name__=="__main__":main()
