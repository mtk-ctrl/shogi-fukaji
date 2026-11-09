"""D68: fresh round, 100 fresh paired openings, 200 audit games, 20 real runner jobs."""
import argparse
from collections import Counter,defaultdict
import json
import math
import statistics
from pathlib import Path
import subprocess
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fukaji.online_cycle import fresh_prefix
from fukaji.online_learning import atomic_json,digest,load_snapshot,validate_game
from scripts import d42_d37_vs_d41_paired100 as d42
from scripts import d68_games as games
from scripts import d41_public_20 as core

BASE=games.BASE
SEED=202611082321
PAIRS=100
LANES=20
THRESHOLD=12

def get_models(work):
    cfg=json.loads((work/"model-ids.json").read_text())
    before,opt,_=load_snapshot(work/"models",BASE)
    after,after_opt,_=load_snapshot(work/"models",cfg["after"])
    if opt["step"]!=11864 or after_opt["step"]<=11864:
        raise ValueError("invalid parent-child optimizer")
    return before,after

def configure(work):
    cfg=json.loads((work/"model-ids.json").read_text())
    d42.BASE=BASE
    d42.AFTER=cfg["after"]
    d42.SEED=SEED
    d42.PAIRS=PAIRS
    d42.LANES=LANES
    d42.get_models=get_models

def prepare(state,baseline_archive,old_selfplay,old_audit,trained,trained_archive_sha,
            source,source_sha,d41,d42_source,d45_selfplay,d45_audit,d47_audit,d48_audit,d50_selfplay,d50_audit,d52_selfplay,d52_audit,d54_selfplay,d54_audit,d56_selfplay,d56_audit,d58_selfplay,d58_audit,d60_selfplay,d60_audit,d62_selfplay,d62_audit,d64_selfplay,d64_audit,d66_selfplay,d66_audit,out):
    for path,sha in ((state,games.STATE_SHA),(baseline_archive,games.D66_TRAIN_SHA),
                     (old_selfplay,games.D43_SELFPLAY_SHA),
                     (old_audit,games.D43_AUDIT_SHA),(trained,trained_archive_sha),
                     (source,source_sha),(d41,games.D41_SHA),(d42_source,games.D42_SHA),
                     (d45_selfplay,games.D45_SELFPLAY_SHA),(d45_audit,games.D45_AUDIT_SHA),
                     (d47_audit,games.D47_AUDIT_SHA),(d48_audit,games.D48_AUDIT_SHA),
                     (d50_selfplay,games.D50_SELFPLAY_SHA),(d50_audit,games.D50_AUDIT_SHA),
                     (d52_selfplay,games.D52_SELFPLAY_SHA),(d52_audit,games.D52_AUDIT_SHA),
                     (d54_selfplay,games.D54_SELFPLAY_SHA),(d54_audit,games.D54_AUDIT_SHA),
                     (d56_selfplay,games.D56_SELFPLAY_SHA),(d56_audit,games.D56_AUDIT_SHA),
                     (d58_selfplay,games.D58_SELFPLAY_SHA),(d58_audit,games.D58_AUDIT_SHA),
                     (d60_selfplay,games.D60_SELFPLAY_SHA),(d60_audit,games.D60_AUDIT_SHA),
                     (d62_selfplay,games.D62_SELFPLAY_SHA),(d62_audit,games.D62_AUDIT_SHA),
                     (d64_selfplay,games.D64_SELFPLAY_SHA),(d64_audit,games.D64_AUDIT_SHA),
                     (d66_selfplay,games.D66_SELFPLAY_SHA),(d66_audit,games.D66_AUDIT_SHA)):
        games.checksum(path,sha)
    formal=json.loads(Path("formal/model.json").read_text())
    if formal["research_snapshot_sha256"]!=BASE or formal["optimizer_step"]!=11864 or formal.get("search_backend_variant")!="d48_pvs_mate":
        raise ValueError("current D64 changed during second learning run")
    with tarfile.open(state,"r:gz") as t:
        meta=json.load(t.extractfile("d17-state/state.json"))
        if meta["research_latest"]!=core.D37:raise ValueError("D37 old splits not verified")
    models=out/"runtime"/"models";models.mkdir(parents=True,exist_ok=True)
    with tarfile.open(baseline_archive,"r:gz") as t:
        parent_report=json.load(t.extractfile("report.json"))
        if parent_report["trained_sha256"]!=BASE or parent_report["optimizer_step_after"]!=11864 or parent_report["schema"]!="d68-learned1000-v1":
            raise ValueError("baseline must be adopted D64/Adam10974")
        for ext in ("npz","json"):
            name=f"{BASE}.{ext}"
            (models/name).write_bytes(t.extractfile(name).read())
    with tarfile.open(trained,"r:gz") as t:
        report=json.load(t.extractfile("report.json"))
        after=report["trained_sha256"]
        if (report["schema"]!="d68-learned1000-v1" or report["baseline"]!=BASE or
            report["optimizer_step_before"]!=11864 or report["optimizer_step_after"]<=11864 or
            report["source_sha256"]!=source_sha):
            raise ValueError("D66 learning data not authenticated")
        for ext in ("npz","json"):
            name=f"{after}.{ext}"
            (models/name).write_bytes(t.extractfile(name).read())
    with tarfile.open(source,"r:gz") as t:
        train_plan=json.load(t.extractfile("plan.json"))
        if train_plan["schema"]!="d68-d66-1000-v1" or len(train_plan["families"])!=1000:
            raise ValueError("second training parents invalid")
    with tarfile.open(old_selfplay,"r:gz") as t:
        old_train_plan=json.load(t.extractfile("plan.json"))
        if old_train_plan["schema"]!="d43-d37-1000-v1" or len(old_train_plan["families"])!=1000:
            raise ValueError("previous 1000 training parents invalid")
    with tarfile.open(old_audit,"r:gz") as t:
        old_audit_plan=json.load(t.extractfile("plan.json"))
        if old_audit_plan["schema"]!="d43-paired200-v1" or len(old_audit_plan["families"])!=100:
            raise ValueError("previous 100 paired audit parents invalid")
    with tarfile.open(d42_source,"r:gz") as t:
        d42_plan=json.load(t.extractfile("plan.json"))
        if d42_plan["schema"]!="fukaji-d42-before-after-20lane-paired100-v1":
            raise ValueError("old 50 paired audit mismatch")
    exclude=set(meta["protected_parents"])|set(meta["excluded_openings"])
    exclude.update(games.older_d40(meta))
    exclude.update(d42.read_d41_parent_ids(d41))
    for plan in (d42_plan,old_train_plan,old_audit_plan,train_plan):
        exclude.update(x["parent_id"] for x in plan["families"])
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
        (d60_audit,"D60 audit",100,"d60-paired200-v1"),
        (d62_selfplay,"D62 train",1000,"d62-d60-1000-v1"),
        (d62_audit,"D62 audit",100,"d62-paired200-v1"),
        (d64_selfplay,"D64 train",1000,"d64-d62-1000-v1"),
        (d64_audit,"D64 audit",100,"d64-paired200-v1"),
        (d66_selfplay,"D66 train",1000,"d66-d64-1000-v1"),
        (d66_audit,"D66 audit",100,"d66-paired200-v1")):
        with tarfile.open(file,"r:gz") as z:
            member=next((x for x in z.getmembers() if x.name.lstrip("./")=="plan.json"),None)
            if member is None:raise ValueError("missing protected "+label)
            extra=json.load(z.extractfile(member))
        if extra["schema"]!=schema or len(extra["families"])!=count:
            raise ValueError("invalid protected plan "+label)
        exclude.update(g["parent_id"] for g in extra["families"])
    frozen=set(exclude)
    families=[]
    for i in range(PAIRS):
        opening,parent=fresh_prefix(SEED+i*1019,exclude)
        if parent in frozen or len(opening)!=6:
            raise ValueError("D62 audit parent is training/old evaluation or repeated")
        families.append({"index":i,"prefix":opening,"parent_id":parent,"lane":i%LANES})
        exclude.add(parent)
    work=out/"runtime"
    atomic_json(work/"model-ids.json",{"before":BASE,"after":after,
        "trained_archive_sha256":trained_archive_sha,
        "training_source_sha256":source_sha})
    configure(work)
    before,after_model=get_models(work)
    plan={"schema":"d68-paired200-v1","before":BASE,"after":after,
          "pairs":PAIRS,"games":200,"lanes":LANES,"seed":SEED,
          "frozen_old_exclusion_count":len(frozen),
          "conditions":{"backend":"cpp_nnue","quiescence":0,
            "normal_ms":200,"long_ms":1000,"max_long_per_side":10,
            "max_plies":512,"exploration_probability":0},
          "decision":{"finish_all":True,"margin_win_count":12,
                      "max_unresolved":10,"otherwise_keep":"D60"},
          "families":families}
    atomic_json(work/"plan.json",plan)
    core.pack_tar(work,out/"runtime.tar.gz")
    atomic_json(out/"receipt.json",{"before":BASE,"after":after,
        "pairs":PAIRS,"plan_sha256":digest((work/"plan.json").read_bytes())})
    print(json.dumps({"planned_games":200,"pairs":PAIRS,"after":after,
                      "old_exclusions":len(frozen)}))

def segment(work,index,side,out):
    configure(work)
    d42.segment(work,index,side,out,36)

def worker(runtime,lane,out):
    if not 0<=lane<LANES:raise ValueError("invalid lane")
    work=out/"runtime";core.safe_extract(runtime,work)
    configure(work)
    get_models(work)
    games_dir=out/"games";games_dir.mkdir(parents=True,exist_ok=True)
    for i in range(lane,PAIRS,LANES):
        for side in (0,1):
            final=games_dir/f"family-{i:03d}-candidate-black{side}.json"
            if final.exists():continue
            for _ in range(24):
                if final.exists():break
                p=subprocess.run([sys.executable,__file__,"segment","--work",str(work),
                    "--index",str(i),"--side",str(side),"--out",str(games_dir)],
                    capture_output=True,text=True,timeout=155)
                if p.returncode:raise RuntimeError(p.stderr[-1800:])
            if not final.exists():raise RuntimeError(f"unfinished family={i} side={side}")
    archive=out/f"lane-{lane:02d}.tar.gz"
    core.pack_tar(games_dir,archive)
    print(json.dumps({"lane":lane,"games":10,"sha256":digest(archive.read_bytes())}))

def collect(runtime,downloads,out):
    work=out/"runtime";core.safe_extract(runtime,work)
    configure(work)
    before,after=get_models(work)
    plan=json.loads((work/"plan.json").read_text())
    if plan["schema"]!="d68-paired200-v1" or len(plan["families"])!=100:
        raise ValueError("bad frozen paired comparison")
    lane_files=sorted(downloads.glob("lane-*.tar.gz"))
    if [p.name for p in lane_files]!=[f"lane-{i:02d}.tar.gz" for i in range(LANES)]:
        raise ValueError("all 20 result jobs are required")
    cfg=json.loads((work/"model-ids.json").read_text())
    counts=Counter();pairs=defaultdict(list);gamehash={};telemetry=[];plies=0
    publication=out/"publication";(publication/"games").mkdir(parents=True,exist_ok=True)
    for lane,archive in enumerate(lane_files):
        expected={f"family-{i:03d}-candidate-black{s}.json"
                  for i in range(lane,PAIRS,LANES) for s in (0,1)}
        with tarfile.open(archive,"r:gz") as t:
            actual={m.name for m in t.getmembers() if m.isfile()}
            if actual!=expected:raise ValueError("wrong paired match shard")
            for name in sorted(actual):
                raw=t.extractfile(name).read();g=json.loads(raw)
                i=g["family_index"];family=plan["families"][i]
                side=g["candidate_black"]
                models=[cfg["after"],BASE] if side else [BASE,cfg["after"]]
                player=0 if side else 1
                if (g["models"]!=models or g["parent_id"]!=family["parent_id"] or
                    g["moves"][:6]!=family["prefix"] or g["candidate_side"]!=player or
                    g["role"]!="audit" or g["lane"]!=lane):
                    raise ValueError("paired provenance mismatch")
                validate_game(g,before)
                if len(g["telemetry"])!=len(g["moves"])-6:raise ValueError("trace incomplete")
                used=[0,0]
                for trow in g["telemetry"]:
                    if trow["budget_ms"] not in (200,1000) or trow.get("backend")!="cpp_nnue" or "research_mate_nodes" not in trow:
                        raise ValueError("invalid elapsed clock or backend")
                    if trow["longthink"]:used[int(trow["side"])]+=1
                    telemetry.append(trow)
                if max(used)>10 or used!=g["conditions"]["longthink_used_per_side"]:
                    raise ValueError("longthink overuse")
                if g["winner"] is None:
                    if g["status"]=="repetition_draw":counts["draws"]+=1;score=.5
                    else:counts["unresolved"]+=1;score=None
                elif g["winner"]==player:counts["after_wins"]+=1;score=1.
                else:counts["before_wins"]+=1;score=0.
                pairs[i].append(score);plies+=len(g["moves"])
                gamehash[name]=digest(raw)
                (publication/"games"/name).write_bytes(raw)
    if len(gamehash)!=200 or len(pairs)!=100 or any(len(v)!=2 for v in pairs.values()):
        raise ValueError("not 100 complete, swapped opening pairs")
    after_wins=counts["after_wins"];before_wins=counts["before_wins"]
    diff=after_wins-before_wins;unresolved=counts["unresolved"]
    score_low=sum(sum(0 if v is None else v for v in pairs[i])/2 for i in range(100))/100
    score_high=sum(sum(1 if v is None else v for v in pairs[i])/2 for i in range(100))/100
    eps=math.sqrt(math.log(40)/(2*100))
    verdict=("after_better" if diff>0 else "before_better") if abs(diff)>=THRESHOLD and unresolved<=10 else "inconclusive_keep_D66"
    times={}
    for budget in (200,1000):
        vals=[float(z["elapsed_ms"]) for z in telemetry if z["budget_ms"]==budget]
        times[str(budget)]={"moves":len(vals),"mean_ms":statistics.mean(vals) if vals else None,
            "max_ms":max(vals) if vals else None,
            "over_budget_100ms":sum(x>budget+100 for x in vals)}
    result={"schema":"d68-paired200-result-v1","before":BASE,"after":cfg["after"],
            "games":200,"opening_pairs":100,"before_wins":before_wins,
            "after_wins":after_wins,"draws":counts["draws"],"unresolved":unresolved,
            "score_lower":score_low,"score_upper":score_high,
            "pair_hoeffding_95_interval":[max(0.,score_low-eps),min(1.,score_high+eps)],
            "difference":diff,"predeclared_verdict":verdict,
            "judgment_is_heuristic":True,"validated_plies":plies,
            "times":times,"game_sha256":gamehash,
            "plan_sha256":digest((work/"plan.json").read_bytes()),
            "formal_changed":False}
    atomic_json(publication/"report.json",result)
    (publication/"plan.json").write_bytes((work/"plan.json").read_bytes())
    archive=out/"D68-D66-before-after-200games.tar.gz"
    core.pack_tar(publication,archive)
    atomic_json(out/"receipt.json",{"sha256":digest(archive.read_bytes()),
                                   "games":200,"before":BASE,"after":cfg["after"],
                                   "decision":verdict})
    print(json.dumps({k:v for k,v in result.items() if k!="game_sha256"},ensure_ascii=False))

def main():
    a=argparse.ArgumentParser();sub=a.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("prepare")
    for flag in ("state","baseline-archive","old-selfplay","old-audit","trained","source","d41","d42","d45-selfplay","d45-audit","d47-audit","d48-audit","d50-selfplay","d50-audit","d52-selfplay","d52-audit","d54-selfplay","d54-audit","d56-selfplay","d56-audit","d58-selfplay","d58-audit","d60-selfplay","d60-audit","d62-selfplay","d62-audit","d64-selfplay","d64-audit","d66-selfplay","d66-audit","out"):p.add_argument("--"+flag,type=Path,required=True)
    for flag in ("trained-sha","source-sha"):p.add_argument("--"+flag,required=True)
    p=sub.add_parser("segment")
    for flag in ("work","out"):p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--index",type=int,required=True);p.add_argument("--side",type=int,choices=(0,1),required=True)
    p=sub.add_parser("worker")
    p.add_argument("--runtime",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--lane",type=int,required=True)
    p=sub.add_parser("collect")
    for flag in ("runtime","downloads","out"):p.add_argument("--"+flag,type=Path,required=True)
    z=a.parse_args()
    if z.cmd=="prepare":
        prepare(z.state,z.baseline_archive,z.old_selfplay,z.old_audit,z.trained,z.trained_sha,z.source,z.source_sha,z.d41,z.d42,z.d45_selfplay,z.d45_audit,z.d47_audit,z.d48_audit,z.d50_selfplay,z.d50_audit,z.d52_selfplay,z.d52_audit,z.d54_selfplay,z.d54_audit,z.d56_selfplay,z.d56_audit,z.d58_selfplay,z.d58_audit,z.d60_selfplay,z.d60_audit,z.d62_selfplay,z.d62_audit,z.d64_selfplay,z.d64_audit,z.d66_selfplay,z.d66_audit,z.out)
    elif z.cmd=="segment":segment(z.work,z.index,z.side,z.out)
    elif z.cmd=="worker":worker(z.runtime,z.lane,z.out)
    else:collect(z.runtime,z.downloads,z.out)
if __name__=="__main__":main()
