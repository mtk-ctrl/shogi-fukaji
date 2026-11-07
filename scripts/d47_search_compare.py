"""D47: non-pruning PVS research against the unchanged D45 native NNUE.

Creates a candidate binary from the pinned official C++ source, changing ONLY
the inner alpha-beta window policy. The official NNUE, USI, rules, and formal
contract files remain unmodified. 150 fresh paired openings x swapped colors.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import shogi
from fukaji.formal_model import load as load_formal
from fukaji.native_nnue import NativeSelector, build_native
from fukaji.online_cycle import fresh_prefix
from fukaji.online_learning import atomic_json, digest, validate_game
from scripts import d42_d37_vs_d41_paired100 as arena
from scripts import d41_public_20 as common

MODEL = "941598f2a9ec926d5f67dadf03582ccef718a56a843070dc1f576cc7eeca44e8"
WEIGHTS = "9faf66a9d1fee9d282e35835628377ab7951bb94335a5d0498384bf88f4d2c26"
SOURCE = ROOT / "native/nnue_search_ffi.cpp"
SEED = 202611071830
PAIRS = 150
LANES = 20
MAX_PLIES = 512
BASELINE = "baseline"
CANDIDATE = "pvs"
GAME_SCHEMA = "d47-pvs-paired300-v1"

# Exact anchor from the owner-approved source: research must not silently patch
# another, concurrently changed C++ implementation.
ORIGINAL = """    double best=-2*kNMate;string word;
    for(const auto& move:legal) {
      double score;
      {MoveGuard guard(*this,move);score=-search(depth-1,-beta,-alpha,ply+1);}
      if(score>best){best=score;word=move;}alpha=max(alpha,best);if(alpha>=beta)break;
    }
"""
PVS = """    double best=-2*kNMate;string word;
    bool first_move=true;
    for(const auto& move:legal) {
      double score;
      {
        MoveGuard guard(*this,move);
        if(first_move) {
          score=-search(depth-1,-beta,-alpha,ply+1);
        } else {
          // Null-window probe is only a preliminary bound, never a skipped move.
          // The smallest representable positive window works for NNUE doubles.
          const double probe=std::nextafter(alpha,std::numeric_limits<double>::infinity());
          score=-search(depth-1,-probe,-alpha,ply+1);
          if(score>alpha && score<beta) {
            // A potentially best quiet/sacrifice move receives a full search.
            score=-search(depth-1,-beta,-alpha,ply+1);
          }
        }
      }
      first_move=false;
      if(score>best){best=score;word=move;}alpha=max(alpha,best);if(alpha>=beta)break;
    }
"""

_MODELS = None

def verified_models():
    global _MODELS
    if _MODELS is None:
        a, spec_a = load_formal()
        b, spec_b = load_formal()
        if (spec_a["research_snapshot_sha256"] != MODEL or
            spec_a["weights_sha256"] != WEIGHTS or
            spec_b["weights_sha256"] != WEIGHTS or
            spec_a["quiescence"] != 0):
            raise ValueError("D45 current NNUE changed; refuse search-only comparison")
        if a is b:
            raise AssertionError("search-only candidate must have a distinct model object")
        _MODELS = a, b
    return _MODELS

def candidate_source():
    original = SOURCE.read_text()
    if original.count(ORIGINAL) != 1:
        raise ValueError("official C++ search anchor changed; no implicit rebase")
    if "std::nextafter" in original or "bool first_move=true;" in original:
        raise ValueError("expected original alpha-beta baseline")
    candidate = original.replace(ORIGINAL, PVS, 1)
    if candidate.replace(PVS, ORIGINAL, 1) != original:
        raise AssertionError("candidate changed anything beyond the PVS search block")
    return candidate

def build_candidate():
    if not shutil.which("g++"):
        raise RuntimeError("g++ required for independent PVS binary")
    content = candidate_source()
    native_dir = ROOT / "native"
    key = digest(content.encode() + (native_dir/"rules_state_ffi.cpp").read_bytes() +
                 (native_dir/"rules_probe.cpp").read_bytes())
    folder = ROOT/"build/d47-pvs"
    folder.mkdir(parents=True, exist_ok=True)
    src = folder/"nnue_search_pvs_ffi.cpp"
    target = folder/("libnnue-pvs-"+key[:20]+".so")
    if not target.exists():
        src.write_text(content)
        temporary = target.with_suffix(".tmp.so")
        args = ["g++","-std=c++17","-O3","-fPIC","-shared",
                "-ffp-contract=off","-Wall","-Wextra","-Werror",
                "-I",str(native_dir),str(src),"-o",str(temporary)]
        result = subprocess.run(args,capture_output=True,text=True,timeout=120)
        if result.returncode:
            raise RuntimeError("PVS candidate failed C++ compile: "+result.stderr[-4000:])
        temporary.replace(target)
    return target

class PairSelector:
    """Independent baseline/candidate native libraries; D45 tensors are identical."""
    def __init__(self):
        self.official = NativeSelector(build_native())
        self.candidate = NativeSelector(build_candidate())
    def __enter__(self):
        return self
    def __exit__(self, kind, value, traceback):
        try:
            self.official.close()
        finally:
            self.candidate.close()
    def backend(self, model):
        before, after = verified_models()
        if model is before:
            return self.official
        if model is after:
            return self.candidate
        raise ValueError("unknown search-only model identity")
    def register(self, model):
        return self.backend(model).register(model)
    def select(self, board, model, **kwargs):
        chosen = self.backend(model)
        result = chosen.select(board,model,**kwargs)
        result["research_search"] = (CANDIDATE if chosen is self.candidate else BASELINE)
        return result

def configure():
    arena.BASE = MODEL
    arena.AFTER = MODEL
    arena.SEED = SEED
    arena.PAIRS = PAIRS
    arena.LANES = LANES
    arena.MAX_PLIES = MAX_PLIES
    arena.get_models = lambda work: verified_models()
    arena.NativeSelector = PairSelector

def sha_checked(path, expected):
    actual = digest(path.read_bytes())
    if actual != expected:
        raise ValueError("input archive SHA mismatch: "+str(path))

HISTORICAL = (
    ("d43-selfplay", "fe0a4eaec246a7e2e32c911c4327232eef83f8a87897e7a3a96538220ced6bbd"),
    ("d43-audit", "c08ccab3f5e9ef90c288e8ccfc5f3951eb188dfc30876af775c2ba3cfb03f510"),
    ("d45-selfplay", "be22c4e92eff0b684f06502dcdf9be52071ab60beab5a4a91edd8605f4d873c4"),
    ("d45-audit", "6531747f10d015bec7817186f6fbed149aa6ac1822d94fbcce7858a7ceae211b"),
)

def prepare(archives, out):
    verified_models()
    if len(archives) != 4:
        raise ValueError("exactly four historical archives required")
    protected = set()
    hashes = {}
    for (label, expected), path in zip(HISTORICAL, archives):
        sha_checked(path, expected)
        hashes[label] = expected
        with tarfile.open(path, "r:gz") as stream:
            plan = json.load(stream.extractfile("plan.json"))
        historical = plan["families"]
        expected_count = 1000 if "selfplay" in label else 100
        if len(historical) != expected_count:
            raise ValueError("invalid history family count: "+label)
        old = {str(row["parent_id"]) for row in historical}
        if len(old) != len(historical):
            raise ValueError("duplicate historical family: "+label)
        protected.update(old)
    families = []
    excluded = set(protected)
    for i in range(PAIRS):
        prefix, parent = fresh_prefix(SEED + 1019*i, excluded)
        if len(prefix) != 6 or parent in protected or parent in {f["parent_id"] for f in families}:
            raise ValueError("opening overlaps protected history")
        families.append({"index":i,"prefix":prefix,"parent_id":parent,"lane":i % LANES})
        excluded.add(parent)
    if len({f["parent_id"] for f in families}) != PAIRS:
        raise ValueError("not 150 independent openings")
    native_hash = digest(SOURCE.read_bytes())
    generated_hash = digest(candidate_source().encode())
    plan = {
        "schema":GAME_SCHEMA, "candidate":CANDIDATE, "baseline":BASELINE,
        "model_snapshot_sha256":MODEL, "inference_weights_sha256":WEIGHTS,
        "baseline_cpp_sha256":native_hash,"candidate_cpp_sha256":generated_hash,
        "historical_archive_sha256":hashes,"protected_parent_count":len(protected),
        "seed":SEED,"opening_pairs":PAIRS,"games":PAIRS*2,"runners":LANES,
        "game_assignment":"index_modulo_20","conditions":{
            "movetime_ms":200,"longthink_ms":1000,"longthink_limit_per_side":10,
            "max_plies":MAX_PLIES,"nnue":"D45 fixed identical for both",
            "cpp":True,"quiescence":0,"no_external_teacher":True,
            "all_legal_moves_considered":True,"exploration_probability":0,
            "same_opening_swapped_colors":True,
            "adoption":"owner decision after audit; never automatic"},
        "families":families
    }
    runtime = out/"runtime"
    runtime.mkdir(parents=True,exist_ok=True)
    atomic_json(runtime/"plan.json",plan)
    common.pack_tar(runtime,out/"runtime.tar.gz")
    atomic_json(out/"receipt.json",{"plan_sha256":digest((runtime/"plan.json").read_bytes()),
                                     "opening_pairs":PAIRS,"games":PAIRS*2,
                                     "runners":LANES,"baseline_cpp_sha256":native_hash,
                                     "candidate_cpp_sha256":generated_hash})
    print(json.dumps({"prepared_pairs":PAIRS,"games":300,"jobs":LANES,
                      "historical_excluded":len(protected)}))

def verify_plan(work):
    plan=json.loads((work/"plan.json").read_text())
    if (plan["schema"]!=GAME_SCHEMA or plan["games"]!=300 or
        plan["model_snapshot_sha256"]!=MODEL or
        plan["inference_weights_sha256"]!=WEIGHTS or
        plan["baseline_cpp_sha256"]!=digest(SOURCE.read_bytes()) or
        plan["candidate_cpp_sha256"]!=digest(candidate_source().encode()) or
        len(plan["families"])!=PAIRS):
        raise ValueError("frozen search-only plan/source mismatch")
    verified_models()
    return plan

def preflight(out):
    verified_models()
    official=build_native()
    candidate=build_candidate()
    source = candidate_source()
    tested=[]
    rng=random.Random(202611071831)
    positions=[]
    board=shogi.Board()
    for i in range(12):
        if i in (0,2,5,9):
            positions.append(list(m.usi() for m in board.move_stack))
        board.push(rng.choice(list(board.legal_moves)))
    positions.append(list(m.usi() for m in board.move_stack))
    before,after=verified_models()
    for i, words in enumerate(positions):
        b=shogi.Board()
        for word in words:b.push(shogi.Move.from_usi(word))
        with NativeSelector(official) as baseline, NativeSelector(candidate) as pvs:
            depth = 3 if i in (0,2) else 2
            a=baseline.select(b,before,max_depth=depth,fixed_depth=True)
            z=pvs.select(b,after,max_depth=depth,fixed_depth=True)
            if (a["depth"]!=depth or z["depth"]!=depth or
                a["move"]!=z["move"] or
                abs(a["score"]-z["score"])>1e-6):
                raise AssertionError("fixed depth PVS changed minimax result: "+
                                     json.dumps({"idx":i,"baseline":a,"pvs":z}))
            tested.append({"plies":len(words),"depth":depth,
                           "baseline_nodes":a["nodes"],"pvs_nodes":z["nodes"]})
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/"preflight.json",
                {"success":True,"fixed_depth_cases":len(tested),"cases":tested,
                 "model":MODEL,"weights_sha256":WEIGHTS,
                 "candidate_cpp_sha256":digest(source.encode()),
                 "original_cpp_sha256":digest(SOURCE.read_bytes())})
    print("D47 preflight fixed-depth search agrees in",len(tested),"positions")

def segment(work,index,side,out):
    configure()
    verify_plan(work)
    arena.segment(work,index,side,out,36)

def worker(runtime,lane,out):
    if not 0<=lane<LANES:raise ValueError("bad lane")
    configure()
    work=out/"runtime"
    common.safe_extract(runtime,work)
    plan=verify_plan(work)
    games=out/"games"
    games.mkdir(parents=True,exist_ok=True)
    local_pairs=[i for i in range(PAIRS) if i % LANES==lane]
    for i in local_pairs:
        for side in (0,1):
            target=games/f"family-{i:03d}-candidate-black{side}.json"
            for attempt in range(24):
                if target.exists():break
                proc=subprocess.run(
                    [sys.executable,__file__,"segment","--work",str(work),
                     "--index",str(i),"--side",str(side),"--out",str(games)],
                    capture_output=True,text=True,timeout=210)
                if proc.returncode:
                    raise RuntimeError("segment failed: "+proc.stderr[-3000:])
            if not target.exists():
                raise RuntimeError("unfinished paired game "+str((i,side)))
    if len(list(games.glob("family-*-candidate-black*.json")))!=len(local_pairs)*2:
        raise ValueError("runner output count mismatch")
    archive=out/f"lane-{lane:02d}.tar.gz"
    common.pack_tar(games,archive)
    print(json.dumps({"lane":lane,"games":len(local_pairs)*2,
                      "sha256":digest(archive.read_bytes())}))

def collect(runtime,downloads,out):
    work=out/"runtime"
    common.safe_extract(runtime,work)
    plan=verify_plan(work)
    archives=sorted(downloads.glob("lane-*.tar.gz"))
    if [p.name for p in archives]!=[f"lane-{i:02d}.tar.gz" for i in range(LANES)]:
        raise ValueError("exactly 20 independent lane archives required")
    before,_=verified_models()
    publication=out/"publication"
    (publication/"games").mkdir(parents=True,exist_ok=True)
    counts=Counter()
    pair_scores=defaultdict(list)
    gamehash={}
    all_times=[]
    total_plies=0
    for lane, archive in enumerate(archives):
        expected={f"family-{i:03d}-candidate-black{s}.json"
                  for i in range(PAIRS) if i % LANES==lane for s in (0,1)}
        with tarfile.open(archive,"r:gz") as stream:
            members={m.name for m in stream.getmembers() if m.isfile()}
            if members!=expected:
                raise ValueError("lane missing/extra games: "+str(lane))
            for name in sorted(members):
                raw=stream.extractfile(name).read()
                game=json.loads(raw)
                i=game["family_index"]
                s=game["candidate_black"]
                candidate_side=0 if s else 1
                if (i % LANES!=lane or
                    game["lane"]!=lane or
                    game["parent_id"]!=plan["families"][i]["parent_id"] or
                    game["moves"][:6]!=plan["families"][i]["prefix"] or
                    game["candidate_side"]!=candidate_side or
                    game["models"]!=[MODEL,MODEL] or
                    game["role"]!="audit"):
                    raise ValueError("candidate/base pairing provenance mismatch")
                validate_game(game,before)
                if len(game["telemetry"])!=len(game["moves"])-6:
                    raise ValueError("missing per-move records")
                used=[0,0]
                for row in game["telemetry"]:
                    side=int(row["side"])
                    expected_variant=CANDIDATE if side==candidate_side else BASELINE
                    if (row.get("research_search")!=expected_variant or
                        row["budget_ms"] not in (200,1000) or
                        row.get("backend")!="cpp_nnue"):
                        raise ValueError("wrong search variant/clock")
                    if row["longthink"]:
                        used[side]+=1
                    all_times.append({"budget":row["budget_ms"],"actual":row["elapsed_ms"],
                                      "variant":expected_variant})
                if (max(used)>10 or
                    used!=game["conditions"]["longthink_used_per_side"]):
                    raise ValueError("longthink limit violation")
                if game["winner"] is None:
                    if game["status"]=="repetition_draw":
                        counts["draws"]+=1
                        score=.5
                    else:
                        counts["unresolved"]+=1
                        score=None
                elif game["winner"]==candidate_side:
                    counts["candidate_wins"]+=1
                    score=1.
                else:
                    counts["baseline_wins"]+=1
                    score=0.
                pair_scores[i].append(score)
                total_plies+=len(game["moves"])
                gamehash[name]=digest(raw)
                (publication/"games"/name).write_bytes(raw)
    if (len(gamehash)!=300 or len(pair_scores)!=PAIRS or
        any(len(v)!=2 for v in pair_scores.values())):
        raise ValueError("300 games from 150 complete opening pairs required")
    n=PAIRS
    lower=sum(sum(0 if v is None else v for v in pair_scores[i])/2 for i in range(n))/n
    upper=sum(sum(1 if v is None else v for v in pair_scores[i])/2 for i in range(n))/n
    eps=math.sqrt(math.log(40)/(2*n))
    stats={}
    for label in (BASELINE,CANDIDATE):
        for ms in (200,1000):
            vals=[r["actual"] for r in all_times
                  if r["variant"]==label and r["budget"]==ms]
            stats[f"{label}-{ms}"]={
                "count":len(vals),"mean_ms":statistics.mean(vals) if vals else None,
                "max_ms":max(vals) if vals else None,
                "over_budget_plus_100ms":sum(v>ms+100 for v in vals)}
    result={
        "schema":"d47-pvs-paired300-result-v1","comparison":"identical-D45-NNUE PVS vs unchanged alpha-beta",
        "model_snapshot_sha256":MODEL,"weights_sha256":WEIGHTS,
        "games":300,"opening_pairs":150,"actual_runner_shards":LANES,
        "baseline_wins":counts["baseline_wins"],
        "candidate_wins":counts["candidate_wins"],
        "draws":counts["draws"],"unresolved":counts["unresolved"],
        "candidate_score_lower":lower,"candidate_score_upper":upper,
        "paired_hoeffding_95_interval":[max(0.,lower-eps),min(1.,upper+eps)],
        "validated_plies":total_plies,"thinking_time":stats,
        "plan_sha256":digest((work/"plan.json").read_bytes()),
        "formal_model_unchanged":True,"adopted":False,"game_sha256":gamehash}
    atomic_json(publication/"report.json",result)
    (publication/"plan.json").write_bytes((work/"plan.json").read_bytes())
    path=out/"D47-PVS-vs-D45-300games.tar.gz"
    common.pack_tar(publication,path)
    atomic_json(out/"receipt.json",{"archive_sha256":digest(path.read_bytes()),
                                     "games":300,"runner_shards":20,
                                     "candidate_wins":counts["candidate_wins"],
                                     "baseline_wins":counts["baseline_wins"]})
    print(json.dumps({k:v for k,v in result.items() if k!="game_sha256"},ensure_ascii=False))

def main():
    p=argparse.ArgumentParser()
    s=p.add_subparsers(dest="cmd",required=True)
    q=s.add_parser("preflight")
    q.add_argument("--out",type=Path,required=True)
    q=s.add_parser("prepare")
    q.add_argument("--history",nargs=4,type=Path,required=True)
    q.add_argument("--out",type=Path,required=True)
    q=s.add_parser("segment")
    q.add_argument("--work",type=Path,required=True)
    q.add_argument("--index",type=int,required=True)
    q.add_argument("--side",type=int,choices=(0,1),required=True)
    q.add_argument("--out",type=Path,required=True)
    q=s.add_parser("worker")
    q.add_argument("--runtime",type=Path,required=True)
    q.add_argument("--lane",type=int,required=True)
    q.add_argument("--out",type=Path,required=True)
    q=s.add_parser("collect")
    q.add_argument("--runtime",type=Path,required=True)
    q.add_argument("--downloads",type=Path,required=True)
    q.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.cmd=="preflight":preflight(a.out)
    elif a.cmd=="prepare":prepare(a.history,a.out)
    elif a.cmd=="segment":segment(a.work,a.index,a.side,a.out)
    elif a.cmd=="worker":worker(a.runtime,a.lane,a.out)
    else:collect(a.runtime,a.downloads,a.out)

if __name__=="__main__":
    main()
