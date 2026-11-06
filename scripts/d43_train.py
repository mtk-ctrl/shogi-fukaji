"""D43 single-writer training: D37/Adam2106 + 1000 new legal selfplay games."""
import argparse
import json
import random
import sys
import tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fukaji.online_learning import (atomic_json,digest,load_snapshot,
                                    validate_game,make_samples,fit,save_snapshot,game_key)
from scripts import d43_games as games
from scripts import d41_public_20 as core

BASE=games.BASE
STATE_SHA=games.STATE_SHA
SEED=games.SEED+77

def train(state,source,source_sha,out):
    games.checksum(state,STATE_SHA)
    games.checksum(source,source_sha)
    out.mkdir(parents=True,exist_ok=True)
    models=out/"models"
    models.mkdir(exist_ok=True)
    with tarfile.open(state,"r:gz") as f:
        meta=json.load(f.extractfile("d17-state/state.json"))
        if meta["research_latest"]!=BASE:raise ValueError("baseline differs from current")
        for ext in ("npz","json"):
            (models/f"{BASE}.{ext}").write_bytes(f.extractfile(f"d17-state/models/{BASE}.{ext}").read())
        raw_old=f.extractfile("d17-state/bootstrap-train.jsonl").read()
    baseline,opt,_=load_snapshot(models,BASE)
    if opt["step"]!=2106:raise ValueError("not D37 optimizer")
    if digest(raw_old)!=meta["bootstrap"]["sha256"]:raise ValueError("old training data SHA")
    with tarfile.open(source,"r:gz") as f:
        plan=json.load(f.extractfile("plan.json"))
        report=json.load(f.extractfile("report.json"))
        names=sorted(n for n in f.getnames() if n.startswith("games/game-") and n.endswith(".json"))
        if (plan["schema"]!="d43-d37-1000-v1" or report["schema"]!="d43-selfplay1000-result-v1" or
            report["games"]!=1000 or report["model"]!=BASE or len(names)!=1000):
            raise ValueError("1000-game sealed source is invalid")
        if report["plan_sha256"]!=digest((json.dumps(plan,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()):
            raise ValueError("plan SHA did not match")
        fresh=[json.load(f.extractfile(n)) for n in names]
    protected=set(meta["protected_parents"])
    keys=set();parents=set()
    for i,g in enumerate(fresh):
        if (g["game_index"]!=i or g["parent_id"]!=plan["families"][i]["parent_id"] or
            g["role"]!="train" or g["models"]!=[BASE,BASE]):
            raise ValueError("game identity differs from plan")
        validate_game(g,baseline,training=True,protected=protected)
        if g["parent_id"] in parents or game_key(g) in keys:
            raise ValueError("duplicated new training experience")
        parents.add(g["parent_id"]);keys.add(game_key(g))
    resolved=[g for g in fresh if g["status"] in ("checkmate","perpetual_check","repetition_draw")]
    if len(resolved)<800:raise ValueError("not enough resolved games for new learning")
    old=[json.loads(line) for line in raw_old.splitlines() if line.strip()]
    for g in old:validate_game(g,baseline,training=True,protected=protected)
    old=random.Random(SEED).sample(old,min(len(old),len(resolved)))
    x,y,sampling=make_samples(resolved+old,baseline,seed=SEED,
             per_game=32,outcome_weight=.75,td_lambda=.8,protected=protected)
    candidate,new_opt,stats=fit(baseline,opt,x,y,epochs=2,lr=.0002,seed=SEED)
    key=save_snapshot(models,candidate,new_opt,{
        "parent":BASE,"purpose":"D43 direct D37 1000 selfplay update",
        "new_games":len(fresh),"new_resolved":len(resolved),
        "historic_replay_games":len(old),"source_release_sha256":source_sha,
        "training":stats,"learning_rate":.0002,
        "outcome_weight":.75,"td_lambda":.8})
    if new_opt["step"]<=opt["step"]:raise ValueError("optimizer not advanced")
    report={"schema":"d43-learned1000-v1","baseline":BASE,
            "trained_sha256":key,"optimizer_step_before":opt["step"],
            "optimizer_step_after":new_opt["step"],
            "new_games":1000,"resolved_new_games":len(resolved),
            "historic_replay_games":len(old),"sample_groups":sampling,
            "fit":stats,"source_sha256":source_sha,
            "strength_proven":False,"formal_model_changed":False}
    atomic_json(out/"report.json",report)
    publication=out/"publication";publication.mkdir(exist_ok=True)
    for name in (f"{key}.npz",f"{key}.json"):
        (publication/name).write_bytes((models/name).read_bytes())
    (publication/"report.json").write_bytes((out/"report.json").read_bytes())
    artifact=out/"D43-D37-1000trained-unadopted.tar.gz"
    core.pack_tar(publication,artifact)
    atomic_json(out/"receipt.json",{"sha256":digest(artifact.read_bytes()),
                "model_sha256":key,"source_sha256":source_sha,
                "optimizer_step_after":new_opt["step"]})
    print(json.dumps(report,ensure_ascii=False))
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--state",type=Path,required=True)
    a.add_argument("--source",type=Path,required=True)
    a.add_argument("--source-sha",required=True)
    a.add_argument("--out",type=Path,required=True)
    x=a.parse_args()
    if len(x.source_sha)!=64:raise ValueError("source_sha must be SHA256")
    train(x.state,x.source,x.source_sha,x.out)
if __name__=="__main__":main()
