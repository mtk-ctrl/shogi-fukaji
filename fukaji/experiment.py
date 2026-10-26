"""One bounded, auditable experiment from *real FUKAJI search games*.

No completed checkmates => blocked, rather than fabricating targets or
training on random-baseline games. Model is research-only and never promoted.
"""
import argparse
import hashlib
import json
from pathlib import Path
from .engine_selfplay import play
from .learning import load_games, fit, partition_rows
from .endgame_branch import branch_from_parents
from .arena import run as arena
import numpy as np

def execute(out, attempts=24, movetime=2, depth=1, max_plies=240, opening_plies=6,
            seed=110000, arena_pairs=4, arena_seed=20261026,
            branch_max_parents=8, branch_max_plies=100, arena_mixes=(50,), arena_max_plies=120):
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    games=root/"search_games.jsonl"
    termination_counts={}
    diagnostic_samples=[]
    with games.open("w") as f:
        for i in range(attempts):
            diagnostic={}
            game=play(seed+i,movetime,max_plies,depth,opening_plies=opening_plies,diagnostic=diagnostic)
            reason=diagnostic.get("reason","unclassified")
            termination_counts[reason]=termination_counts.get(reason,0)+1
            if len(diagnostic_samples)<12 and reason!="checkmate":
                diagnostic_samples.append({"seed":seed+i,**diagnostic})
            if game: f.write(json.dumps(game)+"\n")
    # Fork only from replay-verified mate parents; do not label repetition.
    branch_path=root/"endgame_branches.jsonl"
    branch_report=branch_from_parents(games,branch_path,movetime=movetime,depth=depth,
                                     max_plies=branch_max_plies,max_parents=branch_max_parents)
    combined=root/"combined_verified.jsonl"
    combined.write_bytes(games.read_bytes()+branch_path.read_bytes())
    rows=load_games(combined)
    parts,group_counts=partition_rows(rows)
    report={"generator":"fukaji-alpha-beta-v0","requested_games":attempts,
            "verified_decisive_games":len(rows),"verified_root_games":sum(1 for line in games.read_text().splitlines() if line.strip()),
            "dataset_sha256":hashlib.sha256(combined.read_bytes()).hexdigest(),"branch_report":branch_report,
            "search_milliseconds":movetime,"search_depth":depth,"max_plies":max_plies,
            "opening_plies":opening_plies,"seed":seed,"arena_pairs":arena_pairs,
            "arena_seed":arena_seed,"arena_mixes":list(arena_mixes),"arena_max_plies":arena_max_plies,"branch_max_parents":branch_max_parents,
            "branch_max_plies":branch_max_plies,"model":"linear-value-v0",
            "status":"blocked_insufficient_decisive_games","note":"No fake results; no strength conclusion."}
    # Do not train/promote without independent validation and test games.
    split_counts = {k: len(parts[k]) for k in ('train','validation','test')}
    report["split_game_counts"]=split_counts
    report["split_independent_parent_counts"]=group_counts
    report["termination_counts"]=termination_counts
    report["nondecisive_samples"]=diagnostic_samples
    if all(group_counts[k]>=2 for k in ('train','validation','test')):
        weights,metrics=fit(rows,epochs=4)
        model=root/"value-v0.npz"
        np.savez_compressed(model,weight=weights)
        report["model_sha256"]=hashlib.sha256(model.read_bytes()).hexdigest()
        report["splits"]=metrics
        report["status"]="trained_and_arena_measured"
        # Exploratory mix sensitivity on identical openings and alternating colors.
        # Do not select/promote a model on these results without a new seed.
        report["arena_sweep"]={}
        for mix in arena_mixes:
            result=arena(model,root/f"arena_mix_{mix}.jsonl",pairs=arena_pairs,movetime=movetime,
                         depth=depth,max_plies=arena_max_plies,opening_plies=opening_plies,
                         mix=mix,seed=arena_seed)
            report["arena_sweep"][str(mix)]=result
        if 50 in arena_mixes:
            report["arena"]=report["arena_sweep"]["50"]
        report["arena_sweep_note"]="Exploratory only: no promotion or strength claim without independent confirmation."
    elif len(rows)>=4:
        report["status"]="blocked_insufficient_independent_parent_groups"
    (root/"experiment.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return report

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--out",default="build/experiment")
    a.add_argument("--attempts",type=int,default=24)
    a.add_argument("--movetime",type=int,default=2)
    a.add_argument("--depth",type=int,default=1)
    a.add_argument("--max-plies",type=int,default=240)
    a.add_argument("--opening-plies",type=int,default=6)
    a.add_argument("--seed",type=int,default=110000)
    a.add_argument("--arena-pairs",type=int,default=4)
    a.add_argument("--arena-seed",type=int,default=20261026)
    a.add_argument("--arena-max-plies",type=int,default=120)
    a.add_argument("--arena-mixes",type=int,nargs="+",default=[50])
    a.add_argument("--branch-max-parents",type=int,default=8)
    a.add_argument("--branch-max-plies",type=int,default=100)
    a.add_argument("--request",default=None)
    x=a.parse_args()
    if x.request:
        config=json.loads(Path(x.request).read_text(encoding="utf-8"))
        mapping={"attempts":"attempts","movetime_ms":"movetime","depth":"depth",
                 "max_plies":"max_plies","opening_plies":"opening_plies",
                 "seed":"seed","arena_pairs":"arena_pairs","arena_seed":"arena_seed",
                 "arena_max_plies":"arena_max_plies",
                 "branch_max_parents":"branch_max_parents","branch_max_plies":"branch_max_plies"}
        for key,attr in mapping.items():
            if key in config:
                if type(config[key]) is not int:
                    a.error("request field "+key+" must be integer")
                setattr(x,attr,config[key])
        if "arena_mixes" in config:
            mixes=config["arena_mixes"]
            if (not isinstance(mixes,list) or not (1<=len(mixes)<=3)
                or any(type(m) is not int or not 0<=m<=100 for m in mixes)
                or len(set(mixes))!=len(mixes)):
                a.error("arena_mixes must be 1..3 distinct integer percentages 0..100")
            x.arena_mixes=mixes
        if config.get("branch_variants",6)!=6:
            a.error("branch_variants must equal the implemented 6 forks per parent")
    if not (1<=x.attempts<=100 and 1<=x.movetime<=100 and 1<=x.depth<=3
            and 50<=x.max_plies<=400 and 0<=x.opening_plies<=20
            and 0<=x.seed<=2**31-1 and 0<=x.arena_seed<=2**31-1
            and 1<=x.arena_pairs<=40 and 60<=x.arena_max_plies<=300
            and 1<=len(x.arena_mixes)<=3 and len(set(x.arena_mixes))==len(x.arena_mixes)
            and all(type(m) is int and 0<=m<=100 for m in x.arena_mixes)
            and 1<=x.branch_max_parents<=20
            and 20<=x.branch_max_plies<=200):
        a.error("experiment limits exceeded")
    execute(x.out,x.attempts,x.movetime,x.depth,x.max_plies,x.opening_plies,
            seed=x.seed,arena_pairs=x.arena_pairs,arena_seed=x.arena_seed,
            branch_max_parents=x.branch_max_parents,branch_max_plies=x.branch_max_plies,
            arena_mixes=x.arena_mixes,arena_max_plies=x.arena_max_plies)
if __name__=="__main__":main()
