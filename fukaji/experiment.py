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

def execute(out, attempts=24, movetime=2, depth=1, max_plies=240, opening_plies=6):
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    games=root/"search_games.jsonl"
    termination_counts={}
    diagnostic_samples=[]
    with games.open("w") as f:
        for i in range(attempts):
            diagnostic={}
            game=play(110000+i,movetime,max_plies,depth,opening_plies=opening_plies,diagnostic=diagnostic)
            reason=diagnostic.get("reason","unclassified")
            termination_counts[reason]=termination_counts.get(reason,0)+1
            if len(diagnostic_samples)<12 and reason!="checkmate":
                diagnostic_samples.append({"seed":110000+i,**diagnostic})
            if game: f.write(json.dumps(game)+"\n")
    # Fork only from replay-verified mate parents; do not label repetition.
    branch_path=root/"endgame_branches.jsonl"
    branch_report=branch_from_parents(games,branch_path,movetime=movetime,depth=depth,
                                     max_plies=100,max_parents=8)
    combined=root/"combined_verified.jsonl"
    combined.write_bytes(games.read_bytes()+branch_path.read_bytes())
    rows=load_games(combined)
    parts,group_counts=partition_rows(rows)
    report={"generator":"fukaji-alpha-beta-v0","requested_games":attempts,
            "verified_decisive_games":len(rows),"verified_root_games":sum(1 for line in games.read_text().splitlines() if line.strip()),
            "dataset_sha256":hashlib.sha256(combined.read_bytes()).hexdigest(),"branch_report":branch_report,
            "search_milliseconds":movetime,"search_depth":depth,"max_plies":max_plies,
            "opening_plies":opening_plies,"model":"linear-value-v0",
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
        # Pair/alternate colors and report unresolved separately.
        report["arena"]=arena(model,root/"arena.jsonl",pairs=4,movetime=movetime,
                              depth=depth,max_plies=120,opening_plies=opening_plies,mix=50)
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
    x=a.parse_args()
    if not 1<=x.attempts<=100 or not 1<=x.movetime<=100 or not 1<=x.depth<=3 or not 50<=x.max_plies<=400 or not 0<=x.opening_plies<=20:
        a.error("experiment limits exceeded")
    execute(x.out,x.attempts,x.movetime,x.depth,x.max_plies,x.opening_plies)
if __name__=="__main__":main()
