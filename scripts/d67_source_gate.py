"""Fail-closed D66 completion gate for event-driven D67 NNUE adoption.
Neither a timer nor an LLM guesses the win count, run digest, optimizer step,
or trained model hash. Failed/inconclusive comparisons preserve formal D64.
"""
from __future__ import annotations
import argparse, hashlib, json, os, tarfile
from pathlib import Path
from urllib.request import Request, urlopen

OLD="545dba627046621ec53b681d485978b9054776c717d894dcb0de55bec731d4a2"
CPP="c9340af265368547c37cbecc311d8d3982900885b433d1b4ccdc27fbdd7d041a"
TRAIN_NAME="D66-D64-1000trained-unadopted.tar.gz"
GAMES_NAME="D66-D64-1000games.tar.gz"
AUDIT_NAME="D66-D64-before-after-200games.tar.gz"

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tar_json(path, name):
    with tarfile.open(path,"r:gz") as t:
        item=t.extractfile(name)
        if item is None:raise ValueError("archive missing "+name)
        return json.load(item)

def release_digest(repo, tag, asset, expected_commit, token):
    url=f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    req=Request(url,headers={"Authorization":"Bearer "+token,
                             "Accept":"application/vnd.github+json",
                             "X-GitHub-Api-Version":"2022-11-28"})
    with urlopen(req,timeout=30) as r:meta=json.load(r)
    if meta["tag_name"]!=tag or meta["target_commitish"]!=expected_commit:
        raise ValueError("run/release provenance differs "+tag)
    found=[x for x in meta["assets"] if x["name"]==asset and x["state"]=="uploaded"]
    if len(found)!=1 or not found[0].get("digest","").startswith("sha256:"):
        raise ValueError("missing stable uploaded release digest "+asset)
    return found[0]["digest"][7:]

def evaluate(train, games, audit, plan):
    if (train.get("schema")!="d66-learned1000-v1" or
        train.get("baseline")!=OLD or train.get("optimizer_step_before")!=10974 or
        type(train.get("optimizer_step_after")) is not int or
        not 10974<train["optimizer_step_after"]<=20000 or
        train.get("new_games")!=1000 or train.get("resolved_new_games",0)<800 or
        train.get("formal_model_changed") is not False or
        train.get("formal_search_variant")!="d48_pvs_mate" or
        type(train.get("trained_sha256")) is not str or
        len(train["trained_sha256"])!=64):
        raise ValueError("D66 trained NNUE/optimizer report contract")
    if games.get("schema")!="d66-selfplay1000-result-v1" or games.get("games")!=1000:
        raise ValueError("D66 source 1000 games not fully verified")
    if (plan.get("schema")!="d66-paired200-v1" or
        plan.get("before")!=OLD or plan.get("after")!=train["trained_sha256"] or
        len(plan.get("families",[]))!=100 or plan.get("games")!=200 or
        plan.get("pairs")!=100 or
        len({x["parent_id"] for x in plan["families"]})!=100):
        raise ValueError("D66 100 paired openings altered")
    if plan.get("decision",{}).get("margin_win_count")!=12 or plan.get("decision",{}).get("max_unresolved")!=10:
        raise ValueError("comparison criteria retroactively altered")
    if any(plan.get("conditions",{}).get(k)!=v for k,v in
           (("backend","cpp_nnue"),("normal_ms",200),("long_ms",1000),
            ("max_long_per_side",10),("quiescence",0))):
        raise ValueError("clock or search environment drift")
    if (audit.get("schema")!="d66-paired200-result-v1" or
        audit.get("before")!=OLD or audit.get("after")!=train["trained_sha256"] or
        audit.get("games")!=200 or audit.get("opening_pairs")!=100 or
        audit.get("formal_changed") is not False or
        type(audit.get("validated_plies")) is not int or
        audit["validated_plies"]<200 or
        not isinstance(audit.get("game_sha256"),dict) or
        len(audit["game_sha256"])!=200):
        raise ValueError("200 original legal audit result provenance")
    for x in ("after_wins","before_wins","draws","unresolved"):
        if type(audit.get(x)) is not int or audit[x]<0:raise ValueError("bad audit count "+x)
    if sum(audit[x] for x in ("after_wins","before_wins","draws","unresolved"))!=200:
        raise ValueError("200 audit games not accounted for")
    gap=audit["after_wins"]-audit["before_wins"]
    if audit.get("difference")!=gap:raise ValueError("reported win difference incorrect")
    eligible=gap>=12 and audit["unresolved"]<=10
    if eligible!=(audit.get("predeclared_verdict")=="after_better"):
        raise ValueError("predeclared verdict does not match frozen acceptance")
    fit=train.get("fit",{})
    if type(fit.get("positions")) is not int or fit["positions"]<1000:
        raise ValueError("missing actual learning positions")
    return eligible

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-id",required=True,type=int)
    p.add_argument("--source-sha",required=True)
    p.add_argument("--inputs",default="input")
    p.add_argument("--out",default="build/d67-gate.json")
    p.add_argument("--adopt-output",default=None)
    a=p.parse_args()
    if a.run_id<=0 or len(a.source_sha)!=40 or not all(c in "0123456789abcdef" for c in a.source_sha):
        raise ValueError("bad workflow_run identifier")
    repo=os.environ["GITHUB_REPOSITORY"]
    if repo!="mtk-ctrl/shogi-fukaji":raise ValueError("other repository")
    directory=Path(a.inputs)
    token=os.environ["GH_TOKEN"]
    files={
        "trained":(f"fukaji-d66-d64-trained-{a.run_id}",TRAIN_NAME),
        "games":(f"fukaji-d66-d64-1000-selfplay-{a.run_id}",GAMES_NAME),
        "audit":(f"fukaji-d66-d64-before-after-200games-{a.run_id}",AUDIT_NAME)
    }
    shas={}
    for key,(tag,name) in files.items():
        shas[key]=release_digest(repo,tag,name,a.source_sha,token)
        if sha(directory/name)!=shas[key]:
            raise ValueError("release bytes != immutable GitHub digest "+key)
    train=tar_json(directory/TRAIN_NAME,"report.json")
    games=tar_json(directory/GAMES_NAME,"report.json")
    audit=tar_json(directory/AUDIT_NAME,"report.json")
    plan=tar_json(directory/AUDIT_NAME,"plan.json")
    if train.get("source_sha256")!=shas["games"]:
        raise ValueError("trainer used unapproved games archive")
    adopt=evaluate(train,games,audit,plan)
    data={"schema":"d67-source-gate-v1","source_run":a.run_id,"source_sha":a.source_sha,
          "adopt":adopt,"current_before":OLD,"trained_after":train["trained_sha256"],
          "optimizer_before":10974,"optimizer_after":train["optimizer_step_after"],
          "selfplay_sha256":shas["games"],"train_sha256":shas["trained"],
          "audit_sha256":shas["audit"],
          "wins_after":audit["after_wins"],"wins_before":audit["before_wins"],
          "draws":audit["draws"],"unresolved":audit["unresolved"],
          "validated_plies":audit["validated_plies"],"positions":train["fit"]["positions"]}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
    if a.adopt_output:Path(a.adopt_output).open("a").write("adopt="+str(adopt).lower()+"\n")
    if adopt:
        request={
            "enabled":True,"request_id":"d67-d66-after-success",
            "source_run":a.run_id,"former":OLD,"next":train["trained_sha256"],
            "optimizer_before":10974,"optimizer_after":train["optimizer_step_after"],
            "train_sha256":shas["trained"],"audit_sha256":shas["audit"],
            "games_sha256":shas["games"],"after_wins":audit["after_wins"],
            "before_wins":audit["before_wins"],"draws":audit["draws"],
            "unresolved":audit["unresolved"],"validated_plies":audit["validated_plies"],
            "positions":train["fit"]["positions"]
        }
        request_file=Path("research/d67-adopt/START.json")
        if request_file.exists():raise ValueError("D67 request already exists")
        request_file.parent.mkdir(parents=True,exist_ok=True)
        request_file.write_text(json.dumps(request,sort_keys=True,indent=2)+"\n")
    print(json.dumps(data,ensure_ascii=False))

if __name__=="__main__":main()
