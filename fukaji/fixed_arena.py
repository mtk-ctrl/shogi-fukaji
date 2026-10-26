"""Auditable fixed-model paired arena: isolate time-budget effects from training drift.

Unlike the self-play experiment, this never retrains the model. Every arm uses
one SHA-verified model, the same opening seed, the same engine revision, and
candidate color alternation. Only verified checkmates count as wins.
"""
import argparse
import hashlib
import json
from pathlib import Path
from .arena import run as arena


def execute(model_path, request_path, out):
    config=json.loads(Path(request_path).read_text(encoding="utf-8"))
    required={"source_run_id","model_sha256","arena_seed","arena_pairs",
              "arena_mixes","arena_movetimes_ms","arena_max_plies","depth","opening_plies"}
    if not required.issubset(config):
        raise ValueError("missing required fixed-arena settings")
    sha=config["model_sha256"]
    if not isinstance(sha,str) or len(sha)!=64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("invalid expected model sha256")
    actual=hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    if actual!=sha:
        raise ValueError("fixed model SHA256 mismatch")
    integer_fields={"source_run_id":(1,2**63-1),"arena_seed":(0,2**31-1),
                    "arena_pairs":(1,40),"arena_max_plies":(60,300),
                    "depth":(1,3),"opening_plies":(0,20)}
    for key,(low,high) in integer_fields.items():
        value=config[key]
        if type(value) is not int or not low<=value<=high:
            raise ValueError("invalid "+key)
    mixes=config["arena_mixes"]
    times=config["arena_movetimes_ms"]
    if (not isinstance(mixes,list) or not mixes or len(mixes)>3
        or len(set(mixes))!=len(mixes)
        or any(type(v) is not int or not 0<=v<=100 for v in mixes)):
        raise ValueError("invalid arena_mixes")
    if (not isinstance(times,list) or not times or len(times)>3
        or len(set(times))!=len(times)
        or any(type(v) is not int or not 1<=v<=100 for v in times)):
        raise ValueError("invalid arena_movetimes_ms")
    if len(mixes)*len(times)*config["arena_pairs"]*2>240:
        raise ValueError("arena game budget exceeded")
    root=Path(out)
    root.mkdir(parents=True,exist_ok=True)
    report={"status":"fixed_model_arena_measured","source_run_id":config["source_run_id"],
            "model_sha256":actual,"arena_seed":config["arena_seed"],
            "arena_pairs":config["arena_pairs"],"arena_mixes":mixes,
            "arena_movetimes_ms":times,"arena_max_plies":config["arena_max_plies"],
            "depth":config["depth"],"opening_plies":config["opening_plies"],
            "arms":{},"note":"Exploratory, not an independent proof of strength."}
    for ms in times:
        for mix in mixes:
            key=f"{ms}ms_mix{mix}"
            report["arms"][key]=arena(
                model_path,root/f"{key}.jsonl",pairs=config["arena_pairs"],
                movetime=ms,depth=config["depth"],max_plies=config["arena_max_plies"],
                opening_plies=config["opening_plies"],mix=mix,seed=config["arena_seed"])
    (root/"fixed_arena_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--request",required=True)
    p.add_argument("--out",default="build/fixed_arena")
    a=p.parse_args()
    execute(a.model,a.request,a.out)


if __name__=="__main__":
    main()
