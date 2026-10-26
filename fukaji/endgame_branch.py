"""Replay-audited late-game continuations from independently verified mate games.

Branches inherit the original game's parent_id, so no related branch may
cross train/validation/test boundaries. Branches are genuine FUKAJI search
continuations, with only 1-2 legal seeded exploration moves at the fork.
"""
import hashlib
import json
import random
from pathlib import Path
import shogi
from .engine_selfplay import play

def parent_id(game):
    return hashlib.sha256((game["start_sfen"]+"|"+" ".join(game["moves"])).encode()).hexdigest()

def branch_from_parents(games_path, output, movetime=2, depth=1,
                        max_plies=120, max_parents=12):
    parents=[json.loads(line) for line in Path(games_path).read_text().splitlines() if line.strip()]
    result=[]
    attempts=0
    counts={}
    provenance=[]
    for parent in parents[:max_parents]:
        if set(parent)!={"start_sfen","moves","winner","source"} or parent["source"]!="selfplay":
            raise ValueError("invalid root parent game")
        original=shogi.Board(parent["start_sfen"])
        for word in parent["moves"]:
            move=shogi.Move.from_usi(word)
            if move not in original.legal_moves:
                raise ValueError("illegal parent move")
            original.push(move)
        if not original.is_checkmate() or parent["winner"]!=int(not original.turn):
            raise ValueError("parent has no independently verified checkmate")
        pid=parent_id(parent)
        for distance in (8,16,32):
            cut=max(0,len(parent["moves"])-distance)
            b=shogi.Board(parent["start_sfen"])
            for word in parent["moves"][:cut]:
                b.push_usi(word)
            if b.is_checkmate() or b.is_fourfold_repetition():
                continue
            for variant in range(2):
                attempts+=1
                seed=int(pid[:12],16)+distance*100+variant
                diagnostic={}
                game=play(seed,movetime,max_plies,depth,opening_plies=1+variant,
                          start_sfen=b.sfen(),parent_id=pid,diagnostic=diagnostic)
                reason=diagnostic.get("reason","unclassified")
                counts[reason]=counts.get(reason,0)+1
                if game is not None:
                    result.append(game)
                provenance.append({"parent_id":pid,"fork_ply":cut,"distance":distance,
                                   "variant":variant,"seed":seed,"reason":reason})
    target=Path(output)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w") as f:
        for game in result:
            f.write(json.dumps(game)+"\n")
    report={"verified_parent_games":len(parents),"parents_used":min(len(parents),max_parents),
            "branch_attempts":attempts,"verified_branch_mates":len(result),
            "branch_terminations":counts,
            "branch_dataset_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
            "provenance":provenance,
            "note":"Branches from the same root parent must stay in the same split."}
    return report
