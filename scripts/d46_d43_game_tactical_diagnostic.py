"""Audit a real ShogiDroid sample game from the frozen D43-vs-D37 200-game Release.

No weight, search, training, formal setting or D45 action is changed.
Computed weaknesses are factual diagnostics, not tactical Elo estimates.
"""
from __future__ import annotations
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import shogi

D43="bf8c26f9f8f6adff8a87c46ab04fcf5cd76a7915be241da11b99bb4ae2ac8b9f"
D37="170d6c1ee41d77b69f0b75722ece3deb85f3eea189f70d9789db477a38266db1"
RELEASE_SHA="c08ccab3f5e9ef90c288e8ccfc5f3951eb188dfc30876af775c2ba3cfb03f510"
SAMPLE="family-000-candidate-black0.json"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def analyze(path):
    if sha(path.read_bytes())!=RELEASE_SHA:
        raise ValueError("D43 source Release failed SHA256")
    with tarfile.open(path,"r:gz") as tar:
        plan=json.load(tar.extractfile("plan.json"))
        report=json.load(tar.extractfile("report.json"))
        raw=tar.extractfile("games/"+SAMPLE).read()
        if sha(raw)!=report["game_sha256"][SAMPLE]:
            raise ValueError("original game individual SHA256 mismatch")
        game=json.loads(raw)
    if (len(game["moves"])!=282 or game["status"]!="checkmate" or
        game["winner"]!=1 or game["models"]!=[D37,D43] or
        game["moves"][:6]!=plan["families"][0]["prefix"] or
        game["candidate_side"]!=1 or game["family_index"]!=0):
        raise ValueError("not the exact D43 282-ply sample")
    b=shogi.Board()
    counts={"D37":Counter(),"D43":Counter()}
    examples={"mate1_missed":[],"drop_attacked":[],"drop_capturable_next":[],"depth0":[]}
    for ply,word in enumerate(game["moves"],1):
        side=int(b.turn)
        name="D43" if game["models"][side]==D43 else "D37"
        stats=counts[name]
        mv=shogi.Move.from_usi(word)
        if not b.is_legal(mv):raise ValueError(f"illegal move at ply {ply}")
        if ply<=6:
            stats["random_prefix_moves"]+=1
        else:
            stats["searched_moves"]+=1
            t=game["telemetry"][ply-7]
            if t["ply"]!=ply or int(t["side"])!=side or t["move"]!=word:
                raise ValueError(f"telemetry not matched at ply {ply}")
            depth=int(t["depth"])
            stats["depth_total"]+=depth
            stats["depth_zero"]+=depth==0
            stats["depth_le1"]+=depth<=1
            stats["depth_le2"]+=depth<=2
            stats[f"depth_{depth}"]+=1
            if depth==0 and len(examples["depth0"])<12:
                examples["depth0"].append({"ply":ply,"who":name,"move":word,
                    "root_moves":t["root_moves"],"partial":t["partial_root_moves"]})
            if ply>6 and b.is_check():
                stats["in_check_to_move"]+=1
            # Exact mate in one if an immediate legal move leaves opponent checkmated.
            possible=[]
            for candidate in list(b.legal_moves):
                b.push(candidate)
                is_mate=b.is_checkmate()
                b.pop()
                if is_mate:possible.append(candidate.usi())
            if possible:
                stats["mate1_opportunities"]+=1
                if word not in possible:
                    stats["mate1_missed"]+=1
                    if len(examples["mate1_missed"])<18:
                        examples["mate1_missed"].append({"ply":ply,"who":name,
                            "chosen":word,"one_move_mates":possible[:6],
                            "depth":depth})
                else:
                    stats["mate1_found"]+=1
        if mv.drop_piece_type is not None:
            stats["drops"]+=1
            dest=mv.to_square
            b.push(mv)
            if b.is_attacked_by(b.turn,dest):
                stats["drops_on_attacked_squares"]+=1
                capturable=any(u.to_square==dest and
                    b.piece_at(dest) is not None and
                    b.piece_at(dest).color!=b.turn for u in b.legal_moves)
                if capturable:
                    stats["drops_immediately_capturable"]+=1
                if len(examples["drop_attacked"])<14:
                    examples["drop_attacked"].append({"ply":ply,"who":name,"move":word,
                         "capturable_next":bool(capturable)})
                if capturable and len(examples["drop_capturable_next"])<14:
                    examples["drop_capturable_next"].append({"ply":ply,"who":name,"move":word})
            b.pop()
        b.push(mv)
    if not b.is_checkmate() or int(b.turn)==game["winner"]:
        raise ValueError("winner or terminal wrong")
    for who in ("D37","D43"):
        s=counts[who]
        s["avg_depth"]=round(s["depth_total"]/s["searched_moves"],3)
    return {
       "schema":"d46-real-d43-282ply-tactical-diagnostics-v1",
       "source_release_sha256":RELEASE_SHA,
       "source_game":SAMPLE,"game_plies":len(game["moves"]),
       "winner":"D43","loser":"D37",
       "opening_first_six_moves_source":"six uniform-random legal moves from fukaji/arena.py opening()",
       "stats":{k:dict(v) for k,v in counts.items()},
       "examples":examples,
       "interpretation_limitations":[
          "Mate-in-one counting does not detect mate-in-two or deep forced mates",
          "Drops on attacked squares can be purposeful sacrifices; counts alone do not prove blunders",
          "Original game starts with 6 random legal moves, even though 200-game audit has no exploration thereafter",
          "This is one sample game, not a representative strength estimate"
        ]
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=analyze(a.source)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__":
    main()
