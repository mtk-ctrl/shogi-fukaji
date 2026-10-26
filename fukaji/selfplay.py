"""Generate legal, completed games for pipeline smoke tests.
This RANDOM BASELINE is NOT the FUKAJI search engine; do not use these
games as evidence of strength improvement.
"""
import argparse
import json
import random
from pathlib import Path
import shogi

VALUES = {1:1,2:3,3:3,4:5,5:7,6:8,7:9,8:0,9:2,10:4,11:4,12:6,13:10,14:11}

def play(rng, max_plies):
    b=shogi.Board()
    start=b.sfen()
    moves=[]
    for _ in range(max_plies):
        if b.is_checkmate():
            break
        legal=list(b.legal_moves)
        if not legal:
            break
        weights=[]
        for m in legal:
            w=1.0
            if m.to_square is not None:
                captured=b.piece_at(m.to_square)
                if captured:
                    w+=VALUES.get(captured.piece_type,1)*5
            if m.promotion:
                w+=8
            if m.drop_piece_type:
                w*=0.75
            weights.append(w)
        m=rng.choices(legal,weights=weights,k=1)[0]
        moves.append(m.usi())
        b.push(m)
    if not b.is_checkmate():
        return None
    return {"start_sfen":start,"moves":moves,"winner":int(not b.turn),"source":"selfplay"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--attempts",type=int,default=200)
    ap.add_argument("--max-plies",type=int,default=512)
    ap.add_argument("--seed",type=int,default=20261026)
    ap.add_argument("--out",default="build/selfplay.jsonl")
    a=ap.parse_args()
    if not (1<=a.attempts<=10000 and 50<=a.max_plies<=1024):
        ap.error("attempts 1..10000, max-plies 50..1024")
    rng=random.Random(a.seed)
    target=Path(a.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    completed=0
    with target.open("w",encoding="utf-8") as f:
        for _ in range(a.attempts):
            g=play(rng,a.max_plies)
            if g is not None:
                f.write(json.dumps(g,ensure_ascii=False)+"\n")
                completed+=1
    print(f"Completed verified checkmates: {completed}/{a.attempts}")
    if not completed:
        raise SystemExit("no usable completed games; generation cannot train")

if __name__=="__main__":
    main()
