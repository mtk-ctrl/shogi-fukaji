"""Self-play of the actual FUKAJI bootstrap search (not the random smoke generator)."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import shogi
from .engine import select_move

def play(seed, movetime_ms=25, max_plies=160, max_depth=2):
    rng=random.Random(seed)
    board=shogi.Board()
    start=board.sfen()
    moves=[]
    for ply in range(max_plies):
        if board.is_checkmate():
            break
        if board.is_fourfold_repetition():
            return None
        options=list(board.legal_moves)
        if not options:
            break
        move_text,depth,nodes=select_move(board,movetime_ms,max_depth=max_depth)
        if move_text=="resign":
            break
        move=shogi.Move.from_usi(move_text)
        if move not in board.legal_moves:
            raise RuntimeError("engine selected illegal move")
        moves.append(move_text)
        board.push(move)
    if not board.is_checkmate():
        return None
    return {"start_sfen":start,"moves":moves,"winner":int(not board.turn),"source":"selfplay"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--attempts",type=int,default=4)
    ap.add_argument("--movetime",type=int,default=10)
    ap.add_argument("--depth",type=int,default=2)
    ap.add_argument("--max-plies",type=int,default=160)
    ap.add_argument("--seed",type=int,default=20261026)
    ap.add_argument("--out",default="build/engine-selfplay.jsonl")
    args=ap.parse_args()
    if not (1<=args.attempts<=1000 and 1<=args.movetime<=10000 and 1<=args.depth<=6):
        ap.error("invalid limits")
    dest=Path(args.out)
    dest.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with dest.open("w",encoding="utf-8") as f:
        for i in range(args.attempts):
            game=play(args.seed+i,args.movetime,args.max_plies,args.depth)
            if game:
                f.write(json.dumps(game)+"\n")
                count+=1
    summary={"generator":"fukaji-alpha-beta-v0","attempts":args.attempts,"completed_checkmates":count,"movetime_ms":args.movetime,"depth":args.depth,"max_plies":args.max_plies,"seed":args.seed,"dataset_sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),"note":"No outcome is invented for move limits, resign or repetition."}
    (dest.parent / (dest.stem + ".manifest.json")).write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary))
if __name__=="__main__":main()
