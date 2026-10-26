"""Self-play of the actual FUKAJI bootstrap search (not the random smoke generator)."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import shogi
from .engine import select_move
from .value_model import LinearValue

def play(seed, movetime_ms=25, max_plies=160, max_depth=2, model=None, opening_plies=4):
    rng=random.Random(seed)
    board=shogi.Board()
    start=board.sfen()
    moves=[]
    for ply in range(min(opening_plies,max_plies)):
        if board.is_checkmate() or board.is_fourfold_repetition():
            break
        options=list(board.legal_moves)
        if not options:break
        chosen=options[rng.randrange(len(options))]
        moves.append(chosen.usi())
        board.push(chosen)
    for ply in range(len(moves),max_plies):
        if board.is_checkmate():
            break
        if board.is_fourfold_repetition():
            return None
        options=list(board.legal_moves)
        if not options:
            break
        move_text,depth,nodes=select_move(board,movetime_ms,max_depth=max_depth,model=model)
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
    ap.add_argument("--model",default=None)
    ap.add_argument("--model-mix",type=int,default=50)
    ap.add_argument("--opening-plies",type=int,default=4)
    args=ap.parse_args()
    if not (1<=args.attempts<=1000 and 1<=args.movetime<=10000 and 1<=args.depth<=6 and 0<=args.opening_plies<=20):
        ap.error("invalid limits")
    dest=Path(args.out)
    dest.parent.mkdir(parents=True,exist_ok=True)
    model=LinearValue(args.model,mix_percent=args.model_mix) if args.model else None
    count=0
    with dest.open("w",encoding="utf-8") as f:
        for i in range(args.attempts):
            game=play(args.seed+i,args.movetime,args.max_plies,args.depth,model=model,opening_plies=args.opening_plies)
            if game:
                f.write(json.dumps(game)+"\n")
                count+=1
    summary={"generator":"fukaji-alpha-beta-v0","value_model_sha256":model.sha256 if model else None,"value_mix_percent":args.model_mix if model else 0,"attempts":args.attempts,"completed_checkmates":count,"movetime_ms":args.movetime,"depth":args.depth,"opening_plies":args.opening_plies,"max_plies":args.max_plies,"seed":args.seed,"dataset_sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),"note":"No outcome is invented for move limits, resign or repetition."}
    (dest.parent / (dest.stem + ".manifest.json")).write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary))
if __name__=="__main__":main()
