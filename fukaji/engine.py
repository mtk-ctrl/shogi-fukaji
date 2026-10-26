"""Independent FUKAJI v0 USI search engine.

This deliberately small engine is a bootstrapping search baseline, NOT a port
of KUMOJI search or an NNUE-enabled model. It uses python-shogi's legal rules,
our own negamax alpha-beta and material/position evaluation.
"""
from __future__ import annotations
import argparse
import math
import random
import sys
import time
import shogi
from .value_model import LinearValue

VALUES = {1:100,2:300,3:320,4:480,5:760,6:900,7:600,8:0,9:500,10:520,11:620,12:850,13:1100,14:650}
INF=10**9
MATE=1000000

def evaluate(board, model=None):
    score=0
    for sq in range(81):
        p=board.piece_at(sq)
        if p:
            val=VALUES[p.piece_type]
            if p.piece_type != shogi.KING:
                # Encourage central piece development at modest weight.
                rank=sq % 9
                file=sq // 9
                val += 2*(4-abs(file-4)) + (4-abs(rank-4))
            score += val if p.color==board.turn else -val
    for color in (shogi.BLACK,shogi.WHITE):
        factor=1 if color==board.turn else -1
        for kind,count in board.pieces_in_hand[color].items():
            score+=factor*int(count)*VALUES[kind]
    return model.combine(board,score) if model is not None else score

def ordered_moves(board):
    options=list(board.legal_moves)
    def order(m):
        p=board.piece_at(m.to_square)
        return (VALUES.get(p.piece_type,0) if p else 0) + (200 if m.promotion else 0)
    options.sort(key=order,reverse=True)
    return options

def negamax(board, depth, alpha, beta, deadline, nodes, model=None):
    nodes[0]+=1
    if (nodes[0]&127)==0 and time.monotonic()>=deadline:
        raise TimeoutError()
    if board.is_checkmate():
        return -MATE-depth
    if board.is_fourfold_repetition():
        return 0
    if depth<=0:
        return evaluate(board, model)
    legal=ordered_moves(board)
    if not legal:
        return -MATE-depth if board.is_check() else 0
    best=-INF
    for move in legal:
        board.push(move)
        try:
            score=-negamax(board,depth-1,-beta,-alpha,deadline,nodes,model)
        finally:
            board.pop()
        if score>best:
            best=score
        alpha=max(alpha,best)
        if alpha>=beta:
            break
    return best

def select_move(board, movetime_ms=200, max_depth=5, model=None, avoid_repetition=True):
    moves=ordered_moves(board)
    # At the root, prefer genuinely new legal positions over repeating an
    # earlier position. The old engine repeated 96/100 games to termination.
    # A move is not discarded if every legal move repeats; all moves remain
    # legal and forced repetitions are still reported as unresolved.
    if avoid_repetition and len(moves)>1:
        counts=[]
        for m in moves:
            board.push(m)
            try:
                count=board.transpositions[board.zobrist_hash()]
                # Immediate mate must never be filtered out by cycle avoidance.
                if board.is_checkmate():
                    count=-1
            finally:
                board.pop()
            counts.append(count)
        least=min(counts)
        moves=[m for m,count in zip(moves,counts) if count==least]

    if not moves:
        return "resign",0,0
    deadline=time.monotonic()+max(1,movetime_ms)/1000
    best=moves[0]
    bestscore=0
    nodes=[0]
    complete_depth=0
    for depth in range(1,max_depth+1):
        if time.monotonic()>=deadline:
            break
        current=best
        scorebest=-INF
        try:
            for move in moves:
                if time.monotonic()>=deadline:
                    raise TimeoutError()
                board.push(move)
                try:
                    score=-negamax(board,depth-1,-INF,INF,deadline,nodes,model)
                finally:
                    board.pop()
                if score>scorebest:
                    scorebest=score
                    current=move
        except TimeoutError:
            break
        best,current=current,best
        bestscore=scorebest
        complete_depth=depth
        moves.sort(key=lambda m: 0 if m==best else 1)
    return best.usi(),complete_depth,nodes[0]

def command_loop():
    board=shogi.Board()
    depth=4
    model_path = None
    model_mix = 50
    model = None
    for raw in sys.stdin:
        tokens=raw.strip().split()
        if not tokens:
            continue
        cmd=tokens[0]
        if cmd=="usi":
            print("id name FUKAJI v0 bootstrap")
            print("id author mtk-ctrl")
            print("option name SearchDepth type spin default 4 min 1 max 6")
            print("option name ValueModelFile type string default <empty>")
            print("option name ValueModelMix type spin default 50 min 0 max 100")
            print("usiok",flush=True)
        elif cmd=="isready":
            if model_path:
                try:
                    model=LinearValue(model_path,mix_percent=model_mix)
                    print("info string value_model sha256 "+model.sha256,flush=True)
                except Exception as e:
                    model=None
                    print("info string value_model rejected: "+str(e),flush=True)
            else:
                model=None
            print("readyok",flush=True)
        elif cmd=="usinewgame":
            board=shogi.Board()
        elif cmd=="setoption":
            if "ValueModelFile" in tokens:
                idx=tokens.index("ValueModelFile")+1
                model_path=" ".join(tokens[idx+1:] if idx<len(tokens) and tokens[idx]=="value" else tokens[idx:]) or None
                model=None
            if "ValueModelMix" in tokens and "value" in tokens:
                try: model_mix=max(0,min(100,int(tokens[tokens.index("value")+1])))
                except (ValueError,IndexError): pass
                model=None
            if "SearchDepth" in tokens and "value" in tokens:
                try: depth=max(1,min(6,int(tokens[tokens.index("value")+1])))
                except (ValueError,IndexError): pass
        elif cmd=="position":
            try:
                if len(tokens)<2: raise ValueError("missing position")
                if tokens[1]=="startpos":
                    board=shogi.Board()
                    idx=2
                elif tokens[1]=="sfen":
                    if len(tokens)<6:raise ValueError("invalid SFEN")
                    board=shogi.Board(" ".join(tokens[2:6]))
                    idx=6
                else: raise ValueError("invalid position type")
                if idx<len(tokens):
                    if tokens[idx]!="moves":raise ValueError("unknown position suffix")
                    for word in tokens[idx+1:]:
                        move=shogi.Move.from_usi(word)
                        if move not in board.legal_moves:raise ValueError("illegal recorded move")
                        board.push(move)
            except Exception as e:
                print("info string invalid position: "+str(e),flush=True)
        elif cmd=="go":
            ms=200
            if "movetime" in tokens:
                try:ms=max(1,min(10000,int(tokens[tokens.index("movetime")+1])))
                except (ValueError,IndexError):pass
            move,finished,nodes=select_move(board,ms,depth,model)
            print(f"info depth {finished} nodes {nodes}",flush=True)
            print(f"bestmove {move}",flush=True)
        elif cmd=="quit":
            break

if __name__=="__main__":
    command_loop()
