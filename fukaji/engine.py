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

VALUES = {1:100,2:300,3:320,4:480,5:760,6:900,7:600,8:0,9:500,10:520,11:620,12:850,13:1100,14:650}
INF=10**9
MATE=1000000

def evaluate(board):
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
    return score

def ordered_moves(board):
    options=list(board.legal_moves)
    def order(m):
        p=board.piece_at(m.to_square)
        return (VALUES.get(p.piece_type,0) if p else 0) + (200 if m.promotion else 0)
    options.sort(key=order,reverse=True)
    return options

def negamax(board, depth, alpha, beta, deadline, nodes):
    nodes[0]+=1
    if (nodes[0]&127)==0 and time.monotonic()>=deadline:
        raise TimeoutError()
    if board.is_checkmate():
        return -MATE-depth
    if board.is_fourfold_repetition():
        return 0
    if depth<=0:
        return evaluate(board)
    legal=ordered_moves(board)
    if not legal:
        return -MATE-depth if board.is_check() else 0
    best=-INF
    for move in legal:
        board.push(move)
        try:
            score=-negamax(board,depth-1,-beta,-alpha,deadline,nodes)
        finally:
            board.pop()
        if score>best:
            best=score
        alpha=max(alpha,best)
        if alpha>=beta:
            break
    return best

def select_move(board, movetime_ms=200, max_depth=5):
    moves=ordered_moves(board)
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
                    score=-negamax(board,depth-1,-INF,INF,deadline,nodes)
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
    for raw in sys.stdin:
        tokens=raw.strip().split()
        if not tokens:
            continue
        cmd=tokens[0]
        if cmd=="usi":
            print("id name FUKAJI v0 bootstrap")
            print("id author mtk-ctrl")
            print("option name SearchDepth type spin default 4 min 1 max 6")
            print("usiok",flush=True)
        elif cmd=="isready":
            print("readyok",flush=True)
        elif cmd=="usinewgame":
            board=shogi.Board()
        elif cmd=="setoption":
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
            move,finished,nodes=select_move(board,ms,depth)
            print(f"info depth {finished} nodes {nodes}",flush=True)
            print(f"bestmove {move}",flush=True)
        elif cmd=="quit":
            break

if __name__=="__main__":
    command_loop()
