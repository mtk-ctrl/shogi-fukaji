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
from .relative_value import RelativeValue

VALUES = {1:100,2:300,3:320,4:480,5:760,6:900,7:600,8:0,9:500,10:520,11:620,12:850,13:1100,14:650}
INF=10**9
MATE=1000000

def evaluate(board, model=None, fast_relative=False):
    if fast_relative and isinstance(model, RelativeValue) and model.mix > 0:
        return evaluate_relative_fused(board, model)
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

def evaluate_relative_fused(board, model):
    """Single board scan for classical + frozen relative score (opt-in research)."""
    turn = int(board.turn)
    classical = 0
    logit = model._turn_weight
    for sq in range(81):
        p = board.piece_at(sq)
        if p is None:
            continue
        val = VALUES[p.piece_type]
        if p.piece_type != shogi.KING:
            rank, file = sq % 9, sq // 9
            val += 2 * (4 - abs(file - 4)) + (4 - abs(rank - 4))
        classical += val if p.color == board.turn else -val
        owner = int(p.color) ^ turn
        relative_sq = sq if turn == shogi.BLACK else 80 - sq
        logit += model._piece_weights[(owner * 14 + p.piece_type - 1) * 81 + relative_sq]
    for color in (shogi.BLACK, shogi.WHITE):
        owner = int(color) ^ turn
        factor = 1 if color == board.turn else -1
        for kind, count in board.pieces_in_hand[color].items():
            classical += factor * int(count) * VALUES[kind]
            logit += model._hand_prefix[owner, kind][min(20, int(count))]
    learned = int(round(max(-10.0, min(10.0, logit)) * model.weight_cp))
    return int(round((1 - model.mix) * classical + model.mix * learned))


def fast_legal_moves(board):
    """Conservative legality shortcut, falling back to python-shogi for risk cases.

    When the king is not in check, a non-king move originating outside the
    king's rank/file/diagonals cannot reveal a sliding attack. A non-pawn
    drop cannot reveal an attack either. All other moves, especially king
    moves, pinned pieces, evasions and pawn drops, use the original test.
    """
    king = board.king_squares[board.turn]
    safe = king is not None and not board.is_check()
    if safe:
        kfile, krank = divmod(king, 9)
    for move in board.generate_pseudo_legal_moves():
        if safe:
            if move.drop_piece_type:
                if move.drop_piece_type != shogi.PAWN:
                    yield move
                    continue
            elif move.from_square is not None and move.from_square != king:
                mfile, mrank = divmod(move.from_square, 9)
                if (mfile != kfile and mrank != krank
                        and abs(mfile - kfile) != abs(mrank - krank)):
                    yield move
                    continue
        if not board.is_suicide_or_check_by_dropping_pawn(move):
            yield move


def ordered_moves(board, fast_move_list=False, fast_legal_guard=False):
    # Opt-in research: default path remains the original python-shogi logic.
    source = fast_legal_moves(board) if fast_legal_guard else board.legal_moves
    options=([m for m in source] if fast_move_list or fast_legal_guard
             else list(source))
    def order(m):
        p=board.piece_at(m.to_square)
        return (VALUES.get(p.piece_type,0) if p else 0) + (200 if m.promotion else 0)
    options.sort(key=order,reverse=True)
    return options

def negamax(board, depth, alpha, beta, deadline, nodes, model=None, fast_relative=False, guard_checkmate=False, fast_move_list=False, fast_legal_guard=False):
    nodes[0]+=1
    if (nodes[0]&127)==0 and time.monotonic()>=deadline:
        raise TimeoutError()
    # Opt-in research: a side not in check cannot be checkmated.
    # Keep the legacy path unchanged unless explicitly requested.
    if (not guard_checkmate or board.is_check()) and board.is_checkmate():
        return -MATE-depth
    # A repetition is legal, but a shallow search must not score a
    # self-imposed cycle as equal to genuine positional improvement.
    # The root keeps every legal move available if repetition is forced.
    if board.is_fourfold_repetition():
        return -25000
    if board.transpositions[board.zobrist_hash()] >= 2:
        return -1000
    if depth<=0:
        return evaluate(board, model, fast_relative)
    legal=ordered_moves(board, fast_move_list=fast_move_list, fast_legal_guard=fast_legal_guard)
    if not legal:
        return -MATE-depth if board.is_check() else 0
    best=-INF
    for move in legal:
        board.push(move)
        try:
            score=-negamax(board,depth-1,-beta,-alpha,deadline,nodes,model,fast_relative,guard_checkmate,fast_move_list,fast_legal_guard)
        finally:
            board.pop()
        if score>best:
            best=score
        alpha=max(alpha,best)
        if alpha>=beta:
            break
    return best

def select_move(board, movetime_ms=200, max_depth=5, model=None, avoid_repetition=True, complete_depth=False, fast_relative=False, root_alpha_beta=False, guard_checkmate=False, fast_move_list=False, fast_legal_guard=False, telemetry=None, deadline_from_call=False):
    # Optional research-only timing and whole-call budget; USI defaults unchanged.
    call_started = time.monotonic() if telemetry is not None or deadline_from_call else None
    moves=ordered_moves(board, fast_move_list=fast_move_list, fast_legal_guard=fast_legal_guard)
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

    prepared_at = time.monotonic() if telemetry is not None else None
    if not moves:
        if telemetry is not None:
            telemetry.update(root_preparation_ms=(prepared_at-call_started)*1000,
                             internal_search_ms=0.0, total_call_ms=(prepared_at-call_started)*1000,
                             completed_depth=0, nodes=0)
        return "resign",0,0
    # Diagnostic mode removes wall-clock cutoffs. Both engines must complete
    # the same search depth to isolate evaluation quality from inference cost.
    # This mode is for bounded-depth research, not timed USI competition.
    deadline=(float("inf") if complete_depth else
              (call_started if deadline_from_call else time.monotonic()) + max(1,movetime_ms)/1000)
    search_started = time.monotonic() if telemetry is not None else None
    best=moves[0]
    nodes=[0]
    complete_depth=0
    for depth in range(1,max_depth+1):
        if time.monotonic()>=deadline:
            break
        current=best
        scorebest=-INF
        # Research-only root alpha reuse. Child beta=-alpha is a valid
        # fail-soft bound: a cut-off root move cannot beat the current best.
        # Disabled by default to preserve the original search baseline.
        root_alpha=-INF
        try:
            for move in moves:
                if time.monotonic()>=deadline:
                    raise TimeoutError()
                board.push(move)
                try:
                    child_beta = -root_alpha if root_alpha_beta else INF
                    score=-negamax(board,depth-1,-INF,child_beta,deadline,nodes,model,fast_relative,guard_checkmate,fast_move_list,fast_legal_guard)
                finally:
                    board.pop()
                if score>scorebest:
                    scorebest=score
                    current=move
                if root_alpha_beta and score>root_alpha:
                    root_alpha=score
        except TimeoutError:
            break
        best,current=current,best
        complete_depth=depth
        moves.sort(key=lambda m: 0 if m==best else 1)
    if telemetry is not None:
        returned_at = time.monotonic()
        telemetry.update(root_preparation_ms=(prepared_at-call_started)*1000,
                         internal_search_ms=(returned_at-search_started)*1000,
                         total_call_ms=(returned_at-call_started)*1000,
                         completed_depth=complete_depth, nodes=nodes[0])
    return best.usi(),complete_depth,nodes[0]

def command_loop():
    board=shogi.Board()
    depth=4
    model_path = None
    model_mix = 50
    model_type = "legacy"
    model = None
    fast_relative = False
    root_alpha_beta = False
    guard_checkmate = False
    fast_move_list = False
    fast_legal_guard = False
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
            print("option name ValueModelType type combo default legacy var legacy var relative")
            print("option name FastRelativeEval type check default false")
            print("option name RootAlphaBeta type check default false")
            print("option name GuardCheckmate type check default false")
            print("option name FastMoveList type check default false")
            print("option name FastLegalGuard type check default false")
            print("usiok",flush=True)
        elif cmd=="isready":
            if model_path:
                try:
                    model=(RelativeValue if model_type == "relative" else LinearValue)(
                        model_path, mix_percent=model_mix)
                    print("info string value_model sha256 "+model.sha256,flush=True)
                except Exception as e:
                    model=None
                    print("info string value_model rejected: "+str(e),flush=True)
                    # Never silently fall back to classical evaluation in a model match.
                    raise SystemExit(2)
            else:
                model=None
            print("readyok",flush=True)
        elif cmd=="usinewgame":
            board=shogi.Board()
        elif cmd=="setoption":
            if "ValueModelType" in tokens and "value" in tokens:
                model_type=tokens[tokens.index("value")+1].lower()
                if model_type not in ("legacy", "relative"):
                    raise SystemExit("invalid ValueModelType")
                model=None
            if "ValueModelFile" in tokens:
                idx=tokens.index("ValueModelFile")+1
                model_path=" ".join(tokens[idx+1:] if idx<len(tokens) and tokens[idx]=="value" else tokens[idx:]) or None
                model=None
            if "ValueModelMix" in tokens and "value" in tokens:
                try: model_mix=max(0,min(100,int(tokens[tokens.index("value")+1])))
                except (ValueError,IndexError): pass
                model=None
            if "FastRelativeEval" in tokens and "value" in tokens:
                fast_relative = tokens[tokens.index("value")+1].lower() == "true"
            if "RootAlphaBeta" in tokens and "value" in tokens:
                root_alpha_beta = tokens[tokens.index("value")+1].lower() == "true"
            if "GuardCheckmate" in tokens and "value" in tokens:
                guard_checkmate = tokens[tokens.index("value")+1].lower() == "true"
            if "FastMoveList" in tokens and "value" in tokens:
                fast_move_list = tokens[tokens.index("value")+1].lower() == "true"
            if "FastLegalGuard" in tokens and "value" in tokens:
                fast_legal_guard = tokens[tokens.index("value")+1].lower() == "true"
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
            move,finished,nodes=select_move(board,ms,depth,model,fast_relative=fast_relative,root_alpha_beta=root_alpha_beta,guard_checkmate=guard_checkmate,fast_move_list=fast_move_list,fast_legal_guard=fast_legal_guard)
            print(f"info depth {finished} nodes {nodes}",flush=True)
            print(f"bestmove {move}",flush=True)
        elif cmd=="quit":
            break

if __name__=="__main__":
    command_loop()
