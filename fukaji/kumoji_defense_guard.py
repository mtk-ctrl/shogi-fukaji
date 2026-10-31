"""Research-only defense guard for FUKAJI's historical KUMOJI mate losses.

Run the existing G8 search with a reserved portion of its total move budget.
If its chosen move allows an independently proven immediate enemy checkmate,
try legal alternatives in static-evaluation order. Only an alternative whose
complete opponent legal reply list is checked and contains no immediate mate
may replace the original. Exhaustion is UNKNOWN, never 'safe'.

No edits to production select_move, USI options or G8 weight file.
"""
from __future__ import annotations
import time
import shogi
from .engine import evaluate, select_move
from .kumoji_loss_study import snapshot


class ProbeIncomplete(RuntimeError):
    pass


def _enemy_mates_immediately(board, deadline):
    """Board is after FUKAJI move; prove whether opponent has mate-in-one."""
    for enemy in tuple(board.legal_moves):
        if time.monotonic() >= deadline:
            raise ProbeIncomplete("guard-time-exhausted")
        board.push(enemy)
        try:
            if board.is_checkmate():
                return True
        finally:
            board.pop()
    return False


def _unsafe_root_move(board, move, deadline):
    if time.monotonic() >= deadline:
        raise ProbeIncomplete("guard-time-exhausted")
    before=snapshot(board)
    board.push(move)
    try:
        return _enemy_mates_immediately(board,deadline)
    finally:
        board.pop()
        if snapshot(board)!=before:
            raise AssertionError("guard did not restore root position")


def _safe_candidates_by_static_score(board, chosen, model, fast_relative):
    scored=[]
    before=snapshot(board)
    for move in tuple(board.legal_moves):
        if move.usi()==chosen:
            continue
        board.push(move)
        try:
            score=-evaluate(board,model,fast_relative=fast_relative)
        finally:
            board.pop()
        scored.append((score, move.usi(), move))
    if snapshot(board)!=before:
        raise AssertionError("scoring changed board/history")
    scored.sort(key=lambda x:(-x[0],x[1]))
    return [m for _,_,m in scored]


def select_guarded(board, model, *, movetime_ms=200, reserve_ms=40,
                   depth=3, fast_relative=True, root_alpha_beta=True,
                   fast_move_list=True, search_override=None):
    """Return (legal move, proof/timeout telemetry); research opt-in only.

    Total wall time is measured, not hard-capped by Python timing: exceptionally
    slow individual shogi legal operations can exceed the nominal budget.
    """
    if not 1<=reserve_ms<movetime_ms or not 1<=depth<=6:
        raise ValueError("invalid split of search / guard budgets")
    before=snapshot(board)
    start=time.monotonic()
    deadline=start+movetime_ms/1000
    search=search_override or select_move
    search_telemetry={}
    chosen,completed,nodes=search(
        board,movetime_ms=movetime_ms-reserve_ms,max_depth=depth,model=model,
        avoid_repetition=True,fast_relative=fast_relative,
        root_alpha_beta=root_alpha_beta,fast_move_list=fast_move_list,
        deadline_from_call=True,telemetry=search_telemetry)
    if snapshot(board)!=before:
        raise AssertionError("baseline search altered board")
    result={"original_move":chosen,"move":chosen,"depth":completed,
            "nodes":nodes,"original_allows_mate":None,
            "guard_status":"not_checked_budget_exhausted",
            "guard_researched":0,"guard_replacements":0,
            "search_telemetry":search_telemetry}
    if chosen=="resign":
        result["guard_status"]="resign_no_legal_move"
    else:
        move=shogi.Move.from_usi(chosen)
        if move not in board.legal_moves:
            raise ValueError("baseline selected illegal move")
        try:
            danger=_unsafe_root_move(board,move,deadline)
        except ProbeIncomplete:
            result["guard_status"]="unknown_time_expired"
        else:
            result["original_allows_mate"]=danger
            if not danger:
                result["guard_status"]="verified_no_immediate_mate"
            else:
                result["guard_status"]="confirmed_immediate_mate_no_safe_alt_yet"
                for alternative in _safe_candidates_by_static_score(
                        board,chosen,model,fast_relative):
                    if time.monotonic() >= deadline:
                        result["guard_status"]="unknown_time_expired_after_danger"
                        break
                    try:
                        unsafe=_unsafe_root_move(board,alternative,deadline)
                    except ProbeIncomplete:
                        result["guard_status"]="unknown_time_expired_after_danger"
                        break
                    result["guard_researched"]+=1
                    if not unsafe:
                        result["move"]=alternative.usi()
                        result["guard_replacements"]=1
                        result["guard_status"]="confirmed_avoidable_immediate_mate"
                        break
    if snapshot(board)!=before:
        raise AssertionError("guard changed board/history")
    if result["move"]!="resign" and shogi.Move.from_usi(result["move"]) not in board.legal_moves:
        raise ValueError("guard chose illegal move")
    result["total_ms"]=round((time.monotonic()-start)*1000,3)
    result["over_200_ms"]=result["total_ms"]>movetime_ms
    return result
