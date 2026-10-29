"""D6-1: lightweight NNUE architecture cost gate with synthetic weights.

This is NOT trained G8/NNUE strength comparison. It measures Python runtime
overhead only, without new games, labels, or changes to the engine.
"""
from __future__ import annotations

import json
import random
import statistics
import time

import shogi

from .engine import evaluate
from .nnue_research import DualAccumulator, SmallNNUE


def fixed_positions():
    """Deterministic legal positions; never inspect D5 sealed parents."""
    rng = random.Random(20261030)
    board = shogi.Board()
    result = [(0, board.sfen())]
    for ply in range(1, 81):
        if board.is_game_over():
            break
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(legal[rng.randrange(len(legal))])
        if ply in (20, 60):
            result.append((ply, board.sfen()))
    return result


def timed_median(operation, repeats):
    samples = []
    for _ in range(3):
        start = time.perf_counter_ns()
        for _ in range(repeats):
            operation()
        samples.append((time.perf_counter_ns() - start) / repeats / 1e6)
    return round(statistics.median(samples), 6)


def measure(repeats=30):
    if not 2 <= repeats <= 200:
        raise ValueError("bounded synthetic benchmark only")
    model = SmallNNUE.synthetic_test_model(seed=73)
    cases = []
    for ply, sfen in fixed_positions():
        board = shogi.Board(sfen)
        initial = (board.sfen(), board.zobrist_hash(), len(board.move_stack))
        state = DualAccumulator(board, model)
        state.verify()
        move = next(iter(board.legal_moves))
        def bare_pushpop():
            board.push(move)
            board.pop()
        def incremental_pushpop():
            state.push(move)
            state.pop()
        times = {
            "classical_eval_ms": timed_median(lambda: evaluate(board), repeats),
            "nnue_full_eval_ms": timed_median(lambda: model.logit(board), repeats),
            "nnue_cached_eval_ms": timed_median(state.logit, repeats),
            "board_pushpop_ms": timed_median(bare_pushpop, repeats),
            "nnue_pushpop_ms": timed_median(incremental_pushpop, repeats),
        }
        state.verify()
        if (board.sfen(), board.zobrist_hash(), len(board.move_stack)) != initial:
            raise AssertionError("NNUE benchmark mutated the board")
        times["nnue_pushpop_over_board_ratio"] = round(
            times["nnue_pushpop_ms"] / times["board_pushpop_ms"], 3)
        times["cached_over_classical_eval_ratio"] = round(
            times["nnue_cached_eval_ms"] / times["classical_eval_ms"], 3)
        cases.append({"sample_ply": ply, **times, "board_restored": True})
    if len(cases) != 3:
        raise ValueError("could not construct three fixed legal positions")
    return {
        "scope": "D6_synthetic_nnue_cost_only_not_trained_not_strength",
        "synthetic_untrained_weights": True, "repeats_per_batch": repeats,
        "batches_per_measurement": 3, "position_count": len(cases),
        "new_games": 0, "training": False, "engine_modified": False,
        "sealed_d5_parents_accessed": False, "cases": cases,
        "warning": ("Microbenchmarks on a shared CI runner are approximate. "
                    "No trained NNUE, fixed G8 model, 200ms search or game comparison."),
    }


if __name__ == "__main__":
    print("D6_NNUE_COST " + json.dumps(measure(), sort_keys=True))
