"""D3 research-only G8 tactical probe; never enables quiescence in formal USI.

These small hand-built positions are development diagnostics, NOT an unseen
strength benchmark. Legal recapture and check-evasion facts are independently
checked; they do not establish a unique optimal move or win probability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import shogi

from .engine import VALUES, select_move
from .g8_paired_arena import G8_SHA
from .relative_value import RelativeValue

REQUEST = ".github/fukaji-d3-tactical-request.json"
EXPECTED = {
    "enabled": False, "research_only": True, "adoption_allowed": False,
    "training_allowed": False, "g8_model_run": 40636602223,
    "model_sha256": G8_SHA, "movetime_ms": 200, "depth_cap": 6,
    "mix_percent": 25, "fast_relative": True, "root_alpha_beta": True,
    "deadline_from_call": True, "fast_move_list": False, "runner_jobs": 1,
}
ORDER = ("A", "B", "B", "A")
CASES = (
    ("poison_recapture", "8k/9/4r4/4s4/4R4/9/9/9/8K b - 1"),
    ("direct_capture", "8k/9/4r4/4R4/9/9/9/9/8K w - 1"),
    ("check_evasion", "4k4/9/9/9/4R4/9/9/9/4K4 w - 1"),
    ("quiet_promotion", "8k/9/4P4/9/9/9/9/9/8K b - 1"),
    ("illegal_pawn_drop_mate", "3nkn3/3p1p3/4G4/9/9/9/9/9/8K b P 1"),
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def snapshot(board):
    # python-shogi may retain zero-count transposition keys after pop().
    return (board.sfen(), tuple(board.move_stack),
            {key: value for key, value in board.transpositions.items() if value})


def verify_case(name, board):
    """Verify only objective legal/tactical facts, not a subjective best move."""
    before = snapshot(board)
    if name == "poison_recapture":
        first, reply = (shogi.Move.from_usi(s) for s in ("5e5d", "5c5d"))
        assert first in board.legal_moves
        assert board.piece_at(first.to_square).piece_type == shogi.SILVER
        board.push(first)
        try:
            assert reply in board.legal_moves
            assert board.piece_at(reply.to_square).piece_type == shogi.ROOK
            board.push(reply)
            try:
                assert VALUES[shogi.SILVER] < VALUES[shogi.ROOK]
            finally:
                board.pop()
        finally:
            board.pop()
        evidence = "legal silver capture followed by legal rook recapture; net material negative"
    elif name == "direct_capture":
        move = shogi.Move.from_usi("5c5d")
        assert move in board.legal_moves
        assert board.piece_at(move.to_square).piece_type == shogi.ROOK
        evidence = "legal rook capture"
    elif name == "check_evasion":
        assert board.is_check()
        legal = list(board.legal_moves)
        assert legal
        for move in legal:
            board.push(move)
            try:
                # The formerly checked king cannot remain attacked after a legal evasion.
                king = board.king_squares[shogi.WHITE]
                assert king is not None
                assert not board.is_attacked_by(shogi.BLACK, king)
            finally:
                board.pop()
        evidence = "all enumerated check evasions leave the white king unattacked"
    elif name == "quiet_promotion":
        move = shogi.Move.from_usi("5c5b+")
        assert move in board.legal_moves
        assert board.piece_at(move.to_square) is None
        evidence = "legal noncapture promotion"
    elif name == "illegal_pawn_drop_mate":
        move = shogi.Move.from_usi("P*5b")
        assert move in board.generate_pseudo_legal_moves()
        assert move not in board.legal_moves
        evidence = "pawn-drop mate rejected by legal-move generator"
    else:
        raise ValueError("unknown frozen case")
    if snapshot(board) != before:
        raise AssertionError("oracle changed board/history")
    return {"case": name, "sfen": board.sfen(), "evidence": evidence,
            "oracle_scope": "legal fact only, not a unique best move"}


def frozen_proofs():
    proofs = [verify_case(name, shogi.Board(sfen)) for name, sfen in CASES]
    assert len({x["sfen"] for x in proofs}) == len(proofs)
    return proofs



# These constructed positions are development checks, not held-out puzzles.
MATE_ONE_SFEN = "8k/9/7R1/7L1/9/9/9/9/8K b - 1"
MATE_ONE_COUNTEREXAMPLE_SFEN = "8k/9/7R1/9/9/9/9/9/8K b - 1"


def enumerate_mate_in_one(board):
    """Independent legal-move oracle: checking move and zero legal replies.

    The result is the complete set of mate-in-one moves, not a heuristic
    preference or a proof of longer tactical value.
    """
    before = snapshot(board)
    mates = []
    for move in list(board.legal_moves):
        board.push(move)
        try:
            if board.is_check() and not any(board.legal_moves):
                mates.append(move.usi())
        finally:
            board.pop()
    if snapshot(board) != before:
        raise AssertionError("mate oracle changed board/history")
    return tuple(sorted(mates))


def mate_choice_diagnostic():
    """Run the real research search against a proven one-ply oracle.

    This intentionally uses classical evaluation and unbounded fixed depth 1,
    NOT the frozen G8 model or a 200ms strength test. It checks whether the
    existing root search already recognizes immediate mates with Q OFF/ON.
    """
    results = []
    for name, sfen in (
        ("guarded_mate", MATE_ONE_SFEN),
        ("unguarded_counterexample", MATE_ONE_COUNTEREXAMPLE_SFEN),
    ):
        board = shogi.Board(sfen)
        before = snapshot(board)
        mates = enumerate_mate_in_one(board)
        choices = []
        for enabled in (False, True):
            move, depth, nodes = select_move(
                board, movetime_ms=200, max_depth=1, model=None,
                complete_depth=True, avoid_repetition=False,
                fast_relative=False, root_alpha_beta=True,
                fast_move_list=False, deadline_from_call=True,
                quiescence=enabled,
            )
            if snapshot(board) != before:
                raise AssertionError("search changed board/history")
            if move == "resign" or shogi.Move.from_usi(move) not in board.legal_moves:
                raise AssertionError("search returned an illegal move")
            choices.append({
                "quiescence": enabled, "move": move, "depth": depth,
                "nodes": nodes, "is_proven_mate_in_one": move in mates,
            })
        results.append({
            "case": name, "sfen": sfen, "proven_mate_in_one_moves": list(mates),
            "choices": choices, "model": "classical_no_G8",
            "clock": "fixed_depth_1_unbounded_not_200ms",
            "scope": "constructed_development_oracle_not_strength",
        })
    if "2c2b+" not in results[0]["proven_mate_in_one_moves"]:
        raise AssertionError("guarded mate proof failed")
    if "2c2b+" in results[1]["proven_mate_in_one_moves"]:
        raise AssertionError("unguarded counterexample proof failed")
    return results


def validate_request(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config not in (EXPECTED, dict(EXPECTED, enabled=True)):
        raise ValueError("D3 request differs from fixed protocol")
    return config["enabled"]


def run_probe(model_path, out_path):
    raw_model = Path(model_path).read_bytes()
    if digest(raw_model) != G8_SHA:
        raise ValueError("frozen G8 model SHA mismatch")
    model = RelativeValue(model_path, mix_percent=25)
    if model.sha256 != G8_SHA:
        raise ValueError("G8 loader SHA mismatch")
    proofs = frozen_proofs()
    records = []
    for name, sfen in CASES:
        board = shogi.Board(sfen)
        before = snapshot(board)
        observations = []
        for variant in ORDER:
            timing = {}
            started = time.perf_counter()
            move, depth, nodes = select_move(
                board, movetime_ms=200, max_depth=6, model=model,
                fast_relative=True, root_alpha_beta=True,
                fast_move_list=False, deadline_from_call=True,
                quiescence=(variant == "B"), telemetry=timing)
            outer_ms = (time.perf_counter() - started) * 1000
            if snapshot(board) != before:
                raise AssertionError("D3 engine changed board/history")
            if move == "resign" or shogi.Move.from_usi(move) not in board.legal_moves:
                raise AssertionError("D3 returned illegal move")
            if timing["completed_depth"] != depth or timing["nodes"] != nodes:
                raise AssertionError("D3 telemetry mismatch")
            if timing["total_call_ms"] > outer_ms + 1:
                raise AssertionError("D3 inconsistent wall time")
            observations.append({
                "variant": variant, "quiescence": variant == "B",
                "move": move, "completed_depth": depth, "nodes": nodes,
                "outer_call_ms": round(outer_ms, 3),
                "engine_call_ms": round(timing["total_call_ms"], 3),
                "root_preparation_ms": round(timing["root_preparation_ms"], 3),
                "internal_search_ms": round(timing["internal_search_ms"], 3),
                "over_200ms": outer_ms > 200,
            })
        records.append({"case": name, "sfen": sfen, "observations": observations})
    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=True)
    proof_bytes = (json.dumps(proofs, sort_keys=True, indent=2) + "\n").encode()
    obs_bytes = "".join(json.dumps(x, sort_keys=True) + "\n" for x in records).encode()
    (out / "proofs.json").write_bytes(proof_bytes)
    (out / "observations.jsonl").write_bytes(obs_bytes)
    a = [v for row in records for v in row["observations"] if v["variant"] == "A"]
    b = [v for row in records for v in row["observations"] if v["variant"] == "B"]
    summary = {
        "experiment": "D3_quiescence_G8_development_tactical_ABBA",
        "status": "development_diagnostic_not_playing_strength",
        "model_sha256": G8_SHA, "positions": len(records),
        "observations": len(a) + len(b), "order": list(ORDER),
        "movetime_ms": 200, "depth_cap": 6, "mix_percent": 25,
        "fast_relative": True, "root_alpha_beta": True,
        "fast_move_list": False, "deadline_from_call": True,
        "a_over_200ms": sum(x["over_200ms"] for x in a),
        "b_over_200ms": sum(x["over_200ms"] for x in b),
        "a_median_outer_ms": round(statistics.median(x["outer_call_ms"] for x in a), 3),
        "b_median_outer_ms": round(statistics.median(x["outer_call_ms"] for x in b), 3),
        "a_median_depth": statistics.median(x["completed_depth"] for x in a),
        "b_median_depth": statistics.median(x["completed_depth"] for x in b),
        "proofs_sha256": digest(proof_bytes), "observations_sha256": digest(obs_bytes),
        "independent_optimal_move_proven": False,
        "unseen_test": False, "new_games": 0, "formal_adoption": False,
        "strength_improvement_proven": False,
        "limitations": [
            "These are five hand-built development cases; not independent held-out test positions.",
            "Oracle establishes legal tactical facts, not unique correct moves or game-theoretic values.",
            "Quiescence can increase compute cost; equal wall-clock budget must be audited.",
            "200ms is an engine deadline, not a hard OS response guarantee.",
            "No win-rate or playing-strength claim is justified by this probe.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--check-request")
    parser.add_argument("--model")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(frozen_proofs(), indent=2))
        return
    if args.check_request:
        print(json.dumps({"enabled": validate_request(args.check_request)}))
        return
    if not args.model or not args.out:
        parser.error("model and out required")
    if not validate_request(REQUEST):
        raise ValueError("D3 request is disabled; no real measurements")
    print(json.dumps(run_probe(args.model, args.out), indent=2))


if __name__ == "__main__":
    main()
