"""Research-only, history-bound python-shogi legal fallback snapshot.

No USI I/O, worker creation, clock enforcement, outcome labels or adoption.
The caller must use the authoritative board (including its move history), and
measure go-receive/fallback-ready timestamps outside this helper.
"""
from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class LegalSnapshot:
    position_key: str
    legal_moves: tuple[str, ...]
    in_check: bool
    checkmated: bool
    fourfold_repetition: bool
    history_plies: int


def snapshot_from_board(board):
    """Take an immutable snapshot without pushing or popping a move.

    SFEN alone does not contain repetition history. The key includes the
    current SFEN and the actual move stack; it is an identity guard, not a
    cryptographic proof that an untrusted board is legal.
    """
    original_sfen = board.sfen()
    original_history = tuple(move.usi() for move in board.move_stack)
    moves = tuple(move.usi() for move in board.legal_moves)
    in_check = bool(board.is_check())
    checkmated = bool(board.is_checkmate())
    repeated = bool(board.is_fourfold_repetition())
    if board.sfen() != original_sfen or tuple(m.usi() for m in board.move_stack) != original_history:
        raise RuntimeError('legal snapshot mutated the board')
    if len(moves) != len(set(moves)) or any(not m or m == 'resign' for m in moves):
        raise ValueError('invalid legal snapshot')
    if checkmated and moves:
        raise ValueError('checkmated board has legal moves')
    payload = json.dumps((original_sfen, original_history), separators=(',', ':'),
                         ensure_ascii=True)
    return LegalSnapshot(hashlib.sha256(payload.encode('ascii')).hexdigest(),
                         moves, in_check, checkmated, repeated,
                         len(original_history))


def begin_with_legal_snapshot(supervisor, board, *, go_received_at,
                              fallback_ready_at, budget_ms=200):
    """Start a synthetic response generation with actual python-shogi moves.

    Returns (generation, snapshot). Time spent building the snapshot must be
    included in the caller-supplied fallback_ready_at; this function cannot
    prove a physical 200 ms response deadline.
    """
    snapshot = snapshot_from_board(board)
    generation = supervisor.begin(
        position_key=snapshot.position_key, legal_moves=snapshot.legal_moves,
        go_received_at=go_received_at, fallback_ready_at=fallback_ready_at,
        budget_ms=budget_ms)
    return generation, snapshot
