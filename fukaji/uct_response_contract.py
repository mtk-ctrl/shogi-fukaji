"""Research-only UCT response protocol model; NOT wired into USI.

Legal moves and monotonic timestamps must come from the trusted supervisor.
No worker process, stdout I/O, shogi rule validation, or real-time guarantee.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ResponseReceipt:
    generation: int
    move: str
    go_received_at: float
    fallback_ready_at: float
    prepared_at: float
    sent_at: float
    deadline_at: float
    fallback_ready_in_time: bool
    response_sent_in_time: bool
    worker_stopped: bool


class ResponseSupervisorModel:
    """Single-threaded synthetic protocol; not a production supervisor."""

    def __init__(self):
        self.generation = 0
        self._active = False
        self._sent = None
        self._last_event_at = float('-inf')

    def _clock(self, now):
        if not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError('finite supervisor timestamp required')
        if now < self._last_event_at:
            raise ValueError('supervisor clock moved backwards')
        self._last_event_at = float(now)

    def begin(self, *, position_key, legal_moves, go_received_at,
              fallback_ready_at, budget_ms=200):
        if not isinstance(position_key, str) or not position_key:
            raise ValueError('position_key required')
        if type(budget_ms) is not int or not 1 <= budget_ms <= 60000:
            raise ValueError('invalid budget')
        if not isinstance(go_received_at, (int, float)) or not math.isfinite(go_received_at):
            raise ValueError('invalid go timestamp')
        if not isinstance(fallback_ready_at, (int, float)) or not math.isfinite(fallback_ready_at):
            raise ValueError('invalid fallback timestamp')
        if fallback_ready_at < go_received_at:
            raise ValueError('fallback predates go')
        if go_received_at < self._last_event_at:
            raise ValueError('go timestamp predates previous event')
        moves = tuple(legal_moves)
        if any(not isinstance(m, str) or not m or m == 'resign' for m in moves):
            raise ValueError('invalid legal move snapshot')
        if len(moves) != len(set(moves)):
            raise ValueError('duplicate legal moves')
        if self._active and self._sent is None:
            raise RuntimeError('previous response not sent')
        self.generation += 1
        self._active = True
        self._legal = frozenset(moves)
        self._fallback = moves[0] if moves else 'resign'
        self._choice = self._fallback
        self._position_key = position_key
        self._go_at = float(go_received_at)
        self._ready_at = float(fallback_ready_at)
        self._deadline = self._go_at + budget_ms / 1000.0
        self._prepared = None
        self._sent = None
        self._worker_stopped = False
        self._last_event_at = self._ready_at
        return self.generation

    def offer(self, *, generation, position_key, move, received_at):
        """received_at is supervisor-observed, NEVER worker-reported."""
        self._clock(received_at)
        if not self._active or self._prepared is not None:
            return False
        if generation != self.generation or position_key != self._position_key:
            return False
        if received_at >= self._deadline or move not in self._legal:
            return False
        self._choice = move
        return True

    def prepare(self, *, at):
        """Prepare a move, not evidence that stdout was written."""
        self._clock(at)
        if not self._active:
            raise RuntimeError('no active go')
        if self._prepared is None:
            self._prepared = float(at)
        return self._choice

    def mark_sent(self, *, move, at):
        """Record externally confirmed send; performs no I/O."""
        self._clock(at)
        if self._prepared is None or self._sent is not None:
            raise RuntimeError('response must be prepared and sent once')
        if move != self._choice:
            raise ValueError('sent move differs from prepared choice')
        self._sent = float(at)
        return self.receipt()

    def mark_worker_stopped(self, *, generation, at):
        self._clock(at)
        if generation != self.generation or not self._active:
            return False
        self._worker_stopped = True
        return True

    def receipt(self):
        if self._sent is None:
            raise RuntimeError('send not confirmed')
        return ResponseReceipt(
            self.generation, self._choice, self._go_at, self._ready_at,
            self._prepared, self._sent, self._deadline,
            self._ready_at < self._deadline,
            self._sent <= self._deadline, self._worker_stopped)
