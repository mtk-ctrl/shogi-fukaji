"""Research-only deterministic USI/IPC event model. No real I/O or workers.

Caller-supplied timestamps are supervisor-observed monotonic times, not worker
claims. This module does not enforce a wall-clock deadline or stop a process.
"""
import math

from fukaji.uct_response_contract import ResponseSupervisorModel


class SyntheticIpcModel:
    """Track position/go/stop/bestmove and worker-resource lifetimes separately."""

    def __init__(self):
        self.supervisor = ResponseSupervisorModel()
        self._last_at = float('-inf')
        self._position = None
        self._active_generation = None
        self._pending_move = None
        self._sent = True
        self._workers = {}
        self.failed_sends = 0

    def _stamp(self, at):
        if isinstance(at, bool) or not isinstance(at, (int, float)) or not math.isfinite(at):
            raise ValueError('finite event timestamp required')
        if at < self._last_at:
            raise ValueError('event clock moved backwards')
        self._last_at = float(at)

    def position(self, *, key, legal_moves, at):
        self._stamp(at)
        if not self._sent:
            raise RuntimeError('cannot replace position before bestmove confirmation')
        if not isinstance(key, str) or not key:
            raise ValueError('position key required')
        moves = tuple(legal_moves)
        if any(not isinstance(m, str) or not m or m == 'resign' for m in moves):
            raise ValueError('invalid legal move snapshot')
        if len(set(moves)) != len(moves):
            raise ValueError('duplicate legal moves')
        self._position = (key, moves)

    def go(self, *, at, fallback_ready_at, budget_ms=200):
        self._stamp(at)
        if self._position is None:
            raise RuntimeError('position required before go')
        if not self._sent:
            raise RuntimeError('previous bestmove not confirmed')
        self._stamp(fallback_ready_at)
        key, moves = self._position
        generation = self.supervisor.begin(
            position_key=key, legal_moves=moves,
            go_received_at=at, fallback_ready_at=fallback_ready_at,
            budget_ms=budget_ms)
        self._active_generation = generation
        self._pending_move = None
        self._sent = False
        self._workers[generation] = {'stopped': False, 'released': False}
        return generation

    def worker_offer(self, *, generation, key, move, received_at):
        self._stamp(received_at)
        if self._sent or self._active_generation is None:
            return False
        return self.supervisor.offer(
            generation=generation, position_key=key, move=move,
            received_at=received_at)

    def stop(self, *, at):
        """Prepare exactly one bestmove; does not send bytes or stop a worker."""
        self._stamp(at)
        if self._sent or self._active_generation is None or self._pending_move is not None:
            return False
        self._pending_move = self.supervisor.prepare(at=at)
        return True

    def deadline(self, *, at):
        self._stamp(at)
        if self._active_generation is None or self._sent:
            return False
        if at < self.supervisor._deadline:
            raise ValueError('deadline not yet reached')
        return self.stop(at=at)

    def pending_move(self):
        return self._pending_move

    def send_failed(self, *, at):
        """Record transport failure without claiming a bestmove was sent."""
        self._stamp(at)
        if self._pending_move is None or self._sent:
            raise RuntimeError('no pending bestmove')
        self.failed_sends += 1

    def confirm_sent(self, *, move, at):
        """External caller confirms its write; not proof of GUI receipt."""
        self._stamp(at)
        if self._pending_move is None or self._sent:
            raise RuntimeError('no pending bestmove')
        if move != self._pending_move:
            raise ValueError('confirmed move differs from prepared move')
        receipt = self.supervisor.mark_sent(move=move, at=at)
        self._sent = True
        return receipt

    def worker_stopped(self, *, generation, at):
        self._stamp(at)
        state = self._workers.get(generation)
        if state is None or state['stopped']:
            return False
        state['stopped'] = True
        if generation == self._active_generation:
            self.supervisor.mark_worker_stopped(generation=generation, at=at)
        return True

    def resources_released(self, *, generation, at):
        self._stamp(at)
        state = self._workers.get(generation)
        if state is None or not state['stopped'] or state['released']:
            return False
        state['released'] = True
        return True

    def outstanding_generations(self):
        """Includes already-stopped workers with unconfirmed resource cleanup."""
        return tuple(sorted(g for g, state in self._workers.items()
                            if not state['released']))
