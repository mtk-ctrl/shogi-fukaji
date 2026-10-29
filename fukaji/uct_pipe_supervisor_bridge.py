"""Research-only post-receipt bridge: real child pipe -> synthetic supervisor -> private sink.

This is deliberately a SEQUENTIAL TRACE REPLAY, not a live deadline supervisor:
the fixed child has exited before the supervisor prepares/sends a response.
The sink is a private os.pipe(), never USI stdout or a GUI. No forced stops,
external commands, engine workers, adoption, or hard 200ms guarantee.
"""
from dataclasses import dataclass
import os
import time

from fukaji.uct_pipe_probe import run_pipe_probe
from fukaji.uct_synthetic_ipc import SyntheticIpcModel


@dataclass(frozen=True)
class BridgeProbeResult:
    mode: str
    candidate_from_pipe: str | None
    candidate_accepted_by_supervisor: bool
    prepared_move: str
    sent_confirmed: bool
    fake_sink_received: bool
    gui_delivery_verified: bool
    write_at_ms: float | None
    flush_at_ms: float | None
    fake_sink_read_at_ms: float | None
    response_sent_in_time: bool | None
    child_reaped: bool
    pipes_closed: bool
    worker_resources_released: bool
    outstanding_generations: tuple[int, ...]
    failed_sends: int
    stop_observed_by_pipe: bool


def run_pipe_supervisor_bridge(mode, *, budget_ms=2000, stop_after_ms=None,
                               simulate_sink_closed=False):
    """Replay a bounded pipe receipt through the model; write to private sink.

    The fallback is a trusted FIXTURE ('2g2f'), not a live board snapshot.
    The model's go/fallback-ready clock is set to 0 for trace replay; actual
    go-to-fallback preparation and live stop races are NOT measured.
    """
    if type(simulate_sink_closed) is not bool:
        raise ValueError('invalid sink fixture')
    started = time.monotonic_ns()
    legal = ('2g2f', '7g7f')
    model = SyntheticIpcModel()
    model.position(key='history-a', legal_moves=legal, at=0.0)
    generation = model.go(at=0.0, fallback_ready_at=0.0, budget_ms=budget_ms)
    observation = run_pipe_probe(
        mode, generation=generation, key='history-a',
        legal_moves=legal, budget_ms=budget_ms, stop_after_ms=stop_after_ms)
    accepted = False
    if observation.accepted_move is not None:
        if observation.accepted_at_ms is None:
            raise RuntimeError('accepted pipe candidate has no complete-line timestamp')
        accepted = model.worker_offer(
            generation=generation, key='history-a',
            move=observation.accepted_move,
            received_at=observation.accepted_at_ms / 1000.0)
    now = (time.monotonic_ns() - started) / 1e9
    model.stop(at=now)
    move = model.pending_move()
    payload = ('bestmove ' + move + '\n').encode('ascii')

    read_fd, write_fd = os.pipe()
    write_at = flush_at = read_at = None
    confirmed = sink_received = False
    receipt = None
    if simulate_sink_closed:
        os.close(read_fd)
        read_fd = None
    writer = os.fdopen(write_fd, 'wb', buffering=4096)
    try:
        try:
            writer.write(payload)
            write_at = (time.monotonic_ns() - started) / 1e6
            writer.flush()
            flush_at = (time.monotonic_ns() - started) / 1e6
            receipt = model.confirm_sent(move=move, at=flush_at / 1000.0)
            confirmed = True
            if read_fd is not None:
                received = os.read(read_fd, len(payload))
                read_at = (time.monotonic_ns() - started) / 1e6
                sink_received = received == payload
        except (BrokenPipeError, OSError):
            model.send_failed(at=(time.monotonic_ns() - started) / 1e9)
    finally:
        try:
            writer.close()
        except BrokenPipeError:
            pass
        if read_fd is not None:
            os.close(read_fd)

    # Child termination and resource release are not evidence of delivery.
    if observation.child_reaped:
        model.worker_stopped(generation=generation,
                             at=(time.monotonic_ns() - started) / 1e9)
    if observation.child_reaped and observation.pipes_closed:
        model.resources_released(generation=generation,
                                 at=(time.monotonic_ns() - started) / 1e9)
    return BridgeProbeResult(
        mode, observation.accepted_move, accepted, move, confirmed,
        sink_received, False, write_at, flush_at, read_at,
        receipt.response_sent_in_time if receipt else None,
        observation.child_reaped, observation.pipes_closed,
        not model.outstanding_generations(),
        model.outstanding_generations(), model.failed_sends,
        observation.stop_observed)
