"""Research-only two overlapping finite OS child fixtures; never USI/G8.

No shell, signals, forced termination, arbitrary commands, or hard deadline.
"""
from dataclasses import dataclass
import json
import os
import select
import subprocess
import sys
import time

from fukaji.uct_synthetic_ipc import SyntheticIpcModel


@dataclass(frozen=True)
class OsOverlapResult:
    old_pid: int
    new_pid: int
    simultaneous_alive: bool
    old_alive_when_new_go: bool
    old_offer_rejected: bool
    new_offer_accepted: bool
    observed_generations: tuple[int, ...]
    old_outstanding_at_new_go: tuple[int, ...]
    final_outstanding: tuple[int, ...]
    child_exit_codes: tuple[int | None, int | None]
    children_reaped: tuple[bool, bool]
    pipes_closed: tuple[bool, bool]
    private_responses_received: tuple[bool, bool]
    new_response_flush_ms: float
    new_planned_response_ms: int
    new_flush_lag_ms: float
    gui_delivery_verified: bool = False


def _finite_child(generation, key):
    record = json.dumps({'generation': generation, 'key': key,
                         'move': '7g7f'}, separators=(',', ':'))
    sys.stdout.write(record + '\n')
    sys.stdout.flush()
    ready, _, _ = select.select([sys.stdin], [], [], 1.2)
    if ready:
        sys.stdin.readline()
    time.sleep(0.03)
    return 0


def _private_send(move):
    """Private pipe write/flush/read, NOT formal stdout or GUI delivery."""
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(write_fd, 'wb') as writer:
            writer.write(('bestmove ' + move + '\n').encode('ascii'))
            writer.flush()
            flushed_ns = time.monotonic_ns()
        observed = os.read(read_fd, 128)
        return observed == ('bestmove ' + move + '\n').encode('ascii'), flushed_ns
    finally:
        os.close(read_fd)


def run_os_overlap_probe(*, budget_ms=400, planned_ms=150):
    """Observe two finite OS children, old candidate rejection, and cleanup.

    Real PID/poll/wait/pipe close are distinct from the synthetic ledger.
    Children wait at most 1.2s for cooperative stop and then naturally exit.
    """
    if type(budget_ms) is not int or not 1 <= budget_ms <= 1000:
        raise ValueError('invalid bounded research budget')
    if type(planned_ms) is not int or not 0 <= planned_ms <= budget_ms:
        raise ValueError('invalid planned response')
    model = SyntheticIpcModel()
    key = 'trusted-fixture-position-history'
    moves = ('7g7f', '2g2f')
    now = lambda: time.monotonic_ns() / 1e9
    processes = []
    reaped = [False, False]
    exit_codes = [None, None]
    closed = [False, False]
    old_gen = new_gen = None
    try:
        model.position(key=key, legal_moves=moves, at=now())
        old_gen = model.go(at=now(), fallback_ready_at=now(), budget_ms=budget_ms)
        old = subprocess.Popen([sys.executable, '-u', '-m',
                                'fukaji.uct_os_overlap_probe', '--child',
                                str(old_gen), key], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               close_fds=True, bufsize=0)
        processes.append(old)
        model.stop(at=now())
        old_received, _ = _private_send(model.pending_move())
        if not old_received:
            raise RuntimeError('private old response not received')
        model.confirm_sent(move=model.pending_move(), at=now())
        old_outstanding = model.outstanding_generations()
        old_alive = old.poll() is None

        model.position(key=key, legal_moves=moves, at=now())
        new_go_ns = time.monotonic_ns()
        new_gen = model.go(at=now(), fallback_ready_at=now(), budget_ms=budget_ms)
        new = subprocess.Popen([sys.executable, '-u', '-m',
                                'fukaji.uct_os_overlap_probe', '--child',
                                str(new_gen), key], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               close_fds=True, bufsize=0)
        processes.append(new)
        simultaneous = old.poll() is None and new.poll() is None
        observed = []
        old_rejected = new_accepted = False
        buffers = {old.stdout.fileno(): b'', new.stdout.fileno(): b''}
        owners = {old.stdout.fileno(): old_gen, new.stdout.fileno(): new_gen}
        deadline_ns = new_go_ns + planned_ms * 1_000_000
        while time.monotonic_ns() < deadline_ns and len(observed) < 2:
            remaining = (deadline_ns - time.monotonic_ns()) / 1e9
            ready, _, _ = select.select(list(buffers), [], [],
                                        min(0.01, max(0, remaining)))
            for fd in ready:
                chunk = os.read(fd, 1024)
                if not chunk:
                    continue
                buffers[fd] += chunk
                if len(buffers[fd]) > 4096:
                    raise ValueError('fixed child response exceeded bound')
                if b'\n' in buffers[fd]:
                    line, buffers[fd] = buffers[fd].split(b'\n', 1)
                    msg = json.loads(line)
                    accepted = model.worker_offer(
                        generation=msg['generation'], key=msg['key'],
                        move=msg['move'], received_at=now())
                    observed.append(owners[fd])
                    if owners[fd] == old_gen:
                        old_rejected = not accepted
                    else:
                        new_accepted = accepted
        while time.monotonic_ns() < deadline_ns:
            time.sleep(min(0.002, (deadline_ns - time.monotonic_ns()) / 1e9))
        model.stop(at=now())
        new_received, new_flush_ns = _private_send(model.pending_move())
        if not new_received:
            raise RuntimeError('private new response not received')
        model.confirm_sent(move=model.pending_move(), at=now())
        flush_ms = (new_flush_ns - new_go_ns) / 1e6
    finally:
        # Only the two exact Popen handles created here; never signal/kill.
        for process in processes:
            try:
                process.stdin.write(b'stop\n')
                process.stdin.flush()
            except (OSError, BrokenPipeError):
                pass
        for i, process in enumerate(processes):
            try:
                exit_codes[i] = process.wait(timeout=2.0)
                reaped[i] = True
            except subprocess.TimeoutExpired:
                pass
            process.stdin.close()
            process.stdout.close()
            closed[i] = process.stdin.closed and process.stdout.closed
            if reaped[i]:
                gen = old_gen if i == 0 else new_gen
                model.worker_stopped(generation=gen, at=now())
                if closed[i]:
                    model.resources_released(generation=gen, at=now())
    return OsOverlapResult(old.pid, new.pid, simultaneous, old_alive,
                           old_rejected, new_accepted, tuple(observed),
                           old_outstanding, model.outstanding_generations(),
                           tuple(exit_codes), tuple(reaped), tuple(closed),
                           (old_received, new_received), flush_ms, planned_ms,
                           flush_ms - planned_ms)


def _main(argv):
    if len(argv) != 4 or argv[1] != '--child':
        return 2
    generation = int(argv[2])
    if generation < 1 or argv[3] != 'trusted-fixture-position-history':
        return 2
    return _finite_child(generation, argv[3])


if __name__ == '__main__':
    sys.exit(_main(sys.argv))
