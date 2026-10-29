"""Research-only live finite-child / legal-fallback / private-pipe probe.

NOT connected to official USI, GUI, G8, training, or an engine. No shell,
external command, forced termination, signals, or hard real-time guarantee.
All children are fixed finite fixtures; child-reap is separate from response.
"""
from dataclasses import dataclass
import json
import os
import select
import subprocess
import sys
import time

from fukaji.uct_legal_snapshot import snapshot_from_board
from fukaji.uct_synthetic_ipc import SyntheticIpcModel

_MODES = frozenset(('normal', 'delayed', 'fragmented', 'stale', 'wrong_key',
                    'malformed', 'silent', 'abrupt', 'linger'))
_MAX_LINE = 4096


@dataclass(frozen=True)
class LiveProbeResult:
    mode: str
    fallback_move: str
    candidate_accepted: bool
    selected_move: str
    fallback_ready_at_ms: float
    response_flush_at_ms: float | None
    private_read_at_ms: float | None
    response_sent_in_time: bool | None
    sent_confirmed: bool
    private_sink_received: bool
    gui_delivery_verified: bool
    child_alive_at_response: bool
    child_reaped: bool
    child_exit_code: int | None
    pipes_closed: bool
    stop_request_at_ms: float | None
    worker_resources_released: bool
    outstanding_generations: tuple[int, ...]
    failed_sends: int
    rejected_messages: int
    board_unchanged: bool


def _fixed_child(mode, generation, key):
    """Finite test child: never runs a search or arbitrary commands."""
    if mode == 'abrupt':
        os._exit(7)
    if mode == 'delayed':
        time.sleep(0.12)
    if mode not in ('silent', 'linger'):
        record = {'generation': generation - (mode == 'stale'),
                  'key': key + '-old' if mode == 'wrong_key' else key,
                  'move': '7g7f'}
        raw = (json.dumps(record, separators=(',', ':')) + '\n').encode('ascii')
        if mode == 'malformed':
            raw = b'{broken}\n'
        if mode == 'fragmented':
            sys.stdout.buffer.write(raw[:8]); sys.stdout.buffer.flush()
            time.sleep(0.025)
            sys.stdout.buffer.write(raw[8:]); sys.stdout.buffer.flush()
        else:
            sys.stdout.buffer.write(raw); sys.stdout.buffer.flush()
    # A fixed upper bound prevents unattended research fixtures lingering.
    ready, _, _ = select.select([sys.stdin], [], [], 0.8)
    if ready:
        sys.stdin.readline()  # cooperative stop; not a signal or forced kill
    if mode == 'linger':
        time.sleep(0.10)
    return 0


def run_live_supervisor_probe(mode, *, board, budget_ms=400,
                              respond_after_ms=200, stop_after_ms=None,
                              simulate_sink_closed=False):
    """Supervise a LIVE fixed child and send one bestmove to a PRIVATE pipe.

    All timings use the parent's monotonic clock. go is stamped before the
    real legal snapshot is built. Flush is not GUI receipt. A normal OS/Python
    scheduler can delay the parent beyond the budget: never claim a hard bound.
    """
    if mode not in _MODES:
        raise ValueError('unrecognized finite fixture')
    if type(budget_ms) is not int or not 1 <= budget_ms <= 5000:
        raise ValueError('invalid budget')
    if type(respond_after_ms) is not int or not 0 <= respond_after_ms <= budget_ms:
        raise ValueError('invalid response threshold')
    if stop_after_ms is not None and (type(stop_after_ms) is not int or stop_after_ms < 0):
        raise ValueError('invalid stop threshold')
    if type(simulate_sink_closed) is not bool:
        raise ValueError('invalid private sink fixture')
    start_ns = time.monotonic_ns()
    now_ms = lambda: (time.monotonic_ns() - start_ns) / 1e6
    before_sfen = board.sfen()
    before_moves = tuple(m.usi() for m in board.move_stack)
    model = SyntheticIpcModel()
    snapshot = snapshot_from_board(board)
    ready_ms = now_ms()
    model.position(key=snapshot.position_key, legal_moves=snapshot.legal_moves, at=0.0)
    generation = model.go(at=0.0, fallback_ready_at=ready_ms / 1000.0,
                          budget_ms=budget_ms)
    fallback = snapshot.legal_moves[0] if snapshot.legal_moves else 'resign'
    process = subprocess.Popen(
        [sys.executable, '-u', '-m', 'fukaji.uct_live_supervisor_probe',
         '--child', mode, str(generation), snapshot.position_key],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        close_fds=True, bufsize=0)
    buffer = b''
    accepted = False
    rejected = 0
    sent = received = False
    flush_ms = read_ms = stop_ms = None
    sent_in_time = None
    alive_at_response = False
    reaped = False
    exit_code = None
    pipes_closed = False
    read_fd = write_fd = None
    writer = None
    try:
        child_fd = process.stdout.fileno()
        # Respond at a predetermined time or earlier on explicit stop; do not
        # wait for the child to exit before choosing a legal fallback.
        threshold = min(respond_after_ms, stop_after_ms) if stop_after_ms is not None else respond_after_ms
        while now_ms() < threshold:
            remaining = max(0.0, (threshold - now_ms()) / 1000.0)
            ready, _, _ = select.select([child_fd], [], [], min(0.005, remaining))
            if not ready:
                continue
            chunk = os.read(child_fd, 1024)
            at_ms = now_ms()  # receipt time is parent-owned, after read
            if not chunk:
                break
            buffer += chunk
            if len(buffer) > _MAX_LINE:
                raise ValueError('fixed child payload exceeded limit')
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                try:
                    obj = json.loads(line.decode('utf-8'))
                    valid = (type(obj) is dict and set(obj) == {'generation', 'key', 'move'}
                             and type(obj['generation']) is int
                             and type(obj['key']) is str
                             and type(obj['move']) is str)
                except (UnicodeError, ValueError, TypeError):
                    valid = False
                # A stop/response threshold may expire while select/read blocks.
                # Recheck the parent receipt time before accepting the offer.
                if valid and at_ms < threshold:
                    accepted_now = model.worker_offer(
                        generation=obj['generation'], key=obj['key'],
                        move=obj['move'], received_at=at_ms / 1000.0)
                    accepted |= accepted_now
                    rejected += int(not accepted_now)
                else:
                    rejected += 1
        model.stop(at=now_ms() / 1000.0)
        move = model.pending_move()
        alive_at_response = process.poll() is None
        payload = ('bestmove ' + move + '\n').encode('ascii')
        read_fd, write_fd = os.pipe()
        if simulate_sink_closed:
            os.close(read_fd)
            read_fd = None
        writer = os.fdopen(write_fd, 'wb', buffering=4096)
        write_fd = None
        try:
            writer.write(payload)
            writer.flush()
            flush_ms = now_ms()
            receipt = model.confirm_sent(move=move, at=flush_ms / 1000.0)
            sent = True
            sent_in_time = receipt.response_sent_in_time
            if read_fd is not None:
                data = os.read(read_fd, len(payload))
                read_ms = now_ms()
                received = data == payload
        except (BrokenPipeError, OSError):
            model.send_failed(at=now_ms() / 1000.0)
        # Cooperative stop is sent AFTER the bestmove attempt. This does not
        # terminate or signal any unrelated process or GitHub Actions run.
        stop_ms = now_ms()
        try:
            process.stdin.write(b'stop\n')
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            exit_code = process.wait(timeout=1.0)
            reaped = True
        except subprocess.TimeoutExpired:
            pass  # Never forcibly terminate; report outstanding worker.
    finally:
        if writer is not None:
            try:
                writer.close()
            except (BrokenPipeError, OSError):
                pass
        elif write_fd is not None:
            os.close(write_fd)
        if read_fd is not None:
            os.close(read_fd)
        process.stdin.close()
        process.stdout.close()
        pipes_closed = process.stdin.closed and process.stdout.closed
        if not reaped:
            try:
                exit_code = process.wait(timeout=1.0)
                reaped = True
            except subprocess.TimeoutExpired:
                pass
        if reaped:
            model.worker_stopped(generation=generation, at=now_ms() / 1000.0)
            if pipes_closed:
                model.resources_released(generation=generation, at=now_ms() / 1000.0)
    return LiveProbeResult(
        mode, fallback, accepted, move, ready_ms, flush_ms, read_ms,
        sent_in_time, sent, received, False, alive_at_response, reaped,
        exit_code, pipes_closed, stop_ms, not model.outstanding_generations(),
        model.outstanding_generations(), model.failed_sends, rejected,
        board.sfen() == before_sfen and
        tuple(m.usi() for m in board.move_stack) == before_moves)


def _main(argv):
    if len(argv) != 5 or argv[1] != '--child' or argv[2] not in _MODES:
        return 2
    return _fixed_child(argv[2], int(argv[3]), argv[4])


if __name__ == '__main__':
    sys.exit(_main(sys.argv))
