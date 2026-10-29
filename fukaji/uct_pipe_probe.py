"""Research-only bounded child-process pipe probe; never wired into USI/G8.

The parent owns monotonic receipt times. A pipe read is NOT GUI delivery, and
Popen/pipe cleanup is NOT proof of a hard 200ms deadline or zero OS leaks.
No shell, no user-provided command, no signals, no forced termination.
"""
from dataclasses import dataclass
import json
import os
import select
import subprocess
import sys
import time

_MODES = frozenset(('normal', 'delayed', 'fragmented', 'stale', 'malformed',
                    'truncated', 'duplicate', 'abrupt', 'wrong_key'))
_MAX_BYTES = 4096


@dataclass(frozen=True)
class PipeProbeResult:
    mode: str
    accepted_move: str | None
    fallback_move: str
    lines_observed: int
    rejected_messages: int
    incomplete_bytes: int
    stop_observed: bool
    child_exit_code: int | None
    child_reaped: bool
    pipes_closed: bool
    first_byte_read_ms: float | None
    elapsed_ms: float
    gui_delivery_verified: bool = False


def _worker(mode, generation, key):
    """Finite, trusted test fixture; never invokes an engine or receives code."""
    if mode == 'abrupt':
        os._exit(7)
    if mode == 'delayed':
        time.sleep(0.075)
    record = {'generation': generation - (1 if mode == 'stale' else 0),
              'key': key + '-old' if mode == 'wrong_key' else key, 'move': '7g7f'}
    raw = (json.dumps(record, separators=(',', ':')) + '\n').encode('ascii')
    if mode == 'malformed':
        raw = b'{invalid json}\n'
    if mode == 'truncated':
        raw = raw[:9]
    if mode == 'fragmented':
        sys.stdout.buffer.write(raw[:8]); sys.stdout.buffer.flush()
        time.sleep(0.025)
        sys.stdout.buffer.write(raw[8:]); sys.stdout.buffer.flush()
    else:
        sys.stdout.buffer.write(raw); sys.stdout.buffer.flush()
    if mode == 'duplicate':
        sys.stdout.buffer.write(raw); sys.stdout.buffer.flush()
    return 0


def run_pipe_probe(mode, *, generation=1, key='history-a',
                   legal_moves=('7g7f', '2g2f'), budget_ms=2000,
                   stop_after_ms=None, timeout_s=3.0):
    """Observe one bounded real pipe; do not send USI or stop external runs.

    A synthetic stop is an observation threshold only, NOT a process signal.
    On timeout the probe reports a cleanup error rather than silently claiming
    worker termination. All fixture child modes have finite runtimes.
    """
    if mode not in _MODES:
        raise ValueError('unrecognized fixed worker fixture')
    if type(generation) is not int or generation <= 0:
        raise ValueError('invalid generation')
    if not isinstance(key, str) or not key or len(key) > 128:
        raise ValueError('invalid position key')
    if type(budget_ms) is not int or not 1 <= budget_ms <= 60000:
        raise ValueError('invalid budget')
    if stop_after_ms is not None and (type(stop_after_ms) is not int or stop_after_ms < 0):
        raise ValueError('invalid stop threshold')
    if not isinstance(timeout_s, (int, float)) or not 1 <= timeout_s <= 10:
        raise ValueError('invalid bounded wait')
    moves = tuple(legal_moves)
    if not moves or any(not isinstance(m, str) or not m or m == 'resign' for m in moves):
        raise ValueError('fixture requires validated legal fallback')
    if len(set(moves)) != len(moves):
        raise ValueError('duplicate legal moves')
    started = time.monotonic_ns()
    process = subprocess.Popen(
        [sys.executable, '-u', '-m', 'fukaji.uct_pipe_probe',
         '--child', mode, str(generation), key],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, close_fds=True, bufsize=0)
    buffer = b''
    accepted = None
    count = rejected = 0
    first_read_ms = None
    stopped = False
    exit_code = None
    reaped = False
    try:
        fd = process.stdout.fileno()
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            elapsed_ms = (time.monotonic_ns() - started) / 1e6
            if stop_after_ms is not None and elapsed_ms >= stop_after_ms:
                stopped = True
            ready, _, _ = select.select([fd], [], [], min(0.01, max(0.0, end-time.monotonic())))
            if not ready:
                continue
            chunk = os.read(fd, 1024)
            read_ms = (time.monotonic_ns() - started) / 1e6
            # Recheck after select/read: stop can expire while blocked in select.
            if stop_after_ms is not None and read_ms >= stop_after_ms:
                stopped = True
            if not chunk:
                break
            if first_read_ms is None:
                first_read_ms = read_ms
            buffer += chunk
            if len(buffer) > _MAX_BYTES:
                raise ValueError('bounded pipe payload exceeded')
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                count += 1
                try:
                    obj = json.loads(line.decode('utf-8'))
                    valid = (isinstance(obj, dict) and set(obj) == {'generation', 'key', 'move'}
                             and type(obj['generation']) is int
                             and obj['generation'] == generation
                             and obj['key'] == key
                             and type(obj['move']) is str
                             and obj['move'] in moves)
                except (UnicodeError, ValueError, TypeError, KeyError):
                    valid = False
                if valid and accepted is None and not stopped and read_ms < budget_ms:
                    accepted = obj['move']
                else:
                    rejected += 1
        else:
            raise TimeoutError('bounded child pipe did not reach EOF')
        exit_code = process.wait(timeout=1.0)
        reaped = True
    finally:
        process.stdout.close()
        if not reaped:
            try:
                exit_code = process.wait(timeout=1.0)
                reaped = True
            except subprocess.TimeoutExpired:
                # No signal/forced termination in this research probe.
                pass
    return PipeProbeResult(mode, accepted, moves[0], count, rejected,
                           len(buffer), stopped, exit_code, reaped,
                           process.stdout.closed,
                           first_read_ms,
                           (time.monotonic_ns()-started)/1e6)


def _main(argv):
    if len(argv) != 5 or argv[1] != '--child' or argv[2] not in _MODES:
        return 2
    return _worker(argv[2], int(argv[3]), argv[4])


if __name__ == '__main__':
    sys.exit(_main(sys.argv))
