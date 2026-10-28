"""Research-only UCT Monte Carlo tree search with the existing G8 value model.

No policy network, no stochastic rollout, and no adoption. Terminal checkmate
is exact; nonterminal leaves use a bounded value heuristic. Repetition is a
search penalty, not a labeled draw. Uses python-shogi for legal move rules.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from .engine import evaluate


def _legal_moves_once(board, *, single_pass=False):
    """Materialize legal moves without changing order; opt-in avoids __len__.

    python-shogi's LegalMoveGenerator.__len__ enumerates legal moves. The
    built-in list() requests that length before iterating, so the default
    path can enumerate twice. A comprehension requests only the iterator.
    """
    if single_pass:
        return [move for move in board.legal_moves]
    return list(board.legal_moves)


@dataclass
class Node:
    move: object = None
    visits: int = 0
    value_sum: float = 0.0  # value from the side to move at this node
    children: list = field(default_factory=list)
    expanded: bool = False


def _uct_child(node, exploration):
    """Select from parent's perspective; a child is the opponent's turn."""
    log_parent = math.log(max(2, node.visits))
    return max(
        node.children,
        key=lambda child: (
            float('inf') if child.visits == 0 else
            -child.value_sum / child.visits
            + exploration * math.sqrt(log_parent / child.visits)
        ),
    )


def _best_root_child(root, root_moves, *, linear=False):
    """Choose highest visits, breaking ties by original legal-move order.

    The opt-in path uses the invariant that root.children was created in
    root_moves order and is never reordered. It avoids repeated list.index.
    """
    if linear:
        return max(enumerate(root.children),
                   key=lambda pair: (pair[1].visits, -pair[0]))[1]
    return max(root.children,
               key=lambda child: (child.visits, -root_moves.index(child.move)))


def select_mcts(board, movetime_ms=200, model=None, *, fast_relative=False,
                max_iterations=None, max_tree_depth=12, exploration=1.4,
                seed=0, iteration_trace=None, single_pass_legal_moves=False,
                total_budget_from_call=False, clock_phase_trace=None,
                linear_root_choice=False):
    """Return (legal USI move, iterations, elapsed milliseconds).

    max_iterations can fix the sample budget for reproducible research. When
    supplied, it is still bounded by the wall-clock deadline. The caller
    must record actual iterations and elapsed time, not assume equal nodes.

    Optional iteration_trace is an append-only list of phase timings. It is
    diagnostic instrumentation: extra clock reads and recording change timing,
    so traced runs must NOT be compared directly with untraced 200ms runs.

    total_budget_from_call is opt-in and includes validation, root legal move
    generation, and root setup in the same best-effort wall-clock budget.
    It cannot preempt a long iteration or guarantee a hard deadline.

    linear_root_choice is a research-only opt-in that replaces the quadratic
    root final-choice tie-break with a linear scan in identical move order.
    It does not change the deadline or guarantee 200ms compliance.

    clock_phase_trace is opt-in diagnostic instrumentation, not a time
    compliance measurement. It records root setup, iterations, restoration,
    final move choice and the internal return clock boundary. Its own extra
    clock reads and append overhead change caller-observed elapsed time.
    """
    # Research-only opt-in: start before validation and root move generation.
    # The default clock boundary and return values remain unchanged.
    call_started = time.monotonic() if total_budget_from_call is True else None
    phase_call_started = time.monotonic() if clock_phase_trace is not None else None
    if not (1 <= movetime_ms <= 60000):
        raise ValueError('movetime_ms outside 1..60000')
    if max_iterations is not None and (type(max_iterations) is not int or max_iterations < 1):
        raise ValueError('max_iterations must be a positive integer')
    if type(max_tree_depth) is not int or max_tree_depth < 1:
        raise ValueError('max_tree_depth must be a positive integer')
    if not math.isfinite(exploration) or exploration < 0:
        raise ValueError('invalid exploration constant')
    if iteration_trace is not None and not hasattr(iteration_trace, 'append'):
        raise ValueError('iteration_trace must support append')
    if type(single_pass_legal_moves) is not bool:
        raise ValueError('single_pass_legal_moves must be bool')
    if type(total_budget_from_call) is not bool:
        raise ValueError('total_budget_from_call must be bool')
    if clock_phase_trace is not None and not hasattr(clock_phase_trace, 'append'):
        raise ValueError('clock_phase_trace must support append')
    if type(linear_root_choice) is not bool:
        raise ValueError('linear_root_choice must be bool')

    root_moves = _legal_moves_once(board, single_pass=single_pass_legal_moves)
    if not root_moves:
        elapsed = ((time.monotonic() - call_started) * 1000.0
                   if call_started is not None else 0.0)
        if clock_phase_trace is not None:
            clock_phase_trace.append({
                'terminal_no_legal_moves': True,
                'root_setup_ms': (time.monotonic() - phase_call_started) * 1000.0,
                'root_legal_moves': 0, 'iterations': 0,
            })
        return 'resign', 0, elapsed
    root_sfen = board.sfen()
    root_stack = len(board.move_stack)
    root = Node(expanded=True, children=[Node(move=m) for m in root_moves])
    rng = random.Random(seed)
    start = call_started if call_started is not None else time.monotonic()
    deadline = start + movetime_ms / 1000.0
    root_ready = time.monotonic() if clock_phase_trace is not None else None
    iterations = 0
    longest_iteration_seconds = 0.0
    try:
        while (time.monotonic() < deadline and
               (max_iterations is None or iterations < max_iterations)):
            iteration_started = time.monotonic()
            phase_started = iteration_started
            phases = {} if iteration_trace is not None else None
            node = root
            path = [root]
            pushed = 0
            try:
                while node.expanded and node.children and pushed < max_tree_depth:
                    unexplored = [child for child in node.children if child.visits == 0]
                    child = rng.choice(unexplored) if unexplored else _uct_child(node, exploration)
                    board.push(child.move)
                    pushed += 1
                    node = child
                    path.append(node)
                if phases is not None:
                    now = time.monotonic()
                    phases['selection_ms'] = (now - phase_started) * 1000.0
                    phase_started = now
                # Optional research timing: subdivide the composite expansion.
                # Keep the short-circuit order and the untraced search identical.
                if phases is not None:
                    for name in ('checkmate_ms', 'repetition_ms',
                                 'legal_moves_ms', 'child_nodes_ms', 'push_ms'):
                        phases[name] = 0.0
                if pushed < max_tree_depth and not node.expanded:
                    clock = time.monotonic() if phases is not None else None
                    is_mate = board.is_checkmate()
                    if phases is not None:
                        phases['checkmate_ms'] = (time.monotonic() - clock) * 1000.0
                    if not is_mate:
                        clock = time.monotonic() if phases is not None else None
                        is_repetition = board.is_fourfold_repetition()
                        if phases is not None:
                            phases['repetition_ms'] = (time.monotonic() - clock) * 1000.0
                        if not is_repetition:
                            clock = time.monotonic() if phases is not None else None
                            moves = _legal_moves_once(board, single_pass=single_pass_legal_moves)
                            if phases is not None:
                                phases['legal_moves_ms'] = (time.monotonic() - clock) * 1000.0
                                clock = time.monotonic()
                            node.children = [Node(move=m) for m in moves]
                            node.expanded = True
                            if node.children:
                                child = rng.choice(node.children)
                            if phases is not None:
                                phases['child_nodes_ms'] = (time.monotonic() - clock) * 1000.0
                            if node.children:
                                clock = time.monotonic() if phases is not None else None
                                board.push(child.move)
                                pushed += 1
                                node = child
                                path.append(node)
                                if phases is not None:
                                    phases['push_ms'] = (time.monotonic() - clock) * 1000.0
                if phases is not None:
                    now = time.monotonic()
                    phases['expansion_ms'] = (now - phase_started) * 1000.0
                    phase_started = now
                if board.is_checkmate():
                    value = -1.0
                elif board.is_fourfold_repetition():
                    # Heuristic discouragement only; NOT a result label.
                    value = -0.5
                else:
                    value = math.tanh(evaluate(board, model, fast_relative) / 600.0)
                if phases is not None:
                    now = time.monotonic()
                    phases['leaf_ms'] = (now - phase_started) * 1000.0
                    phase_started = now
                for visited in reversed(path):
                    visited.visits += 1
                    visited.value_sum += value
                    value = -value
                iterations += 1
                if phases is not None:
                    now = time.monotonic()
                    phases['backup_ms'] = (now - phase_started) * 1000.0
                    phase_started = now
            finally:
                for _ in range(pushed):
                    board.pop()
                if phases is not None:
                    phases['undo_ms'] = (time.monotonic() - phase_started) * 1000.0
            # An iteration cannot be interrupted safely while a board is pushed.
            # Reserve time based on observed full-iteration costs instead of
            # beginning another iteration just before the deadline. This is a
            # best-effort guard, not a hard real-time guarantee.
            iteration_seconds = time.monotonic() - iteration_started
            longest_iteration_seconds = max(
                longest_iteration_seconds, iteration_seconds)
            reserve = max(0.003, 1.5 * longest_iteration_seconds)
            if phases is not None:
                iteration_trace.append({
                    'iteration': iterations, 'depth': pushed,
                    'root_legal_moves': len(root_moves),
                    'total_ms': iteration_seconds * 1000.0,
                    'reserve_ms': reserve * 1000.0,
                    'remaining_ms': (deadline - time.monotonic()) * 1000.0,
                    **phases,
                })
            if deadline - time.monotonic() <= reserve:
                break
        loop_finished = time.monotonic() if clock_phase_trace is not None else None
    finally:
        if board.sfen() != root_sfen or len(board.move_stack) != root_stack:
            raise AssertionError('MCTS corrupted caller board')
    verified = time.monotonic() if clock_phase_trace is not None else None
    # Child visits measure how often each root move was examined.
    best = _best_root_child(root, root_moves, linear=linear_root_choice)
    choice_finished = time.monotonic() if clock_phase_trace is not None else None
    elapsed = (time.monotonic() - start) * 1000.0
    if clock_phase_trace is not None:
        return_clock_finished = time.monotonic()
        clock_phase_trace.append({
            'terminal_no_legal_moves': False,
            'root_legal_moves': len(root_moves), 'iterations': iterations,
            'total_budget_from_call': total_budget_from_call,
            'root_setup_ms': (root_ready - phase_call_started) * 1000.0,
            'search_loop_ms': (loop_finished - root_ready) * 1000.0,
            'restore_verify_ms': (verified - loop_finished) * 1000.0,
            'root_choice_ms': (choice_finished - verified) * 1000.0,
            'return_clock_ms': (return_clock_finished - choice_finished) * 1000.0,
            'internal_elapsed_ms': elapsed,
        })
    return best.move.usi(), iterations, elapsed
