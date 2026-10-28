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


def select_mcts(board, movetime_ms=200, model=None, *, fast_relative=False,
                max_iterations=None, max_tree_depth=12, exploration=1.4,
                seed=0):
    """Return (legal USI move, iterations, elapsed milliseconds).

    max_iterations can fix the sample budget for reproducible research. When
    supplied, it is still bounded by the wall-clock deadline. The caller
    must record actual iterations and elapsed time, not assume equal nodes.
    """
    if not (1 <= movetime_ms <= 60000):
        raise ValueError('movetime_ms outside 1..60000')
    if max_iterations is not None and (type(max_iterations) is not int or max_iterations < 1):
        raise ValueError('max_iterations must be a positive integer')
    if type(max_tree_depth) is not int or max_tree_depth < 1:
        raise ValueError('max_tree_depth must be a positive integer')
    if not math.isfinite(exploration) or exploration < 0:
        raise ValueError('invalid exploration constant')

    root_moves = list(board.legal_moves)
    if not root_moves:
        return 'resign', 0, 0.0
    root_sfen = board.sfen()
    root_stack = len(board.move_stack)
    root = Node(expanded=True, children=[Node(move=m) for m in root_moves])
    rng = random.Random(seed)
    start = time.monotonic()
    deadline = start + movetime_ms / 1000.0
    iterations = 0
    longest_iteration_seconds = 0.0
    try:
        while (time.monotonic() < deadline and
               (max_iterations is None or iterations < max_iterations)):
            iteration_started = time.monotonic()
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
                if (pushed < max_tree_depth and not node.expanded
                        and not board.is_checkmate()
                        and not board.is_fourfold_repetition()):
                    moves = list(board.legal_moves)
                    node.children = [Node(move=m) for m in moves]
                    node.expanded = True
                    if node.children:
                        child = rng.choice(node.children)
                        board.push(child.move)
                        pushed += 1
                        node = child
                        path.append(node)
                if board.is_checkmate():
                    value = -1.0
                elif board.is_fourfold_repetition():
                    # Heuristic discouragement only; NOT a result label.
                    value = -0.5
                else:
                    value = math.tanh(evaluate(board, model, fast_relative) / 600.0)
                for visited in reversed(path):
                    visited.visits += 1
                    visited.value_sum += value
                    value = -value
                iterations += 1
            finally:
                for _ in range(pushed):
                    board.pop()
            # An iteration cannot be interrupted safely while a board is pushed.
            # Reserve time based on observed full-iteration costs instead of
            # beginning another iteration just before the deadline. This is a
            # best-effort guard, not a hard real-time guarantee.
            longest_iteration_seconds = max(
                longest_iteration_seconds, time.monotonic() - iteration_started)
            reserve = max(0.003, 1.5 * longest_iteration_seconds)
            if deadline - time.monotonic() <= reserve:
                break
    finally:
        if board.sfen() != root_sfen or len(board.move_stack) != root_stack:
            raise AssertionError('MCTS corrupted caller board')
    # Child visits measure how often each root move was examined.
    best = max(root.children, key=lambda c: (c.visits, -root_moves.index(c.move)))
    return best.move.usi(), iterations, (time.monotonic() - start) * 1000.0
