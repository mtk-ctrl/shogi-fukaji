"""Train-only sampling candidate; not wired into the frozen NNUE trainer.

Keep adjacent side-to-move observations together. This removes parity aliasing
from evenly spaced individual plies without looking at outcome labels.
"""
from __future__ import annotations

import numpy as np


def paired_sample_indices(length, max_positions=16):
    """Return sorted unique indices, preserving the final ply pair.

    Long games use equally spaced adjacent pairs, giving equal representation
    to both mover parities. Caps >= 4 also preserve the first pair. Short
    games retain every observation; cap 2 retains only the final pair.
    """
    if (type(length) is not int or length < 1 or
            type(max_positions) is not int or max_positions < 2 or
            max_positions % 2):
        raise ValueError("positive length and even sample cap >= 2 required")
    if length <= max_positions:
        return np.arange(length, dtype=np.int64)
    if max_positions == 2:
        return np.array([length - 2, length - 1], dtype=np.int64)
    ends = np.linspace(1, length - 1, max_positions // 2, dtype=np.int64)
    return np.column_stack((ends - 1, ends)).reshape(-1)


def paired_training_rows(rows, mapping, max_positions=16):
    """Apply the candidate to train only; leave holdout objects unchanged."""
    for group, x, y in rows:
        if mapping.get(group) not in ("train", "validation", "test"):
            raise ValueError("unknown parent split")
        if mapping[group] == "train":
            selected = paired_sample_indices(len(y), max_positions)
            x, y = x[selected], y[selected]
        yield group, x, y
