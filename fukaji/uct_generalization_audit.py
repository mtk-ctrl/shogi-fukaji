"""Research-only validation for a disjoint UCT timing holdout and ABBA rows.

This module does NOT download source games, prove legal replay, or start runs.
Callers must separately verify the frozen source/model SHA and full legality.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

PHASES = (12, 50, 110)
PASSES = ("off_a", "on_a", "on_b", "off_b")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _sha(value):
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def validate_parent_split(development, holdout, *, source_sha256):
    """Require 12 distinct game parents per split, four at each phase.

    SFEN and game-ID disjointness is metadata validation, NOT legal replay.
    """
    if not _sha(source_sha256):
        raise ValueError("invalid frozen source SHA256")
    result = {}
    for label, rows in (("development", development), ("holdout", holdout)):
        if not isinstance(rows, (list, tuple)) or len(rows) != 12:
            raise ValueError(f"{label} must have 12 parents")
        phases = Counter()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("parent must be a dict")
            if row.get("ply") not in PHASES or type(row.get("ply")) is not int:
                raise ValueError("unexpected parent ply")
            if type(row.get("seed")) is not int or row["seed"] < 0:
                raise ValueError("invalid seed")
            if any(not isinstance(row.get(k), str) or not row[k].strip()
                   for k in ("parent_id", "game_id", "sfen")):
                raise ValueError("missing parent identity")
            if row.get("source_sha256") != source_sha256:
                raise ValueError("source SHA mismatch")
            phases[row["ply"]] += 1
        if any(phases[p] != 4 for p in PHASES):
            raise ValueError("need four parents at each ply")
        for field in ("parent_id", "game_id", "sfen"):
            values = [r[field] for r in rows]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} {field}")
        result[label] = rows
    for field in ("parent_id", "game_id", "sfen"):
        if {r[field] for r in development} & {r[field] for r in holdout}:
            raise ValueError(f"holdout overlaps development by {field}")
    return {"development_parents": 12, "holdout_parents": 12,
            "holdout_by_ply": {p: 4 for p in PHASES},
            "source_sha256": source_sha256, "legality_replayed": False}


def validate_holdout_abba(rows, holdout, *, model_sha256, source_sha256):
    """Check frozen pairing, metadata and 48 observations; no win labels."""
    if not _sha(model_sha256) or not _sha(source_sha256):
        raise ValueError("invalid frozen SHA256")
    if not isinstance(rows, (list, tuple)) or len(rows) != 48:
        raise ValueError("expected 48 observations")
    parents = {p["parent_id"]: p for p in holdout}
    if len(parents) != 12 or len(holdout) != 12:
        raise ValueError("expected 12 distinct holdout parents")
    grouped = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("observation must be a dict")
        p = parents.get(row.get("parent_id"))
        if p is None:
            raise ValueError("unknown parent")
        if (row.get("game_id") != p["game_id"] or
            row.get("sfen") != p["sfen"] or row.get("seed") != p["seed"] or
            row.get("source_sha256") != source_sha256 or
            row.get("model_sha256") != model_sha256 or
            row.get("movetime_ms") != 200):
            raise ValueError("frozen ABBA metadata mismatch")
        if row.get("pass") not in PASSES:
            raise ValueError("unknown pass")
        if type(row.get("iterations")) is not int or row["iterations"] < 0:
            raise ValueError("invalid iteration count")
        elapsed = row.get("external_ms")
        if (type(elapsed) not in (float, int) or
                not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError("invalid elapsed time")
        if (row.get("legal") is not True or
            row.get("board_restored") is not True or
            not isinstance(row.get("move"), str) or
            row["move"] in ("", "resign")):
            raise ValueError("legality/restoration evidence missing")
        if any(k in row for k in ("training_label", "winner", "draw_label")):
            raise ValueError("timing observations cannot be outcome teachers")
        grouped[row["parent_id"]].append(row["pass"])
    for parent_id in parents:
        if grouped[parent_id] != list(PASSES):
            raise ValueError("parent order must be OFF-ON-ON-OFF")
    return {"parents": 12, "observations": 48, "off": 24, "on": 24,
            "legality_replayed": False, "training_labels_created": False}
