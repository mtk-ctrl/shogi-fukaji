"""Select tactical positions from the audited G8/G7 240-game source.

Research-only selection. No teacher labels, no search or adoption.
Full legal move prefixes are retained to reproduce hands and repetitions.
Unavailable categories are reported as shortages, never fabricated.
"""
from __future__ import annotations

from collections import defaultdict

import shogi

CATEGORIES = ("checkmate", "in_check", "repetition_risk", "hand_pieces")


def position_key(board):
    """SFEN position, side and hands, excluding the move counter."""
    return board.sfen().rsplit(" ", 1)[0]


def classify_position(board, occurrence_count):
    """Return overlapping tactical categories, not outcome labels."""
    kinds = []
    if board.is_checkmate():
        kinds.append("checkmate")
    elif board.is_check():
        kinds.append("in_check")
    if occurrence_count >= 3 and not board.is_fourfold_repetition():
        kinds.append("repetition_risk")
    if sum(int(n) for hand in board.pieces_in_hand for n in hand.values()) > 0:
        kinds.append("hand_pieces")
    return kinds


def select_tactical_positions(rows, per_category=3, max_ply=240):
    """Select unique parent pairs and SFENs from independently legal histories.

    The caller must first verify frozen source SHA and all 240 complete
    histories with search_method_probe.load_source(). This function also
    validates each prefix. A terminal checkmate is a terminal test case,
    NOT a position for comparing two move choices.
    """
    if type(per_category) is not int or per_category < 1:
        raise ValueError("per_category must be a positive integer")
    if type(max_ply) is not int or max_ply < 1:
        raise ValueError("max_ply must be a positive integer")
    candidates = {category: [] for category in CATEGORIES}
    for row in rows:
        board = shogi.Board()
        occurrences = defaultdict(int)
        occurrences[position_key(board)] = 1
        prefix = []
        for ply, word in enumerate(row["opening"] + row["moves"], start=1):
            if ply > max_ply:
                break
            move = shogi.Move.from_usi(word)
            if move not in board.legal_moves:
                raise ValueError("illegal source move at ply " + str(ply))
            board.push(move)
            prefix.append(word)
            key = position_key(board)
            occurrences[key] += 1
            kinds = classify_position(board, occurrences[key])
            if not kinds:
                continue
            hand_count = sum(int(n) for h in board.pieces_in_hand for n in h.values())
            item = {
                "category": None, "pair": row["pair"], "g8_black": row["g8_black"],
                "ply": ply, "prefix": list(prefix), "sfen": board.sfen(),
                "in_check": bool(board.is_check()),
                "is_checkmate": bool(board.is_checkmate()),
                "hand_pieces": hand_count, "position_occurrences": occurrences[key],
                "source_status": row["status"],
            }
            for category in kinds:
                candidates[category].append(dict(item, category=category))
    used_pairs, used_sfen, selected = set(), set(), []
    shortages = {}
    for category in CATEGORIES:
        count = 0
        for item in candidates[category]:
            if item["pair"] in used_pairs or item["sfen"] in used_sfen:
                continue
            selected.append(item)
            used_pairs.add(item["pair"])
            used_sfen.add(item["sfen"])
            count += 1
            if count == per_category:
                break
        shortages[category] = per_category - count
    return selected, shortages
