"""G8 selection protocol tests: frozen holdouts, exact quotas, no promotion."""
import hashlib
import unittest
from unittest.mock import patch
from collections import Counter

from fukaji.generation8_selection import frozen_splits, select, target_index


def parent(i):
    return hashlib.sha256(("g8-parent-" + str(i)).encode()).hexdigest()


def fixtures():
    games = []
    for i in range(400):
        status = "checkmate" if i < 258 else "move_limit_unresolved"
        games.append({
            "game_id": i, "parent_id": parent(i), "status": status,
            "winner": (i % 2 if status == "checkmate" else None),
            "opening": ["7g7f"] * 6, "moves": ["2g2f"] * 214,
        })
    prior = {hashlib.sha256(("old-parent-" + str(i)).encode()).hexdigest():
             ("train" if i < 447 else "validation" if i < 502 else "test")
             for i in range(557)}
    return games, prior


class G8SelectionTests(unittest.TestCase):
    def test_frozen_split_and_100_train_only(self):
        games, prior = fixtures()
        splits = frozen_splits(games, prior)
        self.assertEqual(Counter(splits.values()),
                         {"train": 320, "validation": 40, "test": 40})
        self.assertFalse(set(splits) & set(prior))
        with patch("fukaji.generation8_selection.verified_position",
                   side_effect=lambda g, i: "unique-" + str(g["game_id"])):
            candidates = select(games, splits)
        self.assertEqual(len(candidates), 100)
        self.assertEqual(Counter(x["reason_category"] for x in candidates),
                         {"mate_tail": 40, "mate_middle": 20,
                          "unresolved_late": 20, "early_mixed": 20})
        self.assertTrue(all(x["split"] == "train" for x in candidates))
        self.assertEqual(len({x["parent_id"] for x in candidates}), 100)
        self.assertTrue(all(x["winner_for_proven_mates_only"] is None
                            for x in candidates if x["terminal_status"] != "checkmate"))

    def test_overlap_fails_closed(self):
        games, prior = fixtures()
        games[0]["parent_id"] = next(iter(prior))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            frozen_splits(games, prior)

    def test_duplicate_board_fails_closed(self):
        games, prior = fixtures()
        splits = frozen_splits(games, prior)
        with patch("fukaji.generation8_selection.verified_position",
                   return_value="same-position"):
            with self.assertRaisesRegex(ValueError, "insufficient unique"):
                select(games, splits)

    def test_position_selection_range(self):
        games, _ = fixtures()
        for kind in ("mate_tail", "mate_middle", "unresolved_late", "early_mixed"):
            idx = target_index(games[0], kind)
            self.assertTrue(6 <= idx < 220)


if __name__ == "__main__":
    unittest.main()
