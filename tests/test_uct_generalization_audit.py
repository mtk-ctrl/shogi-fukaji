"""Offline metadata tests: no G8 download, 200ms run or new game."""
import copy
import unittest

from fukaji.uct_generalization_audit import (
    PASSES, validate_holdout_abba, validate_parent_split,
)

SOURCE = "60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf"
MODEL = "18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492"


def parents(prefix):
    # Synthetic metadata; SFEN strings are placeholders, NOT legal positions.
    return [{"parent_id": f"{prefix}{i}", "game_id": f"{prefix}-game-{i}",
             "sfen": f"{prefix}-sfen-{i}", "ply": (12, 50, 110)[i // 4],
             "seed": 20279000 + i, "source_sha256": SOURCE}
            for i in range(12)]


def observations(holdout):
    return [{"parent_id": p["parent_id"], "game_id": p["game_id"],
             "sfen": p["sfen"], "seed": p["seed"], "source_sha256": SOURCE,
             "model_sha256": MODEL, "movetime_ms": 200,
             "pass": name, "external_ms": 199.0, "iterations": 0,
             "legal": True, "board_restored": True, "move": "7g7f"}
            for p in holdout for name in PASSES]


class GeneralizationProtocol(unittest.TestCase):
    def setUp(self):
        self.dev, self.hold = parents("dev"), parents("hold")

    def test_disjoint_12_parent_split(self):
        receipt = validate_parent_split(self.dev, self.hold, source_sha256=SOURCE)
        self.assertEqual(receipt["holdout_by_ply"], {12: 4, 50: 4, 110: 4})
        self.assertIs(receipt["legality_replayed"], False)

    def test_overlap_or_wrong_phase_rejected(self):
        hold = copy.deepcopy(self.hold)
        hold[0]["game_id"] = self.dev[0]["game_id"]
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_parent_split(self.dev, hold, source_sha256=SOURCE)
        hold = copy.deepcopy(self.hold)
        hold[0]["ply"] = 50
        with self.assertRaisesRegex(ValueError, "four parents"):
            validate_parent_split(self.dev, hold, source_sha256=SOURCE)

    def test_wrong_source_and_duplicate_sfen_rejected(self):
        hold = copy.deepcopy(self.hold)
        hold[0]["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            validate_parent_split(self.dev, hold, source_sha256=SOURCE)
        hold = copy.deepcopy(self.hold)
        hold[1]["sfen"] = hold[0]["sfen"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_parent_split(self.dev, hold, source_sha256=SOURCE)

    def test_abba_order_and_counts(self):
        rows = observations(self.hold)
        result = validate_holdout_abba(rows, self.hold, model_sha256=MODEL,
                                       source_sha256=SOURCE)
        self.assertEqual((result["off"], result["on"]), (24, 24))
        self.assertIs(result["training_labels_created"], False)
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaisesRegex(ValueError, "order"):
            validate_holdout_abba(rows, self.hold, model_sha256=MODEL,
                                  source_sha256=SOURCE)

    def test_missing_legality_and_teacher_labels_rejected(self):
        rows = observations(self.hold)
        rows[0]["legal"] = False
        with self.assertRaisesRegex(ValueError, "legality"):
            validate_holdout_abba(rows, self.hold, model_sha256=MODEL,
                                  source_sha256=SOURCE)
        rows = observations(self.hold)
        rows[0]["training_label"] = "draw"
        with self.assertRaisesRegex(ValueError, "teachers"):
            validate_holdout_abba(rows, self.hold, model_sha256=MODEL,
                                  source_sha256=SOURCE)

    def test_zero_iteration_with_legal_move_is_allowed(self):
        rows = observations(self.hold)
        self.assertTrue(all(r["iterations"] == 0 for r in rows))
        validate_holdout_abba(rows, self.hold, model_sha256=MODEL,
                              source_sha256=SOURCE)


if __name__ == "__main__":
    unittest.main()
