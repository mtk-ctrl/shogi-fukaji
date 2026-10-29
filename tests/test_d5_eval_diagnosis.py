"""D5 split and outcome-label safety; no real games or G8 training in unit tests."""
import unittest

from fukaji.d5_eval_diagnosis import (
    EXCLUDED, REMAINING, PLIES, auc, frozen_split, pair_bootstrap,
    score_summary, verify_source, feature_snapshot, development_error_cases,
)


class D5EvaluationDiagnosisTests(unittest.TestCase):
    def test_split_excludes_all_d4_and_early_development_pairs(self):
        dev, sealed = frozen_split()
        self.assertEqual(len(dev), 12)
        self.assertEqual(len(sealed), 12)
        self.assertEqual(len(REMAINING), 24)
        self.assertEqual(set(dev) | set(sealed), REMAINING)
        self.assertFalse(set(dev) & set(sealed))
        self.assertFalse(set(dev) & EXCLUDED)
        self.assertEqual(frozen_split(), (dev, sealed))
        self.assertEqual(PLIES, (20, 60, 100))

    def test_auc_and_signs_follow_side_to_move_winner(self):
        rows = [
            {"pair": 1, "outcome": 1, "g8": 200, "classical": 0},
            {"pair": 1, "outcome": -1, "g8": -100, "classical": 0},
            {"pair": 2, "outcome": 1, "g8": 150, "classical": 0},
            {"pair": 2, "outcome": -1, "g8": -300, "classical": 0},
        ]
        self.assertEqual(auc(rows, "g8"), 1.0)
        self.assertEqual(score_summary(rows, "g8")["sign_accuracy"], 1.0)
        self.assertEqual(score_summary(rows, "classical")["sign_accuracy"], 0.0)
        self.assertEqual(score_summary(rows, "classical")["zero_scores"], 4)
        self.assertEqual(pair_bootstrap(rows, "g8"), [1.0, 1.0])

    def test_auc_without_both_classes_is_unknown(self):
        rows = [{"pair": 1, "outcome": 1, "g8": 2}]
        self.assertIsNone(auc(rows, "g8"))
        self.assertIsNone(score_summary([], "g8")["sign_accuracy"])
        self.assertIsNone(pair_bootstrap(rows, "g8"))

    def test_unresolved_not_converted_to_outcome(self):
        # A source with no complete 240-game audit must be rejected,
        # rather than silently reporting zero losses.
        with self.assertRaises(ValueError):
            verify_source([])

    def test_feature_snapshot_is_read_only_and_mover_relative(self):
        import shogi
        board = shogi.Board()
        before = (board.sfen(), board.zobrist_hash(), len(board.move_stack))
        features = feature_snapshot(board)
        self.assertEqual(features["board_material_cp"], 0)
        self.assertEqual(features["hand_material_cp"], 0)
        self.assertIsInstance(features["own_king_zone_attacks"], int)
        self.assertIsInstance(features["opponent_king_zone_attacks"], int)
        self.assertEqual((board.sfen(), board.zobrist_hash(),
                          len(board.move_stack)), before)

    def test_error_cases_exclude_correct_predictions(self):
        def row(pair, g8, classical, outcome):
            return {"pair": pair, "ply": 20, "g8_black": True,
                    "g8": g8, "classical": classical, "outcome": outcome,
                    "features": {"board_material_cp": 0}}
        errors = development_error_cases([
            row(1, 100, 100, 1), row(2, -100, 100, 1),
            row(3, -100, -100, -1),
        ])
        self.assertEqual([r["pair"] for r in errors], [2])
        self.assertFalse(errors[0]["g8_correct"])
        self.assertTrue(errors[0]["classical_correct"])


if __name__ == "__main__":
    unittest.main()
