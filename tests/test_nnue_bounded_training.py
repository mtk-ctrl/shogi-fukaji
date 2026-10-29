"""Preserve training and full-split evaluation while bounding feature storage."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import shogi

from fukaji.learning import FEATURES, iter_games, load_games
from fukaji.nnue_research import SmallNNUE
from fukaji.nnue_train_research import (
    MAX_POSITIONS_PER_GAME, bounded_training_rows, fit_train_only, loss,
)


class BoundedNnueTrainingTests(unittest.TestCase):
    def rows(self, count):
        rng = np.random.default_rng(count)
        x = (rng.random((count, FEATURES)) < 0.015).astype(np.float32)
        x[:, -1] = 1.0
        y = (np.arange(count) % 2).astype(np.float32)
        return x, y

    def test_exact_optimizer_sample_and_holdouts_untouched(self):
        train = self.rows(63)
        validation = self.rows(41)
        test = self.rows(29)
        source = [("tr", *train), ("va", *validation), ("te", *test)]
        rows = list(bounded_training_rows(iter(source),
                    {"tr": "train", "va": "validation", "te": "test"}))
        indexes = np.linspace(0, 62, MAX_POSITIONS_PER_GAME, dtype=np.int64)
        np.testing.assert_array_equal(rows[0][1], train[0][indexes])
        np.testing.assert_array_equal(rows[0][2], train[1][indexes])
        self.assertIs(rows[1][1], validation[0])
        self.assertIs(rows[1][2], validation[1])
        self.assertIs(rows[2][1], test[0])
        self.assertIs(rows[2][2], test[1])
        self.assertLess(rows[0][1].nbytes, train[0].nbytes)

    def test_same_weights_and_optimizer_stats_as_full_training_rows(self):
        samples = [self.rows(n) for n in (1, 17, 31, 63)]
        compact = [(x, y) for _, x, y in bounded_training_rows(
            ((str(i), x, y) for i, (x, y) in enumerate(samples)),
            {str(i): "train" for i in range(len(samples))})]
        full_model, full_stats = fit_train_only(samples, epochs=2, seed=71)
        bounded_model, bounded_stats = fit_train_only(compact, epochs=2, seed=71)
        self.assertEqual(full_stats, bounded_stats)
        for name in ("hidden_weights", "hidden_bias", "output_weights"):
            np.testing.assert_array_equal(getattr(full_model, name),
                                          getattr(bounded_model, name))
        self.assertEqual(full_model.output_bias, bounded_model.output_bias)

    def test_streamed_loss_uses_every_position_and_matches_list(self):
        parts = [self.rows(n) for n in (17, 63)]
        model = SmallNNUE.synthetic_test_model(seed=71)
        expected = loss(parts, model)
        actual = loss(iter(parts), model)
        self.assertEqual(actual, expected)
        self.assertEqual(actual["games"], 2)
        self.assertEqual(actual["positions"], 80)

    def test_iterator_preserves_verified_games_and_rejects_illegal_tail(self):
        # Same constructed legal mate used in independent tactical diagnostics.
        sfen = "8k/9/7R1/7L1/9/9/9/9/8K b - 1"
        board = shogi.Board(sfen)
        move = shogi.Move.from_usi("2c2b+")
        self.assertIn(move, board.legal_moves)
        board.push(move)
        self.assertTrue(board.is_checkmate())
        game = {"start_sfen": sfen, "moves": ["2c2b+"],
                "winner": 0, "source": "selfplay"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "games.jsonl"
            path.write_text(json.dumps(game) + "\n", encoding="utf-8")
            a = load_games(path)
            b = list(iter_games(path))
            self.assertEqual(len(a), 1)
            self.assertEqual(a[0][0], b[0][0])
            np.testing.assert_array_equal(a[0][1], b[0][1])
            np.testing.assert_array_equal(a[0][2], b[0][2])
            # Iterator must validate the complete move list before yielding.
            game["moves"].append("1a1a")
            path.write_text(json.dumps(game) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                next(iter_games(path))


if __name__ == "__main__":
    unittest.main()
