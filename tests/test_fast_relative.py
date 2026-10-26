"""Opt-in fused relative evaluation must exactly match the existing score."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import shogi

from fukaji.engine import evaluate, evaluate_relative_fused, select_move
from fukaji.learning import FEATURES
from fukaji.relative_value import RelativeValue


class FusedRelativeTests(unittest.TestCase):
    def test_exact_score_and_depth_one_move_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen.npz"
            weights = np.random.default_rng(20271027).normal(
                0.0, 0.04, FEATURES).astype(np.float32)
            np.savez(path, weight=weights)
            board = shogi.Board()
            positions = [board.sfen()]
            for ply in range(90):
                legal = list(board.legal_moves)
                if not legal:
                    break
                move = legal[(ply * 17 + 11) % len(legal)]
                board.push(move)
                if ply % 6 == 0 and not board.is_checkmate():
                    positions.append(board.sfen())
            self.assertGreaterEqual(len(positions), 8)
            for mix in (0, 25, 100):
                model = RelativeValue(path, mix_percent=mix)
                for sfen in positions:
                    position = shogi.Board(sfen)
                    old = evaluate(position, model)
                    new = evaluate(position, model, fast_relative=True)
                    self.assertEqual(old, new, (mix, sfen))
                    if mix:
                        self.assertEqual(old, evaluate_relative_fused(position, model))
                position = shogi.Board(positions[0])
                original = select_move(position, max_depth=1, model=model,
                                       complete_depth=True)
                optimized = select_move(position, max_depth=1, model=model,
                                        complete_depth=True, fast_relative=True)
                self.assertEqual(original, optimized)

    def test_default_and_no_model_unchanged(self):
        board = shogi.Board()
        self.assertEqual(evaluate(board), evaluate(board, fast_relative=True))
        self.assertEqual(select_move(board, max_depth=1, complete_depth=True),
                         select_move(board, max_depth=1, complete_depth=True,
                                     fast_relative=True))


if __name__ == "__main__":
    unittest.main()
