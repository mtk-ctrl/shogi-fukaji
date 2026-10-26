"""USI research bridge for the frozen mover-relative FUKAJI candidate."""
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fukaji.learning import FEATURES


class RelativeUSITests(unittest.TestCase):
    def make_model(self, root):
        path = Path(root) / "relative.npz"
        np.savez(path, weight=np.zeros(FEATURES, dtype=np.float32))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_relative_model_loaded_and_legal_bestmove(self):
        with tempfile.TemporaryDirectory() as td:
            path, sha = self.make_model(td)
            transcript = (
                "usi\n"
                "setoption name ValueModelType value relative\n"
                f"setoption name ValueModelFile value {path}\n"
                "setoption name ValueModelMix value 25\n"
                "setoption name SearchDepth value 2\n"
                "isready\n"
                "position startpos moves 7g7f 3c3d\n"
                "go movetime 20\n"
                "quit\n"
            )
            p = subprocess.run(
                [sys.executable, "-m", "fukaji.engine"],
                input=transcript, text=True, capture_output=True, timeout=20,
            )
            self.assertEqual(p.returncode, 0, p.stderr + p.stdout)
            self.assertIn("option name ValueModelType", p.stdout)
            self.assertIn("info string value_model sha256 " + sha, p.stdout)
            self.assertIn("readyok", p.stdout)
            moves = [line.split()[1] for line in p.stdout.splitlines()
                     if line.startswith("bestmove ")]
            self.assertEqual(len(moves), 1)
            import shogi
            board = shogi.Board()
            board.push_usi("7g7f")
            board.push_usi("3c3d")
            self.assertIn(shogi.Move.from_usi(moves[0]), board.legal_moves)

    def test_invalid_model_fails_closed_before_readyok(self):
        p = subprocess.run(
            [sys.executable, "-m", "fukaji.engine"],
            input="setoption name ValueModelType value relative\n"
                  "setoption name ValueModelFile value /nonexistent/fukaji-model.npz\n"
                  "isready\nquit\n",
            text=True, capture_output=True, timeout=10,
        )
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("value_model rejected", p.stdout)
        self.assertNotIn("readyok", p.stdout)


if __name__ == "__main__":
    unittest.main()
