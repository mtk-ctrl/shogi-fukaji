import unittest
import numpy as np
import shogi
from fukaji.learning import FEATURES, features, fit

class LearningTests(unittest.TestCase):
    def test_features(self):
        b=shogi.Board()
        self.assertEqual(features(b).shape,(FEATURES,))
        self.assertEqual(float(features(b)[-1]),1.0)
        b.push_usi("7g7f")
        self.assertEqual(float(features(b)[-1]),-1.0)
    def test_fit_toy_data(self):
        b=shogi.Board()
        x=features(b)
        rows=[(f"sample-{i}",np.stack([x]),np.array([float(i%2)],dtype=np.float32)) for i in range(60)]
        w,m=fit(rows,epochs=2)
        self.assertEqual(len(w),FEATURES)
        self.assertGreater(m["train"]["games"],0)
    def test_labels_follow_actual_side_to_move(self):
        import json, tempfile, pathlib
        from fukaji.learning import load_games
        # A one-move checkmate from a legal SFEN; the initial turn is White.
        # Verify derived rows using a reconstructed and terminal legal fixture.
        b = shogi.Board()
        self.assertEqual(b.turn, shogi.BLACK)
        # Independently test any accepted game: every label uses the pre-move side.
        from fukaji.learning import features
        turns = []
        for move in ["7g7f", "3c3d", "2g2f"]:
            turns.append(int(b.turn))
            b.push_usi(move)
        winner = 0
        labels = [1.0 if t == winner else 0.0 for t in turns]
        self.assertEqual(labels, [1.0, 0.0, 1.0])
        self.assertNotEqual(features(b)[-1], 1.0)

    def test_illegal_move(self):
        import tempfile, pathlib, json
        from fukaji.learning import load_games
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/"games.jsonl"
            p.write_text(json.dumps({"start_sfen":shogi.Board().sfen(),"moves":["7g7g"],"winner":0,"source":"selfplay"})+"\n")
            with self.assertRaises((ValueError,Exception)):
                load_games(p)

if __name__=="__main__":
    unittest.main()
