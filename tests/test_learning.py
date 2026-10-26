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
