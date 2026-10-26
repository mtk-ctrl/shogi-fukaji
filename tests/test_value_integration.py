import tempfile
import unittest
from pathlib import Path
import numpy as np
import shogi
from fukaji.learning import FEATURES, features
from fukaji.engine import evaluate,select_move
from fukaji.value_model import LinearValue

class ValueIntegrationTests(unittest.TestCase):
    def test_zero_model_does_not_change_classical_at_zero_mix(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"zero.npz"
            np.savez_compressed(p,weight=np.zeros(FEATURES,dtype=np.float32))
            b=shogi.Board()
            m=LinearValue(p,mix_percent=0)
            self.assertEqual(evaluate(b,m),evaluate(b))
            self.assertEqual(m.score(b),0)

    def test_turn_perspective_and_nonzero_inference(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"turn.npz"
            w=np.zeros(FEATURES,dtype=np.float32)
            w[-1]=1.0
            np.savez_compressed(p,weight=w)
            m=LinearValue(p,mix_percent=100)
            b=shogi.Board()
            self.assertEqual(m.score(b),100)
            self.assertEqual(evaluate(b,m),100)
            b.push_usi("7g7f")
            self.assertEqual(m.score(b),-100)
            self.assertEqual(evaluate(b,m),-100)
            move,depth,nodes=select_move(b,30,max_depth=2,model=m)
            self.assertIn(shogi.Move.from_usi(move),b.legal_moves)

    def test_sparse_inference_matches_original_dense_dot(self):
        rng = np.random.default_rng(1234)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"weights.npz"
            w=rng.normal(0,0.2,FEATURES).astype(np.float32)
            np.savez_compressed(p,weight=w)
            m=LinearValue(p,mix_percent=25)
            b=shogi.Board()
            for ply in range(40):
                dense=float(np.dot(features(b),w))
                expected=int(round(max(-10.0,min(10.0,dense))*100))
                self.assertLessEqual(abs(m.score(b)-expected),1,
                                     f"ply={ply}")
                options=list(b.legal_moves)
                if not options or b.is_checkmate():break
                b.push(options[int(rng.integers(len(options)))])

    def test_reject_bad_model(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"bad.npz"
            np.savez_compressed(p,weight=np.ones(3,dtype=np.float32))
            with self.assertRaises(ValueError): LinearValue(p)
            np.savez_compressed(p,weight=np.full(FEATURES,np.nan,dtype=np.float32))
            with self.assertRaises(ValueError): LinearValue(p)

if __name__=="__main__":unittest.main()
