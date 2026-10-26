import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from fukaji.arena import run
from fukaji.learning import FEATURES

class ArenaTests(unittest.TestCase):
    def test_paired_real_moves_no_fake_results(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            model=p/"zero.npz"
            np.savez_compressed(model,weight=np.zeros(FEATURES,dtype=np.float32))
            result=run(model,p/"arena.jsonl",pairs=1,movetime=1,depth=1,max_plies=2,opening_plies=2)
            self.assertEqual(result["games"],2)
            self.assertEqual(result["unresolved"],2)
            self.assertIsNone(result["candidate_score_on_resolved"])
            rows=[json.loads(s) for s in (p/"arena.jsonl").read_text().splitlines()]
            self.assertEqual([r["candidate_black"] for r in rows],[True,False])
            self.assertTrue(all(r["winner"] is None for r in rows))
    def test_zero_mix_matches_identical_baseline_with_paired_opening(self):
        # A nonzero learned file at 0% mixing must yield exactly the
        # same legal search and move sequence on both sides.
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            model=p/"nonzero.npz"
            rng=np.random.default_rng(8)
            np.savez_compressed(model,weight=rng.normal(0,0.1,FEATURES).astype(np.float32))
            result=run(model,p/"control.jsonl",pairs=1,movetime=2,
                       depth=1,max_plies=12,opening_plies=4,mix=0,seed=129)
            rows=[json.loads(s) for s in (p/"control.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["opening"],rows[1]["opening"])
            self.assertEqual(rows[0]["moves"],rows[1]["moves"])
            self.assertEqual(rows[0]["winner"],rows[1]["winner"])
            self.assertEqual(result["search_diagnostics"]["candidate_turns"],
                             result["search_diagnostics"]["baseline_turns"])
            self.assertIn("candidate_depth_zero",result["search_diagnostics"])

    def test_bad_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            model=p/"bad.npz"
            np.savez_compressed(model,weight=np.zeros(3,dtype=np.float32))
            with self.assertRaises(ValueError):
                run(model,p/"arena.jsonl",pairs=1)
if __name__=="__main__":unittest.main()
