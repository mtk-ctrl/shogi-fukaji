import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import shogi
from fukaji.engine import select_move
from fukaji.learning import partition_rows
from fukaji.endgame_branch import branch_from_parents

class RealLearningImprovementTests(unittest.TestCase):
    def test_repetition_avoidance_selects_fresh_legal_king_move(self):
        b=shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        cycle=("5i5h","5a5b","5h5i","5b5a")
        for _ in range(2):
            for word in cycle:
                self.assertIn(shogi.Move.from_usi(word),b.legal_moves)
                b.push_usi(word)
        self.assertGreaterEqual(b.transpositions[b.zobrist_hash()],3)
        word,depth,nodes=select_move(b,30,max_depth=1,avoid_repetition=True)
        self.assertIn(shogi.Move.from_usi(word),b.legal_moves)
        self.assertNotEqual(word,"5i5h")
    def test_parent_groups_never_cross_splits(self):
        rows=[]
        for i in range(12):
            key=f"{i:064x}"
            for _ in range(3):
                rows.append((key,np.zeros((1,3)),np.zeros(1)))
        parts,groups=partition_rows(rows)
        self.assertEqual(sum(groups.values()),12)
        self.assertGreaterEqual(min(groups.values()),2)
        self.assertEqual(sum(len(x) for x in parts.values()),36)
        self.assertTrue(all(len(part)%3==0 for part in parts.values()))
    def test_empty_branch_source_is_safe(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/"parents.jsonl").write_text("")
            report=branch_from_parents(root/"parents.jsonl",root/"branches.jsonl")
            self.assertEqual(report["branch_attempts"],0)
            self.assertEqual(report["verified_branch_mates"],0)
    def test_invalid_parent_rejected_before_fork(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            game={"start_sfen":shogi.Board().sfen(),"moves":["7g7f"],"winner":0,"source":"selfplay"}
            (root/"parents.jsonl").write_text(json.dumps(game)+"\n")
            with self.assertRaises(ValueError):
                branch_from_parents(root/"parents.jsonl",root/"branches.jsonl")
if __name__=="__main__":
    unittest.main()
