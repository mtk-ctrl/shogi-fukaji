"""Regression checks for immutable KUMOJI failure provenance and mate proof."""
import tempfile
import unittest
from pathlib import Path
import shogi
from fukaji.kumoji_loss_study import (
    RUN_ID, PROBE_PAIRS, snapshot, has_mate_in_one, audit,
)


class KumojiLossStudyTests(unittest.TestCase):
    def test_sampled_pairs_were_fixed_before_examining_tactics(self):
        self.assertEqual(RUN_ID, 41177748192)
        self.assertEqual(PROBE_PAIRS, tuple(range(0, 50, 5)))

    def test_exact_mate_in_one_probe_keeps_position_and_history(self):
        b=shogi.Board("8k/9/7R1/7L1/9/9/9/9/8K b - 1")
        before=snapshot(b)
        self.assertTrue(has_mate_in_one(b))
        self.assertEqual(snapshot(b),before)

    def test_negative_mate_proof_no_false_positive(self):
        b=shogi.Board()
        before=snapshot(b)
        self.assertFalse(has_mate_in_one(b))
        self.assertEqual(snapshot(b),before)

    def test_corrupt_source_never_bypasses_hash_guard(self):
        with tempfile.TemporaryDirectory() as root:
            a=Path(root)/"games.jsonl"; b=Path(root)/"summary.json"
            a.write_text("{}\n",encoding="utf-8")
            b.write_text("{}\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"SHA256"):
                audit(a,b)


if __name__=="__main__":
    unittest.main()
