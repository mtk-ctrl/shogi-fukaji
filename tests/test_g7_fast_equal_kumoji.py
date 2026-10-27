import copy
import tempfile
import unittest
from pathlib import Path

from fukaji.g7_fast_equal_kumoji import (
    expected_request, validate_request, engine_options,
    PAIRS, SHARDS, SEED, aggregate,
)


class G7EqualKumojiTests(unittest.TestCase):
    def test_fixed_conditions_and_no_adoption(self):
        c = expected_request()
        self.assertEqual(validate_request(c), c)
        self.assertEqual((PAIRS, SHARDS), (32, 20))
        self.assertEqual(c["movetime_ms"], 200)
        self.assertEqual(c["g7_depth"], 4)
        self.assertTrue(c["g7_fast_move_list"])
        self.assertFalse(c["training_allowed"])
        self.assertFalse(c["model_adoption_allowed"])
        self.assertFalse(c["official_engine_changes_allowed"])
        self.assertFalse(c["kumoji_position_knowledge"])
        self.assertFalse(c["kumoji_adaptive_long_think"])
        self.assertFalse(c["kumoji_ponder"])
        self.assertEqual(c["kumoji_search_threads"], 1)
        self.assertNotIn(SEED, (20274327, 20274627, 20274027, 20273727))
        self.assertEqual(sum(len(range(i, PAIRS, SHARDS)) for i in range(SHARDS)), PAIRS)

    def test_reject_any_drift(self):
        for k,v in (("g7_fast_move_list", False),
                    ("kumoji_position_knowledge", True),
                    ("kumoji_adaptive_long_think", True),
                    ("kumoji_ponder", True),
                    ("kumoji_search_threads", 2),
                    ("training_allowed", True),
                    ("model_adoption_allowed", True),
                    ("g7_depth", 2),
                    ("movetime_ms", 100),
                    ("seed", 20274027)):
            c = copy.deepcopy(expected_request())
            c[k] = v
            with self.subTest(k=k), self.assertRaises(ValueError):
                validate_request(c)
        c = expected_request()
        c["extra"] = "unsafe"
        with self.assertRaises(ValueError):
            validate_request(c)

    def test_exact_usi_settings(self):
        a,b = engine_options("model.npz")
        self.assertEqual(a["SearchDepth"], 4)
        self.assertTrue(a["FastMoveList"])
        self.assertTrue(a["FastRelativeEval"])
        self.assertTrue(a["RootAlphaBeta"])
        self.assertFalse(a["USI_Ponder"])
        self.assertEqual(b["SearchThreads"], 1)
        for k in ("PositionKnowledge", "AdaptiveLongThink", "USI_Ponder"):
            self.assertFalse(b[k])

    def test_missing_shards_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "missing independent shard"):
                aggregate(Path(d)/"request.json", d, d)

    def test_workflow_bounded_and_separate(self):
        yml = Path(".github/workflows/g7-fast-equal-kumoji.yml").read_text()
        self.assertIn("max-parallel: 20", yml)
        self.assertIn("fail-fast: false", yml)
        self.assertIn("cancel-in-progress: false", yml)
        self.assertIn("workflow_dispatch:", yml)
        self.assertIn("ref: 807db500ee2739c1eb6a527b614eeea40a5993d6", yml)
        self.assertIn("run-id: 40434466661", yml)
        self.assertIn("fukaji.g7_fast_equal_kumoji", yml)
        self.assertIn("aggregate:", yml)
        self.assertIn("if-no-files-found: error", yml)
        self.assertEqual(yml.count(".github/fukaji-g7-fast-equal-kumoji-request.json"), 3)


if __name__ == "__main__":
    unittest.main()
