"""Fail-closed frozen root-bound arena protocol and paired accounting."""
import unittest
import shogi
from fukaji.root_alpha_arena import validate_request, summarize_pairs, MODEL_SHA, MODEL_RUN


class RootAlphaArenaTests(unittest.TestCase):
    def config(self):
        return {
            "research_only": True, "relative_source_run_id": MODEL_RUN,
            "relative_model_sha256": MODEL_SHA, "arena_seed": 20271327,
            "arena_pairs": 8, "max_plies": 120, "opening_plies": 6,
            "movetime_ms": 200, "depth": 2, "mix_percent": 25,
            "fast_relative": True, "root_alpha_beta_default": False,
            "root_alpha_beta_candidate": True, "training_allowed": False,
        }

    def test_protocol_frozen(self):
        self.assertEqual(validate_request(self.config()), self.config())
        for key, value in (
            ("arena_seed", 0), ("arena_pairs", 16),
            ("max_plies", 160), ("movetime_ms", 1000),
            ("depth", 3), ("mix_percent", 100),
            ("fast_relative", False), ("training_allowed", True),
            ("root_alpha_beta_default", True),
            ("root_alpha_beta_candidate", False),
            ("relative_model_sha256", "bad"),
        ):
            c=self.config()
            c[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_request(c)

    def test_independent_32_game_protocol_is_frozen(self):
        c = self.config()
        c.update({"arena_seed": 20271427, "arena_pairs": 16, "max_plies": 160})
        self.assertEqual(validate_request(c), c)
        for field, wrong in (("arena_pairs", 8), ("max_plies", 120),
                             ("movetime_ms", 1000), ("root_alpha_beta_default", True)):
            bad = dict(c)
            bad[field] = wrong
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_request(bad)

    def test_third_independent_32_game_protocol_is_frozen(self):
        c = self.config()
        c.update({"arena_seed": 20271527, "arena_pairs": 16, "max_plies": 160})
        self.assertEqual(validate_request(c), c)
        for field, wrong in (("arena_pairs", 8), ("max_plies", 120),
                             ("movetime_ms", 1000), ("depth", 3),
                             ("root_alpha_beta_candidate", False),
                             ("training_allowed", True)):
            bad = dict(c)
            bad[field] = wrong
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_request(bad)

    def test_unresolved_is_not_a_win(self):
        head=["7g7f","3c3d"]
        rows=[
            {"opening":head,"candidate_black":True,"status":"checkmate","winner":shogi.BLACK},
            {"opening":head,"candidate_black":False,"status":"move_limit_unresolved","winner":None},
            {"opening":head,"candidate_black":True,"status":"checkmate","winner":shogi.WHITE},
            {"opening":head,"candidate_black":False,"status":"move_limit_unresolved","winner":None},
        ]
        self.assertEqual(summarize_pairs(rows,2),
                         {"candidate_advantage":1,"baseline_advantage":1})
        with self.assertRaises(ValueError):
            summarize_pairs(rows[:-1],2)
        with self.assertRaises(ValueError):
            summarize_pairs([rows[1],rows[0],*rows[2:]],2)


if __name__=="__main__":
    unittest.main()
