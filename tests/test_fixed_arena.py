import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fukaji.fixed_arena import execute


class FixedArenaTests(unittest.TestCase):
    def config(self, model):
        return {
            "source_run_id":40349702347,
            "model_sha256":hashlib.sha256(model.read_bytes()).hexdigest(),
            "arena_seed":20270127,
            "arena_pairs":2,
            "arena_mixes":[0,25],
            "arena_movetimes_ms":[2,20],
            "arena_max_plies":160,
            "depth":1,
            "opening_plies":6,
        }

    def test_same_model_and_openings_across_all_arms(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            model=root/"model.npz"
            model.write_bytes(b"test fixture")
            req=root/"request.json"
            req.write_text(json.dumps(self.config(model)))
            with patch("fukaji.fixed_arena.arena",return_value={"games":4}) as fake:
                report=execute(model,req,root/"out")
            self.assertEqual(set(report["arms"]),{"2ms_mix0","2ms_mix25","20ms_mix0","20ms_mix25"})
            self.assertEqual(fake.call_count,4)
            self.assertEqual([c.kwargs["movetime"] for c in fake.call_args_list],[2,2,20,20])
            self.assertEqual([c.kwargs["mix"] for c in fake.call_args_list],[0,25,0,25])
            self.assertTrue(all(c.kwargs["seed"]==20270127 for c in fake.call_args_list))
            self.assertTrue(all(c.args[0]==model for c in fake.call_args_list))

    def test_complete_depth_request_routes_equal_depth_and_sha(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            model=root/"model.npz"
            model.write_bytes(b"test fixture")
            req=root/"request.json"
            cfg=self.config(model)
            cfg["arena_search_mode"]="complete_depth"
            cfg["arena_movetimes_ms"]=[20]
            req.write_text(json.dumps(cfg))
            with patch("fukaji.fixed_arena.arena",return_value={"games":4}) as fake:
                report=execute(model,req,root/"out")
            self.assertEqual(set(report["arms"]),{"complete_depth1_mix0","complete_depth1_mix25"})
            self.assertFalse(report["wall_clock_cutoff_enabled"])
            self.assertEqual(fake.call_count,2)
            self.assertTrue(all(c.kwargs["search_mode"]=="complete_depth" for c in fake.call_args_list))
            self.assertTrue(all(c.kwargs["depth"]==1 for c in fake.call_args_list))

    def test_mismatched_model_is_rejected_before_arena(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            model=root/"model.npz"
            model.write_bytes(b"test fixture")
            req=root/"request.json"
            cfg=self.config(model)
            cfg["model_sha256"]="0"*64
            req.write_text(json.dumps(cfg))
            with patch("fukaji.fixed_arena.arena") as fake:
                with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
                    execute(model,req,root/"out")
                fake.assert_not_called()

    def test_unsafe_game_budget_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            model=root/"model.npz"
            model.write_bytes(b"test fixture")
            req=root/"request.json"
            cfg=self.config(model)
            cfg["arena_pairs"]=40
            req.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,"budget"):
                execute(model,req,root/"out")


if __name__=="__main__":
    unittest.main()
