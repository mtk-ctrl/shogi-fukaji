import json
import tempfile
import unittest
from pathlib import Path
from fukaji.engine_selfplay import play
from fukaji.audit import audit

class EngineSelfplayTests(unittest.TestCase):
    def test_bounded_search_can_end_without_fabricating_winner(self):
        self.assertIsNone(play(seed=1,movetime_ms=1,max_plies=2,max_depth=1))
    def test_zero_game_manifest_is_auditable_not_trainable(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"games.jsonl";p.write_text("")
            m=Path(d)/"games.manifest.json"
            m.write_text(json.dumps({"generator":"fukaji-alpha-beta-v0","dataset_sha256":hashlib.sha256(b"").hexdigest(),"completed_checkmates":0}))
            self.assertEqual(audit(p,m)["completed_verified_games"],0)
            m.write_text(json.dumps({"generator":"fukaji-alpha-beta-v0","dataset_sha256":"0"*64,"completed_checkmates":0}))
            with self.assertRaises(AssertionError):
                audit(p,m)
if __name__=="__main__":unittest.main()
