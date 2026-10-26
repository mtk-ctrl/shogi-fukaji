import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fukaji.direct_arena import execute, replay_check
from fukaji.learning import FEATURES


class DirectArenaTests(unittest.TestCase):
    def test_sha_verified_equal_depth_color_pairs_and_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            legacy = root / "legacy.npz"
            relative = root / "relative.npz"
            np.savez_compressed(legacy, weight=np.zeros(FEATURES, dtype=np.float32))
            np.savez_compressed(relative, weight=np.full(FEATURES, 0.001, dtype=np.float32))
            cfg = {"relative_source_run_id": 40359123574,
                   "legacy_source_run_id": 40358050404,
                   "relative_model_sha256": hashlib.sha256(relative.read_bytes()).hexdigest(),
                   "legacy_model_sha256": hashlib.sha256(legacy.read_bytes()).hexdigest(),
                   "arena_seed": 20270627, "arena_pairs": 1,
                   "max_plies": 4, "opening_plies": 2,
                   "depth": 1, "mix_percent": 25}
            request = root / "request.json"
            request.write_text(json.dumps(cfg), encoding="utf-8")
            report = execute(relative, legacy, request, root / "result")
            self.assertEqual(report["games"], 2)
            self.assertEqual(report["pairs"], 1)
            self.assertEqual(report["relative_wins"] + report["legacy_wins"] +
                             report["unresolved"], 2)
            rows = [json.loads(line) for line in
                    (root / "result/direct_games.jsonl").read_text().splitlines()]
            self.assertEqual([r["relative_black"] for r in rows], [True, False])
            self.assertEqual(rows[0]["opening"], rows[1]["opening"])
            for row in rows:
                replay_check(row["opening"], row["moves"], row)
            self.assertEqual(report["stats"]["relative_depth_zero"], 0)
            self.assertEqual(report["stats"]["legacy_depth_zero"], 0)
            cfg.update({"search_mode": "timed", "movetime_ms": 1,
                        "depth": 2, "max_plies": 4})
            request.write_text(json.dumps(cfg), encoding="utf-8")
            timed = execute(relative, legacy, request, root / "timed")
            self.assertEqual(timed["games"], 2)
            self.assertEqual(timed["search_mode"], "timed")
            self.assertEqual(timed["movetime_ms"], 1)
            self.assertEqual(timed["depth"], 2)
            self.assertEqual(timed["relative_wins"] + timed["legacy_wins"] +
                             timed["unresolved"], 2)
            self.assertEqual(timed["stats"]["relative_turns"] +
                             timed["stats"]["legacy_turns"], 8)
            self.assertLessEqual(timed["stats"]["relative_depth_sum"] +
                                 timed["stats"]["legacy_depth_sum"], 16)
            cfg["relative_model_sha256"] = "0" * 64
            request.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "relative_model_sha256 mismatch"):
                execute(relative, legacy, request, root / "invalid")

    def test_fabricated_checkmate_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid checkmate winner"):
            replay_check([], [], {"status": "checkmate", "winner": 0})


if __name__ == "__main__":
    unittest.main()
