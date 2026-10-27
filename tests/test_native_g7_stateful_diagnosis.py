"""Guardrails for 80-position stateful research request and aggregation."""
import json
import tempfile
import unittest
from pathlib import Path

from native.g7_stateful_diagnosis import (
    expected_request, validate_request, aggregate, SHARDS, SAMPLES, VARIANTS,
)


class StatefulRequestTests(unittest.TestCase):
    def test_explicit_disabled_and_fixed_request(self):
        request = expected_request()
        self.assertTrue(validate_request(request))
        self.assertFalse(validate_request({**request, "enabled": False}))
        for patch in ({"samples": 16}, {"shards": 19},
                      {"training_allowed": True}, {"adoption_allowed": True}):
            with self.assertRaises(ValueError):
                validate_request({**request, **patch})

    def test_all_20_shards_and_240_rows_required(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for shard in range(SHARDS):
                records = []
                for sample in range(shard, SAMPLES, SHARDS):
                    for mode in VARIANTS:
                        records.append(dict(
                            sample=sample, shard=shard, variant=mode,
                            fixed_move="7g7f", fixed_nodes=10,
                            nodes=100, depth=2, elapsed_ms=201.0,
                            sfen_ms=0, abi_ms=0, state_legal_ms=0,
                            state_push_ms=0, state_pop_ms=0,
                            move_construct_ms=0, order_sort_ms=0, state_open_ms=0))
                (root / f"part-{shard:02d}.json").write_text(json.dumps(dict(
                    shard=shard, positions_sha256="fixed",
                    games_sha256="games", model_sha256="model",
                    seed=20271027, records=records)))
            result = aggregate(root, root / "out")
            self.assertEqual(len(result["records"]), 240)
            self.assertEqual(result["summary"]["native_stateful"]["total_nodes"], 8000)
            (root / "part-19.json").unlink()
            with self.assertRaises(ValueError):
                aggregate(root, root / "out")


if __name__ == "__main__":
    unittest.main()
