"""Lightweight protocol tests; never download G8 or start timed games in CI."""
import json
import tempfile
import unittest
from pathlib import Path

import shogi

from fukaji.uct_parent7_g8_trace import (
    ORDER, PARENT7_SFEN, REQUEST, SEED, summarize, validate_request,
)


class Parent7G8Protocol(unittest.TestCase):
    def test_fixed_sfen_has_legal_moves(self):
        board = shogi.Board(PARENT7_SFEN)
        self.assertTrue(any(board.legal_moves))
        self.assertEqual(board.sfen(), PARENT7_SFEN)

    def test_request_defaults_off_and_rejects_mutation(self):
        self.assertIs(REQUEST["enabled"], False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(REQUEST), encoding="utf-8")
            self.assertIs(validate_request(path), False)
            path.write_text(json.dumps(dict(REQUEST, enabled=True)), encoding="utf-8")
            self.assertIs(validate_request(path), True)
            path.write_text(json.dumps(dict(REQUEST, calls=80)), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_request(path)

    def test_summary_requires_frozen_order_and_phase(self):
        rows = []
        for name in ORDER:
            traced = name.startswith("traced")
            rows.append({
                "pass": name, "traced": traced, "sfen": PARENT7_SFEN,
                "seed": SEED, "target_ply": 50, "iterations": 3,
                "move": "7e8e", "external_ms": 199.0,
                "internal_ms": 198.0, "cpu_ms": 197.0,
                "phase": {"longest_iteration_ms": 1.0} if traced else None,
            })
        result = summarize(rows)
        self.assertEqual(result["trace_calls"], 4)
        self.assertEqual(result["plain_over_200ms"], 0)
        rows[0]["pass"] = "plain_a"
        with self.assertRaises(ValueError):
            summarize(rows)


    def test_summary_rejects_trace_without_phase_and_wrong_seed(self):
        rows = []
        for name in ORDER:
            traced = name.startswith("traced")
            rows.append({
                "pass": name, "traced": traced, "sfen": PARENT7_SFEN,
                "seed": SEED, "target_ply": 50, "iterations": 0,
                "move": "7e8e", "external_ms": 198.0,
                "internal_ms": 197.0, "cpu_ms": 196.0,
                "phase": {} if traced else None,
            })
        self.assertEqual(summarize(rows)["trace_calls"], 4)
        rows[0]["phase"] = None
        with self.assertRaises(ValueError):
            summarize(rows)
        rows[0]["phase"] = {}
        rows[1]["seed"] = SEED + 1
        with self.assertRaises(ValueError):
            summarize(rows)


if __name__ == "__main__":
    unittest.main()
