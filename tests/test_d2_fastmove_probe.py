"""D2 request safety, same-clock ABBA and history-preserving parity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import shogi

from fukaji.d2_fastmove_probe import (
    EXPECTED_REQUEST, ORDER, fixed_depth_parity, measure_position,
    validate_request,
)


class D2FastMoveTests(unittest.TestCase):
    def test_default_disabled_and_frozen_request(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "request.json"
            p.write_text(json.dumps(EXPECTED_REQUEST))
            self.assertFalse(validate_request(p))
            p.write_text(json.dumps(dict(EXPECTED_REQUEST, enabled=True)))
            self.assertTrue(validate_request(p))
            p.write_text(json.dumps(dict(EXPECTED_REQUEST, runner_jobs=2)))
            with self.assertRaises(ValueError):
                validate_request(p)

    def test_abba_differs_only_fast_move_list(self):
        board = shogi.Board()
        item = {"prefix": [], "sfen": board.sfen()}
        legal = next(iter(board.legal_moves)).usi()
        args = []

        def fake_select(board, *, telemetry, **kwargs):
            args.append(kwargs)
            telemetry.update(root_preparation_ms=0.1, internal_search_ms=0.1,
                             total_call_ms=0.2, completed_depth=1, nodes=2)
            return legal, 1, 2

        with patch("fukaji.d2_fastmove_probe.select_move", side_effect=fake_select):
            rows = measure_position(item, None)
        self.assertEqual([r["variant"] for r in rows], list(ORDER))
        self.assertEqual([x["fast_move_list"] for x in args],
                         [False, True, True, False])
        for x in args:
            self.assertTrue(x["deadline_from_call"])
            self.assertTrue(x["fast_relative"])
            self.assertTrue(x["root_alpha_beta"])
            self.assertEqual((x["movetime_ms"], x["max_depth"]), (200, 6))
        self.assertEqual(board.sfen(), shogi.Board().sfen())

    def test_parity_detects_node_mismatch(self):
        b = shogi.Board()
        p = {"category": "opening", "pair": 1, "prefix": [], "sfen": b.sfen()}
        legal = next(iter(b.legal_moves)).usi()
        with patch("fukaji.d2_fastmove_probe.select_move",
                   side_effect=[(legal, 2, 10), (legal, 2, 11)]):
            with self.assertRaises(AssertionError):
                fixed_depth_parity([p], None, per_category=1)

    def test_actual_fixed_depth_one_preserves_history(self):
        b = shogi.Board()
        before = (b.sfen(), tuple(b.move_stack))
        a = __import__("fukaji.engine", fromlist=["select_move"]).select_move(
            b, max_depth=1, complete_depth=True, fast_move_list=False,
            deadline_from_call=True)
        c = __import__("fukaji.engine", fromlist=["select_move"]).select_move(
            b, max_depth=1, complete_depth=True, fast_move_list=True,
            deadline_from_call=True)
        self.assertEqual(a, c)
        self.assertEqual((b.sfen(), tuple(b.move_stack)), before)


if __name__ == "__main__":
    unittest.main()
