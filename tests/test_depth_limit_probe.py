"""D1 protocol, legal history, timing, and A/B/B/A regression tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import shogi

from fukaji.depth_limit_probe import (
    EXPECTED_REQUEST, ORDER, measure_position, select_positions, validate_request,
)
from fukaji.engine import select_move


class DepthLimitProbeTests(unittest.TestCase):
    def test_request_is_disabled_by_default_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "request.json"
            path.write_text(json.dumps(EXPECTED_REQUEST), encoding="utf-8")
            self.assertFalse(validate_request(path))
            path.write_text(json.dumps(dict(EXPECTED_REQUEST, enabled=True)), encoding="utf-8")
            self.assertTrue(validate_request(path))
            path.write_text(json.dumps(dict(EXPECTED_REQUEST, runner_jobs=2)), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_request(path)

    def test_fixed_abba_order_and_only_depth_cap_differs(self):
        self.assertEqual(ORDER, ("A", "B", "B", "A"))
        b = shogi.Board()
        item = {"prefix": [], "sfen": b.sfen()}
        legal = next(iter(b.legal_moves)).usi()
        calls = []

        def fake_select(board, *, max_depth, telemetry, **kwargs):
            calls.append((max_depth, kwargs["movetime_ms"], kwargs["fast_relative"],
                          kwargs["root_alpha_beta"]))
            telemetry.update(root_preparation_ms=0.1, internal_search_ms=0.2,
                             total_call_ms=0.3, completed_depth=1, nodes=3)
            return legal, 1, 3

        with patch("fukaji.depth_limit_probe.select_move", side_effect=fake_select):
            observed = measure_position(item, None)
        self.assertEqual([r["variant"] for r in observed], list(ORDER))
        self.assertEqual([c[0] for c in calls], [2, 6, 6, 2])
        self.assertTrue(all(c[1:] == (200, True, True) for c in calls))
        self.assertEqual(b.sfen(), shogi.Board().sfen())

    def test_actual_engine_optional_timing_and_default_unchanged(self):
        board = shogi.Board()
        before = (board.sfen(), tuple(board.move_stack))
        timing = {}
        word, depth, nodes = select_move(board, movetime_ms=15, max_depth=1,
                                         telemetry=timing)
        self.assertIn(shogi.Move.from_usi(word), board.legal_moves)
        self.assertEqual(timing["completed_depth"], depth)
        self.assertEqual(timing["nodes"], nodes)
        self.assertGreaterEqual(timing["total_call_ms"], timing["root_preparation_ms"])
        self.assertGreaterEqual(timing["total_call_ms"], timing["internal_search_ms"])
        self.assertEqual((board.sfen(), tuple(board.move_stack)), before)
        self.assertEqual(len(select_move(board, movetime_ms=15, max_depth=1)), 3)

    def test_select_does_not_fabricate_missing_categories(self):
        positions, shortages = select_positions([], quotas={
            "opening": 1, "middlegame": 1, "endgame": 1, "tactical": 1})
        self.assertEqual(positions, [])
        self.assertEqual(shortages, {
            "opening": 1, "middlegame": 1, "endgame": 1, "tactical": 1})

    def test_selection_one_parent_and_real_prefix(self):
        board = shogi.Board()
        seq = ["7g7f", "3c3d", "2g2f", "8c8d"]
        for word in seq:
            self.assertIn(shogi.Move.from_usi(word), board.legal_moves)
            board.push_usi(word)
        rows = [{"pair": 0, "g8_black": True, "opening": seq[:2],
                 "moves": seq[2:], "status": "move_limit_unresolved"}]
        with patch.dict("fukaji.depth_limit_probe.TARGET_PLIES", {"opening": 2}):
            positions, shortages = select_positions(rows, quotas={
                "opening": 1, "middlegame": 0, "endgame": 0, "tactical": 0})
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["prefix"], seq[:2])
        self.assertEqual(positions[0]["pair"], 0)
        self.assertEqual(shortages, dict.fromkeys(shortages, 0))
        self.assertEqual(positions[0]["sfen"],
                         shogi.Board().sfen() if not seq else self._after(seq[:2]))

    @staticmethod
    def _after(words):
        b = shogi.Board()
        for word in words:
            b.push_usi(word)
        return b.sfen()

    def test_timing_rejects_illegal_move(self):
        board = shogi.Board()
        item = {"prefix": [], "sfen": board.sfen()}
        with patch("fukaji.depth_limit_probe.select_move", return_value=("resign", 0, 0)):
            with self.assertRaises(ValueError):
                measure_position(item, None)


if __name__ == "__main__":
    unittest.main()
