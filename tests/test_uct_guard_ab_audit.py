"""Offline regression tests for read-only UCT guard ABBA audit."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fukaji.uct_guard_ab_audit import (
    REQUEST, SOURCE_SHA, MODEL_SHA, audit, validate_request,
)


def make_run(root, elapsed, *, alter=False):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(12):
        ply = (12, 50, 110)[i // 4]
        rows.append({
            "pair": i, "target_ply": ply,
            "sfen": "fixed-position-" + str(i if not alter else i+1),
            "hand_pieces": i, "g8_eval_cp": i,
            "in_check": False,
            "methods": {"uct": {
                "elapsed_ms": elapsed, "iterations": 20,
                "move": "7g7f",
            }},
        })
    raw = "".join(json.dumps(r)+"\n" for r in rows).encode("utf-8")
    (root / "position_diagnostics.jsonl").write_bytes(raw)
    summary = {
        "status": "read_only_search_diagnostic_not_strength_or_adoption",
        "diagnostics_sha256": hashlib.sha256(raw).hexdigest(),
        "source_games_sha256": SOURCE_SHA, "g8_model_sha256": MODEL_SHA,
        "source_run": 40637486947, "g8_model_run": 40636602223,
        "positions": 12, "movetime_ms_each": 200,
        "unresolved_source_games_used_as_teacher": False,
        "model_promoted": False, "official_kumoji_changed": False,
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


class GuardAbbaAuditTests(unittest.TestCase):
    def test_request_is_default_off_and_strict(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "request.json"
            path.write_text(json.dumps(REQUEST), encoding="utf-8")
            self.assertFalse(validate_request(path))
            path.write_text(json.dumps(dict(REQUEST, enabled=True)), encoding="utf-8")
            self.assertTrue(validate_request(path))
            path.write_text(json.dumps(dict(REQUEST, positions=13)), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_request(path)

    def test_four_runs_are_paired_and_sha_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, elapsed in (("base-a", 205.0), ("guard-a", 196.0),
                                  ("guard-b", 198.0), ("base-b", 207.0)):
                make_run(root / name, elapsed)
            summary = audit(*(root / n for n in ("base-a", "guard-a", "guard-b",
                                                  "base-b")), root / "out")
            self.assertEqual(summary["baseline_over_200ms"], 24)
            self.assertEqual(summary["guard_over_200ms"], 0)
            self.assertEqual(summary["guard_faster_positions"], 12)
            self.assertEqual(summary["observations_per_variant"], 24)
            raw = (root / "out" / "abba_positions.jsonl").read_bytes()
            self.assertEqual(summary["abba_positions_sha256"],
                             hashlib.sha256(raw).hexdigest())

    def test_rejects_changed_position_and_corrupt_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("base-a", "guard-a", "guard-b", "base-b"):
                make_run(root / name, 200.0, alter=name == "guard-b")
            with self.assertRaisesRegex(ValueError, "position"):
                audit(*(root / n for n in ("base-a", "guard-a", "guard-b",
                                            "base-b")), root / "out")
            make_run(root / "guard-b", 200.0)
            with (root / "guard-b" / "position_diagnostics.jsonl").open("ab") as out:
                out.write(b"\n")
            with self.assertRaisesRegex(ValueError, "SHA"):
                audit(*(root / n for n in ("base-a", "guard-a", "guard-b",
                                            "base-b")), root / "out")


if __name__ == "__main__":
    unittest.main()
