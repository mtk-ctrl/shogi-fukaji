"""D9 frozen tail diagnostic: strict OFF gate and legal/undo smoke."""
import unittest

from fukaji.d9_tail_probe import ORDER, expected_request, run_tail_probe, validate_request


class D9TailProbeTests(unittest.TestCase):
    def test_exact_request_rejects_changed_source_and_flags(self):
        self.assertFalse(validate_request(expected_request(False)))
        self.assertTrue(validate_request(expected_request(True)))
        for key, value in (("enabled", 1), ("training_allowed", True),
                           ("d8_source_artifact", 0), ("max_depth", 4),
                           ("tails", [[36, 225]]), ("movetime_ms", 201)):
            cfg = expected_request(False)
            cfg[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_request(cfg)

    def test_frozen_position_abba_only_changes_clock_and_restores(self):
        observed = []

        def choose(board, **kwargs):
            observed.append((board.sfen(), kwargs["time_check_each_node"],
                             kwargs["max_depth"], kwargs["movetime_ms"]))
            kwargs["telemetry"]["root_preparation_ms"] = 1.0
            return next(iter(board.legal_moves)).usi(), 2, 128

        row = {"opening": [], "moves": ["7g7f"],
               "played": [{"candidate": True, "elapsed_ms": 874.0}]}
        rows = run_tail_probe(None, [(36, 1, row)], select=choose)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({x[0] for x in observed}), 1)
        self.assertEqual([x[1] for x in observed], list(ORDER))
        self.assertEqual({x[2:] for x in observed}, {(3, 200)})
        self.assertEqual([x["original_d8_ms"] for x in rows], [874.0] * 4)


if __name__ == "__main__":
    unittest.main()
