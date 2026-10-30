"""D9 gated clock-probe smoke tests without network or G8 training."""
import unittest

import shogi

from fukaji.d9_clock_probe import ORDER, SEEDS, expected_request, run_probe, summarize, validate_request


class D9ClockProbeTests(unittest.TestCase):
    def test_disabled_default_and_exact_protocol(self):
        self.assertFalse(validate_request(expected_request(False)))
        self.assertTrue(validate_request(expected_request(True)))
        for key, value in (("enabled", 1), ("training_allowed", True),
                           ("movetime_ms", 201), ("max_depth", 4),
                           ("seeds", [1, 2, 3]), ("order", [True, False, True, False])):
            cfg = expected_request(False)
            cfg[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_request(cfg)

    def test_abba_has_same_board_and_legal_moves_with_only_clock_option(self):
        observed = []

        def choose(board, **kwargs):
            observed.append((board.sfen(), kwargs["time_check_each_node"],
                             kwargs["max_depth"], kwargs["movetime_ms"]))
            kwargs["telemetry"]["root_preparation_ms"] = 0.5
            return next(iter(board.legal_moves)).usi(), 2, 123

        rows = run_probe(None, select=choose)
        self.assertEqual(len(rows), len(SEEDS) * len(ORDER))
        for index, seed in enumerate(SEEDS):
            block = observed[index * 4:index * 4 + 4]
            self.assertEqual(len({x[0] for x in block}), 1)
            self.assertEqual([x[1] for x in block], list(ORDER))
            self.assertEqual({x[2:] for x in block}, {(3, 200)})
            self.assertEqual([x["seed"] for x in rows[index * 4:index * 4 + 4]], [seed] * 4)
        summary = summarize(rows)
        self.assertEqual(summary["each_node"]["calls"], 6)
        self.assertEqual(summary["every_128_nodes"]["calls"], 6)
        self.assertEqual(summary["each_node"]["depth3_complete"], 0)


if __name__ == "__main__":
    unittest.main()
