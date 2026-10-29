"""D3 independent legal-fact and explicit-request guard regression tests."""
import json
import tempfile
from pathlib import Path
import unittest

import shogi

from fukaji.d3_tactical_probe import (
    CASES, EXPECTED, ORDER, MATE_ONE_SFEN, MATE_ONE_COUNTEREXAMPLE_SFEN,
    enumerate_mate_in_one, mate_choice_diagnostic,
    poison_capture_choice_diagnostic,
    frozen_proofs, snapshot, validate_request, verify_case,
)


class D3TacticalProbeTests(unittest.TestCase):
    def test_five_frozen_cases_and_unique_sfen(self):
        proofs = frozen_proofs()
        self.assertEqual(len(proofs), 5)
        self.assertEqual(len({x["sfen"] for x in proofs}), 5)
        self.assertTrue(all("not a unique best move" in x["oracle_scope"]
                            for x in proofs))

    def test_oracles_restore_history(self):
        for name, sfen in CASES:
            with self.subTest(case=name):
                board = shogi.Board(sfen)
                before = snapshot(board)
                verify_case(name, board)
                self.assertEqual(snapshot(board), before)

    def test_poison_capture_has_forced_legal_recapture(self):
        board = shogi.Board(dict(CASES)["poison_recapture"])
        board.push(shogi.Move.from_usi("5e5d"))
        self.assertIn(shogi.Move.from_usi("5c5d"), board.legal_moves)

    def test_pawn_drop_mate_is_illegal(self):
        board = shogi.Board(dict(CASES)["illegal_pawn_drop_mate"])
        self.assertNotIn(shogi.Move.from_usi("P*5b"), board.legal_moves)

    def test_independent_mate_in_one_and_defensive_counterexample(self):
        """Enumerate every legal first move, then every legal evasion.

        The lance defends 2b after the rook vacates 2c. Removing it allows
        the defending king to capture the promoted rook; these are constructed
        development positions, not unseen strength-test examples.
        """
        mate_sfen = "8k/9/7R1/7L1/9/9/9/9/8K b - 1"
        board = shogi.Board(mate_sfen)
        before = snapshot(board)
        self.assertFalse(board.is_check())
        expected = shogi.Move.from_usi("2c2b+")
        self.assertIn(expected, board.legal_moves)
        mates = []
        for move in list(board.legal_moves):
            board.push(move)
            try:
                if board.is_checkmate():
                    self.assertTrue(board.is_check())
                    self.assertEqual(list(board.legal_moves), [])
                    mates.append(move.usi())
            finally:
                board.pop()
        self.assertIn(expected.usi(), mates)
        self.assertEqual(snapshot(board), before)

        # With no defending lance, the king can legally capture on 2b.
        unguarded = shogi.Board("8k/9/7R1/9/9/9/9/9/8K b - 1")
        unguarded_before = snapshot(unguarded)
        self.assertIn(expected, unguarded.legal_moves)
        unguarded.push(expected)
        try:
            self.assertFalse(unguarded.is_checkmate())
            self.assertIn(shogi.Move.from_usi("1a2b"), unguarded.legal_moves)
        finally:
            unguarded.pop()
        self.assertEqual(snapshot(unguarded), unguarded_before)

    def test_full_width_mate_in_one_oracle_and_counterexample(self):
        guarded = shogi.Board(MATE_ONE_SFEN)
        unguarded = shogi.Board(MATE_ONE_COUNTEREXAMPLE_SFEN)
        guarded_before, unguarded_before = snapshot(guarded), snapshot(unguarded)
        guarded_mates = enumerate_mate_in_one(guarded)
        unguarded_mates = enumerate_mate_in_one(unguarded)
        self.assertEqual(set(guarded_mates),
                         {"2c1c", "2c1c+", "2c2a+", "2c2b+"})
        self.assertEqual(unguarded_mates, ())
        self.assertEqual(len(guarded_mates), len(set(guarded_mates)))
        self.assertEqual(snapshot(guarded), guarded_before)
        self.assertEqual(snapshot(unguarded), unguarded_before)

    def test_actual_depth_one_search_choice_against_independent_oracle(self):
        diagnostics = mate_choice_diagnostic()
        self.assertEqual([x["case"] for x in diagnostics],
                         ["guarded_mate", "unguarded_counterexample"])
        self.assertEqual(len(diagnostics[0]["choices"]), 2)
        self.assertEqual(len(diagnostics[1]["choices"]), 2)
        for case in diagnostics:
            self.assertEqual([x["quiescence"] for x in case["choices"]],
                             [False, True])
            self.assertEqual(case["model"], "classical_no_G8")
            self.assertEqual(case["clock"], "fixed_depth_1_unbounded_not_200ms")
            for choice in case["choices"]:
                self.assertEqual(choice["depth"], 1)
                self.assertGreater(choice["nodes"], 0)
        # Both variants already recognize immediate checkmate at the normal
        # negamax leaf. This test must not be cited as a D3 strength gain.
        for choice in diagnostics[0]["choices"]:
            self.assertTrue(choice["is_proven_mate_in_one"])

    def test_actual_poison_recapture_choice_is_legal_and_restores_history(self):
        diagnostic = poison_capture_choice_diagnostic()
        self.assertEqual(diagnostic["case"], "poison_recapture")
        self.assertEqual(diagnostic["known_recapture"], "5e5d then 5c5d")
        self.assertEqual(diagnostic["model"], "classical_no_G8")
        self.assertEqual(diagnostic["clock"], "fixed_depth_unbounded_not_200ms")
        self.assertEqual([x["variant"] for x in diagnostic["results"]],
                         ["A_depth1_no_q", "B_depth1_with_q",
                          "control_depth2_no_q"])
        self.assertEqual([x["depth"] for x in diagnostic["results"]], [1, 1, 2])
        for observation in diagnostic["results"]:
            self.assertGreater(observation["nodes"], 0)
            self.assertIsInstance(observation["takes_defended_silver"], bool)
        self.assertEqual([x["move"] for x in diagnostic["results"]],
                         ["5e5d", "5e7e", "5e7e"])
        self.assertEqual([x["takes_defended_silver"] for x in diagnostic["results"]],
                         [True, False, False])
        # A legal recapture is a tactical risk, not proof of a unique best move.


    def test_multiple_legal_recapture_oracles_and_history(self):
        from fukaji.d3_tactical_probe import RECAPTURE_VARIANTS, recapture_legal_oracle
        self.assertEqual(len(RECAPTURE_VARIANTS), 4)
        self.assertEqual(len({row[1] for row in RECAPTURE_VARIANTS}), 4)
        for name, sfen, capture, reply, prey in RECAPTURE_VARIANTS:
            with self.subTest(case=name):
                result = recapture_legal_oracle(name, sfen, capture, reply, prey)
                self.assertEqual(result["capture"], capture)
                self.assertEqual(result["legal_recapture"], reply)
                self.assertEqual(result["scope"],
                                 "legal_two_ply_recapture_only_not_optimal_move")

    def test_multiple_tactical_choices_are_observed_not_assumed(self):
        from fukaji.d3_tactical_probe import tactical_variation_diagnostic
        observations = tactical_variation_diagnostic()
        self.assertEqual(len(observations), 6)
        self.assertEqual(len({x["sfen"] for x in observations}), 6)
        self.assertEqual({x["case"] for x in observations},
                         {"silver_file5", "pawn_file5", "bishop_diagonal",
                          "silver_reversed", "check_evasion", "quiet_promotion"})
        for case in observations:
            self.assertEqual([x["variant"] for x in case["observations"]],
                             ["A_depth1_no_q", "B_depth1_with_q",
                              "control_depth2_no_q"])
            self.assertEqual(case["model"], "classical_no_G8")
            self.assertEqual(case["clock"], "fixed_depth_unbounded_not_200ms")
            for item in case["observations"]:
                self.assertGreater(item["nodes"], 0)
                self.assertGreaterEqual(item["wall_ms"], 0)
                self.assertIn(item["depth"], (0, 1, 2))
            if case["case"] in ("check_evasion", "quiet_promotion"):
                self.assertTrue(all(
                    x["takes_recapturable_piece"] is None
                    for x in case["observations"]))
            else:
                self.assertTrue(all(
                    isinstance(x["takes_recapturable_piece"], bool)
                    for x in case["observations"]))
        # Machine-readable observations are for diagnosis, not proof of Elo.
        print("D3_VARIATIONS_JSON=" + json.dumps(observations, sort_keys=True))


    def test_order_is_abba(self):
        self.assertEqual(ORDER, ("A", "B", "B", "A"))

    def test_request_is_disabled_by_default_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "request.json"
            path.write_text(json.dumps(EXPECTED))
            self.assertFalse(validate_request(path))
            path.write_text(json.dumps(dict(EXPECTED, enabled=True)))
            self.assertTrue(validate_request(path))
            path.write_text(json.dumps(dict(EXPECTED, movetime_ms=201)))
            with self.assertRaises(ValueError):
                validate_request(path)


if __name__ == "__main__":
    unittest.main()
