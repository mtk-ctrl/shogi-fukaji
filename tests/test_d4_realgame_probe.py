"""D4 real-game frozen source gate; no network, games, or model training."""
import unittest

import shogi

from fukaji.d4_realgame_probe import (
    SAMPLES, STAGES, SEED, expected_request, validate_request,
    freeze_cases, board_from_moves, verify_proof,
)
from fukaji.d4_three_only_probe import snapshot
from fukaji.mate_oracle import prove_mate_up_to_three


class D4RealgameProbeTests(unittest.TestCase):
    def test_request_is_disabled_by_default_and_exact(self):
        self.assertFalse(validate_request(expected_request()))
        self.assertTrue(validate_request(expected_request(True)))
        invalid = expected_request(True)
        invalid["movetime_ms"] = 201
        with self.assertRaises(ValueError):
            validate_request(invalid)

    def test_parent_separation_and_fixed_stage_counts(self):
        games = [
            {"pair": i, "g8_black": True, "status": "checkmate",
             "opening": ["7g7f"], "moves": ["3c3d"] * 50}
            for i in range(120)
        ]
        cases = freeze_cases(games)
        self.assertEqual(len(cases), SAMPLES)
        self.assertEqual(len({c["pair"] for c in cases}), SAMPLES)
        self.assertEqual([c["stage"] for c in cases],
                         [stage for stage in STAGES for _ in range(6)])
        self.assertEqual(cases, freeze_cases(games))
        self.assertEqual(SEED, "D4-20261030-first-proof-holdout-v1")

    def test_legal_history_and_illegal_history(self):
        board = board_from_moves(["7g7f", "3c3d"])
        self.assertEqual(len(board.move_stack), 2)
        with self.assertRaises(ValueError):
            board_from_moves(["7g7f", "7g7f"])

    def test_proof_witness_and_restoration(self):
        board = shogi.Board("8k/9/7R1/7L1/9/9/9/9/8K b - 1")
        before = snapshot(board)
        proof = prove_mate_up_to_three(board, max_legal_moves=1000)
        self.assertTrue(proof["mate_in_one"])
        self.assertTrue(proof["mate_in_three"])
        verify_proof(board, proof)
        self.assertEqual(snapshot(board), before)


if __name__ == "__main__":
    unittest.main()
