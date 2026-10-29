"""D4 development generator invariants, independent of the candidate's success."""
import unittest

import shogi

from fukaji.d4_three_only_probe import (
    SEED, positions, to_sfen, discover, snapshot,
)
from fukaji.mate_oracle import prove_mate_up_to_three


class ThreeOnlyDiscoveryTests(unittest.TestCase):
    def test_sfen_builder_and_seed_reproducible(self):
        self.assertEqual(to_sfen({"1a": "k", "2c": "R",
                                  "2d": "L", "9i": "K"}),
                         "8k/9/7R1/7L1/9/9/9/9/K8 b - 1")
        self.assertEqual(list(positions(limit=3)), list(positions(limit=3)))
        self.assertEqual(SEED, 20261030)

    def test_two_frozen_three_only_positions_are_fully_proven(self):
        # Discovery-set fixtures, NOT held-out. These are independent seeded
        # layouts rather than transformations of the previous rook/lance board.
        cases = (
            ("8k/9/9/8N/4N1R2/9/9/9/K8 b - 1",
             {"3e3a+": {"1a1b": {"3a2b"}}}),
            ("5g2k/7L1/8s/4N4/4B2L1/9/9/9/K8 b - 1",
             {"2b2a+": {"1a1b": {"5e1a+", "2a1a"}}}),
        )
        for sfen, expected in cases:
            with self.subTest(sfen=sfen):
                board = shogi.Board(sfen)
                before = snapshot(board)
                proof = prove_mate_up_to_three(board)
                self.assertEqual(proof["mate_in_one"], ())
                found = {
                    p["first"]: {
                        d["reply"]: set(d["mates"]) for d in p["defenses"]
                    } for p in proof["mate_in_three"]
                }
                self.assertEqual(found, expected)
                # Independently replay every reported legal defense and mate.
                for first, defenses in found.items():
                    self.assertIn(shogi.Move.from_usi(first), board.legal_moves)
                    board.push_usi(first)
                    try:
                        self.assertTrue(board.is_check())
                        self.assertEqual(set(defenses),
                                         {m.usi() for m in board.legal_moves})
                        for reply, finishes in defenses.items():
                            board.push_usi(reply)
                            try:
                                for finish in finishes:
                                    self.assertIn(shogi.Move.from_usi(finish),
                                                  board.legal_moves)
                                    board.push_usi(finish)
                                    try:
                                        self.assertTrue(board.is_check())
                                        self.assertFalse(any(board.legal_moves))
                                    finally:
                                        board.pop()
                            finally:
                                board.pop()
                    finally:
                        board.pop()
                self.assertEqual(snapshot(board), before)

    def test_counterexample_without_primary_attacker(self):
        # Deliberately remove the rook from the first discovered layout.
        board = shogi.Board("8k/9/9/8N/4N4/9/9/9/K8 b - 1")
        before = snapshot(board)
        proof = prove_mate_up_to_three(board)
        self.assertEqual(proof["mate_in_one"], ())
        self.assertEqual(proof["mate_in_three"], ())
        self.assertEqual(snapshot(board), before)

    def test_probe_has_bounded_search_and_restores_board(self):
        report = discover(max_positions=3, max_legal_moves=1000, max_examples=1)
        self.assertEqual(report["positions_attempted"], 3)
        self.assertEqual(sum(report["sources"].values()), 3)
        self.assertEqual(
            report["fully_proved"] + report["incomplete"] + report["invalid"], 3)
        for row in report["three_only"]:
            board = shogi.Board(row["sfen"])
            before = snapshot(board)
            proof = prove_mate_up_to_three(board)
            self.assertEqual(proof["mate_in_one"], ())
            self.assertTrue(proof["mate_in_three"])
            self.assertEqual(snapshot(board), before)
            self.assertEqual(len(row["search"]), 2)
            self.assertTrue(all(s["completed_depth"] == s["depth"]
                                for s in row["search"]))


if __name__ == "__main__":
    unittest.main()
