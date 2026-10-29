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
