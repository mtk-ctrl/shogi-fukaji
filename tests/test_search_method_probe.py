"""Small deterministic selection checks for the read-only search probe."""
import unittest

import shogi

from fukaji.search_method_probe import build_positions, _board_from_prefix


class SearchMethodProbeTests(unittest.TestCase):
    def test_selection_restores_actual_history(self):
        seq = ["7g7f", "7c7d", "7f7e", "3c3d", "7e7d"]
        row = {"pair": 0, "g8_black": True, "opening": seq[:2],
               "moves": seq[2:], "status": "move_limit_unresolved"}
        result = build_positions([row], targets=(4,), per_stage=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["target_ply"], 4)
        board = _board_from_prefix(result[0]["prefix"])
        self.assertEqual(board.sfen(), result[0]["sfen"])
        self.assertEqual(len(board.move_stack), 4)

    def test_rejects_illegal_history(self):
        with self.assertRaises(ValueError):
            _board_from_prefix(["7g7f", "7g7f"])

    def test_no_reuse_of_same_parent(self):
        seq = ["7g7f", "7c7d", "7f7e", "3c3d", "7e7d"]
        row = {"pair": 0, "g8_black": True, "opening": seq[:2],
               "moves": seq[2:], "status": "move_limit_unresolved"}
        with self.assertRaises(ValueError):
            build_positions([row], targets=(2, 4), per_stage=1)


if __name__ == "__main__":
    unittest.main()
