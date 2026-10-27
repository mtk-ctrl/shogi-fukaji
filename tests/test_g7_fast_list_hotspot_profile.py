"""FastMoveList profiling preserves fixed-depth search semantics."""
import unittest
import shogi
from fukaji.engine import select_move
from fukaji.g7_search_hotspots import expected_request, validate_request


class G7FastListProfileTests(unittest.TestCase):
    def test_frozen_read_only_protocol(self):
        request = validate_request(expected_request())
        self.assertEqual(request["samples"], 16)
        self.assertEqual(request["fixed_depth"], 2)
        self.assertFalse(request["training_allowed"])
        self.assertFalse(request["adoption_allowed"])

    def test_both_variants_preserve_fixed_depth_and_board(self):
        for moves in ((), ("7g7f", "3c3d")):
            board = shogi.Board()
            for usi in moves:
                move = shogi.Move.from_usi(usi)
                self.assertIn(move, board.legal_moves)
                board.push(move)
            before = board.sfen()
            results = []
            for fast in (False, True):
                result = select_move(board, movetime_ms=200, max_depth=2,
                                     complete_depth=True, root_alpha_beta=True,
                                     fast_move_list=fast)
                self.assertEqual(board.sfen(), before)
                self.assertEqual(result[1], 2)
                self.assertIn(shogi.Move.from_usi(result[0]), board.legal_moves)
                results.append(result)
            self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
