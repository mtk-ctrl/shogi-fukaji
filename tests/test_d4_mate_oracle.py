"""D4: independent, bounded legal proof; never infer a mate from timeout."""
import unittest

import shogi

from fukaji.mate_oracle import ProofIncomplete, prove_mate_up_to_three

GUARDED = "8k/9/7R1/7L1/9/9/9/9/8K b - 1"
UNGUARDED = "8k/9/7R1/9/9/9/9/9/8K b - 1"


def snapshot(board):
    return (
        board.sfen(),
        tuple(move.usi() for move in board.move_stack),
        dict(board.transpositions),
        board.zobrist_hash(),
    )


class D4MateOracleTests(unittest.TestCase):
    def test_four_exhaustively_proven_one_move_mates(self):
        board = shogi.Board(GUARDED)
        before = snapshot(board)
        result = prove_mate_up_to_three(board)
        self.assertEqual(set(result["mate_in_one"]),
                         {"2c1c", "2c1c+", "2c2a+", "2c2b+"})
        self.assertEqual(result["first_moves_checked"],
                         len(tuple(board.legal_moves)))
        self.assertIn("not_best_move", result["scope"])
        self.assertEqual(snapshot(board), before)

    def test_remove_guard_removes_all_one_move_mates(self):
        board = shogi.Board(UNGUARDED)
        before = snapshot(board)
        result = prove_mate_up_to_three(board)
        self.assertEqual(result["mate_in_one"], ())
        # This specific dragon promotion can now be captured by the king.
        board.push_usi("2c2b+")
        self.assertIn(shogi.Move.from_usi("1a2b"), board.legal_moves)
        board.pop()
        self.assertEqual(snapshot(board), before)

    def test_start_position_has_no_checking_mate_in_three(self):
        board = shogi.Board()
        before = snapshot(board)
        result = prove_mate_up_to_three(board)
        self.assertEqual(result["mate_in_one"], ())
        self.assertEqual(result["mate_in_three"], ())
        self.assertEqual(snapshot(board), before)

    def test_budget_exhaustion_is_unknown_and_restores_board(self):
        board = shogi.Board(GUARDED)
        before = snapshot(board)
        with self.assertRaises(ProofIncomplete):
            prove_mate_up_to_three(board, max_legal_moves=1)
        self.assertEqual(snapshot(board), before)
        with self.assertRaises(ValueError):
            prove_mate_up_to_three(board, max_legal_moves=0)
        self.assertEqual(snapshot(board), before)

    def test_every_reported_three_ply_line_covers_all_legal_defenses(self):
        # A proof must cover every defender response; deep search agreement
        # or a single plausible mating line is NOT sufficient evidence.
        for sfen in (GUARDED, UNGUARDED):
            board = shogi.Board(sfen)
            before = snapshot(board)
            result = prove_mate_up_to_three(board)
            for proof in result["mate_in_three"]:
                first = shogi.Move.from_usi(proof["first"])
                self.assertIn(first, board.legal_moves)
                board.push(first)
                try:
                    self.assertTrue(board.is_check())
                    replies = tuple(board.legal_moves)
                    self.assertTrue(replies)
                    self.assertEqual(
                        {entry["reply"] for entry in proof["defenses"]},
                        {reply.usi() for reply in replies})
                    for entry in proof["defenses"]:
                        board.push_usi(entry["reply"])
                        try:
                            self.assertTrue(entry["mates"])
                            for finish in entry["mates"]:
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


if __name__ == "__main__":
    unittest.main()
