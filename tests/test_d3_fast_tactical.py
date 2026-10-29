"""D3 opt-in early tactical filter must be identical to full legal Q choices."""
import random
import unittest

import shogi

from fukaji.d3_tactical_probe import (
    CASES, MATE_ONE_SFEN, MATE_ONE_COUNTEREXAMPLE_SFEN,
    RECAPTURE_VARIANTS, snapshot,
)
from fukaji.engine import ordered_moves, select_move
from fukaji.quiescence import tactical_moves


def test_positions():
    result = [(name, shogi.Board(sfen)) for name, sfen, *_ in RECAPTURE_VARIANTS]
    result += [(name, shogi.Board(sfen)) for name, sfen in CASES]
    result += [("mate_one", shogi.Board(MATE_ONE_SFEN)),
               ("mate_counterexample", shogi.Board(MATE_ONE_COUNTEREXAMPLE_SFEN))]
    # Independently reached legal game states (development tests, not holdout).
    board = shogi.Board()
    rng = random.Random(20261030)
    for ply in range(36):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        if ply % 3 == 0:
            result.append(("reachable_" + str(ply), shogi.Board(board.sfen())))
    return result


class D3FastTacticalTests(unittest.TestCase):
    def test_all_tactical_legal_moves_and_stable_order_match(self):
        positions = test_positions()
        self.assertGreaterEqual(len(positions), 15)
        for name, board in positions:
            with self.subTest(case=name):
                before = snapshot(board)
                original = tactical_moves(board, ordered_moves(board))
                optimized = ordered_moves(board, tactical_only=True)
                self.assertEqual([m.usi() for m in original],
                                 [m.usi() for m in optimized])
                self.assertEqual(snapshot(board), before)
                self.assertTrue(all(m in board.legal_moves for m in optimized))

    def test_fixed_depth_search_move_nodes_and_history_match(self):
        for name, board in test_positions():
            with self.subTest(case=name):
                before = snapshot(board)
                outcomes = []
                for fast in (False, True):
                    move, depth, nodes = select_move(
                        board, movetime_ms=200, max_depth=1, model=None,
                        complete_depth=True, avoid_repetition=False,
                        fast_relative=False, root_alpha_beta=True,
                        fast_move_list=False, deadline_from_call=True,
                        quiescence=True, quiescence_fast_tactical=fast,
                    )
                    self.assertEqual(snapshot(board), before)
                    self.assertIn(shogi.Move.from_usi(move), board.legal_moves)
                    outcomes.append((move, depth, nodes))
                self.assertEqual(outcomes[0], outcomes[1])

    def test_default_and_ordinary_search_unchanged(self):
        name, sfen, *_ = RECAPTURE_VARIANTS[0]
        board = shogi.Board(sfen)
        before = snapshot(board)
        a = select_move(board, max_depth=1, complete_depth=True,
                        avoid_repetition=False, root_alpha_beta=True,
                        quiescence=True)
        b = select_move(board, max_depth=1, complete_depth=True,
                        avoid_repetition=False, root_alpha_beta=True,
                        quiescence=True, quiescence_fast_tactical=False)
        self.assertEqual(a, b)
        self.assertEqual(snapshot(board), before)

    def test_illegal_pawn_drop_mate_never_enters_tactical_list(self):
        board = shogi.Board(dict(CASES)["illegal_pawn_drop_mate"])
        self.assertNotIn(shogi.Move.from_usi("P*5b"),
                         ordered_moves(board, tactical_only=True))


if __name__ == "__main__":
    unittest.main()
