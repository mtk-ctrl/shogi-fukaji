"""Actual python-shogi legal fallback tests, no G8 or production USI."""
import unittest
import shogi
from fukaji.uct_response_contract import ResponseSupervisorModel
from fukaji.uct_legal_snapshot import snapshot_from_board, begin_with_legal_snapshot


class UctLegalSnapshotTests(unittest.TestCase):
    def check_fallback(self, board, expected_check=None):
        original = (board.sfen(), tuple(board.move_stack))
        supervisor = ResponseSupervisorModel()
        generation, snapshot = begin_with_legal_snapshot(
            supervisor, board, go_received_at=0.0, fallback_ready_at=0.004)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), original)
        self.assertEqual(snapshot.legal_moves,
                         tuple(m.usi() for m in board.legal_moves))
        if expected_check is not None:
            self.assertEqual(snapshot.in_check, expected_check)
        if snapshot.legal_moves:
            chosen = supervisor.prepare(at=0.1)  # no UCT iterations
            self.assertIn(shogi.Move.from_usi(chosen), board.legal_moves)
            self.assertNotEqual(chosen, 'resign')
            self.assertFalse(supervisor.offer(
                generation=generation, position_key=snapshot.position_key,
                move='not-a-move', received_at=0.11))
        else:
            chosen = supervisor.prepare(at=0.1)
            self.assertEqual(chosen, 'resign')
        receipt = supervisor.mark_sent(move=chosen, at=0.15)
        self.assertTrue(receipt.response_sent_in_time)
        self.assertEqual((board.sfen(), tuple(board.move_stack)), original)
        return snapshot

    def test_initial_position(self):
        s = self.check_fallback(shogi.Board(), expected_check=False)
        self.assertEqual(s.history_plies, 0)
        self.assertFalse(s.fourfold_repetition)

    def test_in_check_only_evasions(self):
        b = shogi.Board('4k4/9/4R4/9/9/9/9/9/4K4 w - 1')
        self.assertTrue(b.is_check())
        s = self.check_fallback(b, expected_check=True)
        self.assertTrue(s.legal_moves)

    def test_hands_and_history(self):
        b = shogi.Board()
        for usi in ('7g7f', '7c7d', '7f7e', '3c3d', '7e7d'):
            m = shogi.Move.from_usi(usi)
            self.assertIn(m, b.legal_moves)
            b.push(m)
        self.assertEqual(b.pieces_in_hand[shogi.BLACK][shogi.PAWN], 1)
        s = self.check_fallback(b)
        self.assertEqual(s.history_plies, 5)

    def test_pawn_drop_mate_is_not_a_fallback(self):
        b = shogi.Board('4k4/9/3RGR3/9/9/9/9/9/4K4 b P 1')
        forbidden = 'P*5b'
        self.assertNotIn(shogi.Move.from_usi(forbidden), b.legal_moves)
        s = self.check_fallback(b)
        self.assertNotIn(forbidden, s.legal_moves)

    def test_checkmate_has_no_legal_fallback(self):
        b = shogi.Board('4k4/3GRG3/9/9/9/9/9/9/4K4 w - 1')
        self.assertTrue(b.is_checkmate())
        s = self.check_fallback(b, expected_check=True)
        self.assertTrue(s.checkmated)
        self.assertFalse(s.legal_moves)

    def test_fourfold_repetition_requires_history_in_key(self):
        b = shogi.Board('4k4/9/9/9/9/9/9/9/4K4 b - 1')
        for usi in ('5i6i', '5a6a', '6i5i', '6a5a') * 3:
            move = shogi.Move.from_usi(usi)
            self.assertIn(move, b.legal_moves)
            b.push(move)
        self.assertTrue(b.is_fourfold_repetition())
        s = self.check_fallback(b)
        self.assertTrue(s.fourfold_repetition)
        historyless = shogi.Board(b.sfen())
        self.assertEqual(historyless.sfen(), b.sfen())
        self.assertFalse(historyless.is_fourfold_repetition())
        s2 = snapshot_from_board(historyless)
        self.assertNotEqual(s.position_key, s2.position_key)
        # A repetition observation is NOT a win/draw/training label.

    def test_reject_stale_generation_even_if_move_is_legal(self):
        supervisor = ResponseSupervisorModel()
        board = shogi.Board()
        g1, s1 = begin_with_legal_snapshot(supervisor, board,
                                            go_received_at=0.0, fallback_ready_at=0.002)
        first = supervisor.prepare(at=0.03)
        supervisor.mark_sent(move=first, at=0.04)
        board.push(shogi.Move.from_usi(first))
        g2, s2 = begin_with_legal_snapshot(supervisor, board,
                                            go_received_at=0.05, fallback_ready_at=0.052)
        self.assertNotEqual(s1.position_key, s2.position_key)
        self.assertFalse(supervisor.offer(generation=g1, position_key=s1.position_key,
                                          move=first, received_at=0.06))
        self.assertIn(supervisor.prepare(at=0.08), s2.legal_moves)
        self.assertNotEqual(g1, g2)


    def test_real_snapshot_ready_after_deadline_is_not_timely(self):
        board = shogi.Board()
        supervisor = ResponseSupervisorModel()
        _, snapshot = begin_with_legal_snapshot(
            supervisor, board, go_received_at=0.0, fallback_ready_at=0.201)
        self.assertTrue(snapshot.legal_moves)
        move = supervisor.prepare(at=0.21)
        receipt = supervisor.mark_sent(move=move, at=0.22)
        self.assertFalse(receipt.fallback_ready_in_time)
        self.assertFalse(receipt.response_sent_in_time)

if __name__ == '__main__':
    unittest.main()
