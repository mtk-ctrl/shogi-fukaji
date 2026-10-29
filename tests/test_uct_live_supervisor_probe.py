"""Research-only live child and history-bound legal fallback tests."""
import unittest
import shogi
from fukaji.uct_live_supervisor_probe import run_live_supervisor_probe


class LiveSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.board = shogi.Board()

    def test_live_candidate_before_child_reap(self):
        r = run_live_supervisor_probe('normal', board=self.board)
        self.assertTrue(r.candidate_accepted)
        self.assertEqual(r.selected_move, '7g7f')
        self.assertTrue(r.child_alive_at_response)
        self.assertTrue(r.sent_confirmed and r.private_sink_received)
        self.assertLessEqual(r.response_flush_at_ms, r.private_read_at_ms)
        self.assertFalse(r.gui_delivery_verified)
        self.assertTrue(r.child_reaped and r.pipes_closed and r.worker_resources_released)
        self.assertEqual(r.outstanding_generations, ())
        self.assertTrue(r.board_unchanged)

    def test_silent_zero_iteration_fallback(self):
        r = run_live_supervisor_probe('silent', board=self.board)
        self.assertFalse(r.candidate_accepted)
        self.assertEqual(r.selected_move, r.fallback_move)
        self.assertTrue(r.child_alive_at_response and r.private_sink_received)

    def test_delayed_candidate_does_not_block_fallback(self):
        r = run_live_supervisor_probe('delayed', board=self.board, respond_after_ms=10)
        self.assertEqual(r.selected_move, r.fallback_move)
        self.assertTrue(r.private_sink_received and r.child_reaped)

    def test_early_stop_and_lingering_worker_are_separate(self):
        r = run_live_supervisor_probe('linger', board=self.board,
                                      respond_after_ms=100, stop_after_ms=10)
        self.assertTrue(r.child_alive_at_response)
        self.assertTrue(r.private_sink_received)
        self.assertTrue(r.child_reaped and r.worker_resources_released)
        self.assertLess(r.response_flush_at_ms, r.stop_request_at_ms)

    def test_old_generation_old_position_and_corruption_rejected(self):
        for mode in ('stale', 'wrong_key', 'malformed'):
            with self.subTest(mode=mode):
                r = run_live_supervisor_probe(mode, board=self.board)
                self.assertFalse(r.candidate_accepted)
                self.assertEqual(r.selected_move, r.fallback_move)
                self.assertTrue(r.private_sink_received)

    def test_fragmented_line_waits_for_completion(self):
        r = run_live_supervisor_probe('fragmented', board=self.board)
        self.assertTrue(r.candidate_accepted)
        self.assertEqual(r.selected_move, '7g7f')

    def test_abrupt_exit_returns_legal_fallback(self):
        r = run_live_supervisor_probe('abrupt', board=self.board)
        self.assertFalse(r.candidate_accepted)
        self.assertEqual(r.selected_move, r.fallback_move)
        self.assertEqual(r.child_exit_code, 7)
        self.assertTrue(r.child_reaped)

    def test_closed_private_sink_never_claims_sent(self):
        r = run_live_supervisor_probe('normal', board=self.board,
                                      simulate_sink_closed=True)
        self.assertFalse(r.sent_confirmed or r.private_sink_received)
        self.assertIsNone(r.response_sent_in_time)
        self.assertEqual(r.failed_sends, 1)
        self.assertTrue(r.child_reaped)

    def test_sequential_go_without_child_residue(self):
        for mode in ('normal', 'silent'):
            r = run_live_supervisor_probe(mode, board=self.board)
            self.assertTrue(r.child_reaped and r.worker_resources_released)
            self.assertEqual(r.outstanding_generations, ())

    def test_checkmate_no_legal_move_returns_resign(self):
        b = shogi.Board('4k4/3GRG3/9/9/9/9/9/9/4K4 w - 1')
        self.assertTrue(b.is_checkmate())
        r = run_live_supervisor_probe('normal', board=b)
        self.assertEqual(r.fallback_move, 'resign')
        self.assertEqual(r.selected_move, 'resign')
        self.assertTrue(r.board_unchanged)

    def test_invalid_parameters_do_not_spawn(self):
        with self.assertRaises(ValueError):
            run_live_supervisor_probe('untrusted', board=self.board)
        with self.assertRaises(ValueError):
            run_live_supervisor_probe('normal', board=self.board, respond_after_ms=401)


if __name__ == '__main__':
    unittest.main()
