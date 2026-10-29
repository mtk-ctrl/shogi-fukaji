"""Research-only live child and history-bound legal fallback tests."""
import unittest
import shogi
from fukaji.uct_live_supervisor_probe import (run_live_supervisor_probe, LiveProbeSession)


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

    def test_one_supervisor_across_two_go_commands(self):
        session = LiveProbeSession()
        first = run_live_supervisor_probe('normal', board=self.board,
                                          session=session, respond_after_ms=30)
        second = run_live_supervisor_probe('stale', board=self.board,
                                           session=session, respond_after_ms=30)
        self.assertEqual((first.generation, second.generation), (1, 2))
        self.assertEqual(second.selected_move, second.fallback_move)
        self.assertFalse(second.candidate_accepted)
        self.assertEqual(first.outstanding_generations, ())
        self.assertEqual(second.prior_outstanding_generations, ())
        self.assertEqual(second.outstanding_generations, ())
        self.assertTrue(first.board_unchanged and second.board_unchanged)

    def test_synthetic_old_worker_ledger_survives_new_go(self):
        # This is an intentionally SYNTHETIC unreleased generation, NOT
        # a claim that a real OS child remained running.
        session = LiveProbeSession()
        model = session.model
        model.position(key='old-history', legal_moves=('7g7f',), at=0.0)
        old = model.go(at=0.0, fallback_ready_at=0.0, budget_ms=200)
        model.stop(at=0.0)
        model.confirm_sent(move='7g7f', at=0.0)
        result = run_live_supervisor_probe('normal', board=self.board,
                                           session=session, respond_after_ms=30)
        self.assertEqual(result.generation, old + 1)
        self.assertEqual(result.prior_outstanding_generations, (old,))
        self.assertEqual(result.outstanding_generations, (old,))
        self.assertFalse(result.worker_resources_released)
        self.assertTrue(result.child_reaped and result.pipes_closed)

    def test_failed_send_prevents_next_go_in_same_supervisor(self):
        session = LiveProbeSession()
        failed = run_live_supervisor_probe('normal', board=self.board,
                                           session=session, simulate_sink_closed=True,
                                           respond_after_ms=20)
        self.assertFalse(failed.sent_confirmed)
        with self.assertRaises(RuntimeError):
            run_live_supervisor_probe('normal', board=self.board,
                                      session=session, respond_after_ms=20)

    def test_flush_lag_is_observed_not_assumed_zero(self):
        r = run_live_supervisor_probe('silent', board=self.board,
                                      budget_ms=400, respond_after_ms=10)
        self.assertEqual(r.planned_response_at_ms, 10)
        self.assertIsNotNone(r.response_flush_at_ms)
        self.assertGreaterEqual(r.flush_delay_from_plan_ms, 0.0)
        self.assertAlmostEqual(r.flush_delay_from_plan_ms,
                               r.response_flush_at_ms - 10, places=6)
        self.assertAlmostEqual(r.over_budget_at_flush_ms,
                               max(0.0, r.response_flush_at_ms - 400), places=6)
        self.assertFalse(r.gui_delivery_verified)

    def test_invalid_session_rejected_before_child_spawn(self):
        with self.assertRaises(ValueError):
            run_live_supervisor_probe('normal', board=self.board, session=object())


if __name__ == '__main__':
    unittest.main()
