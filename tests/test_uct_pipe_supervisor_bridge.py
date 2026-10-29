"""Fixed finite real-pipe -> synthetic supervisor -> private sink integration tests."""
import unittest

from fukaji.uct_pipe_supervisor_bridge import run_pipe_supervisor_bridge


class BridgeProbeTests(unittest.TestCase):
    def test_normal_candidate_beats_distinct_fallback(self):
        r = run_pipe_supervisor_bridge('normal')
        self.assertEqual(r.candidate_from_pipe, '7g7f')
        self.assertTrue(r.candidate_accepted_by_supervisor)
        self.assertEqual(r.prepared_move, '7g7f')
        self.assertTrue(r.sent_confirmed and r.fake_sink_received)
        self.assertFalse(r.gui_delivery_verified)
        self.assertTrue(r.child_reaped and r.pipes_closed)
        self.assertTrue(r.worker_resources_released)
        self.assertEqual(r.outstanding_generations, ())

    def test_send_flush_and_private_sink_read_are_distinct(self):
        r = run_pipe_supervisor_bridge('normal')
        self.assertLessEqual(r.write_at_ms, r.flush_at_ms)
        self.assertLessEqual(r.flush_at_ms, r.fake_sink_read_at_ms)

    def test_fragmented_line_uses_complete_receipt(self):
        r = run_pipe_supervisor_bridge('fragmented')
        self.assertTrue(r.candidate_accepted_by_supervisor)
        self.assertEqual(r.prepared_move, '7g7f')

    def test_duplicate_does_not_create_second_bestmove(self):
        r = run_pipe_supervisor_bridge('duplicate')
        self.assertEqual(r.prepared_move, '7g7f')
        self.assertTrue(r.sent_confirmed)

    def test_stale_generation_uses_fallback(self):
        r = run_pipe_supervisor_bridge('stale')
        self.assertFalse(r.candidate_accepted_by_supervisor)
        self.assertEqual(r.prepared_move, '2g2f')
        self.assertTrue(r.fake_sink_received)

    def test_old_position_key_uses_fallback(self):
        r = run_pipe_supervisor_bridge('wrong_key')
        self.assertEqual(r.prepared_move, '2g2f')
        self.assertFalse(r.candidate_accepted_by_supervisor)

    def test_malformed_and_truncated_use_fallback(self):
        for mode in ('malformed', 'truncated'):
            with self.subTest(mode=mode):
                r = run_pipe_supervisor_bridge(mode)
                self.assertEqual(r.prepared_move, '2g2f')
                self.assertTrue(r.fake_sink_received)

    def test_deadline_rejects_late_pipe_message(self):
        r = run_pipe_supervisor_bridge('delayed', budget_ms=1)
        self.assertIsNone(r.candidate_from_pipe)
        self.assertEqual(r.prepared_move, '2g2f')
        self.assertFalse(r.response_sent_in_time)
        self.assertTrue(r.fake_sink_received)

    def test_stop_rejects_delayed_pipe_message(self):
        r = run_pipe_supervisor_bridge('delayed', stop_after_ms=1)
        self.assertTrue(r.stop_observed_by_pipe)
        self.assertFalse(r.candidate_accepted_by_supervisor)
        self.assertEqual(r.prepared_move, '2g2f')

    def test_abrupt_child_exit_still_fallback_and_reap(self):
        r = run_pipe_supervisor_bridge('abrupt')
        self.assertEqual(r.prepared_move, '2g2f')
        self.assertTrue(r.child_reaped and r.worker_resources_released)

    def test_broken_private_sink_is_not_confirmed(self):
        r = run_pipe_supervisor_bridge('normal', simulate_sink_closed=True)
        self.assertFalse(r.sent_confirmed)
        self.assertFalse(r.fake_sink_received)
        self.assertIsNone(r.response_sent_in_time)
        self.assertEqual(r.failed_sends, 1)
        self.assertTrue(r.worker_resources_released)

    def test_invalid_fixture_rejected(self):
        with self.assertRaises(ValueError):
            run_pipe_supervisor_bridge('unknown')
        with self.assertRaises(ValueError):
            run_pipe_supervisor_bridge('normal', simulate_sink_closed=1)


if __name__ == '__main__':
    unittest.main()
