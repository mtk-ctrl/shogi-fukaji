"""Small real-pipe probe, NOT a production USI or deadline guarantee."""
import unittest

from fukaji.uct_pipe_probe import run_pipe_probe


class PipeProbeTests(unittest.TestCase):
    def test_normal_message_received_and_child_reaped(self):
        r = run_pipe_probe('normal')
        self.assertEqual(r.accepted_move, '7g7f')
        self.assertEqual((r.lines_observed, r.rejected_messages), (1, 0))
        self.assertEqual(r.child_exit_code, 0)
        self.assertTrue(r.child_reaped and r.pipes_closed)
        self.assertFalse(r.gui_delivery_verified)

    def test_fragmented_line_reassembled(self):
        r = run_pipe_probe('fragmented')
        self.assertEqual(r.accepted_move, '7g7f')
        self.assertEqual(r.lines_observed, 1)
        self.assertEqual(r.incomplete_bytes, 0)

    def test_stale_generation_rejected(self):
        r = run_pipe_probe('stale', generation=4)
        self.assertIsNone(r.accepted_move)
        self.assertEqual(r.rejected_messages, 1)
        self.assertEqual(r.fallback_move, '7g7f')

    def test_old_position_key_rejected(self):
        r = run_pipe_probe('wrong_key')
        self.assertIsNone(r.accepted_move)
        self.assertEqual(r.rejected_messages, 1)

    def test_invalid_json_rejected(self):
        r = run_pipe_probe('malformed')
        self.assertIsNone(r.accepted_move)
        self.assertEqual(r.rejected_messages, 1)

    def test_partial_line_never_accepted(self):
        r = run_pipe_probe('truncated')
        self.assertIsNone(r.accepted_move)
        self.assertEqual(r.lines_observed, 0)
        self.assertGreater(r.incomplete_bytes, 0)

    def test_duplicate_not_double_accepted(self):
        r = run_pipe_probe('duplicate')
        self.assertEqual(r.accepted_move, '7g7f')
        self.assertEqual((r.lines_observed, r.rejected_messages), (2, 1))

    def test_child_abnormal_exit_not_success(self):
        r = run_pipe_probe('abrupt')
        self.assertIsNone(r.accepted_move)
        self.assertEqual(r.child_exit_code, 7)
        self.assertTrue(r.child_reaped and r.pipes_closed)

    def test_parent_observed_deadline_rejects_delayed_message(self):
        r = run_pipe_probe('delayed', budget_ms=1)
        self.assertIsNone(r.accepted_move)
        self.assertEqual(r.rejected_messages, 1)
        self.assertGreater(r.first_byte_read_ms, 1)

    def test_stop_before_delayed_message_rejects(self):
        r = run_pipe_probe('delayed', stop_after_ms=0)
        self.assertTrue(r.stop_observed)
        self.assertIsNone(r.accepted_move)
        self.assertEqual(r.rejected_messages, 1)

    def test_untrusted_command_and_input_rejected(self):
        with self.assertRaises(ValueError):
            run_pipe_probe('shell')
        with self.assertRaises(ValueError):
            run_pipe_probe('normal', legal_moves=('7g7f', '7g7f'))
        with self.assertRaises(ValueError):
            run_pipe_probe('normal', budget_ms=0)


if __name__ == '__main__':
    unittest.main()
