"""Real finite OS child overlap is distinct from synthetic resource ledgers."""
import unittest
from fukaji.uct_os_overlap_probe import run_os_overlap_probe


class OsOverlapTests(unittest.TestCase):
    def test_two_real_children_overlap_and_are_reaped(self):
        result = run_os_overlap_probe()
        self.assertNotEqual(result.old_pid, result.new_pid)
        self.assertTrue(result.old_alive_when_new_go)
        self.assertTrue(result.simultaneous_alive)
        self.assertEqual(result.old_outstanding_at_new_go, (1,))
        self.assertEqual(result.children_reaped, (True, True))
        self.assertEqual(result.child_exit_codes, (0, 0))
        self.assertEqual(result.pipes_closed, (True, True))
        self.assertEqual(result.final_outstanding, ())

    def test_old_os_child_message_is_rejected_by_generation(self):
        result = run_os_overlap_probe()
        self.assertIn(1, result.observed_generations)
        self.assertTrue(result.old_offer_rejected)
        self.assertTrue(all(result.private_responses_received))
        self.assertFalse(result.gui_delivery_verified)

    def test_actual_flush_lag_not_a_deadline_guarantee(self):
        result = run_os_overlap_probe(planned_ms=50)
        self.assertEqual(result.new_planned_response_ms, 50)
        self.assertAlmostEqual(result.new_flush_lag_ms,
                               result.new_response_flush_ms - 50, places=5)
        self.assertTrue(all(result.children_reaped))

    def test_invalid_arguments_do_not_spawn(self):
        for kwargs in ({'planned_ms': 401}, {'budget_ms': 0},
                       {'budget_ms': True}, {'planned_ms': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_os_overlap_probe(**kwargs)


if __name__ == '__main__':
    unittest.main()
