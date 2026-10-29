"""Synthetic IPC ordering tests: no real USI, stdout or worker processes."""
import unittest

from fukaji.uct_synthetic_ipc import SyntheticIpcModel


class SyntheticIpcTests(unittest.TestCase):
    def setup_go(self, moves=('7g7f', '2g2f')):
        model = SyntheticIpcModel()
        model.position(key='history-a', legal_moves=moves, at=0.0)
        generation = model.go(at=0.01, fallback_ready_at=0.012)
        return model, generation

    def test_zero_iteration_stop_returns_legal_fallback(self):
        model, _ = self.setup_go()
        self.assertTrue(model.stop(at=0.03))
        self.assertEqual(model.pending_move(), '7g7f')
        self.assertTrue(model.confirm_sent(move='7g7f', at=0.04).response_sent_in_time)

    def test_candidate_before_stop_and_after_stop(self):
        model, generation = self.setup_go()
        self.assertTrue(model.worker_offer(generation=generation, key='history-a',
                                           move='2g2f', received_at=0.02))
        self.assertTrue(model.stop(at=0.03))
        self.assertFalse(model.worker_offer(generation=generation, key='history-a',
                                            move='7g7f', received_at=0.031))
        self.assertEqual(model.pending_move(), '2g2f')

    def test_stop_send_race_only_one_bestmove(self):
        model, _ = self.setup_go()
        self.assertTrue(model.stop(at=0.03))
        self.assertFalse(model.stop(at=0.031))
        model.confirm_sent(move='7g7f', at=0.04)
        self.assertFalse(model.stop(at=0.041))
        with self.assertRaises(RuntimeError):
            model.confirm_sent(move='7g7f', at=0.042)

    def test_send_failure_not_acknowledged(self):
        model, _ = self.setup_go()
        model.stop(at=0.03)
        model.send_failed(at=0.04)
        self.assertEqual(model.failed_sends, 1)
        with self.assertRaises(RuntimeError):
            model.supervisor.receipt()
        self.assertTrue(model.confirm_sent(move='7g7f', at=0.05).response_sent_in_time)

    def test_late_deadline_and_late_candidate(self):
        model, generation = self.setup_go()
        self.assertFalse(model.worker_offer(generation=generation, key='history-a',
                                            move='2g2f', received_at=0.21))
        self.assertTrue(model.deadline(at=0.211))
        self.assertFalse(model.confirm_sent(move='7g7f', at=0.212).response_sent_in_time)

    def test_deadline_cannot_be_anticipated(self):
        model, _ = self.setup_go()
        with self.assertRaises(ValueError):
            model.deadline(at=0.19)

    def test_old_worker_cannot_update_new_go(self):
        model, old = self.setup_go()
        model.stop(at=0.03)
        model.confirm_sent(move='7g7f', at=0.04)
        model.position(key='history-b', legal_moves=('5g5f',), at=0.05)
        new = model.go(at=0.06, fallback_ready_at=0.062)
        self.assertFalse(model.worker_offer(generation=old, key='history-a',
                                            move='7g7f', received_at=0.07))
        self.assertFalse(model.worker_offer(generation=new, key='history-a',
                                            move='7g7f', received_at=0.071))
        self.assertTrue(model.stop(at=0.08))
        self.assertEqual(model.pending_move(), '5g5f')

    def test_worker_cleanup_not_equivalent_to_send(self):
        model, generation = self.setup_go()
        model.stop(at=0.03)
        receipt = model.confirm_sent(move='7g7f', at=0.04)
        self.assertFalse(receipt.worker_stopped)
        self.assertEqual(model.outstanding_generations(), (generation,))
        self.assertFalse(model.resources_released(generation=generation, at=0.05))
        self.assertTrue(model.worker_stopped(generation=generation, at=0.06))
        self.assertTrue(model.supervisor.receipt().worker_stopped)
        self.assertEqual(model.outstanding_generations(), (generation,))
        self.assertTrue(model.resources_released(generation=generation, at=0.07))
        self.assertEqual(model.outstanding_generations(), ())

    def test_stale_worker_cleanup_does_not_stop_new_worker(self):
        model, old = self.setup_go()
        model.stop(at=0.03)
        model.confirm_sent(move='7g7f', at=0.04)
        model.position(key='history-b', legal_moves=('5g5f',), at=0.05)
        new = model.go(at=0.06, fallback_ready_at=0.062)
        self.assertTrue(model.worker_stopped(generation=old, at=0.07))
        self.assertTrue(model.resources_released(generation=old, at=0.08))
        self.assertEqual(model.outstanding_generations(), (new,))
        model.stop(at=0.09)
        self.assertFalse(model.confirm_sent(move='5g5f', at=0.1).worker_stopped)

    def test_reentrant_position_and_go_rejected_until_send(self):
        model, _ = self.setup_go()
        with self.assertRaises(RuntimeError):
            model.position(key='history-b', legal_moves=('5g5f',), at=0.02)
        with self.assertRaises(RuntimeError):
            model.go(at=0.03, fallback_ready_at=0.032)

    def test_no_go_without_position(self):
        model = SyntheticIpcModel()
        with self.assertRaises(RuntimeError):
            model.go(at=0.01, fallback_ready_at=0.012)

    def test_no_legal_moves_uses_resign_only(self):
        model, _ = self.setup_go(moves=())
        model.stop(at=0.03)
        self.assertEqual(model.pending_move(), 'resign')
        self.assertTrue(model.confirm_sent(move='resign', at=0.04).response_sent_in_time)

    def test_invalid_clock_and_move_confirmation(self):
        model, _ = self.setup_go()
        with self.assertRaises(ValueError):
            model.worker_offer(generation=1, key='history-a',
                               move='7g7f', received_at=0.011)
        model.stop(at=0.03)
        with self.assertRaises(ValueError):
            model.confirm_sent(move='2g2f', at=0.04)
        self.assertTrue(model.confirm_sent(move='7g7f', at=0.05).response_sent_in_time)

    def test_no_resource_cleanup_without_stop(self):
        model, generation = self.setup_go()
        self.assertFalse(model.resources_released(generation=generation, at=0.03))
        self.assertFalse(model.worker_stopped(generation=999, at=0.04))
        self.assertEqual(model.outstanding_generations(), (generation,))


if __name__ == '__main__':
    unittest.main()
