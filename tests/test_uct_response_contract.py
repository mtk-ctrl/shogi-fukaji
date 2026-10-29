"""Synthetic-clock protocol tests; no USI, G8 or real shogi legality."""
import unittest
from fukaji.uct_response_contract import ResponseSupervisorModel


class ResponseContractTests(unittest.TestCase):
    def begin(self, moves=('7g7f', '2g2f'), ready=0.002):
        s = ResponseSupervisorModel()
        g = s.begin(position_key='board-a', legal_moves=moves,
                    go_received_at=0.0, fallback_ready_at=ready)
        return s, g

    def test_zero_iterations_legal_fallback(self):
        s, _ = self.begin()
        move = s.prepare(at=0.18)
        self.assertEqual(move, '7g7f')
        r = s.mark_sent(move=move, at=0.19)
        self.assertTrue(r.response_sent_in_time)
        self.assertFalse(r.worker_stopped)

    def test_early_update_and_late_update(self):
        s, g = self.begin()
        self.assertTrue(s.offer(generation=g, position_key='board-a',
                                move='2g2f', received_at=0.05))
        self.assertFalse(s.offer(generation=g, position_key='board-a',
                                 move='7g7f', received_at=0.201))
        move = s.prepare(at=0.202)
        self.assertEqual(move, '2g2f')
        self.assertFalse(s.mark_sent(move=move, at=0.203).response_sent_in_time)

    def test_stale_position_and_illegal_offer(self):
        s, g = self.begin()
        for generation, key, move, now in (
                (g-1, 'board-a', '2g2f', 0.01),
                (g, 'board-b', '2g2f', 0.02),
                (g, 'board-a', 'illegal', 0.03)):
            self.assertFalse(s.offer(generation=generation, position_key=key,
                                     move=move, received_at=now))
        self.assertEqual(s.prepare(at=0.04), '7g7f')

    def test_terminal_only_resign(self):
        s, g = self.begin(moves=())
        self.assertFalse(s.offer(generation=g, position_key='board-a',
                                 move='7g7f', received_at=0.02))
        move = s.prepare(at=0.03)
        self.assertEqual(move, 'resign')
        self.assertTrue(s.mark_sent(move=move, at=0.04).response_sent_in_time)

    def test_fallback_ready_after_deadline_is_not_success(self):
        s, _ = self.begin(ready=0.21)
        move = s.prepare(at=0.211)
        r = s.mark_sent(move=move, at=0.212)
        self.assertFalse(r.fallback_ready_in_time)
        self.assertFalse(r.response_sent_in_time)

    def test_prepared_not_sent_and_no_duplicate(self):
        s, _ = self.begin()
        move = s.prepare(at=0.18)
        with self.assertRaises(RuntimeError):
            s.receipt()
        with self.assertRaises(ValueError):
            s.mark_sent(move='2g2f', at=0.181)
        self.assertTrue(s.mark_sent(move=move, at=0.19).response_sent_in_time)
        with self.assertRaises(RuntimeError):
            s.mark_sent(move=move, at=0.191)

    def test_cleanup_is_separate(self):
        s, g = self.begin()
        move = s.prepare(at=0.18)
        self.assertFalse(s.mark_sent(move=move, at=0.19).worker_stopped)
        self.assertTrue(s.mark_worker_stopped(generation=g, at=0.3))
        self.assertTrue(s.receipt().worker_stopped)

    def test_old_worker_after_new_position(self):
        s, g = self.begin()
        move = s.prepare(at=0.04)
        s.mark_sent(move=move, at=0.05)
        g2 = s.begin(position_key='board-b', legal_moves=('5g5f',),
                     go_received_at=1.0, fallback_ready_at=1.002)
        self.assertNotEqual(g, g2)
        self.assertFalse(s.offer(generation=g, position_key='board-a',
                                 move='7g7f', received_at=1.01))
        self.assertEqual(s.prepare(at=1.02), '5g5f')

    def test_no_reentrant_go_before_send(self):
        s, _ = self.begin()
        with self.assertRaises(RuntimeError):
            s.begin(position_key='new', legal_moves=('5g5f',),
                    go_received_at=1.0, fallback_ready_at=1.002)

    def test_nonmonotonic_clock_and_invalid_snapshot(self):
        s, g = self.begin()
        with self.assertRaises(ValueError):
            s.offer(generation=g, position_key='board-a', move='7g7f',
                    received_at=0.001)
        with self.assertRaises(ValueError):
            s.prepare(at=float('nan'))
        with self.assertRaises(ValueError):
            s.begin(position_key='b', legal_moves=('x', 'x'),
                    go_received_at=1.0, fallback_ready_at=1.01)

    def test_no_update_after_preparation(self):
        s, g = self.begin()
        move = s.prepare(at=0.1)
        self.assertFalse(s.offer(generation=g, position_key='board-a',
                                 move='2g2f', received_at=0.11))
        self.assertEqual(move, s.prepare(at=0.12))

    def test_new_go_must_follow_prior_send(self):
        s, _ = self.begin()
        move = s.prepare(at=0.18)
        s.mark_sent(move=move, at=0.19)
        with self.assertRaises(ValueError):
            s.begin(position_key='b', legal_moves=('5g5f',),
                    go_received_at=0.1, fallback_ready_at=0.11)


if __name__ == '__main__':
    unittest.main()
