"""History reuse must preserve rules, mixed-model values and search undo."""
import copy
import random
import unittest

import numpy as np
import shogi

from fukaji.online_learning import fresh_model
from fukaji.online_search import Position, select
from fukaji.online_cycle import play


class OnlineHistoryTests(unittest.TestCase):
    def test_mixed_model_history_has_independent_accumulators_and_search_undo(self):
        first, second = fresh_model(51), fresh_model(52)
        b = shogi.Board()
        history = Position(b, first)
        rng = random.Random(13)
        for _ in range(48):
            history.push(rng.choice(list(b.legal_moves)))
        keys, checks, acc = list(history.keys), list(history.checks), history.acc.copy()
        reference = select(b, second, fixed_depth=True, max_depth=2)
        candidate = select(b, second, fixed_depth=True, max_depth=2, history=history)
        for key in ('move', 'depth', 'nodes', 'evaluations', 'score', 'terminal'):
            self.assertEqual(reference[key], candidate[key])
        self.assertEqual((history.keys, history.checks), (keys, checks))
        np.testing.assert_array_equal(history.acc, acc)
        history.verify()
        reused = Position(b, second, history=history)
        full = Position(b, second)
        np.testing.assert_array_equal(reused.acc, full.acc)
        self.assertFalse(np.shares_memory(reused.acc, history.acc))
        self.assertIsNot(reused.keys, history.keys)
        self.assertIsNot(reused.checks, history.checks)

    def test_draw_and_perpetual_check_keep_complete_played_history(self):
        model = fresh_model(5)
        for sfen, words, reason, winner in [
            ('4k4/9/9/9/9/9/9/9/4K4 b - 1',
             ['5i4i', '5a4a', '4i5i', '4a5a'], 'repetition_draw', None),
            ('4k4/9/5R3/9/9/9/9/9/4K4 b - 1',
             ['4c5c', '5a4a', '5c4c', '4a5a'], 'perpetual_check', 1)]:
            b = shogi.Board(sfen)
            history = Position(b, model)
            for word in words * 3:
                history.push(shogi.Move.from_usi(word))
            expected = {'status': reason, 'winner': winner}
            self.assertEqual(Position(b, model, history=history).repetition(), expected)
            self.assertEqual(select(b, model, history=history)['terminal'], expected)

    def test_unrelated_or_stale_history_is_rejected(self):
        model = fresh_model(7)
        b = shogi.Board()
        history = Position(b, model)
        with self.assertRaises(ValueError):
            select(copy.deepcopy(b), model, history=history)
        b.push(shogi.Move.from_usi('7g7f'))
        with self.assertRaises(ValueError):
            select(b, model, history=history)

    def test_timeout_preserves_played_history_and_both_accumulators(self):
        model = fresh_model(9)
        b = shogi.Board()
        history = Position(b, model)
        for word in ['7g7f', '3c3d', '2g2f', '8c8d']:
            history.push(shogi.Move.from_usi(word))
        before = (b.sfen(), +b.transpositions, list(history.keys), list(history.checks))
        row = select(b, model, movetime_ms=1, history=history)
        self.assertIn(shogi.Move.from_usi(row['move']), b.legal_moves)
        self.assertEqual((b.sfen(), +b.transpositions, history.keys, history.checks), before)
        history.verify()

    def test_play_records_per_side_candidate_and_reference(self):
        m = fresh_model(14)
        for flags in ((True, False), (False, True)):
            game = play([m, m], ['a', 'a'], [], role='monitor', parent_id='b'*64,
                        seed=4, movetime=1, max_plies=4, reuse_history=flags)
            self.assertEqual(game['conditions']['reuse_history_by_side'], list(flags))
        with self.assertRaises(ValueError):
            play([m, m], ['a', 'a'], [], role='monitor', parent_id='b'*64,
                 seed=4, reuse_history=(True, 1))


if __name__ == '__main__':
    unittest.main()
