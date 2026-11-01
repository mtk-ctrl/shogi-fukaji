import copy
import unittest
import numpy as np
import shogi
from fukaji.monitor_value_audit import game_metrics
from fukaji.online_learning import fresh_model, parameters
from tests.test_online_learning import mate_game


class MonitorValueAuditTests(unittest.TestCase):
    def test_same_positions_deterministic_and_models_untouched(self):
        game={**mate_game(),'role':'monitor'}; m=fresh_model(29)
        before=[p.copy() for p in parameters(m)]
        a=game_metrics(game,{'a':m,'b':m},seed=1,protected=[game['parent_id']])
        self.assertEqual(a,game_metrics(game,{'a':m,'b':m},seed=1,protected=[game['parent_id']]))
        self.assertEqual(a['metrics']['a'],a['metrics']['b'])
        for p,q in zip(before,parameters(m)):np.testing.assert_array_equal(p,q)

    def test_train_unprotected_and_wrong_outcomes_rejected_unresolved_excluded(self):
        m=fresh_model(29);g={**mate_game(),'role':'monitor'};protected=[g['parent_id']]
        with self.assertRaises(ValueError):game_metrics(mate_game(),{'m':m},seed=1,protected=protected)
        with self.assertRaises(ValueError):game_metrics(g,{'m':m},seed=1,protected=[])
        with self.assertRaises(ValueError):game_metrics({**g,'winner':1},{'m':m},seed=1,protected=protected)
        u={**g,'start_sfen':shogi.STARTING_SFEN,'moves':['7g7f'],
           'winner':None,'status':'move_limit_unresolved'}
        self.assertIsNone(game_metrics(u,{'m':m},seed=1,protected=protected))
