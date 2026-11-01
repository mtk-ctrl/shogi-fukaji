import unittest
import numpy as np
import shogi
from fukaji.stratified_replay_trial import schedules, partition, TOTAL
from fukaji.online_learning import fresh_model, fit, parameters
from tests.test_online_learning import mate_game


class StratifiedReplayTests(unittest.TestCase):
    def test_shared_recent_nested_sources_exact_budget_and_reproducible(self):
        sizes = dict(recent=17, bootstrap=91, historical_nnue=9)
        rows = schedules(sizes, 123)
        n, s = rows['natural_history'], rows['stratified_history']
        np.testing.assert_array_equal(n[:4096], s[:4096])
        for key, indices in rows.items():
            np.testing.assert_array_equal(indices, schedules(sizes,123)[key])
            self.assertEqual(len(indices), TOTAL)
            self.assertTrue(np.all((0<=indices)&(indices<117)))
        self.assertEqual(int(((17<=n)&(n<108)).sum()), 3727)
        self.assertEqual(int(((17<=s)&(s<108)).sum()), 2048)
        np.testing.assert_array_equal(n[4096:4096+2048], s[4096:4096+2048])
        np.testing.assert_array_equal(n[4096+3727:], s[4096+2048:4096+2048+369])
        for bad in [dict(recent=0,bootstrap=1,historical_nnue=1),dict(recent=1,bootstrap=1)]:
            with self.assertRaises(ValueError):schedules(bad,1)

    def test_global_dedup_unresolved_exclusion_and_role_protection(self):
        parent=fresh_model(17); g=mate_game()
        unresolved={**g,'start_sfen':shogi.STARTING_SFEN,'moves':['7g7f'],
                    'winner':None,'status':'move_limit_unresolved'}
        groups=partition([('bootstrap',[g]),('recent',[g,unresolved])],parent,[])
        self.assertEqual(len(groups['bootstrap']),1)
        self.assertFalse(groups['recent'])
        for role in ('monitor','audit'):
            with self.assertRaises(ValueError):
                partition([('recent',[{**unresolved,'role':role}])],parent,[])
        with self.assertRaises(ValueError):partition([('recent',[g])],parent,[g['parent_id']])

    def test_continuation_branches_preserve_parent_and_optimizer(self):
        parent=fresh_model(17)
        x=np.array([[1,2,3]]*128,np.int32); y=np.ones(128,np.float32)
        parent,opt,_=fit(parent,None,x,y,epochs=1)
        before=[p.copy() for p in parameters(parent)]; moments=[p.copy() for p in opt['m']]
        a,ao,_=fit(parent,opt,x,y,epochs=2,seed=29)
        b,bo,_=fit(parent,opt,x,y,epochs=2,seed=29)
        self.assertEqual(ao['step'],opt['step']+2)
        self.assertEqual(ao['step'],bo['step'])
        for p,q in zip(parameters(a),parameters(b)):np.testing.assert_array_equal(p,q)
        for p,q in zip(before,parameters(parent)):np.testing.assert_array_equal(p,q)
        for p,q in zip(moments,opt['m']):np.testing.assert_array_equal(p,q)
