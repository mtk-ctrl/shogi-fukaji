import unittest
import numpy as np
from fukaji.retained_replay_trial import schedules,partition,GROUPS
from fukaji.online_learning import fresh_model
from tests.test_online_learning import mate_game

class RetainedReplayTests(unittest.TestCase):
    def test_exact_exposure_and_shared_control_draws(self):
        sizes=dict(zip(GROUPS,[17,19,11,101,23]));rows=schedules(sizes,31)
        a,b=rows['fresh_only'],rows['retain_two'];bounds=np.cumsum([0]+[sizes[g] for g in GROUPS])
        for name,indices in rows.items():
            self.assertEqual(len(indices),8192)
            np.testing.assert_array_equal(indices,schedules(sizes,31)[name])
            counts=[int(((indices>=lo)&(indices<hi)).sum()) for lo,hi in zip(bounds[:-1],bounds[1:])]
            self.assertEqual(counts,[4096,0,0,2048,2048] if name=='fresh_only' else [2048,1024,1024,2048,2048])
        np.testing.assert_array_equal(a[:2048],b[:2048]);np.testing.assert_array_equal(a[4096:],b[4096:])
        with self.assertRaises(ValueError):schedules({**sizes,'fresh':0},31)

    def test_monitor_and_protected_parents_never_admitted(self):
        parent=fresh_model(31);g=mate_game()
        for role in ('monitor','audit'):
            with self.assertRaisesRegex(ValueError,'training'):
                partition([('fresh',[{**g,'role':role}])],parent,[])
        with self.assertRaises(ValueError):partition([('fresh',[g])],parent,[g['parent_id']])
        with self.assertRaisesRegex(ValueError,'empty'):
            partition([('fresh',[g]),('d30',[g])],parent,[])
