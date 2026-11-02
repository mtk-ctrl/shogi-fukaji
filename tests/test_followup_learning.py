import unittest
import numpy as np
from fukaji.followup_learning import GROUPS,schedule

class FollowupLearningTests(unittest.TestCase):
    def test_budget_and_decay_of_recent_cohorts(self):
        sizes=dict(zip(GROUPS,[11,17,19,31,29]));r=schedule(sizes,33);edges=np.cumsum([0]+list(sizes.values()))
        self.assertEqual(len(r),8192);np.testing.assert_array_equal(r,schedule(sizes,33))
        self.assertEqual([int(((r>=a)&(r<b)).sum()) for a,b in zip(edges[:-1],edges[1:])],[2048,1024,1024,2048,2048])
        with self.assertRaises(ValueError):schedule({**sizes,'fresh':0},33)
