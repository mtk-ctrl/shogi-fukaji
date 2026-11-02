import unittest
import numpy as np
from fukaji.fresh_learning_trial import replacement_schedule
from fukaji.stratified_replay_trial import TOTAL


class FreshLearningScheduleTests(unittest.TestCase):
    def test_only_recent_pool_changes_historical_rows_exact_and_fixed_update_budget(self):
        old=np.arange(TOTAL,dtype=np.int64)
        new=replacement_schedule(old,1700,22000,29)
        self.assertEqual(len(new),TOTAL)
        np.testing.assert_array_equal(new[4096:],old[4096:])
        self.assertTrue(np.all((new[:4096]>=22000)&(new[:4096]<23700)))
        np.testing.assert_array_equal(new,replacement_schedule(old,1700,22000,29))
        with self.assertRaises(ValueError):replacement_schedule(old,0,22000,29)
        with self.assertRaises(ValueError):replacement_schedule(old[:4096],1700,22000,29)
