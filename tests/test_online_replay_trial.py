import unittest
import numpy as np
from fukaji.online_replay_trial import schedules, TOTAL


class ReplayScheduleTests(unittest.TestCase):
    def test_fixed_budget_nested_common_draws_and_only_ratio_changes(self):
        a=schedules(17,91,123);b=schedules(17,91,123)
        for label,rows in a.items():
            np.testing.assert_array_equal(rows,b[label])
            self.assertEqual(len(rows),TOTAL)
            self.assertTrue(np.all((rows>=0)&(rows<108)))
        x=a['recent50_old50'];y=a['recent25_old75']
        self.assertEqual(int((x<17).sum()),512)
        self.assertEqual(int((y<17).sum()),256)
        np.testing.assert_array_equal(x[:256],y[:256])
        np.testing.assert_array_equal(x[512:],y[256:768])
        with self.assertRaises(ValueError):schedules(0,1,123)
