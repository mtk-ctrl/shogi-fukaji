import copy
import unittest
from fukaji.provisional_update import decision_comparisons

class ProvisionalReferenceRoleTests(unittest.TestCase):
    def report(self):
        return dict(comparisons={
            'parent':dict(opponent_id='p',all_game_score_identification_bounds=[.5975,.6175],conclusion='inconclusive'),
            'ablation':dict(opponent_id='a',all_game_score_identification_bounds=[.495,.505],conclusion='inconclusive')})

    def test_explicit_parent_target_keeps_tied_ablation_visible(self):
        r=self.report();before=copy.deepcopy(r)
        with self.assertRaises(ValueError):decision_comparisons(r,'p')
        self.assertEqual(decision_comparisons(r,'p','parent'),['parent']);self.assertEqual(r,before)

    def test_unknown_or_nonparent_target_cannot_weaken_reference(self):
        for label in ('absent','ablation',[],None):
            if label is None:continue
            with self.assertRaises(ValueError):decision_comparisons(self.report(),'p',label)
        with self.assertRaises(ValueError):decision_comparisons({'comparisons':{}},'p','parent')

    def test_no_parent_win_or_proven_bad_control_stops_update(self):
        r=self.report();r['comparisons']['parent']['all_game_score_identification_bounds']=[.5,.5]
        with self.assertRaises(ValueError):decision_comparisons(r,'p','parent')
        r=self.report();r['comparisons']['ablation']['conclusion']='worse_on_this_distribution'
        with self.assertRaises(ValueError):decision_comparisons(r,'p','parent')
