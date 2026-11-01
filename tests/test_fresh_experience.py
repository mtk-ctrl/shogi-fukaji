import copy
import unittest
from fukaji.fresh_experience import validate_plan, PARENT, OPPONENT, SEED
from fukaji.online_cycle import fresh_prefix


class FreshExperiencePlanTests(unittest.TestCase):
    def test_new_train_families_cannot_overlap_protected_monitor_or_change_budget(self):
        _, protected=fresh_prefix(SEED,set())
        excluded={protected};families=[]
        for i in range(32):
            prefix,parent=fresh_prefix(SEED+i*1000,excluded)
            families.append(dict(index=i,prefix=prefix,parent=parent))
        p=dict(parent=PARENT,opponent=OPPONENT,seed=SEED,role='train',planned_games=64,
               worker_cpu=5,excluded_at_start=[],protected_parents=[protected],families=families)
        validate_plan(p)
        self.assertEqual(len({f['parent'] for f in families}),32)
        self.assertNotIn(protected,{f['parent'] for f in families})
        for key,value in [('role','monitor'),('planned_games',66),('worker_cpu',0),('families',families[:-1])]:
            bad=copy.deepcopy(p);bad[key]=value
            with self.assertRaises(ValueError):validate_plan(bad)
