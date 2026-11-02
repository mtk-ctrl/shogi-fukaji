import copy
import unittest
from fukaji.followup_train_collection import families,validate_plan,PARENT,OPPONENT,SEED,CONDITIONS
from fukaji.online_cycle import fresh_prefix

class FollowupPlanTests(unittest.TestCase):
    def test_excludes_frozen_monitor_and_fixes_readonly_collection_budget(self):
        _,protected=fresh_prefix(SEED,set())
        p=dict(schema='fukaji-fresh-train-v1',parent=PARENT,opponent=OPPONENT,seed=SEED,
               role='train',planned_games=64,worker_cpus=[5,6],writer_cpu=7,conditions=CONDITIONS,formal_adoption=False,
               excluded_at_start=[],protected_parents=[protected],families=families([protected]))
        validate_plan(p);self.assertNotIn(protected,{f['parent'] for f in p['families']})
        for key,v in [('role','monitor'),('worker_cpus',[0,1]),('planned_games',4),('families',p['families'][:-1])]:
            bad=copy.deepcopy(p);bad[key]=v
            with self.assertRaises(ValueError):validate_plan(bad)
