import copy
import unittest
from fukaji.next_train_collection import validate_plan,SEED,PARENT,OPPONENT
from fukaji.online_cycle import fresh_prefix

class NextTrainPlanTests(unittest.TestCase):
    def test_excludes_monitor_and_rejects_role_budget_and_opening_changes(self):
        _,protected=fresh_prefix(SEED,set());excluded={protected};families=[]
        for i in range(32):
            prefix,parent=fresh_prefix(SEED+i*1000,excluded);families.append(dict(index=i,prefix=prefix,parent=parent))
        p=dict(schema='fukaji-fresh-train-v1',parent=PARENT,opponent=OPPONENT,seed=SEED,role='train',
               planned_games=64,chunk_families=8,excluded_at_start=[],protected_parents=[protected],families=families,
               conditions=dict(movetime_ms=50,max_additional_plies=256,quiescence=0,backend='cpp_nnue',exploration_probability=.05))
        validate_plan(p);self.assertNotIn(protected,{f['parent'] for f in families})
        for key,v in [('parent',OPPONENT),('role','monitor'),('planned_games',4),('chunk_families',32),('families',families[:-1])]:
            bad=copy.deepcopy(p);bad[key]=v
            with self.assertRaises(ValueError):validate_plan(bad)
