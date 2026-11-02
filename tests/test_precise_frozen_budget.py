import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fukaji.frozen_evaluation import prepare
from fukaji.online_learning import atomic_json


class PreciseFrozenBudgetTests(unittest.TestCase):
    def test_200_families_meet_single_comparison_10point_target_and_insufficient_budget_refused(self):
        from tests.test_online_state import StateRecoveryTests
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'state';StateRecoveryTests().state(root)
            s=json.loads((root/'state.json').read_text());s['excluded_openings']=[]
            atomic_json(root/'state.json',s);op=[dict(label='parent',model=s['research_latest'])]
            cpu=dict(worker_cpus=[0,1,2,3],writer_cpu=4,workers=4,cpu_max='800000 100000')
            with patch('fukaji.frozen_evaluation.cpu_plan',return_value=cpu):
                with self.assertRaises(ValueError):prepare(root,pairs=184,target_half_width=.10,opponents=op)
                with self.assertRaises(ValueError):prepare(root,pairs=257,target_half_width=.10,opponents=op)
                p=prepare(root,pairs=200,target_half_width=.10,opponents=op)
            self.assertEqual(p['planned_games'],400);self.assertEqual(p['minimum_pairs'],185)
            self.assertEqual(p['target_sampling_half_width'],.10)
            self.assertEqual(len({f['parent'] for f in p['families']}),200)
