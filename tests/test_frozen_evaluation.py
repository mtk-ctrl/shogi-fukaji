import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from fukaji.frozen_evaluation import (required_pairs,score_bounds,summarize,prepare,run,
                                    cpu_plan,_commit,_validate_family)
from fukaji.online_learning import load_snapshot,atomic_json
from fukaji.online_cycle import play,lock
from fukaji.online_state import verify,pack,restore


def available():
    try:cpu_plan(1);return bool(shutil.which('g++'))
    except (ValueError,OSError):return False


def fixture(root):
    from test_online_compare import SavedComparisonTests
    return SavedComparisonTests().setup_state(root)


def stats_plan(n=100):
    return {'candidate':'latest','pairs':n,'alpha':.05,'purpose':'strength',
            'conditions':{'backend':'cpp_nnue'},'opponents':[{'label':'a','model':'old'},
                                                          {'label':'b','model':'first'}]}


def stats_family(winner=0,status='checkmate'):
    # candidate_side=0 wins; invert the winner for its white game.
    return [{'candidate_side':side,'winner':None if winner is None else side if winner==0 else 1-side,
             'status':status,'opponent_label':label} for label in ('a','b') for side in (0,1)]


class FrozenEvaluationStatistics(unittest.TestCase):
    def test_pair_budget_has_precision_basis_and_multiple_comparisons(self):
        self.assertEqual(required_pairs(),98)
        epsilon=math.sqrt(math.log(80)/200)
        self.assertAlmostEqual(epsilon,.14802,places=4)
        self.assertLess(epsilon,.15)
        self.assertGreater(required_pairs(.10),required_pairs(.15))
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):prepare(Path(t)/'state',pairs=2)
            self.assertFalse((Path(t)/'state').exists())

    def test_unresolved_never_becomes_half_a_point(self):
        self.assertEqual(score_bounds({'winner':None,'status':'move_limit_unresolved','candidate_side':0}),
                         (0.,1.,'unresolved'))
        self.assertEqual(score_bounds({'winner':None,'status':'repetition_draw','candidate_side':0}),
                         (.5,.5,'draws'))
        result=summarize({i:stats_family(None,'move_limit_unresolved') for i in range(100)},stats_plan())
        for r in result['comparisons'].values():
            self.assertEqual(r['all_game_score_identification_bounds'],[0,1])
            self.assertEqual(r['simultaneous_hoeffding_interval'],[0,1])
            self.assertEqual(r['conclusion'],'inconclusive')

    def test_fixed_endpoint_and_correct_independence_unit(self):
        partial=summarize({i:stats_family() for i in range(2)},stats_plan())
        self.assertFalse(partial['complete'])
        self.assertTrue(all(r['conclusion']=='not_complete' for r in partial['comparisons'].values()))
        finished=summarize({i:stats_family() for i in range(100)},stats_plan())
        a=finished['comparisons']['a']
        self.assertEqual(a['opening_families'],100);self.assertEqual(a['games'],200)
        self.assertEqual(a['wins'],200)
        self.assertAlmostEqual(a['sampling_half_width'],math.sqrt(math.log(80)/200))
        self.assertEqual(a['conclusion'],'improved_on_this_distribution')
        self.assertFalse(finished['research_pointer_changed'])

    def test_symmetric_results_and_duplicate_pair_rejection(self):
        for winner,conclusion in [(0,'improved_on_this_distribution'),(1,'worse_on_this_distribution')]:
            r=summarize({i:stats_family(winner) for i in range(100)},stats_plan())
            self.assertEqual(r['comparisons']['a']['conclusion'],conclusion)
        games=stats_family();games[1]=games[0]
        with self.assertRaises(ValueError):summarize({0:games},stats_plan())


@unittest.skipUnless(available(),'isolated writer+worker CPU and g++ required')
class FrozenEvaluationRecovery(unittest.TestCase):
    def test_all_families_reserved_and_other_comparison_cannot_start(self):
        from fukaji.online_compare import compare_saved
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'state';key=fixture(root)
            plan=prepare(root,name='test',pairs=2,workers=1,movetime=1,max_total_plies=16,purpose='regression')
            state=json.loads((root/'state.json').read_text())
            self.assertTrue(state['paused'])
            self.assertTrue(all(f['parent'] in state['protected_parents'] for f in plan['families']))
            self.assertEqual(verify(root)['monitor_games'],0)
            with self.assertRaises(ValueError):compare_saved(root,key,key,name='other',seed=3,pairs=1)
            with self.assertRaises(ValueError):prepare(root,name='duplicate',pairs=2,workers=1,purpose='regression')
            raw=(root/'test/plan.json').read_bytes();(root/'test/plan.json').write_bytes(raw+b' ')
            with self.assertRaisesRegex(ValueError,'plan checksum'):verify(root)

    def test_atomic_orphan_recovery_checkpoint_and_real_worker_completion(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'state';key=fixture(root)
            plan=prepare(root,name='test',pairs=2,workers=1,movetime=1,max_total_plies=16,purpose='regression')
            model=load_snapshot(root/'models',key)[0];family=plan['families'][0];games=[]
            for side in (0,1):
                g=play([model,model],[key,key],family['prefix'],role='monitor',parent_id=family['parent'],
                       seed=1,movetime=1,max_plies=16-len(family['prefix']),native_backend=True)
                g.update(candidate_side=side,opponent_label='reference',family_index=0,evaluation_name='test')
                games.append(g)
            _validate_family(games,family,plan)
            state=json.loads((root/'state.json').read_text())
            original=atomic_json
            def fail_state(path,data):
                if Path(path).name=='state.json':raise OSError('simulated crash after immutable file publication')
                return original(path,data)
            with lock(root),patch('fukaji.frozen_evaluation.atomic_json',side_effect=fail_state):
                with self.assertRaises(OSError):_commit(root,family,games,plan,state)
            self.assertTrue((root/'test/family-000.jsonl').exists())
            # Restoring a pending evaluation preserves its learning pause.
            receipt=pack(root,Path(t)/'pending.tar.gz')
            restored=Path(t)/'restored';restore(Path(t)/'pending.tar.gz',restored,receipt['sha256'])
            report=run(restored,name='test',checkpoint_every=0)
            self.assertTrue(report['complete']);self.assertEqual(report['completed_games'],4)
            self.assertTrue(report['outcomes_match'])
            self.assertEqual(report['comparisons']['reference']['conclusion'],'regression_test_only')
            state=json.loads((restored/'state.json').read_text())
            self.assertFalse(state['paused']);self.assertNotIn('pending_campaign',state)
            self.assertEqual(state['research_latest'],key)
            self.assertEqual(verify(restored)['monitor_games'],4)
            (restored/'test/report.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'report checksum'):verify(restored)


if __name__=='__main__':unittest.main()
