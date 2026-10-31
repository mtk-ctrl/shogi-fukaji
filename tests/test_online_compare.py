import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fukaji.online_learning import atomic_json, fresh_model, save_snapshot
from fukaji.online_cycle import write_games
from fukaji.online_compare import compare_saved
from fukaji.online_state import verify


class SavedComparisonTests(unittest.TestCase):
    def setup_state(self, root):
        key=save_snapshot(root/'models',fresh_model(1),None,{'parent':None})
        receipt=write_games(root,'bootstrap.jsonl',[{'role':'train','source':'selfplay','parent_id':'train'}])
        atomic_json(root/'state.json',{'schema':1,'formal_adoption':False,'paused':False,
            'history':[key],'research_latest':key,'checkpoint':key,'fixed_anchor':key,
            'bootstrap':receipt,'train_batches':[],'evaluation_batches':[],
            'protected_parents':[],'excluded_openings':[],'used_seeds':[],'cycles':[]})
        return key

    def test_monitor_parents_reserved_and_partial_failures_pause_learning(self):
        for fail in (False,True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);key=self.setup_state(root)
                def play(models,ids,prefix,**kwargs):
                    s=json.loads((root/'state.json').read_text())
                    self.assertTrue(s['paused'])
                    self.assertIn(kwargs['parent_id'],s['protected_parents'])
                    if fail: raise RuntimeError('interrupted game')
                    return {'role':'monitor','source':'selfplay','parent_id':kwargs['parent_id'],
                            'winner':None,'status':'move_limit_unresolved'}
                with patch('fukaji.online_compare.play',side_effect=play):
                    if fail:
                        with self.assertRaises(RuntimeError):compare_saved(root,key,key,name='compare',seed=1,pairs=1)
                    else:
                        report=compare_saved(root,key,key,name='compare',seed=1,pairs=1)
                        self.assertEqual(report['left_score']['unresolved'],2)
                s=json.loads((root/'state.json').read_text())
                self.assertEqual(s['research_latest'],key)
                self.assertEqual(s['paused'],fail)
                self.assertEqual('pending_comparison' in s,fail)
                if not fail:self.assertEqual(verify(root)['monitor_games'],2)
                with self.assertRaises(ValueError):compare_saved(root,key,key,name='compare',seed=1,pairs=1)


if __name__=='__main__':unittest.main()
