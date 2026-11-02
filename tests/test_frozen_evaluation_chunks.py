import json
from pathlib import Path
import shutil
import tempfile
import unittest
from fukaji.frozen_evaluation import prepare,run,cpu_plan
from fukaji.online_cycle import lock
from fukaji.online_learning import atomic_json,digest
from fukaji.online_state import restore,verify


def available():
    try:cpu_plan(1);return bool(shutil.which('g++'))
    except (ValueError,OSError):return False


@unittest.skipUnless(available(),'isolated writer and worker CPUs required')
class FrozenExecutionChunks(unittest.TestCase):
    def test_stopped_partial_archive_resumes_without_replaying_committed_family(self):
        from tests.test_online_state import StateRecoveryTests
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);root=p/'root';StateRecoveryTests().state(root)
            state=json.loads((root/'state.json').read_text());state['excluded_openings']=[]
            atomic_json(root/'state.json',state);key=state['research_latest']
            for bad in (0,257,True):
                with self.assertRaises(ValueError):
                    prepare(root,purpose='regression',pairs=2,workers=1,execution_chunk_families=bad)
            prepare(root,name='chunked',purpose='regression',pairs=2,workers=1,
                movetime=1,max_total_plies=16,candidate_id=key,
                opponents=[dict(label='parent',model=key)],execution_chunk_families=1)
            first=run(root,name='chunked',checkpoint_every=0)
            self.assertFalse(first['complete']);self.assertEqual(first['completed_games'],2)
            self.assertTrue(first['learning_remains_paused']);self.assertFalse((root/'chunked/report.json').exists())
            state=json.loads((root/'state.json').read_text())
            self.assertTrue(state['paused']);self.assertEqual(state['pending_campaign'],'chunked')
            with lock(root):pass  # run returned after the writer/worker boundary.
            before=digest((root/'chunked/family-000.jsonl').read_bytes())
            archive=p/first['checkpoint']['archive'];restored=p/'restored'
            restore(archive,restored,first['checkpoint']['sha256'])
            self.assertEqual(verify(restored)['monitor_games'],2)
            second=run(restored,name='chunked',checkpoint_every=0)
            self.assertTrue(second['complete']);self.assertEqual(second['completed_games'],4)
            self.assertTrue(second['outcomes_match']);self.assertEqual(verify(restored)['monitor_games'],4)
            self.assertEqual(digest((restored/'chunked/family-000.jsonl').read_bytes()),before)
            self.assertFalse(json.loads((restored/'state.json').read_text())['paused'])
