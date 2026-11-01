import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fukaji.online_cycle import lock
from fukaji.online_learning import atomic_json,load_snapshot,make_samples,fit,save_snapshot
from fukaji.online_state import pack
from fukaji.portable_resume import create_index


class PortableResumeTests(unittest.TestCase):
    def fixture(self,p):
        from test_online_compare import SavedComparisonTests
        from test_online_learning import mate_game
        root=p/'state';old=SavedComparisonTests().setup_state(root)
        model,_,_=load_snapshot(root/'models',old);x,y,_=make_samples([mate_game()],model,seed=1)
        model,opt,_=fit(model,None,x,y,epochs=1)
        key=save_snapshot(root/'models',model,opt,{'parent':old})
        state=json.loads((root/'state.json').read_text())
        state['research_latest']=key;state['history'].append(key)
        atomic_json(root/'state.json',state)
        archive=p/'state.tar.gz';receipt=pack(root,archive)
        kwargs={'remote_metadata':{'id':'private-test','shared':False,
            'permissions':[{'type':'user','role':'owner'}]},
            'remote_readback_sha256':receipt['sha256'],'code_commit':'a'*40}
        return root,archive,kwargs

    def test_active_writer_cannot_be_claimed_as_stopped_handoff(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);root,archive,k=self.fixture(p)
            with lock(root):
                with self.assertRaises(BlockingIOError):
                    create_index(archive,p/'handoff.json',stopped_root=root,**k)
                checkpoint=create_index(archive,p/'checkpoint.json',**k)
                self.assertTrue(checkpoint['requires_original_worker_check'])
                self.assertEqual(checkpoint['publication_kind'],'recovery_checkpoint_only')
            stopped=create_index(archive,p/'stopped.json',stopped_root=root,**k)
            self.assertFalse(stopped['requires_original_worker_check'])
            self.assertEqual(stopped['publication_kind'],'stopped_handoff')

    def test_stale_archive_and_public_or_unverified_remote_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);root,archive,k=self.fixture(p)
            state=json.loads((root/'state.json').read_text());state['seed']=99
            atomic_json(root/'state.json',state)
            with self.assertRaisesRegex(ValueError,'stale'):
                create_index(archive,p/'bad.json',stopped_root=root,**k)
            for metadata in ({'id':'x','shared':True,'permissions':[{'type':'user','role':'owner'}]},
                             {'id':'x','shared':False,'permissions':[]},
                             {'id':'x','shared':False,'permissions':[{'type':'anyone','role':'reader'}]}):
                with self.assertRaises(ValueError):
                    create_index(archive,p/'bad.json',**{**k,'remote_metadata':metadata})
            with self.assertRaisesRegex(ValueError,'readback'):
                create_index(archive,p/'bad.json',**{**k,'remote_readback_sha256':'0'*64})
            self.assertFalse((p/'bad.json').exists())

    def test_pending_campaign_index_restores_in_another_directory(self):
        from fukaji.frozen_evaluation import prepare
        from fukaji.online_resume import restore_indexed
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);root,_,k=self.fixture(p)
            with patch('fukaji.frozen_evaluation.cpu_plan',return_value={
                'worker_cpus':[0],'writer_cpu':1,'workers':1,'cpu_max':'max 100000'}):
                plan=prepare(root,name='pending',pairs=2,purpose='regression',workers=1)
            archive=p/'pending.tar.gz';receipt=pack(root,archive)
            k['remote_readback_sha256']=receipt['sha256']
            index=create_index(archive,p/'index.json',stopped_root=root,**k)
            self.assertEqual(index['pending_campaign'],'pending')
            self.assertEqual(index['completed_families'],0)
            self.assertEqual(index['planned_families'],2)
            self.assertEqual(index['evaluation_source_sha256'],plan['source_sha256'])
            result=restore_indexed(p/'index.json',archive,p/'different-environment')
            self.assertTrue(result['resume_identity_verified'])
            self.assertTrue(result['paused'])
            self.assertEqual(result['research_latest'],index['latest_research_model'])
