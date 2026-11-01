import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from fukaji.online_learning import atomic_json, digest, fresh_model, save_snapshot
from fukaji.online_cycle import write_games
from fukaji.online_state import verify, restore, pack


class StateRecoveryTests(unittest.TestCase):
    def state(self, root):
        key = save_snapshot(root/'models', fresh_model(1), None, {'parent': None})
        data = write_games(root, 'train.jsonl', [{'role': 'train', 'source': 'selfplay',
                                               'parent_id': 'train'}])
        atomic_json(root/'state.json', {'schema': 1, 'formal_adoption': False,
            'history': [key], 'research_latest': key, 'checkpoint': key,
            'fixed_anchor': key, 'bootstrap': data, 'train_batches': [],
            'evaluation_batches': [], 'protected_parents': ['monitor'],
            'cycles': [], 'used_seeds': [], 'paused': False})

    def test_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); self.state(p/'old')
            receipt=pack(p/'old', p/'saved.tar.gz')
            result=restore(p/'saved.tar.gz',p/'new',receipt['sha256'])
            self.assertEqual(result,verify(p/'old'))
            self.assertEqual((p/'new/state.json').read_bytes(),(p/'old/state.json').read_bytes())
            with self.assertRaises(ValueError): restore(p/'saved.tar.gz',p/'new',receipt['sha256'])
            with self.assertRaises(ValueError): restore(p/'saved.tar.gz',p/'bad','0'*64)

    def test_corrupt_model_data_and_protected_parent(self):
        for defect in ('model','data','protected','path','report'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp); self.state(p); s=json.loads((p/'state.json').read_text())
                if defect=='model': next((p/'models').glob('*.npz')).write_bytes(b'bad')
                if defect=='data': (p/'train.jsonl').write_text('{}\n')
                if defect=='protected': s['protected_parents'].append('train')
                if defect=='path': s['bootstrap']['path']='../escape.jsonl'
                if defect=='report':
                    (p/'report.json').write_text('{}');s['cycles']=[{'report':'report.json','sha256':'0'*64}]
                atomic_json(p/'state.json',s)
                with self.assertRaises(ValueError):verify(p)

    def test_untrusted_archive_cannot_publish_destination(self):
        for name,kind in [('../escape',tarfile.REGTYPE),('d17-state/link',tarfile.SYMTYPE),
                          ('d17-state/fifo',tarfile.FIFOTYPE),('unexpected/file',tarfile.REGTYPE)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp); a=p/'bad.tar.gz'
                with tarfile.open(a,'w:gz') as t:
                    m=tarfile.TarInfo(name);m.type=kind;m.linkname='/tmp/escape';t.addfile(m,io.BytesIO())
                with self.assertRaises(ValueError):restore(a,p/'new',digest(a.read_bytes()))
                self.assertFalse((p/'new').exists())


class ResumePolicyTests(unittest.TestCase):
    def test_explicit_source_and_publication_required(self):
        from fukaji.online_resume import validate_inputs
        base = {'CYCLES': '1', 'FRESH_BOOTSTRAP': 'false',
                'ALLOW_PUBLIC_STATE_ARTIFACT': 'true'}
        for patch in ({}, {'FRESH_BOOTSTRAP': 'true', 'PUBLISHED_CHECKPOINT': 'true'},
                      {'RESUME_RUN': '1'}, {'RESUME_SHA': 'a'*64},
                      {'FRESH_BOOTSTRAP': 'true', 'ALLOW_PUBLIC_STATE_ARTIFACT': 'false'},
                      {'FRESH_BOOTSTRAP': 'true', 'CYCLES': '4'}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_inputs({**base, **patch})
        for patch in ({'FRESH_BOOTSTRAP': 'true'}, {'PUBLISHED_CHECKPOINT': 'true'},
                      {'RESUME_RUN': '123', 'RESUME_SHA': 'a'*64}):
            validate_inputs({**base, **patch})

    def test_index_mismatch_never_publishes_destination(self):
        from fukaji.online_resume import restore_indexed
        from fukaji.online_learning import load_snapshot, parameters
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); StateRecoveryTests().state(p/'old')
            state = json.loads((p/'old/state.json').read_text())
            # Real saved optimizer, as required for a continuing lineage.
            model, _, meta = load_snapshot(p/'old/models', state['research_latest'])
            opt = {'step': 17, 'm': [np.zeros_like(x) for x in parameters(model)],
                   's': [np.zeros_like(x) for x in parameters(model)]}
            key = save_snapshot(p/'old/models', model, opt, {'parent': None})
            state.update(research_latest=key, checkpoint=key, fixed_anchor=key, history=[key])
            atomic_json(p/'old/state.json', state)
            receipt = pack(p/'old', p/'archive.tar.gz')
            index = {'latest_research_model': key, 'cycles_completed': 0,
                     'optimizer_step': 17, 'state_sha256': receipt['sha256']}
            for field, bad in [('latest_research_model', 'a'*64), ('cycles_completed', 1),
                               ('optimizer_step', 18), ('state_sha256', '0'*64)]:
                atomic_json(p/'index.json', {**index, field: bad})
                with self.subTest(field=field), self.assertRaises((ValueError, FileNotFoundError)):
                    restore_indexed(p/'index.json', p/'archive.tar.gz', p/'new')
                self.assertFalse((p/'new').exists())
            atomic_json(p/'index.json', index)
            result = restore_indexed(p/'index.json', p/'archive.tar.gz', p/'new')
            self.assertTrue(result['resume_identity_verified'])
            self.assertEqual(result['optimizer_step'], 17)
            with self.assertRaises(ValueError):
                restore_indexed(p/'index.json', p/'archive.tar.gz', p/'new')


if __name__=='__main__': unittest.main()
