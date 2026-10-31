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


if __name__=='__main__': unittest.main()
