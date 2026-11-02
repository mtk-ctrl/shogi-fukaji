import json
from pathlib import Path
import tempfile
import unittest
from fukaji.online_learning import atomic_json,digest
from fukaji.online_state import verify,pack,restore


class SupplementalStateTests(unittest.TestCase):
    def test_schema2_roundtrip_counts_extra_train_and_old_schema_rejects_it(self):
        from tests.test_supplemental_training import SupplementalTrainingTests
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);root=p/'root';s=SupplementalTrainingTests().fixture(root)
            s['schema']=2;atomic_json(root/'state.json',s)
            self.assertEqual(verify(root)['train_games'],3)
            r=pack(root,p/'state.tar.gz');restored=restore(p/'state.tar.gz',p/'restored',r['sha256'])
            self.assertEqual(restored['train_games'],3)
            s['schema']=1;atomic_json(root/'state.json',s)
            with self.assertRaisesRegex(ValueError,'schema2'):verify(root)

    def test_even_checksum_consistent_monitor_cannot_enter_supplemental_train(self):
        from tests.test_supplemental_training import SupplementalTrainingTests
        for defect in ('role','protected'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as t:
                root=Path(t);s=SupplementalTrainingTests().fixture(root);s['schema']=2
                path=root/'fresh/family-000.jsonl';games=[json.loads(line) for line in path.read_text().splitlines()]
                for g in games:
                    if defect=='role':g['role']='monitor'
                    else:g['parent_id']='monitor'
                path.write_text(''.join(json.dumps(g)+'\n' for g in games));sha=digest(path.read_bytes())
                s['supplemental_train_batches'][0]['sha256']=sha
                rp=root/'fresh/report.json';r=json.loads(rp.read_text());r['receipts'][0]['sha256']=sha
                atomic_json(rp,r);s['supplemental_collections'][0]['report_sha256']=digest(rp.read_bytes())
                atomic_json(root/'state.json',s)
                with self.assertRaisesRegex(ValueError,'evaluation data'):verify(root)
