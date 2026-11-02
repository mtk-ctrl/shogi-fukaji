import json
from pathlib import Path
import tempfile
import unittest
from fukaji.online_learning import atomic_json,digest
from fukaji.online_cycle import write_games
from fukaji.supplemental_training import verified_receipts


class SupplementalTrainingTests(unittest.TestCase):
    def fixture(self,p):
        from tests.test_online_state import StateRecoveryTests
        StateRecoveryTests().state(p);s=json.loads((p/'state.json').read_text());key=s['research_latest']
        out=p/'fresh';out.mkdir()
        receipt=write_games(out,'family-000.jsonl',[dict(role='train',source='selfplay',parent_id='new')]*2)
        plan=dict(schema='fukaji-fresh-train-v1',parent=key,opponent=key,role='train',
            formal_adoption=False,planned_games=2,families=[dict(index=0,parent='new')])
        atomic_json(out/'plan.json',plan);sha=digest((out/'plan.json').read_bytes())
        atomic_json(out/'report.json',dict(complete=True,outcomes_match=True,completed_games=2,
            plan_sha256=sha,receipts=[receipt]))
        s['supplemental_train_batches']=[dict(**receipt,cohort='fresh_test')]
        s['supplemental_train_batches'][0]['path']='fresh/'+receipt['path']
        s['supplemental_collections']=[dict(plan='fresh/plan.json',plan_sha256=sha,
            report='fresh/report.json',report_sha256=digest((out/'report.json').read_bytes()))]
        return s

    def test_registered_receipts_cover_collection_exactly_and_cannot_be_protected(self):
        for defect in ('none','protected','unregistered','missing','corrupt','path'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as t:
                p=Path(t);s=self.fixture(p)
                if defect=='protected':s['protected_parents'].append('new')
                if defect=='unregistered':s['supplemental_train_batches'][0]['sha256']='0'*64
                if defect=='missing':s['supplemental_train_batches']=[]
                if defect=='corrupt':(p/'fresh/report.json').write_text('{}')
                if defect=='path':s['supplemental_train_batches'][0]['path']='../escape'
                if defect=='none':self.assertEqual(len(verified_receipts(p,s)),1)
                else:
                    with self.assertRaises(ValueError):verified_receipts(p,s)
