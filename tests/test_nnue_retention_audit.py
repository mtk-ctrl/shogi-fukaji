import json
from pathlib import Path
import tempfile
import unittest

from fukaji.online_learning import fresh_model,save_snapshot,atomic_json
from fukaji.online_cycle import write_games
from fukaji.nnue_retention_audit import audit
from test_online_learning import mate_game


class RetentionAuditTests(unittest.TestCase):
    def fixture(self,root,game=None):
        key=save_snapshot(root/'models',fresh_model(91),None,{'parent':None})
        receipt=write_games(root,'bootstrap.jsonl',[game or mate_game()])
        atomic_json(root/'state.json',{'bootstrap':receipt,'train_batches':[],
            'fixed_anchor':key,'research_latest':key,'protected_parents':[]})
        return key

    def test_identical_frozen_models_and_read_only_source(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);key=self.fixture(p);before=(p/'state.json').read_bytes()
            result=audit(p,p/'audit.json',model_ids={'first':key,'latest':key})
            metrics=result['groups']['bootstrap_665']['equal_game_weighted_metrics']
            self.assertEqual(metrics['first'],metrics['latest'])
            self.assertEqual((p/'state.json').read_bytes(),before)
            self.assertFalse(result['monitor_used']);self.assertFalse(result['model_updated'])
            self.assertEqual(result['groups']['bootstrap_665']['resolved_games'],1)
            with self.assertRaises(ValueError):audit(p,p/'audit.json',model_ids={'first':key})

    def test_monitor_or_protected_parent_cannot_be_a_training_diagnostic(self):
        for protected in (False,True):
            with tempfile.TemporaryDirectory() as t:
                p=Path(t);g=mate_game()
                if not protected:g['role']='monitor'
                key=self.fixture(p,g)
                if protected:
                    state=json.loads((p/'state.json').read_text());state['protected_parents']=[g['parent_id']]
                    atomic_json(p/'state.json',state)
                with self.assertRaises(ValueError):audit(p,p/'audit.json',model_ids={'first':key})

    def test_duplicate_game_is_excluded_across_training_batches(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);key=self.fixture(p)
            state=json.loads((p/'state.json').read_text())
            (p/'cycle-001').mkdir()
            state['train_batches']=[write_games(p,'cycle-001/train.jsonl',[mate_game()])]
            atomic_json(p/'state.json',state)
            result=audit(p,p/'audit.json',model_ids={'first':key})
            group=result['groups']['selfplay_cycles_1_to_11']
            self.assertEqual(group['raw_games'],1)
            self.assertEqual(group['unique_games'],0)
            self.assertEqual(group['duplicate_excluded'],1)
            self.assertEqual(group['resolved_games'],0)
            self.assertEqual(group['sampled_positions'],0)

    def test_later_cycle_is_not_mislabeled_as_cycles12_to14(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);key=self.fixture(p)
            state=json.loads((p/'state.json').read_text());(p/'cycle-0015').mkdir()
            state['train_batches']=[write_games(p,'cycle-0015/train.jsonl',[mate_game()])]
            atomic_json(p/'state.json',state)
            result=audit(p,p/'audit.json',model_ids={'first':key})
            self.assertEqual(result['groups']['selfplay_cycle_15']['raw_games'],1)
            self.assertEqual(result['groups']['selfplay_cycles_12_to_14']['raw_games'],0)


if __name__=='__main__':unittest.main()
