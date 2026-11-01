import json
from pathlib import Path
import shutil
import tempfile
import unittest

from fukaji.accumulator_arena import run
from fukaji.frozen_evaluation import cpu_plan
from fukaji.online_state import verify


def available():
    try:cpu_plan(1);return bool(shutil.which('g++'))
    except (OSError,ValueError):return False


@unittest.skipUnless(available(),'writer and worker CPU plus g++ required')
class AccumulatorArenaTests(unittest.TestCase):
    def test_same_tensor_both_colors_backend_labels_parent_protection_and_no_promotion(self):
        from test_online_compare import SavedComparisonTests
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'state';key=SavedComparisonTests().setup_state(root)
            result=run(root,pairs=1,movetime=1,max_total=16,workers=1)
            self.assertEqual(result['legal_replay_games'],2)
            self.assertEqual(result['model_both_sides'],key)
            self.assertFalse(result['strength_proven']);self.assertFalse(result['research_pointer_changed'])
            games=[json.loads(x) for x in (root/'d28-accumulator-screen/pair-000.jsonl').read_text().splitlines()]
            for game in games:
                self.assertEqual(game['models'],[key,key])
                side=game['candidate_side']
                self.assertEqual(game['conditions']['backend_by_side'][side],'cpp_nnue_accumulator')
                self.assertTrue(all(x['backend']=='cpp_nnue_accumulator' for x in game['telemetry'] if x['side']==side))
            self.assertEqual(verify(root)['monitor_games'],2)
            state=json.loads((root/'state.json').read_text());self.assertEqual(state['research_latest'],key)
            self.assertFalse(state['paused']);self.assertNotIn('pending_comparison',state)
