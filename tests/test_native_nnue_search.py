import copy
import ctypes as ct
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import shogi

from fukaji.online_learning import fresh_model,expand_king_relative
from fukaji.online_search import select,Position
from fukaji.native_nnue import NativeSelector,build_native
from fukaji.online_cycle import play
from fukaji.online_compare import compare_saved


@unittest.skipUnless(shutil.which('g++'),'g++ required')
class NativeNNUEOnlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.binary=build_native()
    def setUp(self):self.model=fresh_model(818)

    def test_value_matches_both_perspectives_through_legal_game(self):
        b=shogi.Board();rng=random.Random(737)
        with NativeSelector(self.binary) as native:
            for _ in range(64):
                for p in (0,1):
                    acc=self.model.full_accumulator(b,p)
                    expected=float(np.dot(np.clip(acc,0,1),self.model.output_weights.astype(float))+self.model.output_bias)
                    self.assertAlmostEqual(native.value(b,self.model,p),expected,places=11)
                legal=list(b.legal_moves)
                if not legal:break
                b.push(rng.choice(legal))

    def test_fixed_depth_matches_special_positions_and_exact_restoration(self):
        sfens=[shogi.STARTING_SFEN,'4k4/9/4P4/9/9/9/9/9/4K4 b P 1',
            '4k4/9/4R4/9/9/9/9/9/4K4 w - 1',
            '4k4/9/3+p5/4S4/9/9/9/9/4K4 b - 1',
            '3pkp3/3p1p3/4G4/9/9/9/9/9/4K4 b P 1',
            '7lk/8p/7G1/9/9/9/9/9/4K3R b - 1']
        for sfen in sfens:
            with self.subTest(sfen=sfen), NativeSelector(self.binary) as native:
                b=shogi.Board(sfen);counts=+b.transpositions
                ref=select(b,self.model,max_depth=2,fixed_depth=True)
                got=native.select(b,self.model,max_depth=2,fixed_depth=True)
                self.assertEqual(ref['move'],got['move'])
                self.assertGreater(got['nodes'],0)  # PVS intentionally changes nodes
                self.assertAlmostEqual(ref['score'],got['score'],places=7)
                self.assertEqual(b.sfen(),sfen);self.assertEqual(+b.transpositions,counts)
                native.state.assert_equal(b,full=True)

    def test_random_fixed_depth_and_timed_search_restore_history(self):
        b=shogi.Board();rng=random.Random(311)
        with NativeSelector(self.binary) as native:
            for _ in range(6):
                for _ in range(4):b.push(rng.choice(list(b.legal_moves)))
                sfen=b.sfen();counts=+b.transpositions
                ref=select(b,self.model,max_depth=2,fixed_depth=True)
                got=native.select(b,self.model,max_depth=2,fixed_depth=True)
                self.assertEqual(got['move'],ref['move'])
                self.assertAlmostEqual(got['score'],ref['score'],places=7)
                timed=native.select(b,self.model,movetime_ms=1)
                self.assertIn(shogi.Move.from_usi(timed['move']),b.legal_moves)
                self.assertEqual(b.sfen(),sfen);self.assertEqual(+b.transpositions,counts)
                native.state.assert_equal(b,full=True)

    def test_repetition_continuous_check_and_existing_history(self):
        for sfen,words,reason,winner in [
            ('4k4/9/9/9/9/9/9/9/4K4 b - 1',['5i4i','5a4a','4i5i','4a5a'],'repetition_draw',None),
            ('4k4/9/5R3/9/9/9/9/9/4K4 b - 1',['4c5c','5a4a','5c4c','4a5a'],'perpetual_check',1)]:
            b=shogi.Board(sfen)
            for word in words*3:b.push(shogi.Move.from_usi(word))
            with NativeSelector(self.binary) as native:
                result=native.select(b,self.model,movetime_ms=1)
                self.assertIsNone(result['move'])
                self.assertEqual(result['terminal'],{'status':reason,'winner':winner})
                native.state.assert_equal(b,full=True)

    def test_reject_wrong_architecture_mutation_history_and_invalid_handles(self):
        b=shogi.Board()
        with NativeSelector(self.binary) as native:
            native.value(b,self.model)
            extended,_=expand_king_relative(self.model,None)
            with self.assertRaises(ValueError):native.register(extended)
            for options in ({'quiescence':1},{'movetime_ms':0},{'max_depth':65}):
                with self.assertRaises(ValueError):native.select(b,self.model,**options)
            out=ct.c_double()
            self.assertEqual(native.lib.fukaji_nnue_value(0,0,0,ct.byref(out)),-3)
            self.model.hidden_weights[0,0]+=1
            with self.assertRaisesRegex(ValueError,'frozen'):native.select(b,self.model)
        with self.assertRaises(ValueError):native.select(b,self.model)
        b.push(shogi.Move.from_usi('7g7f'))
        with NativeSelector(self.binary) as native:
            native.value(b,self.model);b.pop()
            with self.assertRaises(ValueError):native.sync(b)

    def test_mixed_game_settings_and_paired_backend_follow_candidate(self):
        g=play([self.model]*2,['fixture']*2,[],role='monitor',parent_id='probe',
               seed=17,movetime=1,max_plies=16,native_backend=(True,False))
        self.assertEqual(g['conditions']['backend_by_side'],['cpp_nnue','python'])
        self.assertTrue(all(t['backend']=='cpp_nnue' for t in g['telemetry'] if t['side']==0))
        from test_online_compare import SavedComparisonTests
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);key=SavedComparisonTests().setup_state(root);settings=[]
            def fake(models,ids,prefix,**kwargs):
                settings.append(kwargs['native_backend'])
                return {'role':'monitor','source':'selfplay','parent_id':kwargs['parent_id'],
                        'winner':None,'status':'move_limit_unresolved'}
            with patch('fukaji.online_compare.play',side_effect=fake):
                r=compare_saved(root,key,key,name='native-only',seed=3,pairs=1,
                                left_backend='cpp_nnue')
            self.assertEqual(settings,[(True,False),(False,True)])
            self.assertEqual(r['backend_by_model'],['cpp_nnue','python'])

    def test_native_handles_can_be_reopened_without_leaks(self):
        for _ in range(20):
            with NativeSelector(self.binary) as native:
                self.assertTrue(np.isfinite(native.value(shogi.Board(),self.model)))

    def test_learning_pipeline_keeps_native_backend_for_all_comparisons(self):
        from fukaji.online_cycle import cycle,write_games
        from fukaji.online_learning import save_snapshot,atomic_json
        from test_online_learning import mate_game
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);key=save_snapshot(root/'models',self.model,None,{'parent':None})
            receipt=write_games(root,'bootstrap.jsonl',[mate_game()])
            atomic_json(root/'state.json',{'formal_adoption':False,'paused':False,'cycles':[],
                'research_latest':key,'checkpoint':key,'fixed_anchor':key,'bootstrap':receipt,
                'excluded_openings':[],'protected_parents':[],'seed':1,'used_seeds':[],
                'train_batches':[],'evaluation_batches':[],'history':[key],'generation':0})
            monitor=[{**mate_game(),'role':'monitor','parent_id':'b'*64,'candidate_side':0}]*4
            with patch('fukaji.online_cycle.play',return_value=mate_game()) as playing, \
                 patch('fukaji.online_cycle.compare',return_value=monitor) as comparing:
                report=cycle(root,pairs=1,max_plies=16,periodic_every=1,audit_pairs=2,native_backend=True)
            self.assertEqual(report['config']['backend'],'cpp_nnue')
            self.assertTrue(all(c.kwargs['native_backend'] is True for c in playing.call_args_list))
            self.assertTrue(all(c.kwargs['native_backend'] is True for c in comparing.call_args_list))
            with self.assertRaises(ValueError):cycle(root,native_backend=True,quiescence=1)


if __name__=='__main__':unittest.main()
