import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import shogi

from fukaji.nnue_research import SmallNNUE, INPUTS, HIDDEN, _piece_index, _hand_index
from fukaji.online_search import Position, select
from fukaji.online_compare import compare_saved
from fukaji.online_cycle import play


def synthetic_material_fixture():
    """Synthetic regression-only weights; never a training or playing model."""
    w=np.zeros((INPUTS,HIDDEN),np.float32);b=np.zeros(HIDDEN,np.float32)
    v=np.zeros(HIDDEN,np.float32);b[0]=.5;v[0]=1.
    values=[0,.005,.01,.015,.02,.025,.035,.045,0,.025,.025,.025,.025,.04,.05]
    for color in (0,1):
        sign=1 if color==0 else -1
        for kind in range(1,15):
            for sq in range(81):
                w[_piece_index(shogi.Piece(kind,color),sq,0),0]=sign*values[kind]
        for kind in range(1,8):
            for count in range(20):w[_hand_index(color,kind,count,0),0]=sign*values[kind]
    return SmallNNUE(w,b,v,-.5)


class QuiescenceTests(unittest.TestCase):
    def setUp(self): self.model=synthetic_material_fixture()

    def test_poisoned_capture_is_seen_through_recapture(self):
        b=shogi.Board('4k4/9/5g3/4p4/9/4R4/9/9/4K4 b - 1')
        plain=select(b,self.model,max_depth=1,fixed_depth=True)
        quiet=select(b,self.model,max_depth=1,fixed_depth=True,quiescence=2)
        self.assertEqual(plain['move'],'5f5d')
        self.assertNotEqual(quiet['move'],'5f5d')
        self.assertGreater(quiet['quiescence_nodes'],0)
        self.assertEqual(plain['quiescence_nodes'],0)
        self.assertEqual(b.sfen(),'4k4/9/5g3/4p4/9/4R4/9/9/4K4 b - 1')

    def test_checked_leaf_cannot_use_stand_pat(self):
        b=shogi.Board('4k4/9/5R3/9/9/9/9/9/4K4 b - 1')
        original=Position.logit
        def guarded(state):
            self.assertFalse(state.board.is_check())
            return original(state)
        with patch.object(Position,'logit',guarded):
            result=select(b,self.model,max_depth=1,fixed_depth=True,quiescence=1)
        self.assertIn(shogi.Move.from_usi(result['move']),b.legal_moves)

    def test_quiet_position_matches_old_leaf_and_bounds_restore(self):
        b=shogi.Board();original=b.sfen();counts=+b.transpositions
        old=select(b,self.model,max_depth=1,fixed_depth=True)
        new=select(b,self.model,max_depth=1,fixed_depth=True,quiescence=2)
        self.assertEqual(old['move'],new['move'])
        self.assertAlmostEqual(old['score'],new['score'],places=7)
        for q in (1,2,4):
            result=select(b,self.model,movetime_ms=1,quiescence=q)
            self.assertIn(shogi.Move.from_usi(result['move']),b.legal_moves)
            self.assertEqual(b.sfen(),original);self.assertEqual(+b.transpositions,counts)
        for q in (-1,5,True,1.5):
            with self.assertRaises(ValueError):select(b,self.model,quiescence=q)
        with self.assertRaises(ValueError):
            play([self.model]*2,['x']*2,[],role='monitor',parent_id='p',seed=1,quiescence=(2,5))

    def test_repetition_and_mate_still_precede_value_approximation(self):
        b=shogi.Board('4k4/9/9/9/9/9/9/9/4K4 b - 1')
        for word in ['5i4i','5a4a','4i5i','4a5a']*3:b.push(shogi.Move.from_usi(word))
        result=select(b,self.model,quiescence=2)
        self.assertIsNone(result['move'])
        self.assertEqual(result['terminal']['status'],'repetition_draw')
        b=shogi.Board('7lk/8p/7G1/9/9/9/9/9/4K3R b - 1')
        result=select(b,self.model,max_depth=1,fixed_depth=True,quiescence=2)
        b.push(shogi.Move.from_usi(result['move']));self.assertTrue(b.is_checkmate())

    def test_paired_comparison_keeps_search_setting_with_candidate(self):
        from test_online_compare import SavedComparisonTests
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);key=SavedComparisonTests().setup_state(root)
            flags=[]
            def fake_play(models,ids,prefix,**kwargs):
                flags.append(kwargs['quiescence'])
                return {'role':'monitor','source':'selfplay','parent_id':kwargs['parent_id'],
                        'winner':None,'status':'move_limit_unresolved'}
            with patch('fukaji.online_compare.play',side_effect=fake_play):
                report=compare_saved(root,key,key,name='q-only',seed=3,pairs=1,
                                     left_quiescence=2,right_quiescence=0)
            self.assertEqual(flags,[(2,0),(0,2)])
            self.assertEqual(report['quiescence_by_model'],[2,0])
            self.assertFalse(report['research_pointer_changed'])


if __name__=='__main__':unittest.main()
