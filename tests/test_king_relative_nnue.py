import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import shogi

from fukaji.nnue_research import (INPUTS, KING_INPUTS, HIDDEN, SmallNNUE,
    DualAccumulator, active_indices, king_relative_indices)
from fukaji.online_learning import (fresh_model, expand_king_relative, parameters,
    make_samples, fit, save_snapshot, load_snapshot, logits_for, probability)
from fukaji.online_search import Position, select
from test_online_learning import mate_game


class KingRelativeTests(unittest.TestCase):
    def setUp(self):
        self.parent = fresh_model(913)
        self.model, _ = expand_king_relative(self.parent, None)
        # Exercise nonzero relation weights, not just the zero-extension case.
        rng = np.random.default_rng(141)
        self.model.hidden_weights[INPUTS:] = rng.normal(0, .002, (KING_INPUTS-INPUTS, HIDDEN))

    def test_relative_features_rotate_and_ignore_absolute_translation(self):
        b = shogi.Board('4k4/9/9/5P3/9/9/9/9/4K4 b - 1')
        # Rotate 180 degrees and swap colors.
        rotated = shogi.Board(); rotated.clear()
        for sq in range(81):
            piece = b.piece_at(sq)
            if piece:
                rotated.set_piece_at(80-sq, shogi.Piece(piece.piece_type, 1-piece.color))
        self.assertEqual(sorted(self.model.active_indices(b, 0)),
                         sorted(self.model.active_indices(rotated, 1)))
        shifted = shogi.Board('9/4k4/9/9/5P3/9/9/9/4K4 b - 1')
        # Only own-king features change when the pawn and enemy king translate.
        enemy = lambda board: sorted(i for i in king_relative_indices(board, 0)
                                     if i >= INPUTS+(KING_INPUTS-INPUTS)//2)
        self.assertEqual(enemy(b), enemy(shifted))
        for board in (b, shogi.Board()):
            for p in (0,1):
                ix = self.model.active_indices(board,p)
                self.assertEqual(len(ix),len(set(ix)))
                self.assertTrue(all(0 <= i < KING_INPUTS for i in ix))
                self.assertEqual(ix[:len(active_indices(board,p))], active_indices(board,p))

    def test_zero_extension_preserves_values_parameters_and_optimizer(self):
        opt = {'step': 123, 'm': [np.full_like(x, .03) for x in parameters(self.parent)],
               's': [np.full_like(x, .04) for x in parameters(self.parent)]}
        model, expanded = expand_king_relative(self.parent, opt)
        self.assertEqual(expanded['step'], 123)
        for kind in ('m','s'):
            np.testing.assert_array_equal(expanded[kind][0][:INPUTS], opt[kind][0])
            self.assertFalse(expanded[kind][0][INPUTS:].any())
            for i in (1,2,3): np.testing.assert_array_equal(expanded[kind][i], opt[kind][i])
        b = shogi.Board(); rng=random.Random(28)
        for _ in range(32):
            for p in (0,1):
                np.testing.assert_array_equal(model.full_accumulator(b,p),self.parent.full_accumulator(b,p))
            self.assertEqual(model.logit(b),self.parent.logit(b))
            legal=list(b.legal_moves)
            if not legal: break
            b.push(rng.choice(legal))
        self.assertFalse(np.shares_memory(expanded['m'][0],opt['m'][0]))
        with self.assertRaises(ValueError): expand_king_relative(model, expanded)

    def test_incremental_captures_drops_promotions_and_king_moves_exact_undo(self):
        sfens=[shogi.STARTING_SFEN, '4k4/9/4P4/9/9/9/9/9/4K4 b P 1',
               '4k4/9/4R4/9/9/9/9/9/4K4 w - 1',
               '4k4/9/3+p5/4S4/9/9/9/9/4K4 b - 1']
        for sfen in sfens:
            b=shogi.Board(sfen); state=Position(b,self.model)
            for move in list(b.legal_moves):
                before=state.acc.copy();counts=+b.transpositions
                state.push(move); state.verify(); state.pop(); state.verify()
                np.testing.assert_array_equal(state.acc,before)
                self.assertEqual(b.sfen(),sfen);self.assertEqual(+b.transpositions,counts)
        for state_type in (Position,DualAccumulator):
            b=shogi.Board(); state=state_type(b,self.model); rng=random.Random(83)
            for _ in range(80):
                legal=list(b.legal_moves)
                if not legal: break
                state.push(rng.choice(legal));state.verify()
                self.assertAlmostEqual(state.logit(),self.model.logit(b),places=8)
            while state.history if isinstance(state,DualAccumulator) else state.undo:
                state.pop();state.verify()
            self.assertEqual(b.sfen(),shogi.STARTING_SFEN)

    def test_same_targets_new_feature_gradient_and_optimizer_roundtrip(self):
        g=mate_game()
        x,y,_=make_samples([g],self.parent,seed=1)
        k,z,_=make_samples([g],self.parent,seed=1,feature_model=self.model)
        np.testing.assert_array_equal(y,z)
        self.assertEqual([list(row[row<INPUTS]) for row in k],
                         [list(row[row<INPUTS]) for row in x])
        params=parameters(self.model)
        pre,_,logits=logits_for(params,k)
        unit=next(i for i in range(HIDDEN) if 0<pre[0,i]<1)
        feature=next(i for i in k[0] if INPUTS<=i<KING_INPUTS)
        analytic=float((probability(logits)[0]-z[0])*params[2][unit]/len(z))
        def loss(delta):
            changed=[a.copy() for a in params];changed[0][feature,unit]+=delta
            values=logits_for(changed,k)[2]
            return float(np.mean(np.logaddexp(0,values)-z*values))
        numeric=(loss(.001)-loss(-.001))/.002
        self.assertAlmostEqual(analytic,numeric,delta=4e-5)
        trained,opt,_=fit(self.model,None,k,z)
        self.assertTrue(np.any(trained.hidden_weights[INPUTS:]!=self.model.hidden_weights[INPUTS:]))
        with tempfile.TemporaryDirectory() as d:
            key=save_snapshot(d,trained,opt,{'parent':None})
            restored,adam,_=load_snapshot(d,key)
            np.testing.assert_array_equal(restored.hidden_weights,trained.hidden_weights)
            self.assertEqual(adam['step'],opt['step'])
            path=Path(d)/(key+'.json'); manifest=json.loads(path.read_text())
            manifest['schema']='fukaji-online-nnue-v1';path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'architecture/schema'):load_snapshot(d,key)

    def test_fixed_depth_and_timeout_restore_with_relation_weights(self):
        b=shogi.Board(mate_game()['start_sfen'])
        rows=[select(b,self.model,max_depth=2,fixed_depth=True,incremental=i) for i in (False,True)]
        self.assertEqual(rows[0]['move'],rows[1]['move'])
        self.assertAlmostEqual(rows[0]['score'],rows[1]['score'],places=7)
        b=shogi.Board();before=b.sfen();counts=+b.transpositions
        row=select(b,self.model,movetime_ms=1)
        self.assertIn(shogi.Move.from_usi(row['move']),b.legal_moves)
        self.assertEqual(b.sfen(),before);self.assertEqual(+b.transpositions,counts)

    def test_full_replay_uses_only_training_receipts_and_excludes_unresolved(self):
        from fukaji.online_feature_trial import train_games
        def game(number, status='checkmate'):
            g=mate_game()
            g['start_sfen']=g['start_sfen'].rsplit(' ',1)[0]+' '+str(number)
            return {**g, 'status':status}
        batches={'bootstrap':[game(i+1) for i in range(60)],
                 'old':[game(i+100) for i in range(3)]+[game(199,'move_limit_unresolved')],
                 **{f'recent-{i}':[game(200+i*4+j) for j in range(4)] for i in range(3)},
                 'monitor':[game(999)]}
        state={'bootstrap':{'path':'bootstrap'},
               'train_batches':[{'path':name} for name in ('old','recent-0','recent-1','recent-2')],
               'evaluation_batches':[{'path':'monitor'}]}
        with patch('fukaji.online_feature_trial.read_games', side_effect=lambda root,r:batches[r['path']]):
            small=train_games(None,state,17)
            full=train_games(None,state,17,full_replay=True)
        self.assertEqual(len(small),64);self.assertEqual(len(full),75)
        self.assertTrue(all(g['status']=='checkmate' for g in full))
        self.assertNotIn(batches['monitor'][0],full)
        self.assertEqual(small[:12],full[:12])


if __name__=='__main__':unittest.main()
