import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import shogi

from fukaji.nnue_research import INPUTS, active_indices
from fukaji.online_learning import (fresh_model, targets, fit, probability,
    save_snapshot, load_snapshot, make_samples, validate_game, atomic_json)
from fukaji.online_search import Position, select, terminal
from fukaji.online_cycle import gate, write_games, read_games, lock, cycle


def mate_game():
    return {"start_sfen": "7lk/8p/7G1/9/9/9/9/9/4K3R b - 1",
            "moves": ["2c1b"], "winner": 0, "status": "checkmate",
            "source": "selfplay", "role": "train", "parent_id": "a"*64}


class OnlineLearningTests(unittest.TestCase):
    def setUp(self):
        self.model = fresh_model(17)

    def test_td_terminal_anchor_and_player_sign(self):
        z = np.array([1., 0., 1.])
        p = [.2, .3, .4]
        np.testing.assert_allclose(targets(p, z, outcome_weight=1), z)
        np.testing.assert_allclose(targets(p, z, outcome_weight=0, td_lambda=1), z)
        np.testing.assert_allclose(targets(p, z, outcome_weight=0, td_lambda=0), [.7, .6, 1])
        np.testing.assert_allclose(targets([.5]*3, [.5]*3), [.5]*3)
        with self.assertRaises(ValueError): targets([float('nan')], [1])

    def test_loss_can_correct_a_previous_win_prediction(self):
        ix = np.array([active_indices(shogi.Board(), 0)], np.int32)
        first, opt, _ = fit(self.model, None, ix, np.array([1.], np.float32), epochs=20, lr=.01)
        before = probability(first.logit(shogi.Board()))
        corrected, _, _ = fit(first, opt, ix, np.array([0.], np.float32), epochs=20, lr=.01)
        self.assertLess(probability(corrected.logit(shogi.Board())), before)
        self.assertGreater(opt['step'], 0)

    def test_holdout_and_bad_results_are_rejected(self):
        g = mate_game()
        self.assertTrue(validate_game(g, self.model)[3])
        with self.assertRaises(ValueError):
            make_samples([g], self.model, seed=1, protected={g['parent_id']})
        for role in ('monitor', 'audit'):
            with self.assertRaises(ValueError):
                make_samples([{**g, 'role': role}], self.model, seed=1)
        with self.assertRaises(ValueError):
            validate_game({**g, 'winner': 1}, self.model)
        with self.assertRaises(ValueError):
            validate_game({**g, 'source': 'external'}, self.model)
        with self.assertRaises(ValueError):
            validate_game({**g, 'moves': ['5c5d']}, self.model)

    def test_unresolved_is_not_a_draw_or_training_label(self):
        g = {**mate_game(), 'start_sfen': shogi.STARTING_SFEN,
             'moves': ['7g7f'], 'winner': None, 'status': 'move_limit_unresolved'}
        self.assertFalse(validate_game(g, self.model)[3])
        with self.assertRaises(ValueError): make_samples([g], self.model, seed=1)
        with self.assertRaises(ValueError):
            validate_game({**g, 'status': 'repetition_draw'}, self.model)

    def test_model_optimizer_roundtrip_and_corruption(self):
        x, y, _ = make_samples([mate_game()], self.model, seed=1)
        m, opt, _ = fit(self.model, None, x, y)
        with tempfile.TemporaryDirectory() as d:
            key = save_snapshot(d, m, opt, {'parent': None})
            loaded, restored, _ = load_snapshot(d, key)
            self.assertEqual(restored['step'], opt['step'])
            self.assertAlmostEqual(loaded.logit(shogi.Board()), m.logit(shogi.Board()), places=7)
            p = Path(d)/(key+'.npz')
            p.write_bytes(p.read_bytes()[:-1]+b'X')
            with self.assertRaises(ValueError): load_snapshot(d, key)

    def test_snapshot_provenance_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            save_snapshot(d, self.model, None, {'parent': None})
            with self.assertRaises(ValueError):
                save_snapshot(d, self.model, None, {'parent': 'forged'})

    def test_sparse_gradient_matches_finite_difference(self):
        from fukaji.online_learning import parameters, logits_for
        params = parameters(self.model)
        ix = np.array([active_indices(shogi.Board(), 0)], np.int32)
        pre, hidden, logits = logits_for(params, ix)
        j = next(i for i in range(32) if 0 < pre[0, i] < 1)
        feature = ix[0, 0]
        analytic = (probability(logits)[0]-1)*params[2][j]
        def loss(delta):
            p = [a.copy() for a in params]; p[0][feature,j] += delta
            value = logits_for(p, ix)[2][0]
            return np.logaddexp(0, value)-value
        numeric = (loss(.001)-loss(-.001))/.002
        self.assertAlmostEqual(float(analytic), float(numeric), delta=3e-5)


class OnlineSearchTests(unittest.TestCase):
    def setUp(self): self.model = fresh_model(71)

    def test_timeout_before_depth_one_keeps_evaluated_best(self):
        from fukaji.engine import ordered_moves
        b = shogi.Board()
        moves = ordered_moves(b, fast_move_list=True, fast_legal_guard=True)
        with patch('fukaji.online_search.time.monotonic',
                   side_effect=[0, 0, 0, 0, 0, .002, .002]), \
             patch.object(Position, 'logit', side_effect=[-1., -2.]):
            result = select(b, self.model, movetime_ms=1)
        self.assertEqual(result['move'], moves[1].usi())
        self.assertEqual(result['depth'], 0)
        self.assertEqual(result['partial_root_moves'], 2)
        self.assertEqual(b.sfen(), shogi.STARTING_SFEN)

    def test_lazy_leaf_preserves_exact_search_and_mate(self):
        for sfen in (shogi.STARTING_SFEN, mate_game()['start_sfen']):
            b = shogi.Board(sfen)
            slow = select(b, self.model, max_depth=2, fixed_depth=True, fast_leaf=False)
            fast = select(b, self.model, max_depth=2, fixed_depth=True, fast_leaf=True)
            for key in ('move', 'nodes', 'evaluations', 'depth', 'score'):
                self.assertEqual(slow[key], fast[key])

    def test_incremental_all_move_types_and_exact_undo(self):
        sfens = [shogi.STARTING_SFEN,
                 '4k4/9/4P4/9/9/9/9/9/4K4 b P 1',
                 '4k4/9/4R4/9/9/9/9/9/4K4 w - 1']
        for sfen in sfens:
            b = shogi.Board(sfen); p = Position(b, self.model)
            for move in list(b.legal_moves):
                acc = p.acc.copy(); counts = +b.transpositions
                p.push(move); p.verify(); p.pop(); p.verify()
                np.testing.assert_array_equal(acc, p.acc)
                self.assertEqual(b.sfen(), sfen); self.assertEqual(+b.transpositions, counts)
        b = shogi.Board(); p = Position(b, self.model); rng = random.Random(7)
        for _ in range(100):
            legal = list(b.legal_moves)
            if not legal: break
            p.push(rng.choice(legal)); p.verify()
        while p.undo: p.pop(); p.verify()
        self.assertEqual(b.sfen(), shogi.STARTING_SFEN)

    def test_repetition_draw_and_continuous_check_loss(self):
        fixtures = [
            ('4k4/9/9/9/9/9/9/9/4K4 b - 1', ['5i4i','5a4a','4i5i','4a5a'], 'repetition_draw', None),
            ('4k4/9/5R3/9/9/9/9/9/4K4 b - 1', ['4c5c','5a4a','5c4c','4a5a'], 'perpetual_check', 1)]
        for sfen, words, reason, winner in fixtures:
            b=shogi.Board(sfen); p=Position(b,self.model)
            for word in words*3: p.push(shogi.Move.from_usi(word))
            self.assertEqual(p.repetition(), {'status': reason, 'winner': winner})
            # Existing played history must work when starting a new search.
            self.assertEqual(Position(b,self.model).repetition(), p.repetition())

    def test_fixed_depth_incremental_matches_full_and_ordering(self):
        b=shogi.Board(mate_game()['start_sfen'])
        rows=[select(b,self.model,max_depth=2,fixed_depth=True,incremental=i,ordering=o)
              for i,o in [(False,False),(True,False),(True,True)]]
        self.assertEqual(len({r['score'] for r in rows}),1)
        for row in rows:
            move=shogi.Move.from_usi(row['move']); self.assertIn(move,b.legal_moves)
            b.push(move); self.assertTrue(b.is_checkmate()); b.pop()

    def test_timeout_restores_history_and_returns_legal_move(self):
        b=shogi.Board(); before=b.sfen(); counts=+b.transpositions
        row=select(b,self.model,movetime_ms=1)
        self.assertIn(shogi.Move.from_usi(row['move']),b.legal_moves)
        self.assertEqual(b.sfen(),before);self.assertEqual(+b.transpositions,counts)


class OnlineCycleTests(unittest.TestCase):
    def test_collapse_and_insufficient_evidence_differ(self):
        losses=[{'winner':1,'candidate_side':0,'status':'checkmate'}]*4
        unresolved=[{'winner':None,'candidate_side':0,'status':'move_limit_unresolved'}]*4
        self.assertEqual(gate(losses)['action'],'rollback')
        self.assertEqual(gate(unresolved)['action'],'provisional')
        self.assertEqual(gate(unresolved,periodic=True)['action'],'pause')
        draws=[{'winner':None,'candidate_side':0,'status':'repetition_draw'}]*8
        self.assertEqual(gate(draws,periodic=True)['action'],'pause')

    def test_immutable_games_hash_and_lock(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); receipt=write_games(root,'games.jsonl',[mate_game()])
            self.assertEqual(read_games(root,receipt),[mate_game()])
            with self.assertRaises(ValueError):write_games(root,'games.jsonl',[mate_game()])
            with lock(root):
                with self.assertRaises(BlockingIOError):
                    with lock(root):pass
            (root/'games.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):read_games(root,receipt)

    def test_cycle_rollback_preserves_formal_anchor_and_evaluation_is_not_replay(self):
        model=fresh_model(71)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            key=save_snapshot(root/'models',model,None,{'parent':None})
            receipt=write_games(root,'bootstrap.jsonl',[mate_game()])
            state={'formal_adoption':False,'paused':False,'cycles':[], 'research_latest':key,
                   'checkpoint':key,'fixed_anchor':key,'bootstrap':receipt,'excluded_openings':[],
                   'protected_parents':[],'seed':1,'used_seeds':[],'train_batches':[],
                   'evaluation_batches':[],'history':[key],'generation':0}
            atomic_json(root/'state.json',state)
            losses=[{**mate_game(),'role':'monitor','winner':0,'candidate_side':1,'parent_id':'b'*64}]*4
            with patch('fukaji.online_cycle.play',return_value=mate_game()), patch('fukaji.online_cycle.compare',return_value=losses):
                report=cycle(root,pairs=1,max_plies=16)
            actual=json.loads((root/'state.json').read_text())
            self.assertEqual(actual['research_latest'],key)
            self.assertEqual(actual['fixed_anchor'],key)
            self.assertFalse(actual['formal_adoption']);self.assertTrue(actual['paused'])
            self.assertIn('b'*64,actual['protected_parents'])
            self.assertEqual(report['decision']['action'],'rollback')


if __name__ == '__main__': unittest.main()
