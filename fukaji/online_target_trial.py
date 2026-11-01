"""One-factor MC/TD trial from an unchanged saved parent and identical experience."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random

import numpy as np
from .online_cycle import lock, read_games
from .online_compare import compare_saved
from .online_learning import atomic_json, game_key, load_snapshot, make_samples, fit, save_snapshot
from .online_state import verify


def prepare(root, *, seed):
    with lock(root) as root:
        verify(root)
        state = json.loads((root/'state.json').read_text())
        if state['paused'] or state.get('pending_comparison'):
            raise ValueError('inspect paused state before training')
        output = root/'d20-target-training'
        if output.exists():
            raise ValueError('immutable trial exists')
        if len(state['train_batches']) < 3:
            raise ValueError('three recent batches required')
        recent = []
        for receipt in state['train_batches'][-3:]:
            recent.extend(g for g in read_games(root, receipt)
                          if g['status'] in ('checkmate', 'perpetual_check', 'repetition_draw'))
        recent = list({game_key(g): g for g in recent}.values())
        if len(recent) != 12:
            raise ValueError('protocol requires twelve recent resolved games')
        bootstrap = read_games(root, state['bootstrap'])
        old = random.Random(seed).sample(bootstrap, 12)
        games = recent+old
        if len({game_key(g) for g in games}) != 24:
            raise ValueError('duplicate experience')
        output.mkdir()
        parent_id = state['research_latest']
        parent, optimizer, _ = load_snapshot(root/'models', parent_id)
        report = {'parent': parent_id, 'seed': seed, 'epochs': 2, 'lr': .001,
                  'game_hashes': [game_key(g) for g in games], 'recent_games': 12,
                  'bootstrap_games': 12, 'formal_adoption': False, 'strength_proven': False,
                  'research_pointer_changed': False, 'candidates': {}}
        previous = None
        for name, weight in (('mc', 1.), ('td', .75)):
            x,y,stats = make_samples(games, parent, seed=seed, outcome_weight=weight,
                                    protected=set(state['protected_parents']))
            if previous is not None and not np.array_equal(x, previous):
                raise ValueError('sample indices differ between arms')
            previous = x.copy()
            candidate, opt, training = fit(parent, optimizer, x, y, seed=seed, epochs=2, lr=.001)
            key = save_snapshot(root/'models', candidate, opt, {
                'parent': parent_id, 'purpose': 'D20 matched MC/TD target comparison',
                'seed': seed, 'outcome_weight': weight, 'td_lambda': .8,
                'train_game_hashes': report['game_hashes'], 'training': training})
            report['candidates'][name] = {'model': key, 'samples': stats, 'training': training}
        report['identical_sample_indices'] = True
        atomic_json(output/'report.json', report)
        return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--seed', type=int, default=202611012020)
    a = p.parse_args()
    training = prepare(a.root, seed=a.seed)
    print(json.dumps(training), flush=True)
    result = compare_saved(a.root, training['candidates']['mc']['model'],
        training['candidates']['td']['model'], name='d20-target-arena',
        seed=a.seed+100000, pairs=8, movetime=100, max_plies=256)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
