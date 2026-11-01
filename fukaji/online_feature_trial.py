"""Matched king-relative feature trial from a continuing research checkpoint.

New relations start at zero. Both arms use identical verified games, sampled
positions, frozen-parent TD labels, optimizer age and optimization conditions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import shogi

from .nnue_research import INPUTS
from .online_cycle import lock, read_games
from .online_compare import compare_saved
from .online_learning import (atomic_json, digest, game_key, load_snapshot, make_samples,
    fit, save_snapshot, expand_king_relative)
from .online_search import Position, select
from .online_state import verify


def train_games(root, state, seed, *, full_replay=False):
    if len(state['train_batches']) < 3:
        raise ValueError('three recent batches required')
    recent = []
    for receipt in state['train_batches'][-3:]:
        recent.extend(g for g in read_games(root, receipt)
                      if g['status'] in ('checkmate', 'perpetual_check', 'repetition_draw'))
    recent = list({game_key(g): g for g in recent}.values())
    if len(recent) != 12:
        raise ValueError('protocol requires twelve recent resolved games')
    recent_ids = {game_key(g) for g in recent}
    historical = list(read_games(root, state['bootstrap']))
    if full_replay:
        for receipt in state['train_batches'][:-3]:
            historical.extend(g for g in read_games(root, receipt)
                              if g['status'] in ('checkmate', 'perpetual_check', 'repetition_draw'))
    historical = list({game_key(g):g for g in historical if game_key(g) not in recent_ids}.values())
    games = recent + (historical if full_replay else random.Random(seed).sample(historical, 52))
    if len({game_key(g) for g in games}) != len(games) or not 64 <= len(games) <= 1000:
        raise ValueError('duplicate training experience')
    return games


def probe(parent, expanded, games):
    positions = []
    seen = set()
    # Both recent and historical selfplay, never an evaluation cohort.
    for game in games[:4] + games[12:16]:
        for fraction in (.25, .5, .75):
            b = shogi.Board(game['start_sfen'])
            for word in game['moves'][:int(len(game['moves'])*fraction)]:
                b.push(shogi.Move.from_usi(word))
            if b.sfen() not in seen:
                positions.append(b); seen.add(b.sfen())
    rows = []
    for b in positions:
        history = Position(b, parent)
        for p in (0, 1):
            np.testing.assert_array_equal(parent.full_accumulator(b,p),expanded.full_accumulator(b,p))
        # Alternate order to reduce one-sided startup/order effects.
        models = [('base',parent),('expanded_zero',expanded)]
        if len(rows)%2: models.reverse()
        results = {name:select(b,model,max_depth=2,fixed_depth=True,history=history)
                   for name,model in models}
        a,c=results['base'],results['expanded_zero']
        if a['move'] != c['move'] or abs(a['score']-c['score']) > 1e-7:
            raise ValueError('zero extension changed fixed-depth decision')
        rows.append({'sfen':b.sfen(),'results':results})
    return {'positions':len(rows),'fixed_depth':2,'moves_and_scores_match':True,
            'elapsed_ms':{name:float(np.mean([r['results'][name]['elapsed_ms'] for r in rows]))
                          for name in ('base','expanded_zero')},
            'rows':rows}


def prepare(root, *, seed=202611012100, full_replay=False, name='d21-feature'):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('single-level trial name required')
    with lock(root) as root:
        verify(root)
        state=json.loads((root/'state.json').read_text())
        if state['paused'] or state.get('pending_comparison'):
            raise ValueError('inspect unfinished state before training')
        out=root/(name+'-training')
        if out.exists(): raise ValueError('immutable trial exists')
        parent_id=state['research_latest']
        parent,opt,_=load_snapshot(root/'models',parent_id)
        expanded,expanded_opt=expand_king_relative(parent,opt)
        games=train_games(root,state,seed,full_replay=full_replay)
        out.mkdir()
        protocol={'parent':parent_id,'seed':seed,'outcome_weight':.75,'td_lambda':.8,
                  'epochs':2,'lr':.001,'recent_games':12,'historical_games':len(games)-12,
                  'full_replay':full_replay,
                  'game_hashes':[game_key(g) for g in games],
                  'implementation_source_sha256':digest(Path(__file__).read_bytes()),
                  'research_pointer_changed':False,'formal_adoption':False,'strength_proven':False}
        atomic_json(out/'protocol.json',protocol)
        measurements=probe(parent,expanded,games)
        atomic_json(out/'zero-extension-probe.json',measurements)
        report={**protocol,'probe':{k:v for k,v in measurements.items() if k!='rows'},'candidates':{}}
        reference_x=reference_y=None
        for name,start,optimizer in (('absolute',parent,opt),('king_relative',expanded,expanded_opt)):
            x,y,stats=make_samples(games,parent,seed=seed,outcome_weight=.75,
                protected=set(state['protected_parents']),feature_model=start)
            if reference_y is None:
                reference_x,reference_y=x.copy(),y.copy()
            else:
                np.testing.assert_array_equal(reference_y,y)
                if len(reference_x)!=len(x) or any(not np.array_equal(a[a<INPUTS],b[b<INPUTS])
                                                  for a,b in zip(reference_x,x)):
                    raise ValueError('sampled positions differ across arms')
            started=time.monotonic()
            candidate,adam,training=fit(start,optimizer,x,y,seed=seed,epochs=2,lr=.001)
            key=save_snapshot(root/'models',candidate,adam,{
                'parent':parent_id,'purpose':'D21 matched king-relative feature trial',
                'seed':seed,'outcome_weight':.75,'td_lambda':.8,'training':training,
                'train_game_hashes':protocol['game_hashes'],'zero_extension':name=='king_relative'})
            report['candidates'][name]={'model':key,'samples':stats,'training':training,
                                        'training_seconds':time.monotonic()-started}
        report['identical_positions_and_labels']=True
        atomic_json(out/'report.json',report)
        return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--seed',type=int,default=202611012100)
    p.add_argument('--full-replay',action='store_true')
    p.add_argument('--name',default='d21-feature')
    a=p.parse_args()
    report=prepare(a.root,seed=a.seed,full_replay=a.full_replay,name=a.name)
    print(json.dumps(report),flush=True)
    result=compare_saved(a.root,report['candidates']['king_relative']['model'],
        report['candidates']['absolute']['model'],name=a.name+'-arena',
        seed=a.seed+100000,pairs=8,movetime=100,max_plies=256)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
