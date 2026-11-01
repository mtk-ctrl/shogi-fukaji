"""D25: matched optimizer budget, varying only recent/historical sample ratio.

The shared pool contains verified TRAIN only. No pointer update or strength
claim is made here. Candidates and exact sample schedules remain immutable.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from .online_cycle import lock, read_games
from .online_learning import (atomic_json, digest, game_key, load_snapshot,
    make_samples, fit, save_snapshot)
from .online_state import verify

SEEDS=(202611012500,202611012501,202611012502)
TOTAL=1024


def schedules(recent_size, historical_size, seed):
    if recent_size<1 or historical_size<1:raise ValueError('both experience pools required')
    rng=np.random.default_rng(seed)
    recent=rng.integers(recent_size,size=TOTAL//2)
    old=rng.integers(historical_size,size=3*TOTAL//4)+recent_size
    return {'recent50_old50':np.concatenate((recent,old[:TOTAL//2])),
            'recent25_old75':np.concatenate((recent[:TOTAL//4],old))}


def prepare(root, *, name='d25-replay-ratio'):
    if not name or Path(name).name!=name or name in ('.','..'):raise ValueError('invalid name')
    with lock(root) as root:
        verify(root);state=json.loads((root/'state.json').read_text())
        if state['paused'] or state.get('pending_comparison') or state.get('pending_campaign'):
            raise ValueError('finish pending evaluation before training')
        if len(state['cycles'])!=14:raise ValueError('D25 requires frozen cycle14 lineage')
        out=root/name
        if out.exists():raise ValueError('immutable replay trial exists')
        parent_id=state['research_latest'];parent,opt,_=load_snapshot(root/'models',parent_id)
        recent=[];old=read_games(root,state['bootstrap'])
        for i,receipt in enumerate(state['train_batches']):
            (recent if i>=11 else old).extend(read_games(root,receipt))
        resolved=lambda games:{game_key(g):g for g in games if g['status'] in
            ('checkmate','perpetual_check','repetition_draw')}
        recent=resolved(recent);old={k:g for k,g in resolved(old).items() if k not in recent}
        if len(recent)!=11 or len(old)!=707:raise ValueError('unexpected frozen experience pools')
        # Each game's pool is sampled once, independently of arm and schedule.
        xs=[];ys=[];recent_rows=0;rows=[]
        for group,games in (('recent',recent),('historical',old)):
            for key,game in sorted(games.items()):
                x,y,_=make_samples([game],parent,seed=int(key[:16],16),per_game=32,
                    outcome_weight=.75,td_lambda=.8,protected=state['protected_parents'])
                rows.append({'group':group,'game_sha256':key,'first_row':sum(len(a) for a in ys),
                             'rows':len(y)})
                xs.append(x);ys.append(y)
                if group=='recent':recent_rows+=len(y)
        width=max(x.shape[1] for x in xs)
        x=np.concatenate([np.pad(a,((0,0),(0,width-a.shape[1])),constant_values=parent.inputs) for a in xs])
        y=np.concatenate(ys);out.mkdir()
        np.savez_compressed(out/'shared-pool.npz',indices=x,labels=y)
        protocol={'schema':'fukaji-replay-ratio-v1','parent':parent_id,'parent_optimizer_step':opt['step'],
            'recent_games':len(recent),'historical_games':len(old),'pool_rows':rows,
            'pool_sha256':digest((out/'shared-pool.npz').read_bytes()),
            'seeds':list(SEEDS),'positions_per_arm':TOTAL,'epochs':2,'batch_size':128,'lr':.001,
            'expected_optimizer_steps_per_arm':16,'outcome_weight':.75,'td_lambda':.8,
            'sampling':'with replacement, uniform over shared sampled position pool within each group',
            'arms_recent_fraction':{'recent50_old50':.5,'recent25_old75':.25},
            'source_sha256':digest(Path(__file__).read_bytes()),'representative_seed':SEEDS[0],
            'all_seeds_reported':True,'formal_adoption':False,'research_pointer_changed':False,
            'strength_proven':False,'monitor_used_for_training':False}
        atomic_json(out/'protocol.json',protocol)
        report={**protocol,'candidates':{}}
        for seed in SEEDS:
            for arm,indices in schedules(recent_rows,len(y)-recent_rows,seed).items():
                candidate,adam,stats=fit(parent,opt,x[indices],y[indices],seed=seed,
                    epochs=2,lr=.001,batch_size=128)
                if adam['step']!=opt['step']+16:raise ValueError('optimizer budgets differ')
                label=f'{arm}_seed{seed}'
                np.save(out/(label+'-rows.npy'),indices,allow_pickle=False)
                key=save_snapshot(root/'models',candidate,adam,{'parent':parent_id,
                    'purpose':'D25 fixed-budget replay-ratio trial','seed':seed,'arm':arm,
                    'training':stats,'pool_sha256':protocol['pool_sha256'],
                    'sample_schedule_sha256':digest((out/(label+'-rows.npy')).read_bytes()),
                    'outcome_weight':.75,'td_lambda':.8})
                report['candidates'][label]={'model':key,'seed':seed,'arm':arm,'training':stats}
                atomic_json(out/'progress.json',{'completed_candidates':len(report['candidates']),
                    'planned_candidates':6})
        atomic_json(out/'report.json',report)
        return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.root),indent=2))


if __name__=='__main__':main()
