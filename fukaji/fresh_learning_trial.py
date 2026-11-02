"""D30: replace repeated recent experience with verified fresh TRAIN outcomes.

Historical draws, parent/Adam, label rule and optimizer budget match D29's
stratified arm. D29 controls are retained rather than retrained or selected.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np

from . import fresh_experience, online_state
from .online_cycle import lock, read_games
from .online_learning import (atomic_json, digest, fit, game_key, load_snapshot,
    make_samples, save_snapshot)
from .online_state import verify
from .stratified_replay_trial import PARENT, SEEDS, TOTAL

NAME='d30-fresh-learning'
COLLECTION_ENGINE='d912c8005fdc78e3fe39726a7ba8737c5989010d5b9b46abccb4dcca92379edb'


def replacement_schedule(old_schedule, fresh_size, old_pool_size, seed):
    if fresh_size<1 or old_pool_size<1 or old_schedule.shape!=(TOTAL,):
        raise ValueError('fixed matched schedule and nonempty fresh pool required')
    recent=np.random.default_rng(seed+1000000).integers(fresh_size,size=TOTAL//2)+old_pool_size
    return np.concatenate((recent,old_schedule[TOTAL//2:]))


def prepare(root, collection):
    root,collection=Path(root),Path(collection)
    with lock(root),lock(collection):
        verify(root);state=json.loads((root/'state.json').read_text())
        if (state['paused'] or state.get('pending_campaign') or state.get('pending_comparison') or
            state['research_latest']!=PARENT or len(state['cycles'])!=15):
            raise ValueError('finish D29 on the frozen parent before ingesting TRAIN')
        if state.get('supplemental_train_batches') or (root/NAME).exists() or (root/'d30-fresh-experience').exists():
            raise ValueError('inspect existing immutable fresh trial instead of reingesting')
        # Refuse use before the standard verifier understands supplemental TRAIN.
        if getattr(online_state,'STATE_SCHEMA_MAX',1)<2:
            raise ValueError('supplemental TRAIN verification support required')
        plan=json.loads((collection/'plan.json').read_text());fresh_experience.validate_plan(plan)
        report=json.loads((collection/'report.json').read_text())
        if (not report.get('complete') or report['completed_games']!=64 or
            report['plan_sha256']!=digest((collection/'plan.json').read_bytes()) or
            plan['source_sha256']!=digest(Path(fresh_experience.__file__).read_bytes()) or
            plan['engine_sha256']!=COLLECTION_ENGINE or len(report['receipts'])!=32):
            raise ValueError('exact completed allowlisted collection required')
        parent,opt,_=load_snapshot(root/'models',PARENT)
        if opt['step']!=1324:raise ValueError('wrong continuation optimizer')
        models={k:load_snapshot(root/'models',k)[0] for k in (PARENT,fresh_experience.OPPONENT)}
        protected=set(state['protected_parents']);games=[]
        for family,receipt in zip(plan['families'],report['receipts']):
            if receipt['path']!=f"family-{family['index']:03d}.jsonl" or family['parent'] in protected:
                raise ValueError('wrong receipt or evaluation parent in fresh TRAIN')
            pair=read_games(collection,receipt)
            fresh_experience.validate_pair(pair,family,plan,models);games.extend(pair)
        if len(games)!=64:raise ValueError('partial collection')
        d29=root/'d29-stratified-replay';prior=json.loads((d29/'report.json').read_text())
        if prior['parent']!=PARENT or digest((d29/'shared-pool.npz').read_bytes())!=prior['pool_sha256']:
            raise ValueError('D29 frozen shared pool changed')
        with np.load(d29/'shared-pool.npz',allow_pickle=False) as pool:
            old_x,old_y=pool['indices'],pool['labels']
        # No unresolved labels, no duplicate/reused trajectories and no monitor.
        old_keys={r['game_sha256'] for r in prior['pool_rows']};unique={}
        for game in games:
            if game_key(game) in old_keys:raise ValueError('fresh trajectory duplicates old TRAIN')
            if game['status'] in ('checkmate','perpetual_check','repetition_draw'):
                unique.setdefault(game_key(game),game)
        if not unique:raise ValueError('no resolved fresh outcome')
        xs=[];ys=[];rows=[];offset=len(old_y)
        for key,game in sorted(unique.items()):
            x,y,_=make_samples([game],parent,seed=int(key[:16],16),per_game=32,
                outcome_weight=.75,td_lambda=.8,protected=protected)
            if x.shape[1]>old_x.shape[1]:raise ValueError('shared sparse width changed')
            xs.append(np.pad(x,((0,0),(0,old_x.shape[1]-x.shape[1])),constant_values=parent.inputs));ys.append(y)
            rows.append(dict(game_sha256=key,first_row=offset,rows=len(y)));offset+=len(y)
        x=np.concatenate([old_x,*xs]);y=np.concatenate([old_y,*ys]);out=root/NAME
        out.mkdir();shutil.copytree(collection,root/'d30-fresh-experience',ignore=shutil.ignore_patterns('.lock'))
        np.savez_compressed(out/'shared-pool.npz',indices=x,labels=y)
        controls={str(seed):prior['candidates'][f'stratified_history_seed{seed}']['model'] for seed in SEEDS}
        protocol=dict(schema='fukaji-fresh-learning-v1',parent=PARENT,parent_optimizer_step=1324,
            final_optimizer_step=1452,positions_per_arm=TOTAL,epochs=2,batch_size=128,lr=.001,
            outcome_weight=.75,td_lambda=.8,seeds=list(SEEDS),representative_seed=SEEDS[0],
            historical_positions_per_arm=4096,fresh_positions_per_arm=4096,
            historical_draws='exact D29 stratified schedules; bootstrap2048 + cycles1-14 NNUE2048',
            recent_replacement='cycle15 27 resolved games replaced by D30 resolved fresh games; pool sizes differ',
            old_pool_size=len(old_y),fresh_pool_size=len(y)-len(old_y),fresh_pool_rows=rows,
            fresh_resolved_unique_games=len(unique),fresh_raw_games=64,
            controls=controls,pool_sha256=digest((out/'shared-pool.npz').read_bytes()),
            collection_plan_sha256=report['plan_sha256'],collection_report_sha256=digest((collection/'report.json').read_bytes()),
            source_sha256=digest(Path(__file__).read_bytes()),formal_adoption=False,
            research_pointer_changed=False,monitor_used_for_training=False,strength_proven=False)
        for seed in SEEDS:
            old=np.load(d29/f'stratified_history_seed{seed}-rows.npy',allow_pickle=False)
            indices=replacement_schedule(old,protocol['fresh_pool_size'],len(old_y),seed)
            np.save(out/f'fresh_seed{seed}-rows.npy',indices,allow_pickle=False)
        state['supplemental_train_batches']=[{**r,'path':'d30-fresh-experience/'+r['path'],
                                              'cohort':'fresh_d30'} for r in report['receipts']]
        state['excluded_openings']=sorted(set(state['excluded_openings'])|{f['parent'] for f in plan['families']})
        state['supplemental_collections']=[dict(plan='d30-fresh-experience/plan.json',
            plan_sha256=report['plan_sha256'],report='d30-fresh-experience/report.json',
            report_sha256=protocol['collection_report_sha256'])]
        state['schema']=2
        atomic_json(root/'state.json',state);verify(root)
        protocol['state_sha256']=digest((root/'state.json').read_bytes())
        atomic_json(out/'protocol.json',protocol)
        return {k:v for k,v in protocol.items() if k!='fresh_pool_rows'}


def train(root):
    with lock(root) as root:
        verify(root);out=root/NAME;plan=json.loads((out/'protocol.json').read_text())
        if ((out/'report.json').exists() or (out/'progress.json').exists() or
            plan['state_sha256']!=digest((root/'state.json').read_bytes()) or
            plan['source_sha256']!=digest(Path(__file__).read_bytes()) or
            plan['pool_sha256']!=digest((out/'shared-pool.npz').read_bytes())):
            raise ValueError('changed input or existing immutable trial')
        parent,opt,_=load_snapshot(root/'models',plan['parent'])
        with np.load(out/'shared-pool.npz',allow_pickle=False) as pool:x,y=pool['indices'],pool['labels']
        report={**plan,'candidates':{}}
        for seed in SEEDS:
            path=out/f'fresh_seed{seed}-rows.npy';indices=np.load(path,allow_pickle=False)
            old=np.load(root/'d29-stratified-replay'/f'stratified_history_seed{seed}-rows.npy',allow_pickle=False)
            expected=replacement_schedule(old,plan['fresh_pool_size'],plan['old_pool_size'],seed)
            if not np.array_equal(expected,indices):raise ValueError('matched sample schedule changed')
            candidate,adam,stats=fit(parent,opt,x[indices],y[indices],epochs=2,lr=.001,batch_size=128,seed=seed)
            if adam['step']!=1452:raise ValueError('optimizer budgets differ')
            key=save_snapshot(root/'models',candidate,adam,dict(parent=PARENT,
                purpose='D30 fresh recent-experience substitution',seed=seed,training=stats,
                pool_sha256=plan['pool_sha256'],sample_schedule_sha256=digest(path.read_bytes()),
                protocol_sha256=digest((out/'protocol.json').read_bytes()),
                matched_control=plan['controls'][str(seed)],outcome_weight=.75,td_lambda=.8))
            _,restored_opt,_=load_snapshot(root/'models',key)
            if restored_opt['step']!=1452:raise ValueError('Adam reload differs')
            report['candidates'][str(seed)]=dict(model=key,training=stats,matched_control=plan['controls'][str(seed)])
            atomic_json(out/'progress.json',dict(completed_candidates=len(report['candidates']),planned_candidates=3))
        atomic_json(out/'report.json',report)
        return report['candidates']


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train']);p.add_argument('--root',required=True)
    p.add_argument('--collection');a=p.parse_args()
    if a.action=='prepare' and not a.collection:p.error('prepare requires completed --collection')
    print(json.dumps(prepare(a.root,a.collection) if a.action=='prepare' else train(a.root),indent=2))


if __name__=='__main__':main()
