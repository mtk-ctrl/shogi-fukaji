"""D31 matched continuation: fresh-only versus retaining two recent cohorts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from . import next_train_collection as collection
from .online_cycle import lock,read_games
from .online_learning import (atomic_json,digest,implementation_digest,load_snapshot,save_snapshot,
                             make_samples,validate_game,game_key,fit,parameters,logits_for)
from .online_state import verify

NAME='d31-retained-replay'
GROUPS=('fresh','d30','cycle15','bootstrap','historical_nnue')
SEEDS=(202611033101,202611033102,202611033103)
TOTAL=8192


def schedules(sizes,seed):
    if set(sizes)!=set(GROUPS) or any(type(n) is not int or n<1 for n in sizes.values()):
        raise ValueError('five nonempty TRAIN cohorts required')
    offsets={};offset=0
    for group in GROUPS:offsets[group]=offset;offset+=sizes[group]
    rng=np.random.default_rng(seed)
    counts=dict(fresh=4096,d30=1024,cycle15=1024,bootstrap=2048,historical_nnue=2048)
    draws={g:rng.integers(sizes[g],size=counts[g])+offsets[g] for g in GROUPS}
    history=np.concatenate((draws['bootstrap'],draws['historical_nnue']))
    return dict(fresh_only=np.concatenate((draws['fresh'],history)),
                retain_two=np.concatenate((draws['fresh'][:2048],draws['d30'],draws['cycle15'],history)))


def partition(batches,parent,protected):
    groups={g:{} for g in GROUPS};seen=set()
    for group,games in batches:
        if group not in groups:raise ValueError('unknown TRAIN cohort')
        for game in games:
            resolved=validate_game(game,parent,training=True,protected=protected)[3];key=game_key(game)
            if key in seen:continue
            seen.add(key)
            if resolved:groups[group][key]=game
    if any(not groups[g] for g in GROUPS):raise ValueError('empty resolved TRAIN cohort')
    return groups


def losses(model,x,y,sizes):
    result={};start=0;params=parameters(model)
    for group in GROUPS:
        end=start+sizes[group];total=0.
        for i in range(start,end,512):
            stop=min(end,i+512);z=logits_for(params,x[i:stop])[2]
            total+=float(np.sum(np.logaddexp(0,z)-y[i:stop]*z))
        result[group]=total/sizes[group];start=end
    return result


def prepare(root):
    with lock(root) as root:
        verify(root);state=json.loads((root/'state.json').read_text());out=root/NAME;c=root/collection.NAME
        if (out.exists() or state['paused'] or state.get('pending_campaign') or state.get('pending_comparison') or
            state['research_latest']!=collection.PARENT or len(state['cycles'])!=15):
            raise ValueError('idle frozen approved lineage and new trial required')
        plan=json.loads((c/'plan.json').read_text());collection.validate_plan(plan)
        report=json.loads((c/'report.json').read_text())
        if (report.get('complete') is not True or report['completed_games']!=64 or
            report['plan_sha256']!=digest((c/'plan.json').read_bytes()) or
            plan['source_sha256']!=digest(Path(collection.__file__).read_bytes()) or
            plan['engine_sha256']!=implementation_digest() or
            plan['state_sha256']!=digest((root/'state.json').read_bytes()) or len(report['receipts'])!=32):
            raise ValueError('verified completed allowlisted collection required')
        models={k:load_snapshot(root/'models',k)[0] for k in (collection.PARENT,collection.OPPONENT)}
        fresh=[]
        for family,receipt in zip(plan['families'],report['receipts']):
            if receipt['path']!=f"family-{family['index']:03d}.jsonl":raise ValueError('wrong family receipt')
            games=read_games(c,receipt);collection.validate_pair(games,family,plan,models);fresh.extend(games)
        parent,opt,_=load_snapshot(root/'models',collection.PARENT)
        if opt['step']!=1452:raise ValueError('wrong optimizer continuation')
        batches=[('fresh',fresh),('d30',sum((read_games(root,r) for r in state['supplemental_train_batches']),[]))]
        batches.extend(('cycle15' if r['path'].startswith('cycle-0015/') else 'historical_nnue',read_games(root,r))
                       for r in state['train_batches'])
        batches.append(('bootstrap',read_games(root,state['bootstrap'])))
        groups=partition(batches,parent,state['protected_parents'])
        if any(len(groups[g])!=n for g,n in dict(d30=55,cycle15=27,bootstrap=665,historical_nnue=53).items()):
            raise ValueError('wrong historical training cohorts')
        xs=[];ys=[];rows=[];sizes={};offset=0
        for group in GROUPS:
            sizes[group]=0
            for key,game in sorted(groups[group].items()):
                x,y,_=make_samples([game],parent,seed=int(key[:16],16),per_game=32,
                    outcome_weight=.75,td_lambda=.8,protected=state['protected_parents'])
                xs.append(x);ys.append(y);rows.append(dict(group=group,game_sha256=key,first_row=offset,rows=len(y)))
                offset+=len(y);sizes[group]+=len(y)
        width=max(x.shape[1] for x in xs)
        x=np.concatenate([np.pad(a,((0,0),(0,width-a.shape[1])),constant_values=parent.inputs) for a in xs]);y=np.concatenate(ys)
        out.mkdir();np.savez_compressed(out/'shared-pool.npz',indices=x,labels=y)
        # Admit only this independently replay-verified collection. Historical receipts remain unchanged.
        state['supplemental_train_batches'].extend({**r,'path':collection.NAME+'/'+r['path'],'cohort':'fresh_d31'} for r in report['receipts'])
        state['supplemental_collections'].append(dict(plan=collection.NAME+'/plan.json',plan_sha256=report['plan_sha256'],
            report=collection.NAME+'/report.json',report_sha256=digest((c/'report.json').read_bytes())))
        state['excluded_openings']=sorted(set(state['excluded_openings'])|{f['parent'] for f in plan['families']})
        atomic_json(root/'state.json',state);verify(root)
        protocol=dict(schema='fukaji-retained-replay-v1',parent=collection.PARENT,parent_optimizer_step=1452,
            final_optimizer_step=1580,positions_per_arm=TOTAL,epochs=2,batch_size=128,lr=.001,outcome_weight=.75,td_lambda=.8,
            seeds=list(SEEDS),representative_seed=SEEDS[0],arms=['fresh_only','retain_two'],
            games={g:len(v) for g,v in groups.items()},pool_sizes=sizes,pool_rows=rows,
            pool_sha256=digest((out/'shared-pool.npz').read_bytes()),source_sha256=digest(Path(__file__).read_bytes()),
            state_sha256=digest((root/'state.json').read_bytes()),engine_sha256=implementation_digest(),
            collection_plan_sha256=report['plan_sha256'],collection_report_sha256=digest((c/'report.json').read_bytes()),
            schedule_counts=dict(fresh_only=[4096,0,0,2048,2048],retain_two=[2048,1024,1024,2048,2048]),
            frozen_teacher='same approved parent for all cohorts',monitor_used_for_training=False,
            formal_adoption=False,research_pointer_changed=False,strength_proven=False,
            parent_cohort_loss=losses(parent,x,y,sizes))
        for seed in SEEDS:
            for arm,indices in schedules(sizes,seed).items():np.save(out/f'{arm}_seed{seed}-rows.npy',indices,allow_pickle=False)
        atomic_json(out/'protocol.json',protocol)
        return {k:v for k,v in protocol.items() if k!='pool_rows'}


def train(root):
    with lock(root) as root:
        verify(root);out=root/NAME;p=json.loads((out/'protocol.json').read_text())
        if (p['source_sha256']!=digest(Path(__file__).read_bytes()) or
            p['state_sha256']!=digest((root/'state.json').read_bytes()) or
            p['pool_sha256']!=digest((out/'shared-pool.npz').read_bytes()) or
            p['engine_sha256']!=implementation_digest() or (out/'report.json').exists()):
            raise ValueError('changed training inputs or completed immutable trial')
        parent,opt,_=load_snapshot(root/'models',p['parent']);report={**p,'candidates':{}}
        with np.load(out/'shared-pool.npz',allow_pickle=False) as pool:x,y=pool['indices'],pool['labels']
        for seed in SEEDS:
            for arm,expected in schedules(p['pool_sizes'],seed).items():
                label=f'{arm}_seed{seed}';path=out/f'{label}-rows.npy';result=out/f'{label}-result.json'
                if not np.array_equal(np.load(path,allow_pickle=False),expected):raise ValueError('sampling schedule changed')
                if result.exists():
                    candidate=json.loads(result.read_text());model,adam,_=load_snapshot(root/'models',candidate['model'])
                    if candidate['protocol_sha256']!=digest((out/'protocol.json').read_bytes()) or adam['step']!=1580:
                        raise ValueError('changed saved training continuation')
                else:
                    model,adam,stats=fit(parent,opt,x[expected],y[expected],seed=seed,epochs=2,lr=.001,batch_size=128)
                    if adam['step']!=1580:raise ValueError('optimizer budgets differ')
                    key=save_snapshot(root/'models',model,adam,dict(parent=p['parent'],purpose='D31 two-batch retention',
                        arm=arm,seed=seed,training=stats,pool_sha256=p['pool_sha256'],
                        sample_schedule_sha256=digest(path.read_bytes()),protocol_sha256=digest((out/'protocol.json').read_bytes()),
                        outcome_weight=.75,td_lambda=.8))
                    restored,adam,_=load_snapshot(root/'models',key)
                    if adam['step']!=1580:raise ValueError('Adam round-trip changed')
                    for a,b in zip(parameters(model),parameters(restored)):np.testing.assert_array_equal(a,b)
                    candidate=dict(model=key,arm=arm,seed=seed,training=stats,cohort_loss=losses(restored,x,y,p['pool_sizes']),
                                   protocol_sha256=digest((out/'protocol.json').read_bytes()))
                    atomic_json(result,candidate)
                report['candidates'][label]=candidate
                atomic_json(out/'progress.json',dict(completed_candidates=len(report['candidates']),planned_candidates=6))
                print(json.dumps(dict(label=label,**candidate)),flush=True)
        atomic_json(out/'report.json',report);return report['candidates']


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train']);p.add_argument('--root',required=True);a=p.parse_args()
    result=(prepare if a.action=='prepare' else train)(a.root)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
