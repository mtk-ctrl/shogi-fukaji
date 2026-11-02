"""D33 bounded continuation from the selected D31 research parent only."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from . import followup_train_collection as collection
from . import retained_replay_trial as prior
from .online_cycle import lock,read_games
from .online_learning import atomic_json,digest,implementation_digest,load_snapshot,save_snapshot,make_samples,fit,parameters
from .online_state import verify

NAME='d33-experience-learning'
COLLECTION='d33-candidate-experience'
GROUPS=('fresh','previous','earlier','bootstrap','historical_nnue')
ADAPTER=dict(zip(GROUPS,prior.GROUPS))
SEEDS=(202611033301,202611033302,202611033303)


def schedule(sizes,seed):
    return prior.schedules({ADAPTER[g]:sizes[g] for g in GROUPS},seed)['retain_two']


def partition(batches,model,protected):
    encoded=prior.partition([(ADAPTER[g],games) for g,games in batches],model,protected)
    return {g:encoded[ADAPTER[g]] for g in GROUPS}


def losses(model,x,y,sizes):
    encoded=prior.losses(model,x,y,{ADAPTER[g]:sizes[g] for g in GROUPS})
    return {g:encoded[ADAPTER[g]] for g in GROUPS}


def prepare(root,bundle):
    root,bundle=Path(root),Path(bundle)
    with lock(root),lock(bundle):
        verify(root);state=json.loads((root/'state.json').read_text());out=root/NAME
        if (out.exists() or (root/COLLECTION).exists() or state['paused'] or state.get('pending_campaign') or
            state.get('pending_comparison') or state['research_latest']!=collection.PARENT):
            raise ValueError('completed D31 and selected D31 parent required before next learning')
        collection.verify(bundle);cp=json.loads((bundle/'plan.json').read_text());cr=json.loads((bundle/'report.json').read_text())
        parent,opt,_=load_snapshot(root/'models',collection.PARENT)
        if opt['step']!=1580:raise ValueError('D31 Adam continuation required')
        new=[]
        for receipt in cr['receipts']:new.extend(read_games(bundle,receipt))
        batches=[('fresh',new)]
        for r in state['supplemental_train_batches']:
            if r['cohort'] not in ('fresh_d31','fresh_d30'):raise ValueError('unexpected prior supplemental TRAIN')
            batches.append(('previous' if r['cohort']=='fresh_d31' else 'earlier',read_games(root,r)))
        batches.append(('bootstrap',read_games(root,state['bootstrap'])))
        batches.extend(('historical_nnue',read_games(root,r)) for r in state['train_batches'])
        groups=partition(batches,parent,state['protected_parents'])
        if {g:len(v) for g,v in groups.items()}!=dict(fresh=46,previous=54,earlier=55,bootstrap=665,historical_nnue=80):
            raise ValueError('exact D33/previous TRAIN cohorts required')
        xs=[];ys=[];rows=[];sizes={};offset=0
        for group in GROUPS:
            sizes[group]=0
            for key,game in sorted(groups[group].items()):
                x,y,_=make_samples([game],parent,seed=int(key[:16],16),per_game=32,outcome_weight=.75,
                    td_lambda=.8,protected=state['protected_parents'])
                xs.append(x);ys.append(y);rows.append(dict(group=group,game_sha256=key,first_row=offset,rows=len(y)))
                sizes[group]+=len(y);offset+=len(y)
        width=max(a.shape[1] for a in xs)
        x=np.concatenate([np.pad(a,((0,0),(0,width-a.shape[1])),constant_values=parent.inputs) for a in xs]);y=np.concatenate(ys)
        out.mkdir();np.savez_compressed(out/'shared-pool.npz',indices=x,labels=y)
        shutil.copytree(bundle,root/COLLECTION,ignore=shutil.ignore_patterns('.lock','models'))
        state['supplemental_train_batches'].extend({**r,'path':COLLECTION+'/'+r['path'],'cohort':'fresh_d33'} for r in cr['receipts'])
        state['supplemental_collections'].append(dict(plan=COLLECTION+'/plan.json',plan_sha256=cr['plan_sha256'],
            report=COLLECTION+'/report.json',report_sha256=digest((root/COLLECTION/'report.json').read_bytes())))
        state['excluded_openings']=sorted(set(state['excluded_openings'])|{f['parent'] for f in cp['families']})
        atomic_json(root/'state.json',state);verify(root)
        p=dict(schema='fukaji-followup-learning-v1',parent=collection.PARENT,parent_optimizer_step=1580,
            final_optimizer_step=1708,seeds=list(SEEDS),representative_seed=SEEDS[0],positions_per_seed=8192,
            epochs=2,batch_size=128,lr=.001,outcome_weight=.75,td_lambda=.8,
            allocation=dict(zip(GROUPS,[2048,1024,1024,2048,2048])),games={g:len(v) for g,v in groups.items()},
            pool_sizes=sizes,pool_rows=rows,pool_sha256=digest((out/'shared-pool.npz').read_bytes()),
            source_sha256=digest(Path(__file__).read_bytes()),sampling_source_sha256=digest(Path(prior.__file__).read_bytes()),
            engine_sha256=implementation_digest(),state_sha256=digest((root/'state.json').read_bytes()),
            parent_cohort_loss=losses(parent,x,y,sizes),formal_adoption=False,research_pointer_changed=False,
            monitor_used_for_training=False,strength_proven=False)
        for seed in SEEDS:np.save(out/f'seed{seed}-rows.npy',schedule(sizes,seed),allow_pickle=False)
        atomic_json(out/'protocol.json',p)
        return {k:v for k,v in p.items() if k!='pool_rows'}


def train(root):
    with lock(root) as root:
        verify(root);out=root/NAME;p=json.loads((out/'protocol.json').read_text())
        if (p['source_sha256']!=digest(Path(__file__).read_bytes()) or p['sampling_source_sha256']!=digest(Path(prior.__file__).read_bytes()) or
            p['state_sha256']!=digest((root/'state.json').read_bytes()) or p['engine_sha256']!=implementation_digest() or
            p['pool_sha256']!=digest((out/'shared-pool.npz').read_bytes()) or (out/'report.json').exists()):
            raise ValueError('changed followup inputs or immutable completed trial')
        parent,opt,_=load_snapshot(root/'models',p['parent']);report={**p,'candidates':{}}
        with np.load(out/'shared-pool.npz',allow_pickle=False) as a:x,y=a['indices'],a['labels']
        for seed in SEEDS:
            path=out/f'seed{seed}-rows.npy';expected=schedule(p['pool_sizes'],seed)
            if not np.array_equal(np.load(path,allow_pickle=False),expected):raise ValueError('sampling schedule changed')
            result=out/f'seed{seed}-result.json'
            if result.exists():
                c=json.loads(result.read_text());_,adam,_=load_snapshot(root/'models',c['model'])
                if c['protocol_sha256']!=digest((out/'protocol.json').read_bytes()) or adam['step']!=1708:
                    raise ValueError('saved candidate lineage changed')
            else:
                model,adam,stats=fit(parent,opt,x[expected],y[expected],seed=seed,epochs=2,lr=.001,batch_size=128)
                if adam['step']!=1708:raise ValueError('wrong optimizer budget')
                key=save_snapshot(root/'models',model,adam,dict(parent=p['parent'],purpose='D33 candidate selfplay continuation',
                    seed=seed,training=stats,pool_sha256=p['pool_sha256'],protocol_sha256=digest((out/'protocol.json').read_bytes()),
                    sample_schedule_sha256=digest(path.read_bytes()),outcome_weight=.75,td_lambda=.8))
                restored,adam,_=load_snapshot(root/'models',key)
                if adam['step']!=1708:raise ValueError('optimizer reload changed')
                for a,b in zip(parameters(model),parameters(restored)):np.testing.assert_array_equal(a,b)
                c=dict(model=key,seed=seed,training=stats,cohort_loss=losses(restored,x,y,p['pool_sizes']),
                    protocol_sha256=digest((out/'protocol.json').read_bytes()))
                atomic_json(result,c)
            report['candidates'][str(seed)]=c
            atomic_json(out/'progress.json',dict(completed_candidates=len(report['candidates']),planned_candidates=3))
            print(json.dumps(c),flush=True)
        atomic_json(out/'report.json',report);return report['candidates']


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train']);p.add_argument('--root',required=True)
    p.add_argument('--bundle');a=p.parse_args()
    if a.action=='prepare' and not a.bundle:p.error('prepare needs verified --bundle')
    print(json.dumps(prepare(a.root,a.bundle) if a.action=='prepare' else train(a.root),indent=2))

if __name__=='__main__':main()
