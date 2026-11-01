"""D24 fixed-budget, resumable NNUE evaluation with one state writer.

Four independent CPU-pinned game workers evaluate frozen models. Opening
families, not individual color-swapped games, are the statistical units.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import time
import shogi

from .online_cycle import lock, fresh_prefix, play, read_games
from .online_learning import atomic_json, digest, load_snapshot, implementation_digest, validate_game
from .online_state import verify, _pack

SCHEMA='fukaji-frozen-evaluation-v1'
_WORKER={}


def required_pairs(half_width=.15, comparisons=2, alpha=.05):
    if not 0<half_width<.5 or not 0<alpha<1 or comparisons<1:
        raise ValueError('invalid precision plan')
    return math.ceil(math.log(2*comparisons/alpha)/(2*half_width**2))


def score_bounds(game):
    if game['winner'] is not None:
        value=float(game['winner']==game['candidate_side'])
        return value,value,'wins' if value else 'losses'
    if game['status']=='repetition_draw':return .5,.5,'draws'
    return 0.,1.,'unresolved'


def summarize(families, plan):
    n=len(families);m=len(plan['opponents']);complete=n==plan['pairs']
    summaries={}
    for opponent in plan['opponents']:
        label=opponent['label'];counts=Counter();lows=[];highs=[];pair_rows=[]
        for index,games in sorted(families.items()):
            selected=[g for g in games if g['opponent_label']==label]
            if len(selected)!=2 or {g['candidate_side'] for g in selected}!={0,1}:
                raise ValueError('incomplete or duplicate color pair')
            bounds=[score_bounds(g) for g in selected]
            counts.update(x[2] for x in bounds)
            low=sum(x[0] for x in bounds)/2;high=sum(x[1] for x in bounds)/2
            lows.append(low);highs.append(high)
            pair_rows.append({'family':index,'lower_score':low,'upper_score':high})
        total=2*n;lower=sum(lows)/n if n else None;upper=sum(highs)/n if n else None
        # Union bound over two one-sided tails and all prespecified opponents.
        epsilon=math.sqrt(math.log(2*m/plan['alpha'])/(2*n)) if n else None
        interval=[max(0.,lower-epsilon),min(1.,upper+epsilon)] if n else None
        conclusion='not_complete'
        if complete and plan['purpose']=='strength':
            conclusion=('improved_on_this_distribution' if interval[0]>.5 else
                        'worse_on_this_distribution' if interval[1]<.5 else 'inconclusive')
        elif complete:conclusion='regression_test_only'
        summaries[label]={'opponent_id':opponent['model'],'opening_families':n,'games':total,
            **{k:counts[k] for k in ('wins','losses','draws','unresolved')},
            'all_game_score_identification_bounds':[lower,upper],
            'simultaneous_hoeffding_interval':interval,'sampling_half_width':epsilon,
            'conclusion':conclusion,'pair_results':pair_rows}
    return {'schema':SCHEMA,'complete':complete,'planned_families':plan['pairs'],
        'completed_families':n,'completed_games':2*m*n,'candidate':plan['candidate'],
        'conditions':plan['conditions'],'comparisons':summaries,
        'alpha_familywise':plan['alpha'],'formal_adoption':False,'research_pointer_changed':False,
        'strength_proven':False,'old_exploratory_games_included':False}


def cpu_plan(workers):
    allowed=sorted(os.sched_getaffinity(0));chosen=[];seen=set()
    for cpu in allowed:
        path=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=((path/'physical_package_id').read_text().strip(),(path/'core_id').read_text().strip())
        if key not in seen:seen.add(key);chosen.append(cpu)
    quota=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    available=len(chosen) if quota[0]=='max' else min(len(chosen),int(quota[0])//int(quota[1]))
    if not 1<=workers<=4 or available<workers+1:
        raise ValueError('insufficient isolated CPU capacity for workers plus writer')
    return {'worker_cpus':chosen[:workers],'writer_cpu':chosen[workers],
            'cpu_max':' '.join(quota),'workers':workers}


def prepare(root, *, name='d24-nnue400', pairs=100, seed=202611012800,
            movetime=100, max_total_plies=512, workers=4, purpose='strength',
            candidate_id=None, opponents=None):
    if not name or Path(name).name!=name or name in ('.','..'):
        raise ValueError('single-level campaign name required')
    if opponents is not None and (not isinstance(opponents,list) or not 1<=len(opponents)<=2 or
        any(set(o)!={'label','model'} or not isinstance(o['label'],str) or not o['label']
            for o in opponents) or len({o['label'] for o in opponents})!=len(opponents)):
        raise ValueError('one or two uniquely labeled explicit opponents required')
    comparisons=len(opponents) if opponents is not None else (2 if purpose=='strength' else 1)
    if (purpose not in ('strength','regression') or not 1<=pairs<=128 or
        (purpose=='strength' and pairs<required_pairs(comparisons=comparisons)) or
        not 1<=movetime<=200 or not 16<=max_total_plies<=512):
        raise ValueError('budget does not satisfy the prespecified precision plan')
    affinity=cpu_plan(workers)
    with lock(root) as root:
        verify(root);state=json.loads((root/'state.json').read_text());out=root/name
        if out.exists() or state['paused'] or state.get('pending_comparison') or state.get('pending_campaign'):
            raise ValueError('existing/paused evaluation must be inspected or resumed')
        candidate=candidate_id or state['research_latest']
        if opponents is None:
            opponents=[{'label':'before_three_cycles_11','model':'ac2d9c51ceb2979f79be5363f3d1f45bac020bcc9d15d58a3fb67c96145e227f'},
                       {'label':'initial_trained_665','model':state['fixed_anchor']}]
            if purpose=='regression':opponents=[{'label':'reference','model':state['fixed_anchor']}]
        for key in {candidate,*[x['model'] for x in opponents]}:
            model,_,_=load_snapshot(root/'models',key)
            if model.inputs!=2548:raise ValueError('campaign native model architecture unsupported')
        excluded=set(state['excluded_openings']);protected=set(state['protected_parents']);families=[]
        for i in range(pairs):
            prefix,parent=fresh_prefix(seed+i*1000,excluded);protected.add(parent)
            families.append({'index':i,'prefix':prefix,'parent':parent})
        plan={'schema':SCHEMA,'name':name,'candidate':candidate,'opponents':opponents,
            'pairs':pairs,'seed':seed,'alpha':.05,'target_sampling_half_width':.15,
            'minimum_pairs':required_pairs(comparisons=len(opponents)),
            'purpose':purpose,'conditions':{'backend':'cpp_nnue','quiescence':0,
                'movetime_ms':movetime,'max_total_plies':max_total_plies},
            'cpu':affinity,'families':families,'original_paused':state['paused'],
            'source_sha256':digest(Path(__file__).read_bytes()),
            'engine_sha256':implementation_digest(),'formal_adoption':False,
            'created_unix':time.time(),'planned_games':pairs*2*len(opponents),
            'frozen_cycles':len(state['cycles']),'frozen_history':state['history'],
            'research_latest_at_start':state['research_latest']}
        out.mkdir();atomic_json(out/'plan.json',plan)
        state.update(paused=True,pending_campaign=name,
            pending_campaign_sha256=digest((out/'plan.json').read_bytes()),
            protected_parents=sorted(protected),excluded_openings=sorted(excluded))
        atomic_json(root/'state.json',state)
        atomic_json(out/'progress.json',{'phase':'prepared','complete':False,'completed_games':0,
            'planned_games':plan['planned_games'],'completed_families':0,'planned_families':pairs})
        return plan


def _validate_family(games, family, plan):
    if len(games)!=2*len(plan['opponents']):raise ValueError('wrong family game count')
    expected={op['label']:op['model'] for op in plan['opponents']}
    keys=[]
    for game in games:
        side=game['candidate_side'];label=game['opponent_label'];keys.append((label,side))
        ids=[plan['candidate'],expected[label]] if side==0 else [expected[label],plan['candidate']]
        if (side not in (0,1) or game['models']!=ids or game['parent_id']!=family['parent'] or
            game['role']!='monitor' or game['source']!='selfplay' or
            game['start_sfen']!=shogi.STARTING_SFEN or game['family_index']!=family['index'] or
            game['evaluation_name']!=plan['name'] or len(game['moves'])>plan['conditions']['max_total_plies'] or
            game['moves'][:len(family['prefix'])]!=family['prefix'] or
            game['conditions']['backend_by_side']!=['cpp_nnue','cpp_nnue'] or
            game['conditions']['quiescence_by_side']!=[0,0] or
            game['conditions']['max_additional_plies']!=plan['conditions']['max_total_plies']-len(family['prefix']) or
            game['conditions']['movetime_ms']!=plan['conditions']['movetime_ms']):
            raise ValueError('family differs from frozen plan')
    if len(keys)!=len(set(keys)):raise ValueError('duplicate matchup/color in family')


def _recover(root, plan, state):
    families={};known={r['path']:r for r in state['evaluation_batches']};changed=False
    for family in plan['families']:
        name=f"{plan['name']}/family-{family['index']:03d}.jsonl";path=root/name
        if not path.exists():continue
        raw=path.read_bytes();receipt={'path':name,'sha256':digest(raw),'games':2*len(plan['opponents'])}
        games=[json.loads(line) for line in raw.splitlines()];_validate_family(games,family,plan)
        if name in known and known[name]!=receipt:raise ValueError('committed family checksum mismatch')
        if name not in known:state['evaluation_batches'].append(receipt);changed=True
        families[family['index']]=games
    if changed:atomic_json(root/'state.json',state)
    return families


def _commit(root, family, games, plan, state):
    _validate_family(games,family,plan)
    name=f"{plan['name']}/family-{family['index']:03d}.jsonl";path=root/name
    if path.exists():raise ValueError('immutable family already exists')
    raw=('\n'.join(json.dumps(g,sort_keys=True) for g in games)+'\n').encode()
    temporary=path.with_suffix('.tmp');temporary.write_bytes(raw);os.replace(temporary,path)
    state['evaluation_batches'].append({'path':name,'sha256':digest(raw),'games':len(games)})
    atomic_json(root/'state.json',state)


def _worker_init(root, plan, cpu_queue, stop):
    cpu=cpu_queue.get();os.sched_setaffinity(0,{cpu})
    from .native_nnue import build_native
    build_native()
    models={key:load_snapshot(Path(root)/'models',key)[0] for key in
            {plan['candidate'],*[o['model'] for o in plan['opponents']]}}
    _WORKER.update(root=root,plan=plan,models=models,cpu=cpu,stop=stop)


def _work(family):
    plan=_WORKER['plan'];models=_WORKER['models'];started=time.monotonic();games=[]
    opponents=list(plan['opponents'])
    if family['index']%2:opponents.reverse()
    for opponent in opponents:
        for black in ((True,False) if family['index']%2==0 else (False,True)):
            if _WORKER['stop'].is_set():raise RuntimeError('evaluation stopped between games')
            ids=([plan['candidate'],opponent['model']] if black else [opponent['model'],plan['candidate']])
            game=play([models[k] for k in ids],ids,family['prefix'],role='monitor',
                parent_id=family['parent'],seed=plan['seed']+family['index'],
                movetime=plan['conditions']['movetime_ms'],
                max_plies=plan['conditions']['max_total_plies']-len(family['prefix']),
                native_backend=True,quiescence=0)
            game.update(candidate_side=0 if black else 1,opponent_label=opponent['label'],
                family_index=family['index'],evaluation_name=plan['name'],worker_cpu=_WORKER['cpu'])
            games.append(game)
    return family['index'],games,time.monotonic()-started


def run(root, *, name='d24-nnue400', checkpoint_every=20):
    root=Path(root)
    from .native_nnue import build_native
    build_native()  # Compile before workers or measured games start.
    with lock(root) as root:
        verify(root);state=json.loads((root/'state.json').read_text());out=root/name
        plan=json.loads((out/'plan.json').read_text())
        if ((state.get('pending_campaign')!=name) or not state['paused'] or
            digest((out/'plan.json').read_bytes())!=state.get('pending_campaign_sha256') or
            plan['source_sha256']!=digest(Path(__file__).read_bytes()) or
            plan['engine_sha256']!=implementation_digest() or
            state['research_latest']!=plan.get('research_latest_at_start',plan['candidate']) or
            state['history']!=plan['frozen_history']):
            raise ValueError('campaign identity/source/state mismatch')
        families=_recover(root,plan,state);original_affinity=os.sched_getaffinity(0)
        started=time.monotonic();durations=[];ctx=mp.get_context('spawn');stop=ctx.Event();queue=ctx.Queue()
        for cpu in plan['cpu']['worker_cpus']:queue.put(cpu)
        os.sched_setaffinity(0,{plan['cpu']['writer_cpu']})
        executor=ProcessPoolExecutor(max_workers=plan['cpu']['workers'],mp_context=ctx,
            initializer=_worker_init,initargs=(str(root.resolve()),plan,queue,stop))
        try:
            futures={executor.submit(_work,f):f for f in plan['families'] if f['index'] not in families}
            for future in as_completed(futures):
                index,games,seconds=future.result();_commit(root,plan['families'][index],games,plan,state)
                families[index]=games;durations.append(seconds)
                progress={'phase':'running','complete':False,'completed_families':len(families),
                    'planned_families':plan['pairs'],'completed_games':len(families)*2*len(plan['opponents']),
                    'planned_games':plan['planned_games'],'session_seconds':time.monotonic()-started,
                    'estimated_remaining_seconds':(plan['pairs']-len(families))*sum(durations)/len(durations)/plan['cpu']['workers']}
                # Counts may be observed; inferential conclusions wait for the fixed endpoint.
                summary=summarize(families,plan)
                progress['counts']={label:{k:value[k] for k in ('wins','losses','draws','unresolved')}
                                    for label,value in summary['comparisons'].items()}
                atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
                if checkpoint_every and len(families)%checkpoint_every==0:
                    archive=root.parent/f'{name}-checkpoint-{len(families):03d}.tar.gz'
                    if not archive.exists():
                        receipt=_pack(root,archive);print(json.dumps({'checkpoint':receipt}),flush=True)
            executor.shutdown(wait=True);executor=None
            report=summarize(families,plan)
            latest=load_snapshot(root/'models',plan['candidate'])[0];moves=0
            for games in families.values():
                for game in games:validate_game(game,latest);moves+=len(game['moves'])
            report.update(legal_replay_moves=moves,outcomes_match=True,session_seconds=time.monotonic()-started,
                          total_wall_seconds=time.time()-plan['created_unix'],
                          plan_sha256=digest((out/'plan.json').read_bytes()),source_sha256=plan['source_sha256'])
            atomic_json(out/'report.json',report)
            state.pop('pending_campaign');state.pop('pending_campaign_sha256');state['paused']=plan['original_paused']
            state.setdefault('frozen_evaluations',[]).append({'report':f'{name}/report.json',
                'sha256':digest((out/'report.json').read_bytes()),'plan':f'{name}/plan.json',
                'plan_sha256':digest((out/'plan.json').read_bytes())})
            atomic_json(root/'state.json',state)
            atomic_json(out/'progress.json',{'phase':'complete','complete':True,'completed_games':plan['planned_games'],
                'planned_games':plan['planned_games'],'completed_families':plan['pairs'],'planned_families':plan['pairs']})
            return report
        except BaseException as error:
            stop.set();atomic_json(out/'failure.json',{'error':repr(error),'completed_families':len(families),
                'learning_remains_paused':True,'research_pointer_changed':False});raise
        finally:
            if executor is not None:executor.shutdown(wait=True,cancel_futures=True)
            os.sched_setaffinity(0,original_affinity);queue.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','start'))
    p.add_argument('--root',required=True);p.add_argument('--name',default='d24-nnue400')
    p.add_argument('--pairs',type=int,default=100);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.mode in ('prepare','start'):
        plan=prepare(a.root,name=a.name,pairs=a.pairs,workers=a.workers)
        print(json.dumps({k:v for k,v in plan.items() if k not in ('families','frozen_history')}),flush=True)
    if a.mode in ('run','start'):print(json.dumps(run(a.root,name=a.name),indent=2))


if __name__=='__main__':main()
