"""D33: isolated next-TRAIN collection during the frozen D31 monitor."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
from .online_cycle import lock,fresh_prefix,play,read_games,write_games
from .online_learning import atomic_json,digest,implementation_digest,load_snapshot,validate_game

PARENT='c8ad43bfd17a1b6bd551cab1073f6ef13d04ad501ea4cf64f3b319a8c259e37a'
OPPONENT='b73f0c24e07f45b27f8e1d1c741c42607c81845477c19a33cb4268fd87cfb330'
SEED=202611033300
CONDITIONS=dict(movetime_ms=50,max_additional_plies=256,quiescence=0,backend='cpp_nnue',exploration_probability=.05)
_WORKER={}


def families(excluded):
    excluded=set(excluded);result=[]
    for i in range(32):
        prefix,parent=fresh_prefix(SEED+i*1000,excluded);result.append(dict(index=i,prefix=prefix,parent=parent))
    return result


def validate_plan(p):
    if (p.get('schema')!='fukaji-fresh-train-v1' or p.get('parent')!=PARENT or p.get('opponent')!=OPPONENT or
        p.get('role')!='train' or p.get('seed')!=SEED or p.get('planned_games')!=64 or
        p.get('formal_adoption') is not False or p.get('worker_cpus')!=[5,6] or p.get('writer_cpu')!=7 or p.get('conditions')!=CONDITIONS or
        p['families']!=families(set(p['excluded_at_start'])|set(p['protected_parents']))):
        raise ValueError('fixed D33 TRAIN source/budget/openings required')


def validate_pair(games,family,plan,models):
    if len(games)!=2:raise ValueError('both TRAIN colors required')
    for side,g in enumerate(games):
        ids=[PARENT,OPPONENT] if side==0 else [OPPONENT,PARENT];c=g['conditions']
        if (g.get('role')!='train' or g.get('models')!=ids or g.get('parent_id')!=family['parent'] or
            g['moves'][:6]!=family['prefix'] or g.get('learning_start_ply')!=6 or len(g['moves'])>262 or
            g.get('seed')!=SEED+family['index']*2+side or c['backend_by_side']!=['cpp_nnue']*2 or
            c['quiescence_by_side']!=[0,0] or c['movetime_ms']!=50 or c['max_additional_plies']!=256 or
            c['exploration_probability']!=.05 or c['exploration_additional_plies']!=24):
            raise ValueError('D33 game differs from frozen collection')
        validate_game(g,models[ids[0]],training=True,protected=plan['protected_parents'])


def verify(output):
    output=Path(output);p=json.loads((output/'plan.json').read_text());r=json.loads((output/'report.json').read_text())
    validate_plan(p)
    if (p['source_sha256']!=digest(Path(__file__).read_bytes()) or p['engine_sha256']!=implementation_digest() or
        r.get('complete') is not True or r.get('outcomes_match') is not True or r.get('completed_games')!=64 or
        r['plan_sha256']!=digest((output/'plan.json').read_bytes()) or len(r['receipts'])!=32):
        raise ValueError('completed allowlisted D33 collection required')
    models={k:load_snapshot(output/'models',k)[0] for k in (PARENT,OPPONENT)};plies=0
    for f,receipt in zip(p['families'],r['receipts']):
        if receipt['path']!=f"family-{f['index']:03d}.jsonl":raise ValueError('wrong TRAIN receipt')
        games=read_games(output,receipt);validate_pair(games,f,p,models);plies+=sum(len(g['moves']) for g in games)
    if plies!=r['legal_replay_moves']:raise ValueError('legal replay count changed')
    return dict(games=64,legal_replay_moves=plies,outcomes=r['outcomes'],formal_adoption=False,learner_ingested=False)


def _init(output,p,q):
    os.sched_setaffinity(0,{q.get()});_WORKER.update(plan=p,models={k:load_snapshot(Path(output)/'models',k)[0] for k in (PARENT,OPPONENT)})


def _work(f):
    m=_WORKER['models'];games=[]
    for side in (0,1):
        ids=[PARENT,OPPONENT] if side==0 else [OPPONENT,PARENT]
        games.append(play([m[k] for k in ids],ids,f['prefix'],role='train',parent_id=f['parent'],
            seed=SEED+f['index']*2+side,movetime=50,max_plies=256,quiescence=0,native_backend=True))
    return f['index'],games


def collect(root,output):
    root,output=Path(root),Path(output)
    from .native_nnue import build_native
    build_native()
    with lock(output):
        if not {5,6,7}.issubset(os.sched_getaffinity(0)):raise ValueError('isolated D33 CPUs unavailable')
        if not (output/'plan.json').exists():
            state=json.loads((root/'state.json').read_text())
            if state.get('pending_campaign')!='d31-retention400':raise ValueError('reserve all D31 monitor parents first')
            (output/'models').mkdir()
            for key in (PARENT,OPPONENT):
                _,adam,_=load_snapshot(root/'models',key)
                if adam['step']!=1580:raise ValueError('wrong frozen D31 candidate')
                for ext in ('.npz','.json'):shutil.copyfile(root/'models'/(key+ext),output/'models'/(key+ext))
            p=dict(schema='fukaji-fresh-train-v1',parent=PARENT,opponent=OPPONENT,seed=SEED,role='train',
                planned_games=64,worker_cpus=[5,6],writer_cpu=7,conditions=CONDITIONS,
                families=families(set(state['excluded_openings'])|set(state['protected_parents'])),
                excluded_at_start=state['excluded_openings'],protected_parents=state['protected_parents'],
                source_sha256=digest(Path(__file__).read_bytes()),engine_sha256=implementation_digest(),formal_adoption=False,
                note='TRAIN only; original research/evaluation state never written by this collector')
            atomic_json(output/'plan.json',p)
        p=json.loads((output/'plan.json').read_text());validate_plan(p)
        if p['source_sha256']!=digest(Path(__file__).read_bytes()) or p['engine_sha256']!=implementation_digest():
            raise ValueError('collection source changed')
        if (output/'report.json').exists():return verify(output)
        models={k:load_snapshot(output/'models',k)[0] for k in (PARENT,OPPONENT)};done={}
        for f in p['families']:
            rp=output/f"family-{f['index']:03d}.jsonl.json"
            if rp.exists():
                receipt=json.loads(rp.read_text());games=read_games(output,receipt)
                if receipt['path']!=f"family-{f['index']:03d}.jsonl":raise ValueError('wrong saved pair receipt')
                validate_pair(games,f,p,models);done[f['index']]=(receipt,games)
            elif (output/f"family-{f['index']:03d}.jsonl").exists():raise ValueError('incomplete pair commit requires inspection')
        ctx=mp.get_context('spawn');q=ctx.Queue()
        for cpu in p['worker_cpus']:q.put(cpu)
        original=os.sched_getaffinity(0);os.sched_setaffinity(0,{7});started=time.monotonic()
        try:
            with ProcessPoolExecutor(max_workers=2,mp_context=ctx,initializer=_init,initargs=(str(output.resolve()),p,q)) as ex:
                futures=[ex.submit(_work,f) for f in p['families'] if f['index'] not in done]
                for future in as_completed(futures):
                    index,games=future.result();validate_pair(games,p['families'][index],p,models)
                    receipt=write_games(output,f'family-{index:03d}.jsonl',games);atomic_json(output/(receipt['path']+'.json'),receipt)
                    done[index]=(receipt,games);progress=dict(completed_games=2*len(done),planned_games=64,seconds=time.monotonic()-started)
                    atomic_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
        finally:os.sched_setaffinity(0,original);q.close()
        games=[g for i in sorted(done) for g in done[i][1]]
        r=dict(complete=True,completed_games=len(games),planned_games=64,receipts=[done[i][0] for i in sorted(done)],
            plan_sha256=digest((output/'plan.json').read_bytes()),outcomes_match=True,
            legal_replay_moves=sum(len(g['moves']) for g in games),outcomes=dict(Counter(g['status'] for g in games)),learner_ingested=False)
        atomic_json(output/'report.json',r);return verify(output)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['collect','verify']);p.add_argument('--root')
    p.add_argument('--output',required=True);a=p.parse_args()
    if a.action=='collect' and not a.root:p.error('collect needs read-only --root')
    print(json.dumps(collect(a.root,a.output) if a.action=='collect' else verify(a.output),indent=2))

if __name__=='__main__':main()
