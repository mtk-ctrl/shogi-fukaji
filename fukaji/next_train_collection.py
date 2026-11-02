"""D31 finite TRAIN collection, four read-only model workers and one writer."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from .online_cycle import lock,fresh_prefix,play,read_games,write_games
from .online_learning import atomic_json,digest,implementation_digest,load_snapshot,validate_game
from .online_state import verify
from .frozen_evaluation import cpu_plan

PARENT='912db54f04429a0a10d3adb7960c23d7fa43ecdead8115c89b1de2e570afb03c'
OPPONENT='ae53ce0d8cf5570421d29256669a39ebad9344b2078d4ea4d07e41dc2100dd16'
NAME='d31-fresh-experience'
SEED=202611033100
_WORKER={}


def validate_plan(p):
    if (p.get('schema')!='fukaji-fresh-train-v1' or p.get('parent')!=PARENT or
        p.get('opponent')!=OPPONENT or p.get('seed')!=SEED or p.get('role')!='train' or
        p.get('planned_games')!=64 or p.get('chunk_families')!=8 or
        p.get('conditions')!=dict(movetime_ms=50,max_additional_plies=256,quiescence=0,
            backend='cpp_nnue',exploration_probability=.05)):
        raise ValueError('fixed D31 TRAIN plan required')
    excluded=set(p['excluded_at_start'])|set(p['protected_parents']);expected=[]
    for i in range(32):
        prefix,parent=fresh_prefix(SEED+i*1000,excluded)
        expected.append(dict(index=i,prefix=prefix,parent=parent))
    if p['families']!=expected:raise ValueError('TRAIN family schedule changed')


def validate_pair(games,family,p,models):
    if len(games)!=2:raise ValueError('both colors required')
    for side,g in enumerate(games):
        ids=[PARENT,OPPONENT] if side==0 else [OPPONENT,PARENT]
        if (g.get('role')!='train' or g.get('parent_id')!=family['parent'] or g.get('models')!=ids or
            g['moves'][:6]!=family['prefix'] or g.get('learning_start_ply')!=6 or len(g['moves'])>262 or
            g.get('seed')!=SEED+family['index']*2+side or
            g['conditions']['backend_by_side']!=['cpp_nnue','cpp_nnue'] or
            g['conditions']['quiescence_by_side']!=[0,0] or g['conditions']['movetime_ms']!=50 or
            g['conditions']['max_additional_plies']!=256 or
            g['conditions']['exploration_probability']!=.05 or g['conditions']['exploration_additional_plies']!=24):
            raise ValueError('TRAIN game differs from frozen D31 protocol')
        validate_game(g,models[ids[0]],training=True,protected=p['protected_parents'])


def _init(root,p,q):
    os.sched_setaffinity(0,{q.get()})
    _WORKER.update(plan=p,models={k:load_snapshot(Path(root)/'models',k)[0] for k in (PARENT,OPPONENT)})


def _pair(family):
    p=_WORKER['plan'];models=_WORKER['models'];games=[]
    for side in (0,1):
        ids=[PARENT,OPPONENT] if side==0 else [OPPONENT,PARENT]
        games.append(play([models[k] for k in ids],ids,family['prefix'],role='train',
            parent_id=family['parent'],seed=SEED+family['index']*2+side,movetime=50,
            max_plies=256,quiescence=0,native_backend=True))
    return family['index'],games


def run(root):
    root=Path(root)
    from .native_nnue import build_native
    build_native()
    with lock(root):
        verify(root);state=json.loads((root/'state.json').read_text());out=root/NAME
        if state['paused'] or state.get('pending_campaign') or state.get('pending_comparison'):
            raise ValueError('research state must be idle')
        if state['research_latest']!=PARENT or len(state['cycles'])!=15:raise ValueError('wrong research lineage')
        source=digest(Path(__file__).read_bytes());engine=implementation_digest()
        if not (out/'plan.json').exists():
            out.mkdir();excluded=set(state['excluded_openings'])|set(state['protected_parents']);families=[]
            for i in range(32):
                prefix,parent=fresh_prefix(SEED+i*1000,excluded);families.append(dict(index=i,prefix=prefix,parent=parent))
            plan=dict(schema='fukaji-fresh-train-v1',parent=PARENT,opponent=OPPONENT,seed=SEED,
                families=families,role='train',planned_games=64,chunk_families=8,cpu=cpu_plan(4),
                protected_parents=state['protected_parents'],excluded_at_start=state['excluded_openings'],
                source_sha256=source,engine_sha256=engine,state_sha256=digest((root/'state.json').read_bytes()),
                conditions=dict(movetime_ms=50,max_additional_plies=256,quiescence=0,
                    backend='cpp_nnue',exploration_probability=.05),formal_adoption=False)
            atomic_json(out/'plan.json',plan)
        plan=json.loads((out/'plan.json').read_text());validate_plan(plan)
        if (plan['source_sha256']!=source or plan['engine_sha256']!=engine or
            plan['state_sha256']!=digest((root/'state.json').read_bytes()) or (out/'report.json').exists()):
            raise ValueError('changed collection inputs or immutable completed collection')
        models={k:load_snapshot(root/'models',k)[0] for k in (PARENT,OPPONENT)};done={}
        for family in plan['families']:
            rp=out/f"family-{family['index']:03d}.jsonl.json"
            if rp.exists():
                receipt=json.loads(rp.read_text());games=read_games(out,receipt)
                if receipt['path']!=f"family-{family['index']:03d}.jsonl":raise ValueError('wrong family receipt')
                validate_pair(games,family,plan,models);done[family['index']]=(receipt,games)
            elif (out/f"family-{family['index']:03d}.jsonl").exists():raise ValueError('uncommitted pair requires inspection')
        remaining=[f for f in plan['families'] if f['index'] not in done][:plan['chunk_families']]
        ctx=mp.get_context('spawn');q=ctx.Queue()
        for cpu in plan['cpu']['worker_cpus']:q.put(cpu)
        started=time.monotonic()
        try:
            with ProcessPoolExecutor(max_workers=4,mp_context=ctx,initializer=_init,initargs=(str(root.resolve()),plan,q)) as ex:
                for future in as_completed([ex.submit(_pair,f) for f in remaining]):
                    index,games=future.result();validate_pair(games,plan['families'][index],plan,models)
                    receipt=write_games(out,f'family-{index:03d}.jsonl',games)
                    atomic_json(out/(receipt['path']+'.json'),receipt);done[index]=(receipt,games)
                    atomic_json(out/'progress.json',dict(completed_games=len(done)*2,planned_games=64,session_seconds=time.monotonic()-started))
                    print(json.dumps(dict(completed_games=len(done)*2,planned_games=64,session_seconds=time.monotonic()-started)),flush=True)
        finally:q.close()
        all_games=[g for i in sorted(done) for g in done[i][1]]
        summary=dict(complete=len(done)==32,completed_games=len(all_games),planned_games=64,
            outcomes=dict(Counter(g['status'] for g in all_games)),session_seconds=time.monotonic()-started)
        if summary['complete']:
            report=dict(**summary,receipts=[done[i][0] for i in sorted(done)],
                plan_sha256=digest((out/'plan.json').read_bytes()),outcomes_match=True,
                legal_replay_moves=sum(len(g['moves']) for g in all_games),learner_ingested=False,strength_proven=False)
            atomic_json(out/'report.json',report)
        return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    print(json.dumps(run(a.root),indent=2))

if __name__=='__main__':main()
