"""Bounded paired integration screen: identical NNUE, full vs delta compute.

This is a fault/timing screen, not a statistical strength promotion gate.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import time

from .frozen_evaluation import cpu_plan
from .native_nnue import NativeSelector,build_native
from .native_accumulator import AccumulatorSelector,build_accumulator
from .online_cycle import lock,write_games,fresh_prefix,_play,gate
from .online_learning import atomic_json,digest,load_snapshot,validate_game
from .online_state import verify

_WORK={}


class Switch:
    def __init__(self,fast_side):
        self.fast_side=fast_side;self.base=NativeSelector();self.fast=AccumulatorSelector()
    def select(self,board,*args,**kwargs):
        return (self.fast if board.turn==self.fast_side else self.base).select(board,*args,**kwargs)
    def close(self):
        try:self.base.close()
        finally:self.fast.close()


def _initialize(root,key,queue):
    cpu=queue.get();os.sched_setaffinity(0,{cpu})
    _WORK.update(model=load_snapshot(Path(root)/'models',key)[0],key=key,cpu=cpu)


def _pair(index,prefix,parent,seed,movetime,max_total):
    games=[]
    for side in ((0,1) if index%2==0 else (1,0)):
        selectors=Switch(side)
        try:
            game=_play([_WORK['model']]*2,[_WORK['key']]*2,prefix,role='monitor',
                parent_id=parent,seed=seed+index,movetime=movetime,
                max_plies=max_total-len(prefix),native_sides=(True,True),native_selector=selectors)
        finally:selectors.close()
        game['conditions']['backend_by_side']=[
            'cpp_nnue_accumulator' if s==side else 'cpp_nnue' for s in (0,1)]
        game.update(candidate_side=side,worker_cpu=_WORK['cpu'],pair_index=index)
        games.append(game)
    return index,games


def run(root,*,name='d28-accumulator-screen',pairs=16,seed=202611022810,
        movetime=100,max_total=512,workers=4):
    if (not name or Path(name).name!=name or name in ('.','..') or
        not 1<=pairs<=16 or not 1<=movetime<=200 or not 16<=max_total<=512):
        raise ValueError('bounded integration screen required')
    affinity=cpu_plan(workers);base=build_native();fast=build_accumulator()
    with lock(root) as root:
        verify(root);state=json.loads((root/'state.json').read_text());out=root/name
        if out.exists() or state['paused'] or state.get('pending_comparison') or state.get('pending_campaign'):
            raise ValueError('finish existing research before integration screen')
        key=state['research_latest'];excluded=set(state['excluded_openings']);protected=set(state['protected_parents'])
        families=[]
        for i in range(pairs):
            prefix,parent=fresh_prefix(seed+i*1000,excluded);protected.add(parent)
            families.append({'index':i,'prefix':prefix,'parent':parent})
        out.mkdir();plan={'model_both_sides':key,'pairs':pairs,'seed':seed,'movetime_ms':movetime,
            'max_total_plies':max_total,'families':families,'cpu':affinity,
            'candidate_backend':'cpp_nnue_accumulator','reference_backend':'cpp_nnue',
            'base_binary_sha256':digest(base.read_bytes()),'fast_binary_sha256':digest(fast.read_bytes()),
            'source_sha256':digest(Path(__file__).read_bytes()),'evidence_stage':'integration_fault_screen_only',
            'formal_adoption':False,'strength_proven':False,'research_pointer_changed':False}
        atomic_json(out/'plan.json',plan)
        state.update(paused=True,pending_comparison=name,protected_parents=sorted(protected),
                     excluded_openings=sorted(excluded));atomic_json(root/'state.json',state)
        ctx=mp.get_context('spawn');queue=ctx.Queue()
        for cpu in affinity['worker_cpus']:queue.put(cpu)
        original=os.sched_getaffinity(0);os.sched_setaffinity(0,{affinity['writer_cpu']})
        started=time.monotonic();groups={}
        try:
            with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=_initialize,
                    initargs=(str(root.resolve()),key,queue)) as executor:
                futures=[executor.submit(_pair,f['index'],f['prefix'],f['parent'],seed,movetime,max_total)
                         for f in families]
                for future in as_completed(futures):
                    i,games=future.result();receipt=write_games(root,f'{name}/pair-{i:03d}.jsonl',games)
                    state['evaluation_batches'].append(receipt);atomic_json(root/'state.json',state)
                    groups[i]=games;atomic_json(out/'progress.json',{'completed_pairs':len(groups),
                        'planned_pairs':pairs,'seconds':time.monotonic()-started})
            games=[g for i in sorted(groups) for g in groups[i]];model=load_snapshot(root/'models',key)[0]
            for game in games:validate_game(game,model)
            timing={}
            for kind in ('cpp_nnue','cpp_nnue_accumulator'):
                rows=[x for g in games for x in g['telemetry'] if x['backend']==kind]
                timing[kind]={'moves':len(rows),'mean_ms':statistics.mean(x['elapsed_ms'] for x in rows),
                    'mean_depth':statistics.mean(x['depth'] for x in rows),
                    'mean_nodes':statistics.mean(x['nodes'] for x in rows),'max_ms':max(x['elapsed_ms'] for x in rows)}
            report={**plan,'counts':gate(games),'timing':timing,'legal_replay_games':len(games),
                'legal_replay_plies':sum(len(g['moves']) for g in games),'outcomes_match':True,
                'seconds':time.monotonic()-started,'pair_results':[gate(groups[i]) for i in sorted(groups)]}
            atomic_json(out/'report.json',report)
            state.pop('pending_comparison');state['paused']=False;atomic_json(root/'state.json',state)
            return report
        except BaseException as error:
            atomic_json(out/'failure.json',{'error':repr(error),'completed_pairs':len(groups),
                'learning_remains_paused':True,'research_pointer_changed':False})
            raise
        finally:os.sched_setaffinity(0,original);queue.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    print(json.dumps(run(a.root),indent=2))


if __name__=='__main__':main()
