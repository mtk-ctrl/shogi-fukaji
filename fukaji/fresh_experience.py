"""Finite next-TRAIN collection while a separate frozen monitor runs.

Only the output bundle is written. The research root, models, optimizer and
evaluation are read-only. A later learner must explicitly ingest the receipt;
collection itself never promotes a model or relabels monitor games as TRAIN.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

from .online_cycle import fresh_prefix, lock, play, read_games, write_games
from .online_learning import atomic_json, digest, implementation_digest, load_snapshot, validate_game
from .stratified_replay_trial import PARENT

OPPONENT = '2161b2ba9498a1c29bd6a986547a26cb6cc7126a8bcbe8faf87df906c9314753'
SEED = 202611023000


def validate_plan(plan):
    if (plan['parent']!=PARENT or plan['opponent']!=OPPONENT or plan['seed']!=SEED or
        plan.get('role')!='train' or plan.get('planned_games')!=64 or plan.get('worker_cpu')!=5):
        raise ValueError('wrong fixed collection budget/identity')
    excluded=set(plan['excluded_at_start'])|set(plan['protected_parents']);expected=[]
    for i in range(32):
        prefix,parent=fresh_prefix(SEED+i*1000,excluded)
        expected.append(dict(index=i,prefix=prefix,parent=parent))
    if plan['families']!=expected:
        raise ValueError('collection opening schedule changed')


def validate_pair(games, family, plan, models):
    if len(games) != 2:
        raise ValueError('both colors required')
    for side, game in enumerate(games):
        ids = [plan['parent'],plan['opponent']] if side==0 else [plan['opponent'],plan['parent']]
        if (game.get('role')!='train' or game.get('parent_id')!=family['parent'] or
            game.get('models')!=ids or game['moves'][:6]!=family['prefix'] or
            game.get('learning_start_ply')!=6 or len(game['moves'])>262 or
            game.get('seed')!=plan['seed']+family['index']*2+side or
            game['conditions']['backend_by_side']!=['cpp_nnue','cpp_nnue'] or
            game['conditions']['quiescence_by_side']!=[0,0] or
            game['conditions']['movetime_ms']!=50 or
            game['conditions']['max_additional_plies']!=256):
            raise ValueError('TRAIN collection differs from fixed plan')
        validate_game(game,models[ids[0]],training=True,protected=plan['protected_parents'])


def collect(root, output):
    root, output = Path(root),Path(output)
    with lock(output):
        source = digest(Path(__file__).read_bytes()); engine = implementation_digest()
        if not (output/'plan.json').exists():
            state = json.loads((root/'state.json').read_text())
            if state['research_latest']!=PARENT or len(state['cycles'])!=15:
                raise ValueError('frozen cycle15 collection source required')
            excluded=set(state['excluded_openings'])|set(state['protected_parents']);families=[]
            for i in range(32):
                prefix,parent=fresh_prefix(SEED+i*1000,excluded)
                families.append(dict(index=i,prefix=prefix,parent=parent))
            plan=dict(schema='fukaji-fresh-train-v1',parent=PARENT,opponent=OPPONENT,
                seed=SEED,families=families,protected_parents=state['protected_parents'],
                excluded_at_start=state['excluded_openings'],role='train',planned_games=64,
                worker_cpu=5,source_sha256=source,engine_sha256=engine,
                research_state_changed=False,model_updated=False,formal_adoption=False,
                conditions=dict(movetime_ms=50,max_additional_plies=256,quiescence=0,
                                backend='cpp_nnue',exploration_probability=.05))
            atomic_json(output/'plan.json',plan)
        plan=json.loads((output/'plan.json').read_text())
        if (plan['source_sha256']!=source or plan['engine_sha256']!=engine or
            plan['parent']!=PARENT or plan['opponent']!=OPPONENT or plan['seed']!=SEED):
            raise ValueError('resume requires exact frozen collection implementation and models')
        validate_plan(plan)
        if (output/'report.json').exists():
            raise ValueError('completed collection is immutable')
        if plan['worker_cpu'] not in os.sched_getaffinity(0):
            raise ValueError('reserved collection CPU unavailable')
        os.sched_setaffinity(0,{plan['worker_cpu']})
        from .native_nnue import build_native
        build_native()
        models={key:load_snapshot(root/'models',key)[0] for key in (PARENT,OPPONENT)}
        receipts=[];all_games=[];started=time.monotonic()
        for family in plan['families']:
            name=f"family-{family['index']:03d}.jsonl";receipt_path=output/(name+'.json')
            if (output/name).exists():
                if not receipt_path.exists():
                    raise ValueError('incomplete pair commit; inspect instead of replaying')
                receipt=json.loads(receipt_path.read_text());games=read_games(output,receipt)
            else:
                games=[]
                for side in (0,1):
                    ids=[PARENT,OPPONENT] if side==0 else [OPPONENT,PARENT]
                    games.append(play([models[k] for k in ids],ids,family['prefix'],
                        role='train',parent_id=family['parent'],seed=SEED+family['index']*2+side,
                        movetime=50,max_plies=256,quiescence=0,native_backend=True))
                validate_pair(games,family,plan,models)
                receipt=write_games(output,name,games);atomic_json(receipt_path,receipt)
            validate_pair(games,family,plan,models)
            receipts.append(receipt);all_games.extend(games)
            progress=dict(completed_games=len(all_games),planned_games=64,
                          session_seconds=time.monotonic()-started,
                          outcomes=dict(Counter(g['status'] for g in all_games)))
            atomic_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
        if len(all_games)!=64:
            raise ValueError('incomplete fixed collection')
        report=dict(**progress,complete=True,receipts=receipts,
            plan_sha256=digest((output/'plan.json').read_bytes()),
            legal_replay_moves=sum(len(g['moves']) for g in all_games),outcomes_match=True,
            learner_ingested=False,model_updated=False,strength_proven=False,
            note='all 64 retained; unresolved must be excluded by later learner')
        atomic_json(output/'report.json',report)
        return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(collect(a.root,a.output),indent=2))


if __name__=='__main__':main()
