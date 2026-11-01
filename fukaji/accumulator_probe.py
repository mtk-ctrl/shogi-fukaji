"""Read-only paired speed probe on TRAIN positions, never monitor labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import statistics

import shogi

from .native_nnue import NativeSelector,build_native
from .native_accumulator import AccumulatorSelector,build_accumulator
from .online_cycle import read_games
from .online_learning import atomic_json,digest,game_key,load_snapshot,validate_game


def probe(root,output,*,seed=202611022800):
    root=Path(root);output=Path(output)
    if output.exists():raise ValueError('immutable probe exists')
    state_raw=(root/'state.json').read_bytes();state=json.loads(state_raw)
    key=state['research_latest'];model,_,_=load_snapshot(root/'models',key)
    groups={'bootstrap':read_games(root,state['bootstrap']),
            'recent_cycle15':read_games(root,state['train_batches'][-1])}
    rng=random.Random(seed);positions=[];seen=set()
    for label,games in groups.items():
        games=list({game_key(g):g for g in games if g['status']=='checkmate'}.values());rng.shuffle(games)
        count=0
        for game in games:
            validate_game(game,model,training=True,protected=state['protected_parents'])
            first=game.get('learning_start_ply',0)
            ply=first+(len(game['moves'])-first)//2
            board=shogi.Board(game['start_sfen'])
            for word in game['moves'][:ply]:board.push(shogi.Move.from_usi(word))
            if board.sfen() in seen:continue
            seen.add(board.sfen());positions.append({'group':label,'game_sha256':game_key(game),
                'ply':ply,'sfen':board.sfen()});count+=1
            if count==16:break
        if count!=16:raise ValueError('sixteen unique TRAIN positions per group required')
    base_binary=build_native();fast_binary=build_accumulator();output.mkdir(parents=True)
    plan={'model':key,'seed':seed,'positions':positions,'fixed_depth':2,'fixed_order':'ABBAABBA',
        'timed_ms':100,'timed_order':'ABBA','monitor_used':False,'model_updated':False,
        'source_sha256':digest(Path(__file__).read_bytes()),'state_snapshot_sha256':digest(state_raw),
        'base_binary_sha256':digest(base_binary.read_bytes()),'fast_binary_sha256':digest(fast_binary.read_bytes()),
        'purpose':'same decisions/nodes plus paired efficiency; no strength claim',
        'preplanned_speed_screen':1.10,'formal_adoption':False}
    atomic_json(output/'plan.json',plan);rows=[]
    for i,pos in enumerate(positions):
        board=shogi.Board(pos['sfen'])
        with NativeSelector(base_binary) as a,AccumulatorSelector(fast_binary) as b:
            selectors={'A':a,'B':b}
            for selector in selectors.values():selector.select(board,model,fixed_depth=True,max_depth=1)
            fixed=[{'arm':arm,**selectors[arm].select(board,model,fixed_depth=True,max_depth=2)}
                   for arm in plan['fixed_order']]
            reference=fixed[0]
            for result in fixed:
                if (result['move']!=reference['move'] or result['nodes']!=reference['nodes'] or
                    result['evaluations']!=reference['evaluations'] or
                    abs(result['score']-reference['score'])>1e-7):
                    raise ValueError('fixed-depth decision/node/value mismatch')
            timed=[{'arm':arm,**selectors[arm].select(board,model,movetime_ms=100)} for arm in 'ABBA']
            a.state.assert_equal(board,full=True);b.state.assert_equal(board,full=True)
        rows.append({**pos,'fixed':fixed,'timed':timed})
        atomic_json(output/'progress.json',{'completed_positions':i+1,'planned_positions':32})
    def metrics(kind,arm):
        results=[x for row in rows for x in row[kind] if x['arm']==arm]
        return {'observations':len(results),'mean_ms':statistics.mean(x['elapsed_ms'] for x in results),
            'mean_depth':statistics.mean(x['depth'] for x in results),
            'mean_nodes':statistics.mean(x['nodes'] for x in results),
            'max_ms':max(x['elapsed_ms'] for x in results),'over110ms':sum(x['elapsed_ms']>110 for x in results)}
    summary={kind:{arm:metrics(kind,arm) for arm in 'AB'} for kind in ('fixed','timed')}
    ratio=summary['fixed']['A']['mean_ms']/summary['fixed']['B']['mean_ms']
    report={'schema':'fukaji-accumulator-probe-v1','model':key,'positions':32,'all_fixed_depth_match':True,
        'same_nodes_and_evaluations':True,'summary':summary,'fixed_speed_ratio':ratio,
        'passes_speed_screen':ratio>=plan['preplanned_speed_screen'],
        'strength_proven':False,'formal_adoption':False,'monitor_used':False,'rows':rows}
    atomic_json(output/'report.json',report)
    return {k:v for k,v in report.items() if k!='rows'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(probe(a.root,a.output),indent=2))


if __name__=='__main__':main()
