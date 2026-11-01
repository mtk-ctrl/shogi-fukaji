"""Bounded fixed-depth equivalence/cost probe using only saved train parents."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
from statistics import mean,median

import shogi
from .native_nnue import NativeSelector,build_native
from .online_learning import load_snapshot,atomic_json,game_key,implementation_digest
from .online_cycle import read_games
from .online_search import Position,select
from .online_state import verify


def probe(root, *, seed=202611012300,output_name='d23-native-probe.json'):
    if Path(output_name).name!=output_name or not output_name.endswith('.json'):
        raise ValueError('single-level JSON probe name required')
    root=Path(root);verify(root);out=root/output_name
    if out.exists():raise ValueError('immutable native probe exists')
    state=json.loads((root/'state.json').read_text())
    key=state['research_latest'];model=load_snapshot(root/'models',key)[0]
    rng=random.Random(seed)
    bootstrap=read_games(root,state['bootstrap']);recent=[]
    for receipt in state['train_batches'][-3:]:recent.extend(read_games(root,receipt))
    games=rng.sample(bootstrap,4)+rng.sample(recent,4)
    protected=set(state['protected_parents'])
    if any(g['role']!='train' or g['source']!='selfplay' or g['parent_id'] in protected for g in games):
        raise ValueError('probe source is not isolated training experience')
    binary=build_native();rows=[]
    report={'model':key,'seed':seed,'depth':2,'game_hashes':[game_key(g) for g in games],
            'implementation_sha256':implementation_digest(),
            'new_games':0,'formal_adoption':False,'strength_proven':False,'rows':rows}
    for game in games:
        for fraction in (.15,.4,.65,.9):
            board=shogi.Board(game['start_sfen'])
            for word in game['moves'][:int(len(game['moves'])*fraction)]:board.push(shogi.Move.from_usi(word))
            if not list(board.legal_moves):continue
            history=Position(board,model)
            with NativeSelector(binary) as native:
                # Model/history setup is outside both measured searches.
                native.sync(board);native.register(model)
                for p in (0,1):
                    if abs(native.value(board,model,p)-model.logit(board,model.full_accumulator(board,p)))>1e-10:
                        raise ValueError('native NNUE value mismatch')
                searches=[('python',lambda:select(board,model,max_depth=2,fixed_depth=True,history=history)),
                          ('native',lambda:native.select(board,model,max_depth=2,fixed_depth=True,history=history))]
                if len(rows)%2:searches.reverse()
                results={name:fn() for name,fn in searches}
            row={'sfen':board.sfen(),'game_sha256':game_key(game),'results':results};rows.append(row)
            a,b=results['python'],results['native']
            if a['move']!=b['move'] or abs(a['score']-b['score'])>1e-7 or a['nodes']!=b['nodes']:
                report['complete']=False;atomic_json(out,report)
                raise ValueError('native fixed-depth move/score/nodes mismatch; inspect saved probe')
            if len(rows)%8==0:print(json.dumps({'verified_positions':len(rows)}),flush=True)
    report.update(complete=True,positions=len(rows),moves_scores_nodes_match=True,
        mean_ms={k:mean(r['results'][k]['elapsed_ms'] for r in rows) for k in ('python','native')},
        median_ms={k:median(r['results'][k]['elapsed_ms'] for r in rows) for k in ('python','native')},
        max_score_error=max(abs(r['results']['python']['score']-r['results']['native']['score']) for r in rows))
    report['speed_ratio']=report['mean_ms']['python']/report['mean_ms']['native']
    atomic_json(out,report)
    return {k:v for k,v in report.items() if k not in ('rows','game_hashes')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--output-name',default='d23-native-probe.json')
    a=p.parse_args();print(json.dumps(probe(a.root,output_name=a.output_name),indent=2))


if __name__=='__main__':main()
