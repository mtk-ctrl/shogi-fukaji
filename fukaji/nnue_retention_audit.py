"""Read-only training-retention diagnosis; never uses monitor/audit as labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from .nnue_research import INPUTS
from .online_learning import (load_snapshot,validate_game,parameters,logits_for,
    probability,game_key,atomic_json,digest)
from .online_cycle import read_games


def audit(root, output, *, seed=202611012801, model_ids=None):
    root=Path(root);output=Path(output)
    if output.exists():raise ValueError('immutable retention audit exists')
    raw_state=(root/'state.json').read_bytes();state=json.loads(raw_state)
    ids=model_ids or {'first_trained_665':state['fixed_anchor'],
        'before_three_cycles_11':'ac2d9c51ceb2979f79be5363f3d1f45bac020bcc9d15d58a3fb67c96145e227f',
        'latest_14':state['research_latest']}
    models={label:load_snapshot(root/'models',key)[0] for label,key in ids.items()}
    if any(m.inputs!=INPUTS for m in models.values()):raise ValueError('matching absolute architectures required')
    target_parent=next(iter(models.values()));params={k:parameters(m) for k,m in models.items()}
    groups={'bootstrap_665':read_games(root,state['bootstrap']),
            'selfplay_cycles_1_to_11':[],'selfplay_cycles_12_to_14':[]}
    for receipt in state['train_batches']:
        number=int(receipt['path'].split('/')[0].split('-')[1])
        groups['selfplay_cycles_1_to_11' if number<=11 else 'selfplay_cycles_12_to_14'].extend(read_games(root,receipt))
    rng=np.random.default_rng(seed);details=[];summary={};seen=set()
    for group,games in groups.items():
        metrics={k:[] for k in models};unresolved=0;positions=0;duplicates=0;resolved_count=0
        for game in games:
            ix,_,z,resolved=validate_game(game,target_parent,training=True,protected=state['protected_parents'])
            key=game_key(game)
            if key in seen:duplicates+=1;continue
            seen.add(key)
            if not resolved:unresolved+=1;continue
            resolved_count+=1
            first=game.get('learning_start_ply',0)
            if not 0<=first<len(ix):raise ValueError('invalid learning boundary')
            starts=rng.choice(max(1,len(ix)-first-1),min(8,max(1,len(ix)-first-1)),replace=False)+first
            selected=sorted({int(j) for i in starts for j in (i,min(i+1,len(ix)-1))})
            width=max(len(ix[i]) for i in selected);x=np.full((len(selected),width),INPUTS,np.int32)
            for row,i in enumerate(selected):x[row,:len(ix[i])]=ix[i]
            labels=z[selected];positions+=len(labels);row={'group':group,'game_sha256':game_key(game),
                'sampled_plies':selected,'positions':len(labels),'metrics':{}}
            for name,p in params.items():
                logits=logits_for(p,x)[2].astype(float)
                loss=float(np.mean(np.logaddexp(0,logits)-labels*logits))
                brier=float(np.mean((probability(logits)-labels)**2))
                metrics[name].append((loss,brier));row['metrics'][name]={'outcome_bce':loss,'brier':brier}
            details.append(row)
        summary[group]={'raw_games':len(games),'unique_games':len(games)-duplicates,
            'duplicate_excluded':duplicates,'resolved_games':resolved_count,
            'unresolved_excluded':unresolved,'sampled_positions':positions,
            'equal_game_weighted_metrics':{name:{'outcome_bce':float(np.mean([v[0] for v in rows])),
                                                'brier':float(np.mean([v[1] for v in rows]))}
                for name,rows in metrics.items() if rows}}
        print(json.dumps({'audited_group':group,'resolved_games':resolved_count}),flush=True)
    report={'schema':'fukaji-retention-diagnosis-v2','seed':seed,'models':ids,'groups':summary,'games':details,
        'deduplication':'game_key across all training groups before sampling; first occurrence retained',
        'labels':'verified actual outcome only; both players; no TD mixture',
        'source':'bootstrap/train_batches only','monitor_used':False,'model_updated':False,
        'strength_claim':False,'purpose':'training retention/adaptation diagnosis, not unseen strength test',
        'state_snapshot_sha256':digest(raw_state),'source_sha256':digest(Path(__file__).read_bytes())}
    atomic_json(output,report)
    return {k:v for k,v in report.items() if k!='games'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(audit(a.root,a.output),indent=2))


if __name__=='__main__':main()
