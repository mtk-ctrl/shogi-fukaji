"""Read-only, common-outcome prediction diagnostic; never a strength test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fukaji.online_cycle import read_games
from fukaji.online_learning import load_snapshot,validate_game,game_key,parameters,logits_for,atomic_json
from fukaji.retained_replay_trial import NAME


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--campaign',required=True);p.add_argument('--trial',default=NAME);p.add_argument('--out',type=Path,required=True);p.add_argument('--cpu',type=int,default=5)
    a=p.parse_args();os.sched_setaffinity(0,{a.cpu});r=json.loads((a.root/a.trial/'report.json').read_text())
    cp=a.root/a.campaign;plan=json.loads((cp/'plan.json').read_text());report=json.loads((cp/'report.json').read_text())
    if not report['complete']:raise ValueError('only completed frozen monitor may be diagnosed')
    models={'learning_parent':load_snapshot(a.root/'models',r['parent'])[0]}
    models.update({label:load_snapshot(a.root/'models',c['model'])[0] for label,c in r['candidates'].items()})
    losses={k:[] for k in models};nrows=0;resolved=0;unresolved=0;seen=set()
    state=json.loads((a.root/'state.json').read_text());receipts={r['path']:r for r in state['evaluation_batches']}
    for family in plan['families']:
        name=f"{a.campaign}/family-{family['index']:03d}.jsonl";receipt=receipts[name]
        for g in read_games(a.root,receipt):
            if g['role']!='monitor':raise ValueError('diagnostic requires monitor roles')
            key=game_key(g)
            seen.add(key)
            ix,_,z,ok=validate_game(g,models['learning_parent'])
            if not ok:unresolved+=1;continue
            resolved+=1;first=g.get('learning_start_ply',0);available=max(1,len(ix)-first-1)
            starts=first+np.random.default_rng(int(key[:16],16)).choice(available,min(16,available),replace=False)
            selected=sorted({int(j) for i in starts for j in (i,min(i+1,len(ix)-1))})
            width=max(len(ix[i]) for i in selected);x=np.full((len(selected),width),2548,np.int32)
            for row,i in enumerate(selected):x[row,:len(ix[i])]=ix[i]
            y=z[selected];nrows+=len(y)
            for label,model in models.items():
                logits=logits_for(parameters(model),x)[2];value=float(np.mean(np.logaddexp(0,logits)-y*logits))
                if not np.isfinite(value):raise ValueError('nonfinite monitor diagnostic')
                losses[label].append(value)
    result=dict(campaign=a.campaign,trial=a.trial,parent_model_sha256=r['parent'],candidate_models={k:v['model'] for k,v in r['candidates'].items()},resolved_games=resolved,unresolved_excluded=unresolved,sampled_positions=nrows,unique_trajectories=len(seen),
        mean_game_outcome_bce={k:float(np.mean(v)) for k,v in losses.items()},
        labels='observed outcome only; no training',all_seeds_reported=True,
        candidate_selection_changed=False,purpose='development monitor prediction diagnostic, not playing strength',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(a.out,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
