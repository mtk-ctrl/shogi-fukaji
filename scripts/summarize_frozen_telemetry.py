"""Summarize measured search time and depth from a completed frozen campaign."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fukaji.online_cycle import read_games
from fukaji.online_learning import atomic_json
from fukaji.frozen_evaluation import _validate_family


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--campaign',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    state=json.loads((a.root/'state.json').read_text());out=a.root/a.campaign
    plan=json.loads((out/'plan.json').read_text());report=json.loads((out/'report.json').read_text())
    if not report['complete']:raise ValueError('complete fixed endpoint required')
    receipts={r['path']:r for r in state['evaluation_batches']};groups=defaultdict(list);plies=0;games=0
    for family in plan['families']:
        batch=read_games(a.root,receipts[f"{a.campaign}/family-{family['index']:03d}.jsonl"])
        _validate_family(batch,family,plan)
        for game in batch:
            games+=1;plies+=len(game['moves'])
            if len(game['telemetry'])!=len(game['moves'])-len(family['prefix']):raise ValueError('missing telemetry')
            for row in game['telemetry']:
                key=game['opponent_label']+('/candidate' if row['side']==game['candidate_side'] else '/opponent')
                groups[key].append(row)
    result=dict(campaign=a.campaign,games=games,replayed_plies=plies,nominal_ms=plan['conditions']['movetime_ms'],groups={})
    for label,rows in groups.items():
        times=np.array([r['elapsed_ms'] for r in rows]);nodes=np.array([r['nodes'] for r in rows])
        if not np.isfinite(times).all() or (times<0).any():raise ValueError('invalid measured time')
        result['groups'][label]=dict(moves=len(rows),mean_ms=float(np.mean(times)),p99_ms=float(np.quantile(times,.99)),
            max_ms=float(np.max(times)),over_nominal=int((times>result['nominal_ms']).sum()),mean_nodes=float(np.mean(nodes)),
            completed_depth_histogram=dict(Counter(r['depth'] for r in rows)))
    atomic_json(a.out,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
