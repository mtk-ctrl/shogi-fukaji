"""Train-only paired diagnostic. Does not claim unseen accuracy or strength."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shogi
from fukaji.online_learning import load_snapshot, atomic_json, implementation_digest
from fukaji.online_cycle import read_games
from fukaji.online_search import select

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
a=p.parse_args(); root=Path(a.root);state=json.loads((root/'state.json').read_text())
model=load_snapshot(root/'models',state['fixed_anchor'])[0]
games=read_games(root,state['bootstrap']); rows=[]
for i in (0,1,2,3):
    game=games[i];b=shogi.Board(game['start_sfen'])
    upto=min(len(game['moves'])-4, 24 if i<2 else len(game['moves'])-12)
    for word in game['moves'][:upto]:b.push(shogi.Move.from_usi(word))
    variants=[('old_full_legal',True,False),('first_legal',True,True),
              ('first_legal',True,True),('old_full_legal',True,False),
              ('full_nnue_recompute',False,True)]
    expected=None
    for name,inc,fast in variants:
        result=select(b,model,incremental=inc,fast_leaf=fast,max_depth=2,fixed_depth=True)
        exact=(result['move'],result['nodes'],result['evaluations'])
        if expected is None:expected=(exact,result['score'])
        assert exact==expected[0] and abs(result['score']-expected[1])<1e-5,(i,name,result,expected)
        rows.append(dict(position=i,ply=upto,parent_id=game['parent_id'],mode='fixed_depth_2',variant=name,**result))
    for name,inc,fast in variants[:4]:
        result=select(b,model,incremental=inc,fast_leaf=fast,movetime_ms=20)
        rows.append(dict(position=i,ply=upto,parent_id=game['parent_id'],mode='20ms',variant=name,**result))
report=dict(model=state['fixed_anchor'],implementation_sha256=implementation_digest(),
            train_diagnostic_only=True,strength_proven=False,rows=rows)
atomic_json(a.out,report)
for mode in ('fixed_depth_2','20ms'):
 for variant in sorted({r['variant'] for r in rows if r['mode']==mode}):
  subset=[r for r in rows if r['mode']==mode and r['variant']==variant]
  print(mode,variant,len(subset),round(sum(r['elapsed_ms'] for r in subset)/len(subset),3),
        'depth',round(sum(r['depth'] for r in subset)/len(subset),2),flush=True)
