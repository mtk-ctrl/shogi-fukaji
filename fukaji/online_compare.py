"""Finite paired model comparison; reserve monitor parents before playing.

Uses saved, frozen models and never promotes either model. An interrupted
comparison leaves learning paused rather than allowing its monitor into train.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .online_cycle import lock, fresh_prefix, play, write_games, gate
from .online_learning import atomic_json, load_snapshot, digest, implementation_digest
from .online_state import verify


def compare_saved(root, left, right, *, name, seed, pairs=8, movetime=100, max_plies=256,
                  left_quiescence=0, right_quiescence=0,
                  left_backend='python',right_backend='python'):
    if (not name or Path(name).name != name or name in ('.', '..') or
        not 1 <= pairs <= 16 or not 1 <= movetime <= 200 or not 16 <= max_plies <= 512):
        raise ValueError('invalid bounded comparison')
    if any(type(q) is not int or not 0 <= q <= 4 for q in (left_quiescence, right_quiescence)):
        raise ValueError('quiescence depths must be in 0..4')
    backends=(left_backend,right_backend)
    if any(b not in ('python','cpp_nnue') for b in backends) or any(
        b=='cpp_nnue' and q!=0 for b,q in zip(backends,(left_quiescence,right_quiescence))):
        raise ValueError('native backend requires basic quiescence=0 search')
    with lock(root) as root:
        verify(root)
        state = json.loads((root/'state.json').read_text())
        out = root/name
        if out.exists() or state.get('pending_comparison'):
            raise ValueError('comparison exists or unfinished; inspect before retrying')
        out.mkdir()
        a, _, _ = load_snapshot(root/'models', left)
        b, _, _ = load_snapshot(root/'models', right)
        original_paused = state['paused']
        state['paused'] = True
        state['pending_comparison'] = name
        atomic_json(root/'state.json', state)
        excluded = set(state['excluded_openings'])
        protected = set(state['protected_parents'])
        planned = []
        for pair in range(pairs):
            prefix, parent = fresh_prefix(seed+pair*1000, excluded)
            protected.add(parent)
            planned.append((prefix, parent))
        # Reserve every parent first. A crash cannot make these training starts.
        state['excluded_openings'] = sorted(excluded)
        state['protected_parents'] = sorted(protected)
        atomic_json(root/'state.json', state)
        manifest = {'left': left, 'right': right, 'seed': seed, 'pairs': pairs,
                    'movetime_ms': movetime, 'max_plies': max_plies,
                    'role': 'monitor', 'formal_adoption': False,
                    'strength_proven': False, 'planned_parents': [p for _,p in planned]}
        manifest['quiescence_by_model'] = [left_quiescence, right_quiescence]
        manifest['backend_by_model'] = list(backends)
        manifest['implementation_sha256'] = implementation_digest()
        manifest['comparison_source_sha256'] = digest(Path(__file__).read_bytes())
        atomic_json(out/'manifest.json', manifest)
        started = time.monotonic()
        games, receipts = [], []
        for pair, (prefix,parent) in enumerate(planned):
            current = []
            for black in (True,False):
                models = [a,b] if black else [b,a]
                ids = [left,right] if black else [right,left]
                quiescence = ((left_quiescence, right_quiescence) if black else
                              (right_quiescence, left_quiescence))
                native = tuple(b=='cpp_nnue' for b in (backends if black else backends[::-1]))
                game = play(models, ids, prefix, role='monitor', parent_id=parent,
                            seed=seed+pair, movetime=movetime, max_plies=max_plies,
                            quiescence=quiescence,native_backend=native)
                game['candidate_side'] = 0 if black else 1
                current.append(game)
            receipt = write_games(root, f'{name}/pair-{pair:03d}.jsonl', current)
            state['evaluation_batches'].append(receipt)
            atomic_json(root/'state.json', state)
            games.extend(current); receipts.append(receipt)
            atomic_json(out/'progress.json', {'completed_pairs': pair+1,
                'planned_pairs': pairs, 'seconds': time.monotonic()-started,
                'left_score': gate(games)})
            print(json.dumps({'completed_pairs': pair+1, 'left_score': gate(games)}), flush=True)
        # Pair outcomes remain grouped; unresolved games never count as draws.
        pair_results = []
        for pair in range(pairs):
            group = games[pair*2:pair*2+2]
            pair_results.append({'parent': group[0]['parent_id'], **gate(group)})
        report = {**manifest, 'evaluation': receipts, 'left_score': gate(games),
                  'pair_results': pair_results, 'seconds': time.monotonic()-started,
                  'research_pointer_changed': False}
        atomic_json(out/'report.json', report)
        state.pop('pending_comparison')
        state['paused'] = original_paused
        atomic_json(root/'state.json', state)
        return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--left',required=True)
    p.add_argument('--right',required=True);p.add_argument('--name',required=True)
    p.add_argument('--seed',required=True,type=int);p.add_argument('--pairs',type=int,default=8)
    p.add_argument('--movetime',type=int,default=100);p.add_argument('--max-plies',type=int,default=256)
    p.add_argument('--left-quiescence',type=int,default=0)
    p.add_argument('--right-quiescence',type=int,default=0)
    p.add_argument('--left-backend',choices=('python','cpp_nnue'),default='python')
    p.add_argument('--right-backend',choices=('python','cpp_nnue'),default='python')
    a=p.parse_args()
    print(json.dumps(compare_saved(a.root,a.left,a.right,name=a.name,seed=a.seed,
                    pairs=a.pairs,movetime=a.movetime,max_plies=a.max_plies,
                    left_quiescence=a.left_quiescence,right_quiescence=a.right_quiescence,
                    left_backend=a.left_backend,right_backend=a.right_backend),indent=2))


if __name__=='__main__':main()
