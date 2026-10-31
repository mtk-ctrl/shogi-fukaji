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


def compare_saved(root, left, right, *, name, seed, pairs=8, movetime=100, max_plies=256):
    if (not name or Path(name).name != name or name in ('.', '..') or
        not 1 <= pairs <= 16 or not 1 <= movetime <= 200 or not 16 <= max_plies <= 512):
        raise ValueError('invalid bounded comparison')
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
                game = play(models, ids, prefix, role='monitor', parent_id=parent,
                            seed=seed+pair, movetime=movetime, max_plies=max_plies)
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
    a=p.parse_args()
    print(json.dumps(compare_saved(a.root,a.left,a.right,name=a.name,seed=a.seed,
                    pairs=a.pairs,movetime=a.movetime,max_plies=a.max_plies),indent=2))


if __name__=='__main__':main()
