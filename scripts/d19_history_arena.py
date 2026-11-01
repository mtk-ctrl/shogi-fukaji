"""Finite new-opening comparison: frozen NNUE, only history reuse differs."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fukaji.online_cycle import lock, fresh_prefix, play, write_games, gate
from fukaji.online_learning import atomic_json, load_snapshot, implementation_digest
from fukaji.online_state import verify


def run(root, out, *, pairs=8, seed=2026110190, movetime=50):
    with lock(root) as root:
        verify(root)
        state = json.loads((root/'state.json').read_text())
        name = 'd19-history-50ms'
        folder = root/name
        if folder.exists() or state.get('pending_comparison'):
            raise ValueError('existing comparison; inspect before restarting')
        folder.mkdir()
        was_paused = state['paused']
        state.update(paused=True, pending_comparison=name)
        atomic_json(root/'state.json', state)
        excluded = set(state['excluded_openings'])
        protected = set(state['protected_parents'])
        planned = [fresh_prefix(seed+i*1000, excluded) for i in range(pairs)]
        protected.update(parent for _, parent in planned)
        state['excluded_openings'] = sorted(excluded)
        state['protected_parents'] = sorted(protected)
        atomic_json(root/'state.json', state)
        key = state['research_latest']
        model = load_snapshot(root/'models', key)[0]
        manifest = {'model': key, 'seed': seed, 'pairs': pairs,
                    'movetime_ms': movetime, 'max_plies': 256, 'role': 'monitor',
                    'planned_parents': [parent for _, parent in planned],
                    'implementation_sha256': implementation_digest(),
                    'formal_adoption': False, 'strength_proven': False}
        atomic_json(folder/'manifest.json', manifest)
        started = time.monotonic()
        games, receipts = [], []
        for pair, (prefix, parent) in enumerate(planned):
            current = []
            for side in (0, 1):
                flags = (side == 0, side == 1)
                t = time.monotonic()
                g = play([model, model], [key, key], prefix, role='monitor',
                         parent_id=parent, seed=seed+pair, movetime=movetime,
                         max_plies=256, reuse_history=flags)
                current.append({**g, 'candidate_side': side,
                                'whole_game_seconds': time.monotonic()-t})
            receipt = write_games(root, f'{name}/pair-{pair:03d}.jsonl', current)
            receipts.append(receipt)
            state['evaluation_batches'].append(receipt)
            atomic_json(root/'state.json', state)
            games.extend(current)
            atomic_json(folder/'progress.json', {'completed_pairs': pair+1,
                        'score': gate(games), 'seconds': time.monotonic()-started})
            print(json.dumps({'completed_pairs': pair+1, 'score': gate(games)}), flush=True)
        telemetry = []
        for candidate in (False, True):
            rows = [r for g in games for r in g['telemetry']
                    if (r['side'] == g['candidate_side']) == candidate]
            times = sorted(r['elapsed_ms'] for r in rows)
            telemetry.append({'reuse_history': candidate, 'moves': len(rows),
                'depth0': sum(r['depth'] == 0 for r in rows),
                'median_ms': statistics.median(times), 'mean_ms': statistics.mean(times),
                'max_ms': max(times), 'mean_depth': statistics.mean(r['depth'] for r in rows)})
        report = {**manifest, 'evaluation': receipts, 'candidate_score': gate(games),
            'pair_results': [gate(games[i:i+2]) for i in range(0, len(games), 2)],
            'telemetry': telemetry, 'seconds': time.monotonic()-started,
            'research_pointer_changed': False}
        atomic_json(folder/'report.json', report)
        atomic_json(out, report)
        state.pop('pending_comparison')
        state['paused'] = was_paused
        atomic_json(root/'state.json', state)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    run(args.root, args.out)
