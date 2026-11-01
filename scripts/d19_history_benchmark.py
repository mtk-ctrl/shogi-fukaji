"""Preselected train-only ABBA search benchmark; not an unseen strength test."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shogi
from fukaji.online_learning import atomic_json, load_snapshot, implementation_digest
from fukaji.online_search import Position, select


def run(root, manifest, out):
    manifest_path = Path(manifest)
    manifest = json.loads(manifest_path.read_text())
    model = load_snapshot(Path(root)/'models', manifest['model'])[0]
    rows = []
    for i, item in enumerate(manifest['positions']):
        board = shogi.Board(item['start_sfen'])
        history = Position(board, model)
        for word in item['moves']:
            history.push(shogi.Move.from_usi(word))
        # This history is already maintained for terminal rules in BOTH arms
        # of online_cycle.play(). There is no extra candidate setup to hide.
        for mode in ('depth2', '50ms', '100ms'):
            expected = None
            for reuse in (False, True, True, False):
                started = time.monotonic()
                row = select(board, model, history=history if reuse else None,
                             max_depth=2 if mode == 'depth2' else 64,
                             fixed_depth=mode == 'depth2',
                             movetime_ms=50 if mode != '100ms' else 100)
                external_ms = (time.monotonic()-started)*1000
                if mode == 'depth2':
                    exact = tuple(row[k] for k in ('move', 'depth', 'nodes', 'evaluations', 'score'))
                    if expected is None:
                        expected = exact
                    assert exact == expected, (i, exact, expected)
                history.verify()
                rows.append({'position': i, 'ply': len(item['moves']), 'mode': mode,
                             'reuse_history': reuse, 'external_ms': external_ms, **row})
        print(json.dumps({'completed_positions': i+1}), flush=True)
    summary = []
    for mode in ('depth2', '50ms', '100ms'):
        for reuse in (False, True):
            subset = [r for r in rows if r['mode'] == mode and r['reuse_history'] == reuse]
            summary.append({'mode': mode, 'reuse_history': reuse, 'calls': len(subset),
                'mean_ms': statistics.mean(r['external_ms'] for r in subset),
                'median_ms': statistics.median(r['external_ms'] for r in subset),
                'depth0': sum(r['depth'] == 0 for r in subset),
                'mean_depth': statistics.mean(r['depth'] for r in subset),
                'nodes': sum(r['nodes'] for r in subset),
                'over_budget': None if mode == 'depth2' else sum(
                    r['external_ms'] > int(mode[:-2]) for r in subset)})
    report = {'model': manifest['model'], 'implementation_sha256': implementation_digest(),
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'train_diagnostic_only': True, 'strength_proven': False,
        'exact_depth2_matches': len(manifest['positions']), 'summary': summary, 'rows': rows}
    atomic_json(out, report)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    run(args.root, args.manifest, args.out)
