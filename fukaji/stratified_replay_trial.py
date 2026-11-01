"""D29: matched continuation learning, varying historical source exposure only.

TRAIN outcomes and frozen-parent TD labels only. Preparation precedes training;
all seeds are retained, and no research/default pointer is changed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from .online_cycle import lock, read_games
from .online_learning import (atomic_json, digest, game_key, load_snapshot,
    make_samples, fit, save_snapshot, validate_game)
from .online_state import verify

SEEDS = (202611022900, 202611022901, 202611022902)
TOTAL = 8192
PARENT = 'ae53ce0d8cf5570421d29256669a39ebad9344b2078d4ea4d07e41dc2100dd16'
NAME = 'd29-stratified-replay'
GROUPS = ('recent', 'bootstrap', 'historical_nnue')


def partition(batches, parent, protected):
    """Validate even excluded records; first occurrence wins across TRAIN groups."""
    groups = {name: {} for name in GROUPS}
    seen = set()
    for group, games in batches:
        if group not in groups:
            raise ValueError('unknown training cohort')
        for game in games:
            resolved = validate_game(game, parent, training=True, protected=protected)[3]
            key = game_key(game)
            if key in seen:
                continue
            seen.add(key)
            if resolved:
                groups[group][key] = game
    return groups


def schedules(sizes, seed):
    """Exact position counts, shared recent draws and nested source draws."""
    if set(sizes) != set(GROUPS) or any(not isinstance(n, int) or n < 1 for n in sizes.values()):
        raise ValueError('three nonempty position pools required')
    half = TOTAL // 2
    natural_boot = int(np.floor(half * sizes['bootstrap'] /
        (sizes['bootstrap'] + sizes['historical_nnue']) + .5))
    counts = {'natural_history': natural_boot, 'stratified_history': half // 2}
    rng = np.random.default_rng(seed)
    recent = rng.integers(sizes['recent'], size=half)
    boot = rng.integers(sizes['bootstrap'], size=max(counts.values())) + sizes['recent']
    old = (rng.integers(sizes['historical_nnue'], size=half-min(counts.values()))
           + sizes['recent'] + sizes['bootstrap'])
    return {arm: np.concatenate((recent, boot[:n], old[:half-n])) for arm, n in counts.items()}


def prepare(root):
    with lock(root) as root:
        verify(root)
        raw_state = (root/'state.json').read_bytes(); state = json.loads(raw_state)
        if (state['paused'] or state.get('pending_comparison') or state.get('pending_campaign')
            or len(state['cycles']) != 15 or state['research_latest'] != PARENT):
            raise ValueError('requires idle frozen cycle15 lineage')
        out = root/NAME
        if out.exists():
            raise ValueError('immutable trial exists')
        parent, opt, _ = load_snapshot(root/'models', PARENT)
        if opt['step'] != 1324:
            raise ValueError('wrong parent optimizer')
        batches = [('bootstrap', read_games(root, state['bootstrap']))]
        for receipt in state['train_batches']:
            number = int(receipt['path'].split('/')[0].split('-')[1])
            batches.append(('recent' if number == 15 else 'historical_nnue', read_games(root, receipt)))
        groups = partition(batches, parent, state['protected_parents'])
        if {k: len(v) for k, v in groups.items()} != dict(recent=27, bootstrap=665, historical_nnue=53):
            raise ValueError('unexpected frozen training cohorts')
        xs = []; ys = []; rows = []; sizes = {}; offset = 0
        for group in GROUPS:
            sizes[group] = 0
            for key, game in sorted(groups[group].items()):
                x, y, _ = make_samples([game], parent, seed=int(key[:16], 16), per_game=32,
                    outcome_weight=.75, td_lambda=.8, protected=state['protected_parents'])
                rows.append(dict(group=group, game_sha256=key, first_row=offset, rows=len(y)))
                xs.append(x); ys.append(y); offset += len(y); sizes[group] += len(y)
        width = max(x.shape[1] for x in xs)
        x = np.concatenate([np.pad(a, ((0, 0), (0, width-a.shape[1])),
                                   constant_values=parent.inputs) for a in xs])
        y = np.concatenate(ys)
        out.mkdir()
        np.savez_compressed(out/'shared-pool.npz', indices=x, labels=y)
        plan = dict(schema='fukaji-stratified-replay-v1', parent=PARENT,
            parent_optimizer_step=1324, final_optimizer_step=1452,
            positions_per_arm=TOTAL, expected_optimizer_steps_per_arm=128,
            epochs=2, batch_size=128, lr=.001, outcome_weight=.75, td_lambda=.8,
            seeds=list(SEEDS), representative_seed=SEEDS[0], all_seeds_reported=True,
            pool_rows=rows, pool_sizes=sizes, games={k:len(v) for k,v in groups.items()},
            pool_sha256=digest((out/'shared-pool.npz').read_bytes()),
            state_sha256=digest(raw_state), source_sha256=digest(Path(__file__).read_bytes()),
            sampling='replacement within sampled-position pools; 4096 recent plus 4096 historical; '
                     'natural historical source counts rounded to nearest position; stratified 2048+2048',
            arms=list(schedules(sizes, SEEDS[0])), monitor_used_for_training=False,
            formal_adoption=False, research_pointer_changed=False, strength_proven=False)
        for seed in SEEDS:
            for arm, indices in schedules(sizes, seed).items():
                np.save(out/f'{arm}_seed{seed}-rows.npy', indices, allow_pickle=False)
        atomic_json(out/'protocol.json', plan)
        return {k:v for k,v in plan.items() if k!='pool_rows'}


def train(root):
    with lock(root) as root:
        verify(root); out = root/NAME
        plan = json.loads((out/'protocol.json').read_text())
        if (digest((root/'state.json').read_bytes()) != plan['state_sha256'] or
            digest(Path(__file__).read_bytes()) != plan['source_sha256'] or
            digest((out/'shared-pool.npz').read_bytes()) != plan['pool_sha256']):
            raise ValueError('frozen trial inputs changed')
        if (out/'report.json').exists() or (out/'progress.json').exists():
            raise ValueError('inspect immutable completed/partial trial instead of rerunning')
        parent, opt, _ = load_snapshot(root/'models', plan['parent'])
        with np.load(out/'shared-pool.npz', allow_pickle=False) as pool:
            x, y = pool['indices'], pool['labels']
        report = {**plan, 'candidates': {}}
        for seed in SEEDS:
            for arm, expected in schedules(plan['pool_sizes'], seed).items():
                label = f'{arm}_seed{seed}'; path = out/(label+'-rows.npy')
                indices = np.load(path, allow_pickle=False)
                if not np.array_equal(indices, expected):
                    raise ValueError('sample schedule changed')
                candidate, adam, stats = fit(parent, opt, x[indices], y[indices], seed=seed,
                    epochs=plan['epochs'], lr=plan['lr'], batch_size=plan['batch_size'])
                if adam['step'] != plan['final_optimizer_step']:
                    raise ValueError('optimizer budgets differ')
                key = save_snapshot(root/'models', candidate, adam, dict(parent=PARENT,
                    purpose='D29 historical source-stratified replay', seed=seed, arm=arm,
                    training=stats, pool_sha256=plan['pool_sha256'],
                    sample_schedule_sha256=digest(path.read_bytes()),
                    protocol_sha256=digest((out/'protocol.json').read_bytes()),
                    outcome_weight=.75, td_lambda=.8))
                # Verify model and Adam round-trip immediately, before any match.
                restored, restored_opt, _ = load_snapshot(root/'models', key)
                if restored_opt['step'] != adam['step'] or restored.inputs != parent.inputs:
                    raise ValueError('snapshot reload differs')
                report['candidates'][label] = dict(model=key, arm=arm, seed=seed, training=stats)
                atomic_json(out/'progress.json', dict(completed_candidates=len(report['candidates']),
                                                      planned_candidates=6))
        atomic_json(out/'report.json', report)
        return report['candidates']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('action', choices=['prepare', 'train']); a = p.parse_args()
    print(json.dumps((prepare if a.action == 'prepare' else train)(Path(a.root)), indent=2))


if __name__ == '__main__':
    main()
