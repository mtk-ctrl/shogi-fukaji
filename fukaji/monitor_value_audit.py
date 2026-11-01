"""Read-only NNUE outcome prediction on a completed monitor campaign.

This is a development diagnosis, never a training-label export or adoption test.
Every model sees the same sampled positions from the joint played distribution.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from .frozen_evaluation import _validate_family
from .online_cycle import read_games
from .online_learning import (atomic_json, digest, game_key, load_snapshot,
    logits_for, parameters, probability, validate_game)
from .online_state import verify


def game_metrics(game, models, *, seed, protected):
    if game.get('role') != 'monitor' or game.get('parent_id') not in protected:
        raise ValueError('protected monitor games only; never TRAIN')
    first_model = next(iter(models.values()))
    ix, _, outcome, resolved = validate_game(game, first_model, training=False)
    if not resolved:
        return None
    first = game.get('learning_start_ply', 0)
    if not 0 <= first < len(ix):
        raise ValueError('invalid recorded opening boundary')
    rng = np.random.default_rng(seed)
    n = max(1, len(ix)-first-1)
    starts = rng.choice(n, min(8, n), replace=False)+first
    selected = sorted({int(j) for i in starts for j in (i, min(i+1, len(ix)-1))})
    x = np.full((len(selected), max(len(ix[i]) for i in selected)), first_model.inputs, np.int32)
    for row, i in enumerate(selected):
        x[row, :len(ix[i])] = ix[i]
    y = outcome[selected]
    metrics = {}
    for label, model in models.items():
        if model.inputs != first_model.inputs:
            raise ValueError('matching feature representations required')
        logits = logits_for(parameters(model), x)[2].astype(float)
        metrics[label] = dict(outcome_bce=float(np.mean(np.logaddexp(0, logits)-y*logits)),
                             brier=float(np.mean((probability(logits)-y)**2)))
    return dict(game_sha256=game_key(game), sampled_plies=selected, positions=len(y), metrics=metrics)


def audit(root, name, output, *, seed=202611022950):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('immutable monitor diagnosis exists')
    verify(root)
    state_raw = (root/'state.json').read_bytes(); state = json.loads(state_raw)
    matching = [r for r in state.get('frozen_evaluations', []) if r['report'] == name+'/report.json']
    if len(matching) != 1 or state.get('pending_campaign'):
        raise ValueError('completed registered campaign required')
    plan = json.loads((root/name/'plan.json').read_text())
    ids = {'candidate':plan['candidate'], **{o['label']:o['model'] for o in plan['opponents']}}
    models = {label:load_snapshot(root/'models',key)[0] for label,key in ids.items()}
    families = {f['index']:f for f in plan['families']}; details = []; excluded = 0
    for receipt in state['evaluation_batches']:
        if receipt['path'].split('/')[0] != name:
            continue
        games = read_games(root, receipt)
        _validate_family(games, families[games[0]['family_index']], plan)
        for game in games:
            row = game_metrics(game, models, seed=seed+int(game_key(game)[:12],16),
                               protected=set(state['protected_parents']))
            if row is None:
                excluded += 1
            else:
                details.append({**row,'family_index':game['family_index'],
                                'opponent_label':game['opponent_label']})
    if len(details)+excluded != plan['planned_games']:
        raise ValueError('incomplete joint monitor population')
    summaries = {}
    for label in models:
        by_family = {}
        for row in details:
            by_family.setdefault(row['family_index'],[]).append(row['metrics'][label])
        summaries[label] = {metric:float(np.mean([np.mean([g[metric] for g in games])
                                                  for games in by_family.values()]))
                            for metric in ('outcome_bce','brier')}
    report = dict(schema='fukaji-monitor-value-diagnosis-v1', seed=seed, models=ids,
        resolved_games=len(details), unresolved_excluded=excluded, games=details,
        equal_family_weighted_metrics=summaries, labels='verified actual outcome only',
        population='same joint played-position distribution for all models; monitor development diagnosis',
        training_used=False, models_updated=False, strength_claim=False,
        state_snapshot_sha256=digest(state_raw), source_sha256=digest(Path(__file__).read_bytes()),
        plan_sha256=matching[0]['plan_sha256'])
    atomic_json(output, report)
    return {k:v for k,v in report.items() if k!='games'}
