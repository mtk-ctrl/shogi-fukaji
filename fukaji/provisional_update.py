"""Explicit research-pointer decision after a completed frozen experiment.

A positive finite score can justify continued research without claiming a
confidence interval proves strength. The formal engine and rollback stay fixed.
"""
import json
from pathlib import Path
from .online_cycle import lock
from .online_learning import atomic_json,digest,load_snapshot
from .online_state import verify


def decision_comparisons(report, parent, primary_label=None):
    """Keep ablation controls visible without mistaking them for the old model."""
    comparisons=report['comparisons']
    if not comparisons:
        raise ValueError('nonempty completed comparisons required')
    if primary_label is None:
        required=list(comparisons)
    else:
        if (not isinstance(primary_label,str) or primary_label not in comparisons or
            comparisons[primary_label]['opponent_id']!=parent):
            raise ValueError('primary comparison must be against current research parent')
        required=[primary_label]
        if any(r['conclusion']=='worse_on_this_distribution' for label,r in comparisons.items() if label!=primary_label):
            raise ValueError('diagnostic control shows established inferiority')
    if any(comparisons[label]['all_game_score_identification_bounds'][0]<=.5 for label in required):
        raise ValueError('no positive lower score bound against required reference')
    return required


def apply(root, *, campaign, name, reason, primary_label=None):
    if not reason.strip() or any(not s or Path(s).name!=s or s in ('.','..') for s in (campaign,name)):
        raise ValueError('explicit reason and single-level names required')
    with lock(root) as root:
        verify(root);state=json.loads((root/'state.json').read_text())
        if state['paused'] or state.get('pending_campaign') or state.get('pending_comparison'):
            raise ValueError('finish pending evaluation before research update')
        receipts=[r for r in state.get('frozen_evaluations',[]) if r['report']==campaign+'/report.json']
        if len(receipts)!=1:raise ValueError('registered completed experiment required')
        report=json.loads((root/receipts[0]['report']).read_text())
        plan=json.loads((root/receipts[0]['plan']).read_text())
        parent=state['research_latest'];candidate=plan['candidate']
        if (not report.get('complete') or report['candidate']!=candidate or candidate in state['history'] or
            plan['research_latest_at_start']!=parent or
            set(report['comparisons'])!={o['label'] for o in plan['opponents']} or
            any(report['comparisons'][o['label']]['opponent_id']!=o['model'] for o in plan['opponents'])):
            raise ValueError('no unselected positive-score candidate from current parent')
        required=decision_comparisons(report,parent,primary_label)
        model,opt,manifest=load_snapshot(root/'models',candidate)
        old,old_opt,_=load_snapshot(root/'models',parent)
        if (manifest['parent']!=parent or model.inputs!=old.inputs or opt is None or old_opt is None or
            opt['step']<=old_opt['step']):raise ValueError('continuation lineage/Adam mismatch')
        path=root/'decisions'/(name+'.json')
        if path.exists():raise ValueError('immutable research decision already exists')
        decision=dict(schema='fukaji-provisional-update-v1',parent=parent,candidate=candidate,
            rollback_checkpoint=state['checkpoint'],fixed_anchor=state['fixed_anchor'],
            reason=reason,formal_adoption=False,research_only=True,strength_proven=False,
            rule='positive lower score-identification bound against explicit references; not a confidence-interval strength claim',
            primary_opponent_label=primary_label,required_comparisons=required,
            diagnostic_comparisons=[label for label in report['comparisons'] if label not in required],
            full_comparison_report_retained=True,
            evaluation_report=receipts[0]['report'],evaluation_sha256=receipts[0]['sha256'],
            candidate_optimizer_step=opt['step'],parent_optimizer_step=old_opt['step'],
            cycle_counter_unchanged=True)
        atomic_json(path,decision)
        previous=json.loads(json.dumps(state));state['research_latest']=candidate;state['history'].append(candidate)
        state.setdefault('research_updates',[]).append(dict(report=path.relative_to(root).as_posix(),sha256=digest(path.read_bytes())))
        atomic_json(root/'state.json',state)
        try:verify(root)
        except Exception:
            atomic_json(root/'state.json',previous);raise
        return decision
