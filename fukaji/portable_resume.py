"""Private, readback-verified resume indexes with an explicit writer boundary.

A running writer's checkpoint preserves progress but is not a stopped handoff.
This module never uploads data or starts research. Connector reads supply the
observed remote permissions/hash, then the existing archive verifier checks it.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import re
import tempfile

from .online_cycle import lock
from .online_learning import atomic_json, digest, load_snapshot
from .online_state import restore


def create_index(archive, output, *, remote_metadata, remote_readback_sha256,
                 code_commit, stopped_root=None):
    archive=Path(archive);output=Path(output)
    if output.exists():raise ValueError('immutable index already exists')
    permissions=remote_metadata.get('permissions')
    if (remote_metadata.get('shared') is not False or not remote_metadata.get('id') or
        not isinstance(permissions,list) or len(permissions)!=1 or
        permissions[0].get('type')!='user' or permissions[0].get('role')!='owner'):
        raise ValueError('verified owner-only remote metadata required')
    sha=digest(archive.read_bytes())
    if remote_readback_sha256!=sha:
        raise ValueError('remote readback does not match local archive')
    if not re.fullmatch('[0-9a-f]{40}',code_commit):raise ValueError('exact code commit required')
    # A non-blocking flock rejects a stopped-handoff claim while this state has
    # an active writer. Checkpoint-only indexes intentionally omit that claim.
    boundary=lock(stopped_root) if stopped_root is not None else nullcontext()
    with boundary, tempfile.TemporaryDirectory() as tmp:
        recovered=Path(tmp)/'state';counts=restore(archive,recovered,sha)
        state_raw=(recovered/'state.json').read_bytes();state=json.loads(state_raw)
        if stopped_root is not None and (Path(stopped_root)/'state.json').read_bytes()!=state_raw:
            raise ValueError('archive is stale relative to stopped writer state')
        _,optimizer,_=load_snapshot(recovered/'models',state['research_latest'])
        if optimizer is None:raise ValueError('trained optimizer required for continuation')
        pending=state.get('pending_campaign')
        if state.get('pending_comparison'):
            raise ValueError('ordinary incomplete comparisons have no portable resume runner')
        plan=None
        if pending:
            plan=json.loads((recovered/pending/'plan.json').read_text())
            families=[r for r in state['evaluation_batches']
                if re.fullmatch(re.escape(pending)+r'/family-\d{3}\.jsonl',r['path'])]
            resume=f'python -m fukaji.frozen_evaluation run --root RESTORED_ROOT --name {pending}'
        else:families=[];resume=None
        index={'schema':'fukaji-portable-resume-v1','state_sha256':sha,
            'archive_drive_id':remote_metadata['id'],'state_bytes':archive.stat().st_size,
            'latest_research_model':state['research_latest'],'cycles_completed':len(state['cycles']),
            'optimizer_step':optimizer['step'],
            'code_commit':code_commit,'remote_owner_only_verified':True,
            'remote_readback_sha_verified':True,'archive_restore_verified':True,
            'archive_state_sha256':digest(state_raw),'counts':counts,'pending_campaign':pending,
            'completed_families':len(families),'planned_families':None if plan is None else plan['pairs'],
            'resume_command':resume,'requires_original_worker_check':stopped_root is None,
            'publication_kind':'recovery_checkpoint_only' if stopped_root is None else 'stopped_handoff',
            'original_local_session_is_not_a_portable_handle':True,
            'formal_adoption':False}
        if plan is not None:
            index.update(plan_sha256=digest((recovered/pending/'plan.json').read_bytes()),
                evaluation_source_sha256=plan['source_sha256'],engine_sha256=plan['engine_sha256'])
        atomic_json(output,index)
        return index
