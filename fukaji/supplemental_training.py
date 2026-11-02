"""Verify explicitly registered, self-generated TRAIN collection receipts.

Research state schema2 prevents old readers silently ignoring these batches.
Legal replay/source admission belongs to the collector and ingest transaction;
this verifier checks immutable manifests, model provenance and receipt coverage.
"""
from __future__ import annotations
import json
from pathlib import Path
import re
from .online_learning import digest, load_snapshot


def verified_receipts(root, state):
    from .online_state import safe_path
    root=Path(root);declared={};actual={}
    for collection in state.get('supplemental_collections',[]):
        pp=safe_path(root,collection['plan']);rp=safe_path(root,collection['report'])
        if (digest(pp.read_bytes())!=collection['plan_sha256'] or
            digest(rp.read_bytes())!=collection['report_sha256'] or pp.parent!=rp.parent):
            raise ValueError('supplemental collection checksum/path mismatch')
        p=json.loads(pp.read_text());r=json.loads(rp.read_text())
        if (p.get('schema')!='fukaji-fresh-train-v1' or p.get('role')!='train' or
            p.get('formal_adoption') is not False or r.get('complete') is not True or
            r.get('outcomes_match') is not True or r.get('plan_sha256')!=collection['plan_sha256'] or
            r.get('completed_games')!=p.get('planned_games') or
            not 1<=p['planned_games']<=256 or
            len(p['families'])*2!=p['planned_games'] or len(r['receipts'])!=len(p['families'])):
            raise ValueError('invalid completed supplemental TRAIN collection')
        for key in (p['parent'],p['opponent']):load_snapshot(root/'models',key)
        parents=[f['parent'] for f in p['families']]
        if len(set(parents))!=len(parents) or set(parents)&set(state['protected_parents']):
            raise ValueError('supplemental TRAIN parent repeated or protected')
        for family,receipt in zip(p['families'],r['receipts']):
            if receipt['path']!=f"family-{family['index']:03d}.jsonl" or receipt['games']!=2:
                raise ValueError('wrong supplemental family receipt')
            path=(rp.parent/receipt['path']).relative_to(root).as_posix();safe_path(root,path)
            if path in declared:raise ValueError('duplicate supplemental receipt')
            declared[path]={k:receipt[k] for k in ('sha256','games')}
    for receipt in state.get('supplemental_train_batches',[]):
        path=receipt['path'];safe_path(root,path)
        if path in actual or not re.fullmatch('[a-z0-9_]{1,64}',receipt.get('cohort','')):
            raise ValueError('duplicate or invalid supplemental TRAIN cohort')
        actual[path]={k:receipt[k] for k in ('sha256','games')}
    if declared!=actual:
        raise ValueError('supplemental TRAIN receipts differ from registered collection')
    return state.get('supplemental_train_batches',[])
