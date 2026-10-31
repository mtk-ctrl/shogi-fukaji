"""Verify and restore durable D17 research snapshots without reinitializing.

Archives contain only regular files under d17-state/ and optional source/.
Validation happens in a temporary sibling before the destination is published.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

from .online_learning import digest, load_snapshot, atomic_json
from .online_cycle import read_games


def verify(root):
    root = Path(root)
    state = json.loads((root / 'state.json').read_text())
    if state.get('schema') != 1 or state.get('formal_adoption') is not False:
        raise ValueError('not a D17 research state')
    model_ids = set(state['history']) | {
        state['research_latest'], state['checkpoint'], state['fixed_anchor']}
    for path in (root / 'models').glob('*.npz'):
        model_ids.add(path.stem)
    for key in model_ids:
        load_snapshot(root / 'models', key)
    protected = set(state['protected_parents'])
    train, monitor = [], []
    for receipt in [state['bootstrap'], *state['train_batches']]:
        safe_path(root, receipt['path'])
        train.extend(read_games(root, receipt))
    for receipt in state['evaluation_batches']:
        safe_path(root, receipt['path'])
        monitor.extend(read_games(root, receipt))
    if any(g.get('role') != 'train' or g.get('source') != 'selfplay' or
           g.get('parent_id') in protected for g in train):
        raise ValueError('evaluation data in training')
    if any(g.get('role') not in ('monitor', 'audit') or
           g.get('parent_id') not in protected for g in monitor):
        raise ValueError('unprotected evaluation data')
    for receipt in state['cycles']:
        report_path = safe_path(root, receipt['report'])
        if digest(report_path.read_bytes()) != receipt['sha256']:
            raise ValueError('cycle report checksum mismatch')
        report = json.loads(report_path.read_text())
        if report.get('candidate'):
            load_snapshot(root / 'models', report['candidate'])
    if len(state['used_seeds']) != len(set(state['used_seeds'])):
        raise ValueError('duplicate generation seeds')
    return {'models': len(model_ids), 'train_games': len(train),
            'monitor_games': len(monitor), 'cycles': len(state['cycles']),
            'research_latest': state['research_latest'], 'paused': state['paused'],
            'formal_adoption': False, 'strength_proven': False,
            'legal_replay_performed': False}


def safe_path(root, name):
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('unsafe state path')
    result = Path(root).joinpath(*path.parts)
    if result.resolve() != result.absolute() or not result.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('state path escapes root')
    return result


def restore(archive, destination, expected_sha):
    archive, destination = Path(archive), Path(destination)
    if destination.exists():
        raise ValueError('destination exists; never overwrite research state')
    if digest(archive.read_bytes()) != expected_sha:
        raise ValueError('state archive checksum mismatch')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        with tarfile.open(archive) as t:
            members = t.getmembers()
            if len(members) > 20000 or sum(m.size for m in members) > 200_000_000:
                raise ValueError('state exceeds reviewed size')
            seen = set()
            for member in members:
                name = PurePosixPath(member.name)
                if (name.is_absolute() or '..' in name.parts or '\\' in member.name or
                    not name.parts or name.parts[0] not in ('d17-state', 'source') or
                    not (member.isfile() or member.isdir()) or member.name in seen):
                    raise ValueError('unsafe state archive')
                seen.add(member.name)
            t.extractall(tmp, filter='data')
        result = verify(Path(tmp) / 'd17-state')
        os.rename(Path(tmp) / 'd17-state', destination)
    return result


def pack(root, archive):
    root, archive = Path(root), Path(archive)
    result = verify(root)
    if archive.exists():
        raise ValueError('archive exists; immutable output required')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'w:gz') as t:
        t.add(root, arcname='d17-state', filter=lambda m: None if m.name.endswith('/.lock') else m)
    receipt = {**result, 'archive': archive.name, 'sha256': digest(archive.read_bytes()),
               'bytes': archive.stat().st_size}
    # A new environment must be able to restore it before publication succeeds.
    with tempfile.TemporaryDirectory() as tmp:
        assert restore(archive, Path(tmp) / 'restored', receipt['sha256']) == result
    atomic_json(archive.with_suffix(archive.suffix + '.json'), receipt)
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('verify', 'restore', 'pack'))
    p.add_argument('--root', required=True)
    p.add_argument('--archive')
    p.add_argument('--sha256')
    a = p.parse_args()
    if a.mode == 'verify': result = verify(a.root)
    elif a.mode == 'pack': result = pack(a.root, a.archive)
    else: result = restore(a.archive, a.root, a.sha256)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
