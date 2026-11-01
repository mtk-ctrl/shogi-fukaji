"""Explicit research resume sources; never infer latest from a public checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile

from .online_learning import load_snapshot
from .online_state import restore


def validate_inputs(env):
    if env['CYCLES'] not in ('1', '2', '3'):
        raise ValueError('cycles outside 1..3')
    run, sha = env.get('RESUME_RUN', ''), env.get('RESUME_SHA', '')
    fresh = env.get('FRESH_BOOTSTRAP') == 'true'
    published = env.get('PUBLISHED_CHECKPOINT') == 'true'
    if sum((bool(run), fresh, published)) != 1:
        raise ValueError('choose exactly one explicit source: prior run, fresh bootstrap, or historical public checkpoint; private latest requires local indexed restore')
    if run and (not run.isdigit() or not re.fullmatch('[0-9a-f]{64}', sha)):
        raise ValueError('resume requires run ID and checksum')
    if not run and sha:
        raise ValueError('checksum without run')
    if env.get('ALLOW_PUBLIC_STATE_ARTIFACT') != 'true':
        raise ValueError('public repository Actions artifacts require explicit publication opt-in; use local private storage otherwise')


def restore_indexed(index, archive, destination):
    index = json.loads(Path(index).read_text())
    # Required identity, not merely archive integrity: reject stale/wrong lineage.
    expected = (index['latest_research_model'], index['cycles_completed'], index['optimizer_step'])
    if (not re.fullmatch('[0-9a-f]{64}', expected[0]) or
        type(expected[1]) is not int or expected[1] < 0 or
        type(expected[2]) is not int or expected[2] < 0):
        raise ValueError('invalid resume identity')
    destination = Path(destination)
    if destination.exists():
        raise ValueError('destination exists; never overwrite research state')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        staged = Path(tmp)/'restored'
        result = restore(archive, staged, index['state_sha256'])
        _, optimizer, _ = load_snapshot(staged/'models', result['research_latest'])
        actual = (result['research_latest'], result['cycles'],
                  None if optimizer is None else optimizer['step'])
        if actual != expected:
            raise ValueError('restored model/cycle/optimizer differs from resume index')
        os.rename(staged, destination)
    return {**result, 'optimizer_step': expected[2], 'resume_identity_verified': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('validate-actions', 'restore'))
    parser.add_argument('--index'); parser.add_argument('--archive'); parser.add_argument('--root')
    args = parser.parse_args()
    if args.mode == 'validate-actions':
        validate_inputs(os.environ)
    else:
        if not all((args.index, args.archive, args.root)):
            parser.error('restore requires --index, --archive and --root')
        print(json.dumps(restore_indexed(args.index, args.archive, args.root), indent=2))


if __name__ == '__main__': main()
