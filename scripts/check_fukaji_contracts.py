"""Protect the unchanged formal implementation and D17 entrypoint links."""
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
contract = json.loads((root/'research/d17-fast-self-learning/formal-contract.json').read_text())
for name, expected in contract.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
for name in ('AGENTS.md', 'docs/00_開発ルーター.md', 'docs/ROADMAP.md'):
    p = root/name
    for target in re.findall(r'\]\(([^)]+)\)', p.read_text()):
        if '://' not in target and not target.startswith('#'):
            assert (p.parent/target.split('#')[0]).exists(), (name,target)
workflow = (root/'.github/workflows/online-nnue-cycle.yml').read_text()
assert 'workflow_dispatch:' in workflow and 'schedule:' not in workflow
assert 'contents: read' in workflow and 'cancel-in-progress: false' in workflow
assert 'online-nnue-cycle.yml' in (root/'.github/workflows/README.md').read_text()
receipt = json.loads((root/'research/d17-fast-self-learning/latest-state.json').read_text())
archive = (root/'research/d17-fast-self-learning'/receipt['archive']).resolve()
assert archive.is_relative_to((root/'research').resolve()), 'archive outside research'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['sha256'], 'saved state missing/corrupt'
assert 'latest-state.json' in workflow and 'fresh_bootstrap:' in workflow

assert 'published_checkpoint:' in workflow and 'allow_public_state_artifact:' in workflow
assert 'python -m fukaji.online_resume validate-actions' in workflow
assert 'if: always() && inputs.allow_public_state_artifact' in workflow
assert 'Restore durable repository state by default' not in workflow

print('Formal contract, router links, durable state and explicit resume/publication: OK')
