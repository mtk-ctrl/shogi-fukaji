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
print('Formal contract, router links and bounded workflow: OK')
