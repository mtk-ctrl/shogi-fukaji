"""Create an owner-private source/runtime bundle with the pinned inference model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import io

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fukaji.formal_model import load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--source-commit',required=True)
    args=p.parse_args()
    if len(args.source_commit)!=40 or any(c not in '0123456789abcdef' for c in args.source_commit):
        p.error('source-commit must be a full Git SHA')
    _,spec=load()
    files={p.relative_to(ROOT).as_posix():p for folder in ('fukaji','native')
           for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.h','.hpp')}
    for name in ('requirements.txt','run-fukaji.sh','run-fukaji-legacy.sh','scripts/run_hybrid.py','scripts/build_hybrid.py','scripts/hybrid_export_weights.py','research/hybrid/nnue_eval.h','research/hybrid/probe.cpp','THIRD_PARTY_NOTICES.md','LICENSE','formal/model.json','docs/現行エンジン仕様.md',
                 'docs/D34_FORMAL_NNUE_RELEASE_20261103.md','docs/decisions/2026-11-03_D33_NNUE正式採用.md',
                 'scripts/verify_formal_release.py',spec['weights_file']):
        files[name]=ROOT/name
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sorted(files.items())}
    slug=spec['version'].lower().replace(' ','-')
    receipt=dict(schema='fukaji-owner-private-bundle-v1',version=spec['version'],
                 source_commit=args.source_commit,model=spec,files=hashes,
                 platform='Linux x86_64; Python 3.12, g++ C++17; pip install -r requirements.txt')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.out,'w:gz') as archive:
        for name,path in sorted(files.items()):archive.add(path,arcname=slug+'/'+name,recursive=False)
        raw=(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode()
        entry=tarfile.TarInfo(slug+'/BUNDLE.json');entry.size=len(raw);entry.mode=0o644
        archive.addfile(entry,io.BytesIO(raw))
    print(json.dumps(dict(path=str(args.out.resolve()),sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
                          bytes=args.out.stat().st_size,files=len(files)),ensure_ascii=False))

if __name__=='__main__':main()
