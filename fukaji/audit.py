"""Audit provenance and legal replay of FUKAJI engine-generated games."""
import argparse
import hashlib
import json
from pathlib import Path
from .learning import load_games

def audit(games, manifest):
    path=Path(games)
    m=json.loads(Path(manifest).read_text())
    assert m["generator"]=="fukaji-alpha-beta-v0", "not real engine"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==m["dataset_sha256"], "input hash mismatch"
    lines=[line for line in path.read_text().splitlines() if line.strip()]
    assert len(lines)==m["completed_checkmates"], "count mismatch"
    rows=load_games(path)
    assert len(rows)==len(lines), "duplicate games or invalid data"
    return {"completed_verified_games":len(rows),"dataset_sha256":m["dataset_sha256"],"status":"verified","note":"Zero finished games is valid for bounded CI smoke, but cannot train."}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--games",required=True)
    ap.add_argument("--manifest",required=True)
    a=ap.parse_args()
    print(json.dumps(audit(a.games,a.manifest),indent=2))
if __name__=="__main__":main()
