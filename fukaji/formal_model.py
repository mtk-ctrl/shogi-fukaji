"""Pinned inference-only model adopted by the owner; no training state loaded."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from .nnue_research import SmallNNUE,INPUTS,HIDDEN

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'formal/model.json'


def manifest():
    return json.loads(MANIFEST.read_text())


def load(path=None, *, specification=None):
    spec=specification or manifest()
    path=Path(path) if path is not None else ROOT/spec['weights_file']
    if not path.is_absolute():path=ROOT/path
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=spec['weights_sha256']:
        raise ValueError('approved NNUE SHA256 mismatch')
    with np.load(io.BytesIO(raw),allow_pickle=False) as data:
        if set(data.files)!={'w','b','v','o'} or data['o'].shape!=(1,):
            raise ValueError('inference-only NNUE tensors required')
        model=SmallNNUE(data['w'],data['b'],data['v'],data['o'][0])
    if model.inputs!=INPUTS or spec['architecture']!=[INPUTS,HIDDEN,1]:
        raise ValueError('approved NNUE architecture mismatch')
    return model,spec
