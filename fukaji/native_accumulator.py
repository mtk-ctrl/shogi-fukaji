"""Opt-in NNUE accumulator backend; frozen reference source remains unchanged."""
from functools import lru_cache
import hashlib
import subprocess

from .native_nnue import NativeSelector,ROOT


@lru_cache(maxsize=1)
def build_accumulator():
    names=['nnue_accumulator_ffi.cpp','nnue_search_ffi.cpp','rules_state_ffi.cpp','rules_probe.cpp']
    files=[ROOT/'native'/name for name in names]
    sha=hashlib.sha256(b'\n'.join(p.read_bytes() for p in files)).hexdigest()
    folder=ROOT/'build/native-accumulator';folder.mkdir(parents=True,exist_ok=True)
    target=folder/f'libnnue-acc-{sha[:20]}.so'
    if not target.exists():
        temporary=target.with_suffix('.tmp.so')
        result=subprocess.run(['g++','-std=c++17','-O3','-fPIC','-shared','-ffp-contract=off',
            '-Wall','-Wextra','-Werror',str(files[0]),'-o',str(temporary)],
            capture_output=True,text=True,timeout=90)
        if result.returncode:raise RuntimeError('accumulator build failed: '+result.stderr[-3000:])
        temporary.replace(target)
    return target


class AccumulatorSelector(NativeSelector):
    def __init__(self,binary=None):
        super().__init__(binary or build_accumulator())
        for old,new in [('fukaji_nnue_search','fukaji_nnue_accumulator_search'),
                        ('fukaji_nnue_value','fukaji_nnue_accumulator_value')]:
            reference=getattr(self.lib,old);candidate=getattr(self.lib,new)
            candidate.argtypes=reference.argtypes;candidate.restype=reference.restype
            setattr(self.lib,old,candidate)

    def select(self,*args,**kwargs):
        result=super().select(*args,**kwargs)
        return {**result,'backend':'cpp_nnue_accumulator'}
