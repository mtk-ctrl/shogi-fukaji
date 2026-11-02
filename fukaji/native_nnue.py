"""D23 native NNUE search, shared by research and the approved formal USI with explicit, reversible rule state.

Only trained absolute-feature NNUE tensors are accepted. The native legal core
is reused, never its G7 evaluation. No silent fallback to a different search.
"""
from __future__ import annotations

import copy
import ctypes as ct
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import shogi

from .nnue_research import SmallNNUE, INPUTS
from native.g7_stateful_bridge import StatefulRules

ROOT=Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def build_native():
    if not shutil.which('g++'):
        raise RuntimeError('g++ required for native NNUE')
    files=[ROOT/'native'/name for name in ('nnue_search_ffi.cpp','rules_state_ffi.cpp','rules_probe.cpp')]
    source=hashlib.sha256(b'\n'.join(p.read_bytes() for p in files)).hexdigest()
    folder=ROOT/'build/native-nnue';folder.mkdir(parents=True,exist_ok=True)
    target=folder/f'libnnue-{source[:20]}.so'
    if not target.exists():
        temporary=target.with_suffix('.tmp.so')
        command=['g++','-std=c++17','-O3','-fPIC','-shared','-ffp-contract=off',
                 '-Wall','-Wextra','-Werror',str(files[0]),'-o',str(temporary)]
        result=subprocess.run(command,capture_output=True,text=True,timeout=90)
        if result.returncode:
            raise RuntimeError('native NNUE build failed: '+result.stderr[-3000:])
        temporary.replace(target)
    return target


class NativeSelector:
    """One game's native board history and at most two frozen model handles."""
    def __init__(self, binary=None):
        self.binary=Path(binary) if binary is not None else build_native()
        self.lib=ct.CDLL(str(self.binary))
        self.lib.fukaji_nnue_model_open.argtypes=[ct.POINTER(ct.c_float),ct.c_size_t,
            ct.POINTER(ct.c_double),ct.POINTER(ct.c_float),ct.c_double,ct.POINTER(ct.c_uint64)]
        self.lib.fukaji_nnue_model_open.restype=ct.c_int
        self.lib.fukaji_nnue_model_close.argtypes=[ct.c_uint64]
        self.lib.fukaji_nnue_model_close.restype=ct.c_int
        self.lib.fukaji_nnue_value.argtypes=[ct.c_uint64,ct.c_uint64,ct.c_int,ct.POINTER(ct.c_double)]
        self.lib.fukaji_nnue_value.restype=ct.c_int
        self.lib.fukaji_nnue_search.argtypes=[ct.c_uint64,ct.c_uint64,ct.c_int,ct.c_int,
                                             ct.c_int,ct.c_void_p,ct.c_size_t]
        self.lib.fukaji_nnue_search.restype=ct.c_int
        self.lib.fukaji_nnue_search_interruptible.argtypes=self.lib.fukaji_nnue_search.argtypes+[ct.c_void_p]
        self.lib.fukaji_nnue_search_interruptible.restype=ct.c_int
        self.models={};self.state=None;self.moves=[];self.start_sfen=None;self.closed=False

    def register(self, model):
        if self.closed: raise ValueError('native selector is closed')
        if not isinstance(model,SmallNNUE) or model.inputs!=INPUTS:
            raise ValueError('native prototype requires the 2548-input absolute NNUE')
        hasher=hashlib.sha256()
        for array in (model.hidden_weights,model.hidden_bias,model.output_weights):
            hasher.update(array.tobytes())
        hasher.update(model.output_bias.hex().encode())
        signature=hasher.hexdigest()
        if model in self.models and self.models[model][1]!=signature:
            raise ValueError('native NNUE model must remain frozen')
        if model not in self.models:
            w=np.ascontiguousarray(model.hidden_weights,np.float32)
            b=np.ascontiguousarray(model.hidden_bias,np.float64)
            v=np.ascontiguousarray(model.output_weights,np.float32)
            handle=ct.c_uint64()
            rc=self.lib.fukaji_nnue_model_open(w.ctypes.data_as(ct.POINTER(ct.c_float)),w.size,
                b.ctypes.data_as(ct.POINTER(ct.c_double)),v.ctypes.data_as(ct.POINTER(ct.c_float)),
                model.output_bias,ct.byref(handle))
            if rc or not handle.value: raise RuntimeError(f'native model rejected: {rc}')
            self.models[model]=(handle.value,signature)
        return self.models[model][0]

    def sync(self, board):
        if self.closed: raise ValueError('native selector is closed')
        words=[m.usi() for m in board.move_stack]
        if self.state is None:
            initial=copy.deepcopy(board)
            while initial.move_stack:initial.pop()
            self.start_sfen=initial.sfen()
            self.state=StatefulRules(self.binary,initial,audit=False)
        if len(words)<len(self.moves) or words[:len(self.moves)]!=self.moves:
            raise ValueError('native selector cannot switch game histories')
        for word in words[len(self.moves):]:
            self.state.push(word);self.moves.append(word)
        self.state.assert_equal(board)

    def value(self, board, model, perspective=None):
        self.sync(board);key=self.register(model);out=ct.c_double()
        p=board.turn if perspective is None else perspective
        if p not in (0,1):raise ValueError('invalid perspective')
        rc=self.lib.fukaji_nnue_value(key,self.state.handle.value,int(p),ct.byref(out))
        if rc:raise RuntimeError(f'native NNUE value failed: {rc}')
        return out.value

    def select(self, board, model, *, movetime_ms=50,max_depth=64,fixed_depth=False,
               history=None,ordering=True,quiescence=0,cancel=None):
        if not 1<=movetime_ms<=10000 or not 1<=max_depth<=64:
            raise ValueError('invalid native search bounds')
        if quiescence!=0 or not ordering:
            raise ValueError('native prototype supports basic ordered search with quiescence=0')
        if history is not None and (history.board is not board or history.keys[-1]!=board.zobrist_hash()):
            raise ValueError('unrelated native search history')
        started=time.monotonic();original=board.sfen();counts=+board.transpositions
        self.sync(board);key=self.register(model)
        setup_ms=(time.monotonic()-started)*1000
        budget=max(1,int(movetime_ms-setup_ms))
        output=ct.create_string_buffer(16384)
        args=(key,self.state.handle.value,budget,max_depth,int(fixed_depth),output,len(output))
        rc=(self.lib.fukaji_nnue_search(*args) if cancel is None else
            self.lib.fukaji_nnue_search_interruptible(*args,cancel.handle))
        if rc<0 or rc>=len(output):raise RuntimeError(f'native NNUE search failed: {rc}')
        result=json.loads(output.raw[:rc])
        self.state.assert_equal(board)
        if board.sfen()!=original or +board.transpositions!=counts:
            raise AssertionError('native search changed caller board/history')
        if result['move'] is not None and shogi.Move.from_usi(result['move']) not in board.legal_moves:
            raise AssertionError('native search returned an illegal move')
        result['native_elapsed_ms']=result['elapsed_ms']
        result['native_setup_ms']=setup_ms
        result['elapsed_ms']=(time.monotonic()-started)*1000
        result['backend']='cpp_nnue'
        return result

    def close(self):
        errors=[]
        for handle,_ in self.models.values():
            if self.lib.fukaji_nnue_model_close(handle):errors.append('model close')
        self.models.clear()
        if self.state is not None:
            try:self.state.close()
            except RuntimeError as error:errors.append(str(error))
            self.state=None
        self.closed=True
        if errors:raise RuntimeError('; '.join(errors))

    def __enter__(self):return self
    def __exit__(self,exc_type,exc,tb):self.close()
