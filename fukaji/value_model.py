"""Explicit, optional inference adapter for FUKAJI linear value-v0.

Returns a score from the SIDE TO MOVE perspective, matching negamax.
This is a verification bridge, not NNUE, and no strength gain is presumed.
"""
import hashlib
from pathlib import Path
import numpy as np
from .learning import FEATURES, features

class LinearValue:
    def __init__(self, path, weight_cp=100, mix_percent=50):
        if not 0 <= mix_percent <= 100:
            raise ValueError("mix_percent outside 0..100")
        if not 0 < weight_cp <= 2000:
            raise ValueError("weight_cp outside 1..2000")
        self.path = str(path)
        self.sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as data:
            if set(data.files) != {"weight"}:
                raise ValueError("model must contain only weight")
            w = data["weight"]
        if w.shape != (FEATURES,) or not np.isfinite(w).all():
            raise ValueError("incorrect/nonfinite model shape")
        self.weights = np.asarray(w,dtype=np.float32).copy()
        self.weight_cp = float(weight_cp)
        self.mix = mix_percent/100.0

    def score(self, board):
        # Model was trained on side-to-move WDL targets, not Black targets.
        # The logit is a centipawn-like heuristic, not a calibrated cp value.
        logit = float(np.dot(features(board),self.weights))
        return int(round(max(-10.0,min(10.0,logit))*self.weight_cp))

    def combine(self, board, classical):
        # The 0% control must not spend time evaluating a model it ignores.
        if self.mix == 0:
            return int(classical)
        learned = self.score(board)
        return int(round((1-self.mix)*classical + self.mix*learned))
