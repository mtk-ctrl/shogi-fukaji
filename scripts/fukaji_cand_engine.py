"""研究専用USIエンジン：正式版と同じ探索・USI。NNUE重みだけを FUKAJI_CAND_NPZ の候補に差し替える。
正式版のソース・重み・USI既定は変更しない（候補は環境変数で明示したときだけ使う）。"""
import io, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fukaji.formal_model as fm
from fukaji.nnue_research import SmallNNUE, INPUTS, HIDDEN as FORMAL_HIDDEN

_orig = fm.load
def load(path=None, *, specification=None):
    cand = os.environ.get("FUKAJI_CAND_NPZ")
    if not cand:
        return _orig(path, specification=specification)
    with np.load(cand, allow_pickle=False) as d:
        if set(d.files) != {"w", "b", "v", "o"} or d["o"].shape != (1,):
            raise ValueError("inference-only NNUE tensors required")
        hidden = int(d["w"].shape[1])
        if hidden != FORMAL_HIDDEN:  # 研究専用：隠れ層を広げた候補は専用ネイティブライブラリで動かす
            from scripts.h_widen_lib import build_wide_native, use_hidden
            import fukaji.native_nnue as nn
            lib = build_wide_native(hidden); use_hidden(hidden)
            nn.build_native = lambda: lib
        model = SmallNNUE(d["w"], d["b"], d["v"], d["o"][0])
    if model.inputs != INPUTS:
        raise ValueError("architecture mismatch")
    return model, specification or fm.manifest()
fm.load = load
import fukaji.formal_usi as usi
if __name__ == "__main__":
    import fukaji.engine as engine
    engine.command_loop()
