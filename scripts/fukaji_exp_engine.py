"""研究専用USIエンジン：正式版と同じ深路USIだが、実験探索(native/research)のC++を使う。
FUKAJI_SEARCH_EXP=0 なら正式版と同じ探索。正式版のソース・重み・USI既定は変更しない。"""
import hashlib, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fukaji.native_nnue as nn

def build_exp():
    files = [ROOT/"native"/"research"/"nnue_search_exp_ffi.cpp", ROOT/"native"/"rules_state_ffi.cpp",
             ROOT/"native"/"rules_probe.cpp"]
    h = hashlib.sha256(b"\n".join(p.read_bytes() for p in files)).hexdigest()
    folder = ROOT/"build"/"native-nnue-exp"; folder.mkdir(parents=True, exist_ok=True)
    target = folder/f"libnnue-exp-{h[:20]}.so"
    if not target.exists():
        tmp = target.with_suffix(".tmp.so")
        cmd = ["g++","-std=c++17","-O3","-fPIC","-shared","-ffp-contract=off","-Wall","-Wextra","-Werror",
               str(files[0]),"-o",str(tmp)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode: raise RuntimeError("exp native build failed: "+r.stderr[-3000:])
        tmp.replace(target)
    return target

nn.build_native = build_exp
import fukaji.engine as engine
if __name__ == "__main__":
    engine.command_loop()
