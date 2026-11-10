"""研究専用：隠れ層を広げたNNUEを扱うための部品（正式版のソース・重み・契約は変更しない）。
- build_wide_native(H): native/nnue_search_ffi.cpp の kNHidden だけをHに置換した研究用ライブラリをbuild/に作る。
- use_hidden(H): nnue_research / online_learning のHIDDEN定数をHに差し替える（読み込み・学習の前に呼ぶ）。
"""
import hashlib, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def build_wide_native(H):
    src = (ROOT / "native/nnue_search_ffi.cpp").read_text()
    if src.count("kNHidden=32") != 1:
        raise RuntimeError("unexpected native source")
    text = src.replace("kNHidden=32", f"kNHidden={H}")
    key = hashlib.sha256((text + str(H)).encode()).hexdigest()[:16]
    out = ROOT / "build/native-wide"; out.mkdir(parents=True, exist_ok=True)
    cpp = out / f"ffi{H}-{key}.cpp"; so = out / f"libnnue-h{H}-{key}.so"
    if not so.exists():
        cpp.write_text(text)
        r = subprocess.run(["g++", "-std=c++17", "-O3", "-fPIC", "-shared", "-ffp-contract=off",
                            "-I", str(ROOT / "native"), str(cpp), "-o", str(so) + ".tmp"],
                           capture_output=True, text=True, timeout=180)
        if r.returncode:
            raise RuntimeError("wide native build failed: " + r.stderr[-2000:])
        Path(str(so) + ".tmp").replace(so)
    return so


def use_hidden(H):
    import fukaji.nnue_research as nr
    import fukaji.online_learning as ol
    nr.HIDDEN = H; ol.HIDDEN = H
