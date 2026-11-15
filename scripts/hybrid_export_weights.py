"""FUKAJI推論用npz(w,b,v,o)を、ハイブリッドエンジン用の生バイナリへ変換する。
形式: "FKJNNUE1"(8B) + inputs(u32=2548 または 17576=玉相対拡張) + hidden(u32=128) + w[inputs][128] + b[128] + v[128] + o  (すべてfloat32, リトルエンディアン)"""
import argparse, struct
import numpy as np
p = argparse.ArgumentParser(); p.add_argument("npz"); p.add_argument("out"); a = p.parse_args()
if a.npz == "formal":   # 現行の正式ハイブリッドNNUE(formal/model.json の hybrid節。無ければ最上位)
    import json, pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = json.loads((root / "formal/model.json").read_text())
    a.npz = str(root / spec["hybrid"].get("weights_file", spec["weights_file"]))
with np.load(a.npz, allow_pickle=False) as d:
    w, b, v, o = d["w"].astype("<f4"), d["b"].astype("<f4"), d["v"].astype("<f4"), d["o"].astype("<f4")
    v2 = [d[k].astype("<f4") for k in ("l2", "b2", "w3")] if "l2" in d.files else None   # G2(研究): 両視点＋2層目
H = w.shape[1]   # 隠れ層: 128(正式) または 256(研究)
assert H in (128, 256) and w.shape[0] in (2548, 17576, 3848) and b.shape == (H,) and v.shape == (H,) and o.shape == (1,)
with open(a.out, "wb") as f:
    if v2 is None:
        f.write(b"FKJNNUE1" + struct.pack("<II", w.shape[0], H))
    else:
        l2, b2, w3 = v2
        h2 = l2.shape[1]
        assert w.shape == (3848, 128) and l2.shape == (2 * H, h2) and b2.shape == (h2,) and w3.shape == (h2,) and 1 <= h2 <= 64
        f.write(b"FKJNNUE2" + struct.pack("<III", w.shape[0], H, h2))
    f.write(w.tobytes()); f.write(b.tobytes()); f.write(v.tobytes()); f.write(o.tobytes())
    if v2 is not None:
        f.write(np.ascontiguousarray(l2).tobytes()); f.write(b2.tobytes()); f.write(w3.tobytes())   # L2 は [2H][h2] の順
print("wrote", a.out)
