"""FUKAJI推論用npz(w,b,v,o)を、ハイブリッドエンジン用の生バイナリへ変換する。
形式: "FKJNNUE1"(8B) + inputs(u32=2548) + hidden(u32=128) + w[2548][128] + b[128] + v[128] + o  (すべてfloat32, リトルエンディアン)"""
import argparse, struct
import numpy as np
p = argparse.ArgumentParser(); p.add_argument("npz"); p.add_argument("out"); a = p.parse_args()
with np.load(a.npz, allow_pickle=False) as d:
    w, b, v, o = d["w"].astype("<f4"), d["b"].astype("<f4"), d["v"].astype("<f4"), d["o"].astype("<f4")
assert w.shape == (2548, 128) and b.shape == (128,) and v.shape == (128,) and o.shape == (1,)
with open(a.out, "wb") as f:
    f.write(b"FKJNNUE1" + struct.pack("<II", 2548, 128))
    f.write(w.tobytes()); f.write(b.tobytes()); f.write(v.tobytes()); f.write(o.tobytes())
print("wrote", a.out)
