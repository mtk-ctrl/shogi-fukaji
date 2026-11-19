"""距離帯特徴(入力4784)のPython実装とC++評価器(probe)の評価値が一致するかの検査。乱数重み・乱数局面。棋力の実験ではない。
使い方: python scripts/band_parity.py --kumoji-src <雲路のチェックアウト> --work build/band-parity"""
import argparse, random, subprocess, sys, os
from pathlib import Path
import numpy as np, shogi
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from fukaji.nnue_research import SmallNNUE, BAND_INPUTS, HIDDEN
p = argparse.ArgumentParser(); p.add_argument("--kumoji-src", type=Path, required=True); p.add_argument("--work", type=Path, required=True)
p.add_argument("--positions", type=int, default=400); a = p.parse_args()
subprocess.run([sys.executable, str(ROOT / "scripts/build_hybrid.py"), "--kumoji-src", str(a.kumoji_src), "--work", str(a.work), "--probe"], check=True)
rng = np.random.default_rng(7)
model = SmallNNUE(rng.normal(0, .08, (BAND_INPUTS, HIDDEN)), rng.normal(0, .08, HIDDEN), rng.normal(0, .3, HIDDEN), .05)
wfile = a.work / "w.bin"; npz = a.work / "w.npz"
np.savez(npz, w=model.hidden_weights, b=model.hidden_bias.astype(np.float32), v=model.output_weights, o=np.array([model.output_bias], np.float32))
subprocess.run([sys.executable, str(ROOT / "scripts/hybrid_export_weights.py"), str(npz), str(wfile)], check=True)
random.seed(11); boards = []
while len(boards) < a.positions:
    b = shogi.Board()
    for _ in range(random.randint(2, 160)):
        mv = list(b.legal_moves)
        if not mv: break
        b.push(random.choice(mv))
    boards.append(b)
sfens = "\n".join(b.sfen() for b in boards) + "\n"
worst = 0.0
for inc in ("0", "1"):
    env = dict(os.environ, FUKAJI_NNUE_FILE=str(wfile), FUKAJI_NNUE_INCREMENTAL=inc)
    out = subprocess.run([str(a.work / "build/hybrid-probe")], input=sfens, capture_output=True, text=True, env=env, check=True).stdout.split("\n")
    for b, line in zip(boards, out):
        cpp = float(line.split()[0]); py = model.logit(b)
        worst = max(worst, abs(cpp - py))
        if abs(cpp - py) > 2e-3: raise SystemExit(f"MISMATCH inc={inc} {b.sfen()} cpp={cpp} py={py}")
print(f"OK: {a.positions}局面 x (全計算, 差分更新) 最大差 {worst:.2e}")
