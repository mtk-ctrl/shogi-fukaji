"""ハイブリッドエンジン(雲路の探索＋深路E9のNNUE評価)を作る。雲路のコードは変更せず、固定コミットの複製に最小パッチを当てる。
使い方: python scripts/build_hybrid.py --kumoji-src <雲路のチェックアウト> --work build/hybrid [--probe]
パッチ: engine/strategy/evaluation.h の FeatureEvaluator::operator() だけ。FUKAJI_NNUE_FILEがあればNNUE、無ければ雲路本来の評価。"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--kumoji-src", type=Path, required=True)
p.add_argument("--work", type=Path, default=ROOT / "build/hybrid")
p.add_argument("--probe", action="store_true")
a = p.parse_args()
src, work = a.kumoji_src.resolve(), a.work.resolve()
if work.exists():
    shutil.rmtree(work)
shutil.copytree(src, work, ignore=shutil.ignore_patterns(".git", "build", "android", "__pycache__"))
shutil.copy(ROOT / "research/hybrid/nnue_eval.h", work / "engine/strategy/nnue_eval.h")
ev = work / "engine/strategy/evaluation.h"
t = ev.read_text()
old = "    int operator()(const rules::Snapshot& s) const {return evaluate(s,params_,reuse_mask_).total;}"
new = ("    int operator()(const rules::Snapshot& s) const {\n"
       "        if (nnue_hybrid::enabled()) return nnue_hybrid::eval_black(s);\n"
       "        return evaluate(s,params_,reuse_mask_).total;\n    }")
assert t.count(old) == 1, "evaluation.h operator() not found exactly once"
t = t.replace(old, new)
inc = '#include "strategy/evaluation_v2_features.h"\n'
assert t.count(inc) == 1
t = t.replace(inc, inc + '#include "strategy/nnue_eval.h"\n')
ev.write_text(t)
subprocess.run(["git", "init", "-q"], cwd=work, check=True)  # 親リポジトリのgit applyに巻き込まれないよう独立させる
out = work / "build/kumoji-hybrid"
subprocess.run([sys.executable, "scripts/build.py", "--output", "build/kumoji-hybrid"], cwd=work, check=True)
print(out)
if a.probe:
    cmd = json.loads((work / "build/kumoji-hybrid.build.json").read_text())["command"]
    cmd = [c if not c.endswith("engine/main.cpp") else str(ROOT / "research/hybrid/probe.cpp") for c in cmd]
    cmd[cmd.index("-o") + 1] = str(work / "build/hybrid-probe")
    subprocess.run(cmd, cwd=work, check=True)
    print(work / "build/hybrid-probe")
