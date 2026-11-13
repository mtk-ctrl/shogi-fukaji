"""正式エンジン(ハイブリッド): 雲路の探索(固定コミット)＋深路の正式NNUE評価。USIで起動する。
- 重み・設定は formal/model.json (hybrid節) から読む。雲路ソースは固定コミットを取得(env FUKAJI_KUMOJI_SRC で既存チェックアウトを指定可)。
- 初回は雲路を取得してビルドする(数分)。GUIから使う前に `python3 scripts/run_hybrid.py --prepare` を実行しておくこと。
- ビルド中の出力はstderrへ(USIのstdoutを汚さない)。
ライセンス: 雲路はGPL-3.0。ビルドした実行ファイルを配布する場合は対応ソースを併せて提供する(THIRD_PARTY_NOTICES.md)。"""
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    spec = json.loads((ROOT / "formal/model.json").read_text())
    h = spec["hybrid"]
    work = ROOT / "build/hybrid"
    binary = work / "build/kumoji-hybrid"
    wbin = ROOT / "build/fukaji-hybrid-weights.bin"
    npz = ROOT / spec["weights_file"]
    if sha(npz) != spec["weights_sha256"]:
        raise SystemExit("formal NNUE SHA256 mismatch")
    stamp = json.dumps({"ref": h["kumoji_ref"], "eval_h": sha(ROOT / "research/hybrid/nnue_eval.h"),
                        "build": sha(ROOT / "scripts/build_hybrid.py")}, sort_keys=True)
    stamp_file = work / "stamp.json"
    if not (binary.exists() and stamp_file.exists() and stamp_file.read_text() == stamp):
        src = os.environ.get("FUKAJI_KUMOJI_SRC")
        if src:
            src = Path(src)
        else:
            src = ROOT / "build/kumoji-src"
            if not src.exists():
                src.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "clone", "-q", h["kumoji_repo"], str(src)], check=True, stdout=sys.stderr)
            subprocess.run(["git", "-C", str(src), "fetch", "-q", "origin", h["kumoji_ref"]], stdout=sys.stderr)
            subprocess.run(["git", "-C", str(src), "checkout", "-q", h["kumoji_ref"]], check=True, stdout=sys.stderr)
        subprocess.run([sys.executable, str(ROOT / "scripts/build_hybrid.py"), "--kumoji-src", str(src), "--work", str(work)],
                       check=True, stdout=sys.stderr)
        stamp_file.write_text(stamp)
    shaf = wbin.with_suffix(".sha")
    if not (wbin.exists() and shaf.exists() and shaf.read_text() == spec["weights_sha256"]):
        subprocess.run([sys.executable, str(ROOT / "scripts/hybrid_export_weights.py"), str(npz), str(wbin)], check=True, stdout=sys.stderr)
        shaf.write_text(spec["weights_sha256"])
    return spec, binary, wbin, work


def main():
    spec, binary, wbin, work = prepare()
    if "--prepare" in sys.argv:
        print("prepared", binary, file=sys.stderr)
        return
    env = dict(os.environ)
    env.setdefault("FUKAJI_NNUE_FILE", str(wbin))
    env.setdefault("FUKAJI_NNUE_SCALE", str(spec["hybrid"]["nnue_scale"]))
    os.chdir(work)
    os.execve(str(binary), [str(binary)] + sys.argv[1:], env)


if __name__ == "__main__":
    main()
