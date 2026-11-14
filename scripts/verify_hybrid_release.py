"""正式ハイブリッドエンジン(run-fukaji.sh)の実起動検証。棋力実験ではない。
1) USI起動・isready  2) 実際に2局(各30手)を合法手で指せる  3) NNUEが有効(NNUEなしの同ビルドと評価値が異なる)"""
import argparse, json, os, queue, subprocess, sys, threading, time
from pathlib import Path
import shogi
ROOT = Path(__file__).resolve().parents[1]


class Eng:
    def __init__(self, cmd, env=None, cwd=None):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env, cwd=cwd)
        self.q = queue.Queue(); self.lines = []
        threading.Thread(target=lambda: [self.q.put(l.strip()) for l in self.p.stdout], daemon=True).start()
    def send(self, s): self.p.stdin.write(s + "\n"); self.p.stdin.flush()
    def until(self, prefix, timeout=60):
        end = time.monotonic() + timeout; out = []
        while True:
            l = self.q.get(timeout=max(.01, end - time.monotonic())); out.append(l)
            if l.startswith(prefix): return out
    def close(self):
        try: self.send("quit"); self.p.wait(10)
        except Exception: self.p.kill()


def last_cp(lines):
    cp = None
    for l in lines:
        if l.startswith("info") and " score cp " in l:
            cp = int(l.split(" score cp ")[1].split()[0])
    return cp


def usi_move_ok(board, mv):
    try: m = shogi.Move.from_usi(mv)
    except Exception: return False
    return m in board.legal_moves


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    spec = json.loads((ROOT / "formal/model.json").read_text())
    res = {"weights_sha256": spec["hybrid"].get("weights_sha256", spec["weights_sha256"]), "kumoji_ref": spec["hybrid"]["kumoji_ref"]}
    e = Eng([str(ROOT / "run-fukaji.sh")])
    e.send("usi"); res["usi_id"] = [l for l in e.until("usiok") if l.startswith("id name")]
    e.send("isready"); e.until("readyok", 90)
    plies = 0
    for g in range(2):
        board = shogi.Board(); moves = []; e.send("usinewgame")
        for ply in range(30):
            e.send("position startpos" + (" moves " + " ".join(moves) if moves else "")); e.send("go movetime 100")
            mv = e.until("bestmove ", 20)[-1].split()[1]
            assert usi_move_ok(board, mv), (g, ply, mv)
            board.push(shogi.Move.from_usi(mv)); moves.append(mv); plies += 1
            if board.is_game_over(): break
    res["legal_usi_plies"] = plies
    # NNUE有効性: 同じ局面を深さ固定で、NNUEあり/なしの同一バイナリで比べる
    binary = ROOT / "build/hybrid/build/kumoji-hybrid"
    wbin = ROOT / "build/fukaji-hybrid-weights.bin"
    diffs = 0; n = 0
    for pos in ("startpos moves 7g7f 3c3d 2g2f", "startpos moves 7g7f 8c8d 2g2f 8d8e 2f2e", "startpos moves 2g2f 3c3d 7g7f 4a3b 3i4h"):
        scores = []
        for use in (True, False):
            env = dict(os.environ)
            if use: env["FUKAJI_NNUE_FILE"] = str(wbin); env["FUKAJI_NNUE_SCALE"] = str(spec["hybrid"]["nnue_scale"])
            else: env.pop("FUKAJI_NNUE_FILE", None)
            x = Eng([str(binary)], env=env, cwd=str(ROOT / "build/hybrid"))
            x.send("usi"); x.until("usiok"); x.send("setoption name PositionKnowledge value false"); x.send("isready"); x.until("readyok", 60)
            x.send("position " + pos); x.send("go depth 6"); lines = x.until("bestmove ", 60); x.close()
            scores.append(last_cp(lines))
        n += 1; diffs += scores[0] != scores[1]
    res["nnue_active_positions"] = diffs; res["nnue_checked_positions"] = n
    assert plies >= 30 and diffs == n, res
    e.close()
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n"); print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
