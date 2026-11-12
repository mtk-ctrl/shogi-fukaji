"""E2のラベル付け(ハイブリッド版): 局面ごとに、雲路の探索＋深路NNUE評価のハイブリッドエンジンを固定深さで探索し、
その評価値(側手番視点のcp)をロジット(cp/scale)に戻して、e2_label.pyと同じ形式で出力する。教師=深路の評価＋借用した探索(雲路の評価値・指し手は使わない: 評価は常にNNUE)。
1 shard分。環境: E2_DEPTH(既定8), HYB_ENGINE(ハイブリッド実行ファイル), FUKAJI_NNUE_FILE, FUKAJI_NNUE_SCALE。"""
from pathlib import Path
import argparse, json, multiprocessing as mp, os, re, subprocess, sys, time
import shogi
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.e2_data import load_games, pick_plies

DEPTH = int(os.environ.get("E2_DEPTH", "8"))
MOVETIME_MS = int(os.environ.get("E2_MOVETIME_MS", "0"))  # >0なら固定深さでなく固定時間(終盤の詰み絡みで深さ固定は数十秒〜分かかる局面があるため)
SCALE = float(os.environ.get("FUKAJI_NNUE_SCALE", "600"))
_ENG = None


class Engine:
    def __init__(self):
        exe = os.environ["HYB_ENGINE"]
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                  cwd=os.environ.get("HYB_CWD") or None, bufsize=1)
        self.send("usi"); self.wait("usiok")
        self.send("setoption name PositionKnowledge value false")
        self.send("setoption name USI_Ponder value false")
        self.send("isready"); self.wait("readyok")

    def send(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def wait(self, token, collect=None):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine closed")
            if collect is not None:
                collect.append(line)
            if line.startswith(token):
                return line

    def score(self, sfen, depth):
        self.send("usinewgame"); self.send(f"position sfen {sfen}")
        self.send(f"go movetime {MOVETIME_MS}" if MOVETIME_MS else f"go depth {depth}")
        lines = []
        self.wait("bestmove", lines)
        best = None
        for line in lines:
            m = re.match(r"info depth (\d+) .*?score (cp|mate) (-?\d+)", line)
            if m and (MOVETIME_MS or int(m.group(1)) >= depth):
                best = (m.group(2), int(m.group(3)), int(m.group(1)))
        return best

    def close(self):
        try:
            self.send("quit"); self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def work(item):
    global _ENG
    if _ENG is None:
        _ENG = Engine()
    g = item["game"]; plies = set(pick_plies(item))
    if not plies:
        return []
    board = shogi.Board(g["start_sfen"]) if g.get("start_sfen") not in (None, shogi.STARTING_SFEN) else shogi.Board()
    rows = []
    for i, tok in enumerate(g["moves"]):
        if i in plies and not board.is_game_over():
            r = _ENG.score(board.sfen(), DEPTH)
            if r is not None:
                kind, v, reached = r
                logit = (20.0 if v > 0 else -20.0) if kind == "mate" else max(-20.0, min(20.0, v / SCALE))
                rows.append({"key": item["key"], "ply": i, "score": logit * 1000.0, "depth": reached, "mate": kind == "mate"})
        board.push(shogi.Move.from_usi(tok))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", type=Path, required=True)
    p.add_argument("--archives", type=Path, nargs="+", required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-games", type=int, default=0, help="試験用: この棋譜数まで")
    a = p.parse_args()
    items = load_games(a.parts, a.archives, a.state)
    mine = [x for i, x in enumerate(items) if i % a.shards == a.shard]
    if a.max_games:
        mine = mine[:a.max_games]
    print(f"shard {a.shard}: games {len(mine)}/{len(items)} planned positions {sum(len(pick_plies(x)) for x in mine)} depth {DEPTH}", flush=True)
    t0 = time.time(); rows = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        limit = float(os.environ.get("E2_TIME_LIMIT_MIN", "0")) * 60
        for n, r in enumerate(pool.imap_unordered(work, mine, chunksize=1), 1):
            rows += r
            if n % 25 == 0:
                print(f"  {n}/{len(mine)} games, {len(rows)} labels, {time.time()-t0:.0f}s", flush=True)
            if limit and time.time() - t0 > limit:  # 時間切れ: そこまでのラベルを出力して終える(部分出力)
                print(f"time limit {limit/60:.0f} min reached at {n}/{len(mine)} games; writing partial labels", flush=True)
                pool.terminate()
                break
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / f"labels-{a.shard:02d}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"done {len(rows)} labels in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
