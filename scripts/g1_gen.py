"""G1 大量データ生成: ハイブリッド(雲路の探索＋深路NNUE)の短時間自己対局を大量に行い、
各局面の探索評価値(手番側cp)と対局結果を棋譜ごとに記録する。1 shard分(コア数ぶん並列)。
- 教師は深路NNUE＋借用した探索の値と、対局の勝敗だけ。雲路の評価値・定跡・局面知識は使わない(PositionKnowledge=false)。
- 開局は固定乱数のランダム合法手(定跡なし)。以後は同一エンジンが両者を指す。
- 出力: games-XX.jsonl.gz  1行=1局 {"k":開局手数,"m":[指し手],"s":[各手の直前局面の評価cp または null],"w":勝者(0先手/1後手/null),"r":理由}
  詰みの読み筋は ±30000 として記録する。
環境: HYB_ENGINE, HYB_CWD, FUKAJI_NNUE_FILE, FUKAJI_NNUE_SCALE。"""
from pathlib import Path
import argparse, gzip, json, multiprocessing as mp, os, random, re, subprocess, sys, time
import shogi

MATE_CP = 30000
INFO = re.compile(r"info depth (\d+) .*?score (cp|mate) (-?\d+)")


class Engine:
    def __init__(self, kumoji=False):
        """kumoji=True: 同じ実行ファイルをNNUEなしで起動＝雲路本体(練習相手。局面知識は相手本来の設定で有効)。"""
        env = dict(os.environ)
        if kumoji:
            env.pop("FUKAJI_NNUE_FILE", None)
        self.p = subprocess.Popen([os.environ["HYB_ENGINE"]], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
                                  stderr=subprocess.DEVNULL, text=True, cwd=os.environ.get("HYB_CWD") or None, bufsize=1)
        self.send("usi"); self.wait("usiok")
        for name in ("USI_Ponder", "AdaptiveLongThink") + (() if kumoji else ("PositionKnowledge",)):
            self.send(f"setoption name {name} value false")
        self.send("isready"); self.wait("readyok")

    def send(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def wait(self, token):
        lines = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine closed")
            lines.append(line)
            if line.startswith(token):
                return lines

    def go(self, moves, movetime):
        self.send("position startpos moves " + " ".join(moves))
        self.send(f"go movetime {movetime}")
        lines = self.wait("bestmove")
        score = None
        for line in lines:
            m = INFO.match(line)
            if m:
                v = int(m.group(3))
                score = (MATE_CP if v > 0 else -MATE_CP) if m.group(2) == "mate" else max(-MATE_CP, min(MATE_CP, v))
        return lines[-1].split()[1], score


def key(board):
    return " ".join(board.sfen().split()[:3])


def play(engine, rng, opening_plies, movetime, max_plies, opp=None, hybrid_color=None):
    """opp(雲路本体)があれば hybrid_color 側だけ深路が指す。相手番の評価値は記録しない(雲路の評価を教師にしない)。"""
    while True:   # 定跡なしのランダム開局
        board, moves = shogi.Board(), []
        for _ in range(opening_plies):
            legal = list(board.legal_moves)
            if not legal:
                break
            m = rng.choice(legal); board.push(m); moves.append(m.usi())
        if len(moves) == opening_plies and not board.is_check() and any(board.legal_moves):
            break
    engine.send("usinewgame")
    if opp:
        opp.send("usinewgame")
    scores = [None] * opening_plies
    seen = {}
    for k in (key(board),):
        seen[k] = 1
    while len(moves) < max_plies:
        if opp and int(board.turn) != hybrid_color:
            token, score = opp.go(moves, movetime)[0], None
        else:
            token, score = engine.go(moves, movetime)
        if token == "resign":
            return moves, scores, int(board.turn ^ 1), "resign"
        if token == "win":
            return moves, scores, int(board.turn), "declare_win"
        move = shogi.Move.from_usi(token)
        if move not in board.legal_moves:
            return moves, scores, None, "illegal"       # 学習には使わない
        board.push(move); moves.append(token); scores.append(score)
        k = key(board); seen[k] = seen.get(k, 0) + 1
        if seen[k] >= 4:
            return moves, scores, None, "repetition"    # 千日手(連続王手の区別はせず引き分け扱い。学習側で結果0.5)
        if not any(board.legal_moves):
            return moves, scores, int(board.turn ^ 1), "checkmate"
    return moves, scores, None, "move_limit"


def worker(args):
    wid, a = args
    rng = random.Random(a.seed * 100003 + a.shard * 1009 + wid)
    engine = Engine()
    opp = Engine(kumoji=True) if a.opponent == "kumoji" else None
    out, t0, positions = [], time.time(), 0
    while time.time() - t0 < a.minutes * 60 and len(out) < a.max_games:
        hc = len(out) % 2 if opp else None
        moves, scores, winner, reason = play(engine, rng, a.opening_plies, a.movetime_ms, a.max_plies, opp, hc)
        out.append({"k": a.opening_plies, "m": moves, "s": scores, "w": winner, "r": reason, **({"h": hc} if opp else {})})
        positions += sum(s is not None for s in scores)
        if len(out) % 50 == 0:
            print(f"worker {wid}: games {len(out)} positions {positions} elapsed {time.time()-t0:.0f}s", flush=True)
    try:
        engine.send("quit")
    except Exception:
        pass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--minutes", type=float, required=True)
    p.add_argument("--movetime-ms", type=int, default=50)
    p.add_argument("--opening-plies", type=int, default=8)
    p.add_argument("--max-plies", type=int, default=320)
    p.add_argument("--max-games", type=int, default=10**9)
    p.add_argument("--opponent", choices=["self", "kumoji"], default="self")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    with mp.Pool(a.workers) as pool:
        parts = pool.map(worker, [(w, a) for w in range(a.workers)])
    games = [g for part in parts for g in part]
    a.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out / f"games-{a.shard:02d}.jsonl.gz", "wt") as f:
        for g in games:
            f.write(json.dumps(g, separators=(",", ":")) + "\n")
    reasons = {}
    for g in games:
        reasons[g["r"]] = reasons.get(g["r"], 0) + 1
    summary = {"schema": "g1-gen-part-v1", "shard": a.shard, "seed": a.seed, "movetime_ms": a.movetime_ms, "games": len(games),
               "positions": sum(sum(s is not None for s in g["s"]) for g in games), "reasons": reasons, "opponent": a.opponent,
               "hybrid_wins": sum(g.get("h") is not None and g["w"] == g["h"] for g in games),
               "mean_plies": sum(len(g["m"]) for g in games) / max(1, len(games))}
    (a.out / f"summary-{a.shard:02d}.json").write_text(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
