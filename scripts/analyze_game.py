"""棋譜の指定した手を、正式エンジン(ハイブリッド)で検討して標準出力に出す。研究・閲覧専用。
各手について 局面前の最善手とPV、実際の手を指した後の評価、指定した代替手を指した後の評価を、先手視点のcpで示す。"""
import argparse, json, subprocess, sys
import shogi
import shogi.KIF as K


def jp(board, usi):
    try:
        return K.Exporter.kif_move_from(usi, board).strip()
    except Exception:
        return usi


def pv_jp(board, pv, n=8):
    b = shogi.Board(board.sfen())
    out = []
    for t in pv[:n]:
        try:
            out.append(jp(b, t)); b.push_usi(t)
        except Exception:
            break
    return " ".join(out)


class Engine:
    def __init__(self, cmd):
        self.p = subprocess.Popen([cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send("usi"); self.wait("usiok")
        for k, v in (("PositionKnowledge", "false"), ("AdaptiveLongThink", "false"), ("USI_Ponder", "false")):
            self.send(f"setoption name {k} value {v}")
        self.send("isready"); self.wait("readyok")

    def send(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def wait(self, tok):
        for l in self.p.stdout:
            if l.startswith(tok):
                return
        raise RuntimeError("engine closed")

    def search(self, moves, ms):
        self.send("usinewgame")
        self.send("position startpos" + (" moves " + " ".join(moves) if moves else ""))
        self.send(f"go movetime {ms}")
        last = None
        for l in self.p.stdout:
            if l.startswith("info") and " score " in l and " pv " in l:
                last = l
            if l.startswith("bestmove"):
                return l.split()[1], last
        raise RuntimeError("engine closed")


def parse_info(line):
    if not line:
        return None, [], None
    t = line.split()
    score = None
    if "score" in t:
        i = t.index("score")
        score = (t[i + 1], int(t[i + 2]))
    depth = int(t[t.index("depth") + 1]) if "depth" in t else None
    pv = t[t.index("pv") + 1:] if "pv" in t else []
    return score, pv, depth


def sente_view(score, turn_after):
    """turn_after: 評価した局面の手番(shogi.BLACK/WHITE)。cpを先手視点にする。"""
    if score is None:
        return None
    kind, v = score
    s = v if turn_after == shogi.BLACK else -v
    return f"詰み{'先手' if s > 0 else '後手'}勝ち({abs(v)})" if kind == "mate" else f"{s:+d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--moves", required=True)
    ap.add_argument("--plies", required=True, help="検討する手数(1始まり)のカンマ区切り")
    ap.add_argument("--alts", default="{}", help='JSON {"67":["1d1b+"]}')
    ap.add_argument("--ms", type=int, default=3000)
    a = ap.parse_args()
    moves = a.moves.split()
    alts = json.loads(a.alts)
    e = Engine(a.engine)
    for ply in [int(x) for x in a.plies.split(",")]:
        before = moves[:ply - 1]
        board = shogi.Board()
        for t in before:
            board.push_usi(t)
        actual = moves[ply - 1]
        side = "先手" if board.turn == shogi.BLACK else "後手"
        print(f"=== {ply}手目 {side} 実際: {jp(board, actual)} ({actual})")
        print("局面前SFEN:", board.sfen())
        best, info = e.search(before, a.ms)
        sc, pv, d = parse_info(info)
        print(f"  局面前の最善: {jp(board, best)} 評価{sente_view(sc, board.turn)} 深さ{d}  PV: {pv_jp(board, pv)}")
        for label, mv in [("実際の手", actual)] + [("代替", x) for x in alts.get(str(ply), [])]:
            b2 = shogi.Board(board.sfen())
            if shogi.Move.from_usi(mv) not in b2.legal_moves:
                print(f"  {label} {mv}: 非合法"); continue
            name = jp(b2, mv); b2.push_usi(mv)
            _, info2 = e.search(before + [mv], a.ms)
            sc2, pv2, d2 = parse_info(info2)
            print(f"  {label} {name}の後: 先手視点{sente_view(sc2, b2.turn)} 深さ{d2}  続き: {pv_jp(b2, pv2)}")
    e.send("quit")


if __name__ == "__main__":
    main()
