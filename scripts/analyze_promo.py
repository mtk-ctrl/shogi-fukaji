"""成れるのに成らなかった手(香・桂・角・飛)で、成った場合との評価差を正式エンジンで測る。研究・閲覧専用。
各ケースで「実際(不成)の後」と「成った後」を同じ時間で探索し、指した側の視点のcp差(成り−不成)を出す。"""
import argparse, json, statistics
import shogi
from analyze_game import Engine, parse_info


def mover_view(score, turn_after, mover):
    if score is None:
        return None
    kind, v = score
    s = v if turn_after == mover else -v    # turn_after側の評価 → 指した側の視点
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--ms", type=int, default=2000)
    a = ap.parse_args()
    e = Engine(a.engine)
    res = []
    for c in json.load(open(a.cases, encoding="utf-8")):
        b = shogi.Board()
        for t in c["before"]:
            b.push_usi(t)
        mover = b.turn
        vals = {}
        for k in ("actual", "alt"):
            b2 = shogi.Board(b.sfen()); b2.push_usi(c[k])
            _, info = e.search(c["before"] + [c[k]], a.ms)
            sc, _, _ = parse_info(info)
            vals[k] = mover_view(sc, b2.turn, mover)
        d = None if None in vals.values() else vals["alt"] - vals["actual"]
        res.append({**{k: c[k] for k in ("game", "ply", "side", "piece")}, "不成": vals["actual"], "成": vals["alt"], "差(成-不成)": d})
        print(f"{c['game']} {c['ply']}手目 {c['side']} {c['piece']} 不成{vals['actual']} 成{vals['alt']} 差{d}", flush=True)
    ds = [r["差(成-不成)"] for r in res if r["差(成-不成)"] is not None]
    print("=== まとめ ===")
    for p in ("l", "n", "b", "r"):
        x = [r["差(成-不成)"] for r in res if r["piece"] == p and r["差(成-不成)"] is not None]
        if x:
            print(f"{p}: {len(x)}件 平均差{statistics.mean(x):+.0f} 中央値{statistics.median(x):+.0f} 成の方が100cp以上良い{sum(v>=100 for v in x)}件 不成の方が100cp以上良い{sum(v<=-100 for v in x)}件")
    print(f"全体: {len(ds)}件 平均差{statistics.mean(ds):+.0f} 成の方が100cp以上良い{sum(v>=100 for v in ds)}件 不成の方が100cp以上良い{sum(v<=-100 for v in ds)}件")
    e.send("quit")


if __name__ == "__main__":
    main()
