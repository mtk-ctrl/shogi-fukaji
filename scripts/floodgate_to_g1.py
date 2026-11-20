"""Floodgateの対局記録(floodgate-results.jsonl)を、G1学習データ(games-XX.jsonl.gz＋summary-XX.json)の形に変換する。
2026-11-20本人決定：学習の対象は「自己対局」と「Floodgateでの対局」。雲路本体との対局は学習に使わない。
- 教師は深路の手番の探索値(our_search.score_black_view)と勝敗だけ。相手エンジンの評価値・指し手の評価は使わない(相手手番のsはnull)。
- 記録側の learning_eligible=False(従来の安全側の既定)は、この本人決定により学習に使う。変換結果には出典を残す。
使い方: python scripts/floodgate_to_g1.py --in <jsonlのあるディレクトリ> --out <出力> [--scale 2400]"""
import argparse, gzip, json
from pathlib import Path


def convert(rec, only="all"):
    if rec.get("kind") != "floodgate_external_match":
        return None
    moves = rec.get("moves_usi") or []
    if len(moves) < 20:
        return None
    ours = rec.get("your_turn")                       # "+"=先手 / "-"=後手
    me = 0 if ours == "+" else 1 if ours == "-" else None
    if me is None:
        return None
    result = (rec.get("result") or "").lower()
    if result == "win":
        w = me
    elif result == "lose":
        w = 1 - me
    elif result in ("draw", "sennichite", "jishogi"):
        w = None
    else:
        return None                                   # 結果不明は使わない
    if only == "lost" and w != 1 - me:
        return None                                   # 負けた対局だけ使う(2026-11-20本人決定)
    if only == "win" and w != me:
        return None
    scores = [None] * len(moves)
    for e in rec.get("our_search", []):
        p, black = e.get("ply"), e.get("score_black_view")
        if not isinstance(p, int) or not (1 <= p <= len(moves)) or not isinstance(black, (int, float)):
            continue
        if abs(black) >= 29000:                       # 詰みスコアは学習に使わない
            continue
        scores[p - 1] = int(black) if me == 0 else -int(black)   # 直前局面の手番側から見たcp
    if not any(s is not None for s in scores):
        return None
    return {"k": 0, "m": moves, "s": scores, "w": w, "r": rec.get("cause") or "floodgate", "h": me, "g": rec.get("game_id")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scale", type=float, default=2400.0)
    p.add_argument("--only", choices=["all", "lost", "win"], default="lost", help="使う対局。既定は負けた対局だけ")
    p.add_argument("--repeat", type=int, default=1, help="学習で比率を上げたいときに同じ局を繰り返し書く(量が少ないため)")
    p.add_argument("--min-games", type=int, default=0, help="使える対局がこれ未満なら出力せず終了コード3(まとまった量になるまで待つ)")
    a = p.parse_args()
    seen, games, rows = set(), [], 0
    for f in sorted(a.inp.rglob("floodgate-results.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rows += 1
            rec = json.loads(line)
            gid = rec.get("game_id")
            if gid in seen:
                continue
            g = convert(rec, a.only)
            if g:
                seen.add(gid); games.append(g)
    if len(games) < a.min_games:
        print(json.dumps({"waiting": True, "games": len(games), "min_games": a.min_games}, ensure_ascii=False))
        raise SystemExit(3)
    a.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out / "games-00.jsonl.gz", "wt") as fh:
        for _ in range(max(1, a.repeat)):
            for g in games:
                fh.write(json.dumps(g, separators=(",", ":")) + "\n")
    positions = sum(sum(s is not None for s in g["s"]) for g in games)
    summary = {"source": "floodgate", "records_read": rows, "games": len(games), "positions": positions, "only": a.only, "repeat": a.repeat,
               "nnue_scale": a.scale, "note": "雲路本体との対局は含まない。相手手番の評価値は使わない。"}
    (a.out / "summary-00.json").write_text(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
