"""hybrid-match の各シャード(shard-NN.json)から、全局をKIF(UTF-8とShift_JIS)とCSVにして1つのZIPにまとめる。
研究用の取り出し専用。正式版・対局結果は変更しない。
使い方: python scripts/export_kifu.py --parts parts --out kifu/<run_id>.zip --name-a 候補 --name-b 現行 """
import argparse, csv, io, json, zipfile
from pathlib import Path
import shogi
import shogi.KIF as K

START = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-a", default="HYBRID")
    ap.add_argument("--label-b", default="OPP")
    ap.add_argument("--name-a", default="候補")
    ap.add_argument("--name-b", default="現行")
    a = ap.parse_args()
    names = {a.label_a: a.name_a, a.label_b: a.name_b}
    games = []
    for f in sorted(Path(a.parts).rglob("shard-*.json")):
        games += json.loads(f.read_text())["games"]
    games.sort(key=lambda g: (g["pair"], g["black"] != a.label_a))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [["file", "pair", "black", "white", "winner", "reason", "plies"]]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for g in games:
            black = g["black"]
            white = a.label_b if black == a.label_a else a.label_a
            win = "b" if g["winner"] == black else "w" if g["winner"] == white else None
            # 開局が詰みで終わっていないか・合法かを再生で確かめる
            board = shogi.Board()
            for t in g["moves"]:
                board.push_usi(t)
            s = {"names": [names[black], names[white]], "sfen": START, "moves": g["moves"]}
            if win:
                s["win"] = win
            kif = K.Exporter.kif(s)
            stem = f"pair{g['pair']:02d}_先手{names[black]}_後手{names[white]}"
            z.writestr(f"kif_utf8/{stem}.kif", kif)
            z.writestr(f"kif_sjis/{stem}.kif", kif.encode("cp932", errors="replace"))
            rows.append([stem + ".kif", g["pair"], names[black], names[white],
                         names.get(g["winner"], "引分/未決着"), g["reason"], g["plies"]])
        buf = io.StringIO()
        csv.writer(buf).writerows(rows)
        z.writestr("index.csv", "﻿" + buf.getvalue())
    print(f"{len(games)}局を {out} に書いた")


if __name__ == "__main__":
    main()
