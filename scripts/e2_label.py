"""E2: 選んだ局面を現行正式モデル＋正式探索(固定深さ4)で評価してラベル化する。1 shard分。"""
from pathlib import Path
import argparse, json, multiprocessing as mp, sys, time
import shogi
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fukaji.formal_model import load
from fukaji.native_nnue import NativeSelector
from scripts.e2_data import load_games, pick_plies

_MODEL = None
DEPTH = 4


def work(item):
    global _MODEL
    if _MODEL is None:
        _MODEL = load()[0]
    g = item["game"]; plies = pick_plies(item)
    if not plies:
        return []
    board = shogi.Board(g["start_sfen"]) if g.get("start_sfen") not in (None, shogi.STARTING_SFEN) else shogi.Board()
    sel = NativeSelector(); rows = []
    try:
        want = set(plies)
        for i, tok in enumerate(g["moves"]):
            if i in want and not board.is_game_over():
                r = sel.select(board, _MODEL, movetime_ms=10000, max_depth=DEPTH, fixed_depth=True)
                if r.get("score") is not None and r["depth"] == DEPTH:
                    rows.append({"key": item["key"], "ply": i, "score": r["score"], "depth": r["depth"],
                                 "mate": abs(r["score"]) > 900000})
            board.push(shogi.Move.from_usi(tok))
    finally:
        sel.close()
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
    a = p.parse_args()
    items = load_games(a.parts, a.archives, a.state)
    mine = [x for i, x in enumerate(items) if i % a.shards == a.shard]
    print(f"shard {a.shard}: games {len(mine)}/{len(items)} planned positions {sum(len(pick_plies(x)) for x in mine)}", flush=True)
    t0 = time.time(); rows = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(work, mine, chunksize=2), 1):
            rows += r
            if n % 50 == 0:
                print(f"  {n}/{len(mine)} games, {len(rows)} labels, {time.time()-t0:.0f}s", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / f"labels-{a.shard:02d}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"done {len(rows)} labels in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
