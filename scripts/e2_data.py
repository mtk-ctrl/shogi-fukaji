"""E2共通: 蒸留に使う局面の決定的な選び方（深路=NNUE主軸。雲路の知識は使わない）。
ラベルは現行正式モデルの正式探索(固定深さ4)の評価値。雲路の指し手・評価値は使わない。
"""
from pathlib import Path
import hashlib, json, os, random, sys, tarfile
import shogi
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fukaji.online_learning import game_key
from scripts.nnue_round_train import convert

KUMOJI_STRIDE = 3
SELFPLAY_PER_GAME = 13
VARIANT = int(os.environ.get("E2_VARIANT", "0"))  # 周回ごとに別の局面を選ぶ(0=初回と同じ)


def load_games(parts: Path, archives, protected_state: Path):
    out = []
    kg = []
    for f in sorted(Path(parts).rglob("shard-*.json")):
        kg += json.loads(f.read_text())["games"]
    seen = set()
    for c in map(convert, kg):
        if not c:
            continue
        g, fk = c
        k = game_key(g)
        if k not in seen:
            seen.add(k); out.append({"key": k, "game": g, "kind": "kumoji", "fk": fk})
    for f in sorted(archives):
        with tarfile.open(f) as t:
            names = sorted(n for n in t.getnames() if n.startswith("games/game-") and n.endswith(".json"))
            for n in names:
                g = json.load(t.extractfile(n)); k = game_key(g)
                if k not in seen:
                    seen.add(k); out.append({"key": k, "game": g, "kind": "selfplay", "fk": None})
    out.sort(key=lambda x: x["key"])
    return out


def pick_plies(item):
    """この棋譜のどの局面(手を指す前)にラベルを付けるか。決定的。"""
    g = item["game"]; n = len(g["moves"])
    lo = 6
    if n <= lo:
        return []
    if item["kind"] == "kumoji":
        fk = item["fk"]
        return [i for i in range(lo, n) if (i % 2 == 0) == (fk == shogi.BLACK) and (i - lo) % KUMOJI_STRIDE == VARIANT % KUMOJI_STRIDE] + \
               [i for i in range(lo, n) if (i % 2 == 0) != (fk == shogi.BLACK) and (i - lo) % KUMOJI_STRIDE == (VARIANT + 1) % KUMOJI_STRIDE]
    rng = random.Random(int(item["key"][:12], 16) + 7919 * VARIANT)
    pool = list(range(lo, n))
    rng.shuffle(pool)
    return sorted(pool[:SELFPLAY_PER_GAME])
