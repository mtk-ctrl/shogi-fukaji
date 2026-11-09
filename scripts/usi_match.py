"""深路(FUKAJI)対雲路(KUMOJI)のUSI直接対局：独立開局×先後交替ペア。

- 雲路のアリーナ(benchmarks/arena.py)のUSI通信・勝敗判定(Engine/Adjudicator)をそのまま使い、
  開始局面だけを固定乱数の6手開局にする（毎局同じ棋譜になるのを避ける）。
- 両エンジンとも `go movetime N` の同一条件。先読みはどちらも無効。雲路の長考は無効（条件を揃える）。
- 研究専用。どちらのエンジン・重み・既定値も変更しない。雲路の知識は教師にしない。
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, random, sys, time
from pathlib import Path

import shogi


def load_arena(kumoji_src: Path):
    path = kumoji_src / "benchmarks" / "arena.py"
    spec = importlib.util.spec_from_file_location("kumoji_arena", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(kumoji_src / "benchmarks"))
    sys.path.insert(0, str(kumoji_src))
    spec.loader.exec_module(mod)
    return mod


def openings(count: int, plies: int, seed: int):
    """合法な重複しない開局を count 個。王手・終局を含まない。"""
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < count:
        board, moves = shogi.Board(), []
        for _ in range(plies):
            legal = list(board.legal_moves)
            if not legal:
                break
            m = rng.choice(legal)
            board.push(m)
            moves.append(m.usi())
        if len(moves) != plies or board.is_check() or board.is_game_over():
            continue
        key = tuple(moves)
        if key in seen:
            continue
        seen.add(key)
        out.append(moves)
    return out


def play_game(arena, black, white, opening, max_plies):
    """arena.play_game と同じ判定で、開局棋譜から対局する。"""
    board = shogi.Board()
    adjudicator = arena.Adjudicator(board)
    moves = []
    for token in opening:
        board.push(shogi.Move.from_usi(token))
        moves.append(token)
        terminal = adjudicator.after_move(board)
        if terminal is not None:
            raise ValueError("opening ended the game")
    win = lambda e: (white if e is black else black).label  # 相手側ラベル

    def result(winner, reason, **extra):
        return {"winner": winner, "reason": reason, "plies": len(moves),
                "black": black.label, "opening": opening, "moves": moves,
                "final_sfen": board.sfen(), **extra}

    for ply in range(len(opening), max_plies):
        engine = black if board.turn == shogi.BLACK else white
        token = engine.bestmove(moves)
        if token == "resign":
            return result(win(engine), "resign")
        if token == "win":
            if not arena.can_declare_win(board):
                return result(win(engine), "invalid_declaration", illegal_by=engine.label)
            return result(engine.label, "declare_win")
        try:
            move = shogi.Move.from_usi(token)
        except Exception:
            return result(win(engine), f"invalid_usi:{token}", illegal_by=engine.label)
        if move not in board.legal_moves:
            return result(win(engine), f"illegal_move:{token}", illegal_by=engine.label)
        board.push(move)
        moves.append(token)
        terminal = adjudicator.after_move(board)
        if terminal is not None:
            color, reason = terminal
            winner = None if color is None else (black.label if color == shogi.BLACK else white.label)
            return result(winner, reason)
    return result(None, "move_limit")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cmd_play(a):
    arena = load_arena(Path(a.kumoji_src))
    ops = openings(a.pairs, a.opening_plies, a.seed)
    go = f"go movetime {a.movetime_ms}"
    k_opts = {"AdaptiveLongThink": False, "USI_Ponder": False,
              "PositionKnowledge": True,
              "PositionKnowledgeFile": str(Path(a.kumoji_src) / "position-knowledge-v1.tsv")}
    # A=雲路, B=深路。engine-b のオプションは設定しない（深路は既定=現行正式版）。
    kumoji = arena.Engine(a.kumoji, "KUMOJI", k_opts, go)
    fukaji = arena.Engine(a.fukaji, "FUKAJI", {}, go)
    games = []
    try:
        for p in range(a.shard, a.pairs, a.shards):
            for first_black in ("FUKAJI", "KUMOJI"):
                black, white = (fukaji, kumoji) if first_black == "FUKAJI" else (kumoji, fukaji)
                black.configure_game(a.seed + p * 4)
                white.configure_game(a.seed + p * 4 + 1)
                g = play_game(arena, black, white, ops[p], a.max_plies)
                g["pair"] = p
                games.append(g)
                print(f"pair {p:02d} black={first_black}: {g['winner'] or 'draw'} {g['reason']} {g['plies']}", flush=True)
    finally:
        kumoji.close(); fukaji.close()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"shard-{a.shard:02d}.json").write_text(json.dumps({
        "schema": "fukaji-vs-kumoji-usi-part-v1", "shard": a.shard, "go": go,
        "kumoji_options": k_opts, "kumoji_sha256": sha(a.kumoji),
        "games": games}, ensure_ascii=False) + "\n")


def replay_ok(g):
    board = shogi.Board()
    for t in g["moves"]:
        m = shogi.Move.from_usi(t)
        if m not in board.legal_moves:
            return False
        board.push(m)
    return board.sfen() == g["final_sfen"] or g["reason"] in ("declare_win", "resign") or "illegal" in g["reason"] or g["reason"].startswith("invalid")


def cmd_aggregate(a):
    games = []
    for f in sorted(Path(a.parts).rglob("shard-*.json")):
        games += json.loads(f.read_text())["games"]
    n = len(games)
    wins = {"FUKAJI": 0, "KUMOJI": 0}
    for g in games:
        if g["winner"] in wins:
            wins[g["winner"]] += 1
    draws = n - wins["FUKAJI"] - wins["KUMOJI"]
    illegal = [g["pair"] for g in games if "illegal_by" in g]
    reasons = {}
    for g in games:
        reasons[g["reason"]] = reasons.get(g["reason"], 0) + 1
    pairs = {}
    for g in games:
        pairs.setdefault(g["pair"], []).append(g)
    pair_score = [sum((1 if x["winner"] == "FUKAJI" else 0.5 if x["winner"] is None else 0) for x in v)
                  for v in pairs.values()]
    summary = {"schema": "fukaji-vs-kumoji-usi-result-v1", "games": n, "pairs": len(pairs),
               "fukaji_wins": wins["FUKAJI"], "kumoji_wins": wins["KUMOJI"], "draws_or_unresolved": draws,
               "fukaji_score": (wins["FUKAJI"] + 0.5 * draws) / n if n else None,
               "pair_sweeps_fukaji": sum(s == 2 for s in pair_score),
               "pair_sweeps_kumoji": sum(s == 0 for s in pair_score),
               "illegal_pairs": illegal, "reasons": reasons,
               "replay_verified": all(replay_ok(g) for g in games),
               "avg_plies": sum(g["plies"] for g in games) / n if n else None}
    Path(a.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if n != a.expect_games or not summary["replay_verified"]:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    q = sub.add_parser("play")
    q.add_argument("--kumoji", required=True); q.add_argument("--kumoji-src", required=True)
    q.add_argument("--fukaji", required=True)
    q.add_argument("--shard", type=int, default=0); q.add_argument("--shards", type=int, default=1)
    q.add_argument("--pairs", type=int, default=50); q.add_argument("--opening-plies", type=int, default=6)
    q.add_argument("--movetime-ms", type=int, default=200); q.add_argument("--max-plies", type=int, default=300)
    q.add_argument("--seed", type=int, default=202611091)
    q.add_argument("--out", required=True)
    r = sub.add_parser("aggregate")
    r.add_argument("--parts", required=True); r.add_argument("--out", required=True)
    r.add_argument("--expect-games", type=int, default=100)
    a = p.parse_args()
    cmd_play(a) if a.mode == "play" else cmd_aggregate(a)


if __name__ == "__main__":
    main()
