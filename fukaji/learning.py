"""Strict self-play outcome learning baseline (not yet an NNUE engine).
Dataset: JSONL {start_sfen, moves:[USI], winner:0|1|null, source:"selfplay"}.
Only completed wins are used for value supervision; no external engine labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import shogi

PIECES = 14
FEATURES = 81 * PIECES * 2 + 2 * 7 * 20 + 1

def features(board):
    x = np.zeros(FEATURES, dtype=np.float32)
    for sq in range(81):
        p = board.piece_at(sq)
        if p is None:
            continue
        kind = p.piece_type - 1
        assert 0 <= kind < PIECES
        x[(int(p.color) * PIECES + kind) * 81 + sq] = 1.0
    base = 81 * PIECES * 2
    for color in (shogi.BLACK, shogi.WHITE):
        for kind in range(1, 8):
            count = int(board.pieces_in_hand[color].get(kind, 0))
            for j in range(min(20, count)):
                x[base + (color * 7 + kind - 1) * 20 + j] = 1.0
    x[-1] = 1.0 if board.turn == shogi.BLACK else -1.0
    return x

def load_games(path):
    seen = set()
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        g = json.loads(line)
        required = {"start_sfen", "moves", "winner", "source"}
        if not (set(g) == required or set(g) == required | {"parent_id"}):
            raise ValueError(f"line {line_no}: invalid game schema")
        if g["source"] != "selfplay":
            raise ValueError("external/human game rejected")
        if g["winner"] not in (0, 1, None) or type(g["winner"]) not in (int, type(None)):
            raise ValueError("invalid winner")
        if not isinstance(g["moves"], list) or not g["moves"]:
            raise ValueError("empty game")
        key = hashlib.sha256((g["start_sfen"] + "|" + " ".join(g["moves"])).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        group_key = g.get("parent_id", key)
        if not isinstance(group_key,str) or len(group_key)!=64 or any(c not in "0123456789abcdef" for c in group_key):
            raise ValueError(f"line {line_no}: invalid parent_id")
        b = shogi.Board(g["start_sfen"])
        positions = []
        turns = []
        for move_text in g["moves"]:
            move = shogi.Move.from_usi(move_text)
            if move not in b.legal_moves:
                raise ValueError(f"line {line_no}: illegal move {move_text}")
            positions.append(features(b))
            turns.append(int(b.turn))
            b.push(move)
        # No invented wins for unfinished games; no reward for draws.
        if g["winner"] is not None:
            if not b.is_game_over():
                raise ValueError(f"line {line_no}: reported win without terminal position")
            # python-shogi does not attribute every end condition; mate/no-legal-moves only.
            if not b.is_checkmate():
                raise ValueError(f"line {line_no}: terminal reason not verified checkmate")
            if g["winner"] != int(not b.turn):
                raise ValueError(f"line {line_no}: inconsistent winner")
        else:
            continue
        rows.append((group_key, np.stack(positions), np.array([1.0 if turn == g["winner"] else 0.0 for turn in turns], dtype=np.float32)))
    return rows

def partition_rows(rows):
    """Deterministic independent *parent-game* partition, not position split.

    Hold out at least two independent parents per validation/test split when
    six or more distinct parents exist. Assignments are frozen within each
    experiment; do not use the held-out test split for model selection.
    """
    groups=sorted({key for key,_,_ in rows},key=lambda key: hashlib.sha256(("fukaji-group-v2|"+key).encode()).hexdigest())
    n=len(groups)
    if n>=6:
        heldout=max(2,round(n*0.1))
        val_n=test_n=heldout
    elif n>=3:
        val_n=test_n=1
    elif n==2:
        val_n,test_n=0,1
    else:
        val_n=test_n=0
    train_n=n-val_n-test_n
    labels={key: ("train" if i<train_n else "validation" if i<train_n+val_n else "test") for i,key in enumerate(groups)}
    parts={k: [] for k in ("train","validation","test")}
    for row in rows:
        parts[labels[row[0]]].append(row[1:])
    group_counts={k:sum(v==k for v in labels.values()) for k in parts}
    return parts,group_counts

def fit(rows, epochs=12, lr=0.05, seed=42):
    if not rows:
        raise ValueError("no verified completed self-play games")
    parts,group_counts = partition_rows(rows)
    if not parts["train"]:
        raise ValueError("no train games")
    # Small linear value model is only the integration baseline, not an NNUE.
    train = parts["train"]
    rng = np.random.default_rng(seed)
    w = np.zeros(FEATURES, dtype=np.float32)
    for epoch in range(epochs):
        for index in rng.permutation(len(train)):
            x, y = train[index]
            logits = np.clip(x @ w, -20, 20)
            p = 1.0 / (1.0 + np.exp(-logits))
            gradient = (x.T @ (p - y)) / len(y) + 0.0001 * w
            w -= lr * gradient.astype(np.float32)
    def evaluate(part):
        if not part:
            return {"games": 0, "samples": 0, "log_loss": None}
        losses = []
        count = 0
        for x, y in part:
            logits = np.clip(x @ w, -20, 20)
            losses.append(float(np.logaddexp(0, logits).sum() - (y * logits).sum()))
            count += len(y)
        return {"games": len(part), "samples": count, "log_loss": sum(losses) / count}
    return w, {k: evaluate(v) for k, v in parts.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True)
    ap.add_argument("--out", default="build/model")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    if not (1 <= args.epochs <= 100):
        ap.error("epochs must be 1..100")
    games = load_games(args.games)
    w, metrics = fit(games, epochs=args.epochs)
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "value-v0.npz", weight=w)
    source_sha256 = hashlib.sha256(Path(args.games).read_bytes()).hexdigest()
    model_sha256 = hashlib.sha256((target / "value-v0.npz").read_bytes()).hexdigest()
    report = {"model": "linear-value-v0", "not_nnue": True, "engine_integration": False, "seed": 42, "verified_completed_games": len(games), "source_sha256": source_sha256, "model_sha256": model_sha256, "splits": metrics, "note": "Accuracy on self-play outcomes does not establish chess strength."}
    (target / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
