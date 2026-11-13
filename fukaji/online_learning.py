"""Outcome-anchored TD(lambda), sparse NNUE training and immutable snapshots.

All targets use side-to-move probabilities. The frozen parent predicts the
bootstrap targets; gradients never flow through the target. This is value
policy iteration with alpha-beta, not an AlphaZero policy-network replica.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import shogi

from .nnue_research import SmallNNUE, INPUTS, HIDDEN, KING_INPUTS
from .nnue_train_research import SOURCE_ZIP_SHA, MERGED_SHA
from .online_search import Position, terminal

SCHEMA = "fukaji-online-nnue-v1"
KING_SCHEMA = "fukaji-online-nnue-king-relative-v1"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def implementation_digest():
    base = Path(__file__).parent
    return digest(b"\n".join((base/name).read_bytes() for name in
                  ("nnue_research.py", "online_learning.py", "online_search.py", "online_cycle.py",
                   "native_nnue.py", "../native/nnue_search_ffi.cpp",
                   "../native/rules_state_ffi.cpp", "../native/rules_probe.cpp")))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fresh_model(seed):
    rng = np.random.default_rng(seed)
    return SmallNNUE(rng.normal(0, .02, (INPUTS, HIDDEN)),
                     np.full(HIDDEN, .1), rng.normal(0, .02, HIDDEN))


def parameters(model):
    return [model.hidden_weights.copy(), model.hidden_bias.astype(np.float32),
            model.output_weights.copy(), np.array([model.output_bias], dtype=np.float32)]


def expand_king_relative(parent, optimizer):
    """Preserve trained parameters/moments and zero only the additional rows."""
    if parent.king_relative:
        raise ValueError('parent already has king-relative features')
    def expand(array):
        return np.concatenate((array, np.zeros((KING_INPUTS-INPUTS, HIDDEN), dtype=array.dtype)))
    model = SmallNNUE(expand(parent.hidden_weights), parent.hidden_bias,
                      parent.output_weights, parent.output_bias)
    opt = None if optimizer is None else {
        'step': optimizer['step'],
        'm': [expand(optimizer['m'][0]), *[x.copy() for x in optimizer['m'][1:]]],
        's': [expand(optimizer['s'][0]), *[x.copy() for x in optimizer['s'][1:]]]}
    return model, opt


def expand_king_zone(parent, optimizer):
    """玉周辺±2升の小特徴へ拡張。追加行だけゼロ(拡張直後の評価は親と一致)。"""
    from .nnue_research import ZONE_INPUTS
    if parent.king_relative or parent.king_zone:
        raise ValueError('parent already expanded')
    def expand(array):
        return np.concatenate((array, np.zeros((ZONE_INPUTS-INPUTS, HIDDEN), dtype=array.dtype)))
    model = SmallNNUE(expand(parent.hidden_weights), parent.hidden_bias,
                      parent.output_weights, parent.output_bias)
    opt = None if optimizer is None else {
        'step': optimizer['step'],
        'm': [expand(optimizer['m'][0]), *[x.copy() for x in optimizer['m'][1:]]],
        's': [expand(optimizer['s'][0]), *[x.copy() for x in optimizer['s'][1:]]]}
    return model, opt


def save_snapshot(root, model, optimizer, metadata):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    arrays = dict(zip(("w", "b", "v", "o"), parameters(model)))
    if optimizer is not None:
        arrays["step"] = np.array(optimizer["step"], dtype=np.int64)
        for i in range(4):
            arrays[f"m{i}"] = optimizer["m"][i]
            arrays[f"s{i}"] = optimizer["s"][i]
    buff = io.BytesIO()
    np.savez_compressed(buff, **arrays)
    raw = buff.getvalue()
    key = digest(raw)
    path = root / (key + ".npz")
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("immutable snapshot collision")
    if not path.exists():
        temp = path.with_suffix(".tmp")
        temp.write_bytes(raw)
        os.replace(temp, path)
    manifest = {"schema": KING_SCHEMA if model.king_relative else SCHEMA,
                "sha256": key, "architecture": [model.inputs, HIDDEN, 1],
                "implementation_sha256": implementation_digest(),
                "research_only": True, "formal_adoption": False, **metadata}
    manifest_path = root / (key + ".json")
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("snapshot provenance cannot be overwritten")
    atomic_json(manifest_path, manifest)
    load_snapshot(root, key)
    return key


def load_snapshot(root, key):
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("invalid snapshot ID")
    root = Path(root)
    raw = (root / (key + ".npz")).read_bytes()
    manifest = json.loads((root / (key + ".json")).read_text())
    if digest(raw) != key or manifest.get("sha256") != key or manifest.get("schema") not in (SCHEMA, KING_SCHEMA):
        raise ValueError("snapshot checksum/schema mismatch")
    if manifest.get("formal_adoption") is not False or manifest.get("research_only") is not True:
        raise ValueError("not a research snapshot")
    with np.load(io.BytesIO(raw), allow_pickle=False) as a:
        model = SmallNNUE(a["w"], a["b"], a["v"], a["o"][0])
        if (manifest.get('architecture') != [model.inputs, HIDDEN, 1] or
            manifest['schema'] != (KING_SCHEMA if model.king_relative else SCHEMA)):
            raise ValueError('snapshot architecture/schema mismatch')
        opt = None
        if "step" in a:
            opt = {"step": int(a["step"]), "m": [a[f"m{i}"].copy() for i in range(4)],
                   "s": [a[f"s{i}"].copy() for i in range(4)]}
            if opt["step"] < 0:
                raise ValueError("invalid optimizer step")
            for p, m, s in zip(parameters(model), opt["m"], opt["s"]):
                if m.shape != p.shape or s.shape != p.shape or not np.isfinite(m).all() or not np.isfinite(s).all() or (s < 0).any():
                    raise ValueError("invalid optimizer state")
    return model, opt, manifest


def probability(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -30, 30)))


def targets(probabilities, result, *, outcome_weight=.75, td_lambda=.8):
    """Forward lambda return; final p is a verified terminal rule outcome.

    result[t] is the actual outcome for the player to move at t. In a
    two-player zero-sum game the next player's value is complemented.
    """
    p, z = np.asarray(probabilities, float), np.asarray(result, float)
    if len(p) != len(z) or not len(z) or not np.isfinite(p).all() or not np.isfinite(z).all():
        raise ValueError("invalid trajectory")
    if np.any((p < 0) | (p > 1)) or np.any((z < 0) | (z > 1)):
        raise ValueError("probability out of range")
    if not 0 <= outcome_weight <= 1 or not 0 <= td_lambda <= 1:
        raise ValueError("invalid target mixture")
    ret = np.empty_like(z)
    ret[-1] = z[-1]
    for t in range(len(z)-2, -1, -1):
        ret[t] = (1-td_lambda)*(1-p[t+1]) + td_lambda*(1-ret[t+1])
    return outcome_weight*z + (1-outcome_weight)*ret


def game_key(game):
    return digest((game["start_sfen"] + "|" + " ".join(game["moves"])).encode())


def opening_key(start_sfen, moves):
    return digest((start_sfen + "|" + " ".join(moves[:6])).encode())


def validate_game(game, model, *, training=False, protected=(), feature_model=None):
    if game.get("source") != "selfplay" or game.get("role") not in ("train", "monitor", "audit"):
        raise ValueError("untrusted game source or role")
    if training and (game["role"] != "train" or game.get("parent_id") in protected):
        raise ValueError("evaluation parent cannot enter training")
    if type(game.get("winner")) not in (int, type(None)) or game["winner"] not in (0, 1, None):
        raise ValueError("invalid winner")
    if not game.get("moves"):
        raise ValueError("empty trajectory")
    b = shogi.Board(game["start_sfen"])
    state = Position(b, model)
    ix, turns, probs = [], [], []
    for word in game["moves"]:
        # The next move's independent legal-membership check rejects play after
        # mate/no legal moves. Only repetition needs a separate prefix guard.
        if state.repetition() is not None:
            raise ValueError("move after terminal position")
        ix.append((feature_model or model).active_indices(b, b.turn))
        turns.append(b.turn)
        probs.append(float(probability(state.logit())))
        state.push(shogi.Move.from_usi(word))
    end = terminal(state, list(b.legal_moves))
    expected = end or {"status": "move_limit_unresolved", "winner": None}
    if (game["status"], game["winner"]) != (expected["status"], expected["winner"]):
        raise ValueError("result differs from legal replay")
    resolved = expected["status"] in ("checkmate", "perpetual_check", "repetition_draw")
    z = np.array([.5 if game["winner"] is None else float(t == game["winner"]) for t in turns])
    return ix, np.array(probs), z, resolved


def bootstrap_games(archive):
    raw = Path(archive).read_bytes()
    if digest(raw) != SOURCE_ZIP_SHA:
        raise ValueError("bootstrap artifact checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        games_raw = z.read("merged_verified_mates.jsonl")
        mapping = json.loads(z.read("frozen_parent_splits.json"))
    if digest(games_raw) != MERGED_SHA:
        raise ValueError("bootstrap games checksum mismatch")
    games, excluded = [], set()
    for line in games_raw.splitlines():
        game = json.loads(line)
        excluded.add(opening_key(game["start_sfen"], game["moves"]))
        if mapping[game["parent_id"]] == "train":
            games.append({**game, "role": "train", "status": "checkmate"})
    if len(games) != 665:
        raise ValueError("frozen train partition changed")
    return games, {k for k, v in mapping.items() if v != "train"}, excluded


def make_samples(games, parent, *, seed, per_game=32, outcome_weight=.75,
                 td_lambda=.8, protected=(), feature_model=None):
    rng = np.random.default_rng(seed)
    samples, labels, stats, seen = [], [], {"games": 0, "resolved": 0, "unresolved_excluded": 0}, set()
    for game in games:
        key = game_key(game)
        if key in seen:
            continue
        seen.add(key)
        ix, probs, z, resolved = validate_game(game, parent, training=True, protected=protected,
                                             feature_model=feature_model)
        stats["games"] += 1
        if not resolved:
            stats["unresolved_excluded"] += 1
            continue
        stats["resolved"] += 1
        y = targets(probs, z, outcome_weight=outcome_weight, td_lambda=td_lambda)
        # Adjacent pairs retain both players. Random positions change each cycle.
        first = game.get("learning_start_ply", 0)
        if type(first) is not int or not 0 <= first < len(ix):
            raise ValueError("invalid learning boundary")
        available = max(1, len(ix)-first-1)
        starts = first + rng.choice(available, min(max(1, per_game//2), available), replace=False)
        selected = sorted({int(j) for i in starts for j in (i, min(i+1, len(ix)-1))})
        samples.extend(ix[i] for i in selected)
        labels.extend(y[i] for i in selected)
    if not samples:
        raise ValueError("no verified resolved train experience")
    width = max(map(len, samples))
    indices = np.full((len(samples), width), (feature_model or parent).inputs, dtype=np.int32)
    for i, row in enumerate(samples):
        indices[i, :len(row)] = row
    return indices, np.asarray(labels, np.float32), stats


def logits_for(params, ix):
    w, b, v, o = params
    padded = np.vstack((w, np.zeros((1, HIDDEN), np.float32)))
    pre = padded[ix].sum(axis=1) + b
    hidden = np.clip(pre, 0, 1)
    return pre, hidden, hidden @ v + o[0]


def fit(parent, optimizer, samples, labels, *, epochs=2, lr=.001, seed=1,
        batch_size=128):
    if not (1 <= epochs <= 20 and 0 < lr <= .01 and 1 <= batch_size <= 1024):
        raise ValueError("training bounds exceeded")
    if samples.ndim != 2 or labels.shape != (len(samples),) or not len(labels) or not np.isfinite(labels).all() or np.any((labels < 0) | (labels > 1)) or np.any((samples < 0) | (samples > parent.inputs)):
        raise ValueError("invalid sparse batch")
    params = parameters(parent)
    if optimizer is None:
        opt = {"step": 0, "m": [np.zeros_like(p) for p in params], "s": [np.zeros_like(p) for p in params]}
    else:
        opt = {"step": optimizer["step"], "m": [x.copy() for x in optimizer["m"]], "s": [x.copy() for x in optimizer["s"]]}
    rng = np.random.default_rng(seed)

    def loss():
        total = 0.
        for i in range(0, len(labels), 512):
            logits = logits_for(params, samples[i:i+512])[2]
            total += float(np.sum(np.logaddexp(0, logits)-labels[i:i+512]*logits))
        return total/len(labels)

    history = [{"epoch": 0, "loss": loss()}]
    max_grad = 0.
    for epoch in range(epochs):
        order = rng.permutation(len(labels))
        for start in range(0, len(order), batch_size):
            selected = order[start:start+batch_size]
            ix, y = samples[selected], labels[selected]
            pre, hidden, logits = logits_for(params, ix)
            error = (probability(logits)-y)/len(y)
            dh = error[:, None]*params[2][None, :]*((pre > 0) & (pre < 1))
            gw = np.zeros((parent.inputs+1, HIDDEN), np.float32)
            np.add.at(gw, ix.reshape(-1), np.repeat(dh, ix.shape[1], axis=0))
            grads = [gw[:-1], dh.sum(axis=0), hidden.T @ error, np.array([error.sum()], np.float32)]
            norm = float(np.sqrt(sum(np.sum(g.astype(float)**2) for g in grads)))
            if not np.isfinite(norm):
                raise ValueError("nonfinite gradient")
            max_grad = max(max_grad, norm)
            opt["step"] += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                g = g / max(1., norm)  # global norm cap 1; no score-target clipping
                opt["m"][i] = .9*opt["m"][i] + .1*g
                opt["s"][i] = .999*opt["s"][i] + .001*g*g
                p -= lr*((opt["m"][i]/(1-.9**opt["step"])) /
                         (np.sqrt(opt["s"][i]/(1-.999**opt["step"]))+1e-8) + 1e-5*p)
        history.append({"epoch": epoch+1, "loss": loss()})
    candidate = SmallNNUE(params[0], params[1], params[2], params[3][0])
    delta = float(np.sqrt(sum(np.sum((a.astype(float)-b)**2) for a,b in zip(params, parameters(parent)))))
    return candidate, opt, {"positions": len(labels), "optimizer_step": opt["step"],
                            "loss_history": history, "max_gradient_norm": max_grad,
                            "parameter_delta_l2": delta, "strength_proven": False}
