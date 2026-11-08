"""D64: v0.4.1 D62-adopted NNUE selfplay move generator; retains D41's proven clock and checkpoint logic.
Only the inherited D62 optimizer step differs. This module does not publish or adopt.
"""
from __future__ import annotations
import json, random, tarfile
from pathlib import Path
import shogi
from fukaji.online_learning import atomic_json,load_snapshot,validate_game
from fukaji.online_search import Position,terminal
from fukaji.native_nnue import NativeSelector
D64_PARENT = "e04e85dfebb507131df591bf0206e27771afe6944f3249124dfaf6eab5de51fb"
SEED = 202611081820
PAIRS = 20
MAX_PLIES = 512
NORMAL_MS = 200
LONG_MS = 1000
MAX_LONG_PER_SIDE = 10
def need_long(board, ply: int, legal, used: int) -> bool:
    if used >= MAX_LONG_PER_SIDE or ply < 12:
        return False
    captures = sum(board.piece_type_at(m.to_square) != shogi.NONE for m in legal)
    promotions = sum(m.promotion for m in legal)
    return bool(board.is_check() or captures >= 2 or
                (promotions >= 2 and ply >= 24) or
                (ply >= 48 and len(legal) >= 32 and ply % 16 == 0))


def pack_tar(folder: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as t:
        for item in sorted(folder.rglob("*")):
            if item.is_file():
                t.add(item, arcname=item.relative_to(folder))

def segment(runtime: Path, lane: int, game_side: int, out: Path, search_limit: int) -> None:
    plan = json.loads((runtime / "plan.json").read_text())
    if plan["baseline"] != D64_PARENT or len(plan["families"]) != PAIRS:
        raise ValueError("invalid plan")
    family = plan["families"][lane]["openings"][game_side]
    model, opt, _ = load_snapshot(runtime / "models", D64_PARENT)
    if opt["step"] != 10080:
        raise ValueError("D62 Adam10080 optimizer changed")
    out.mkdir(parents=True, exist_ok=True)
    complete = out / f"game-{game_side}.json"
    partial = out / f"game-{game_side}.partial.json"
    if complete.exists():
        return
    if partial.exists():
        state = json.loads(partial.read_text())
        if state["baseline"] != D64_PARENT or state["parent_id"] != family["parent_id"]:
            raise ValueError("unexpected resume state")
        moves = list(state["moves"])
        trace = list(state["telemetry"])
        used = list(state["used"])
        state["moves"] = moves
        state["telemetry"] = trace
        state["used"] = used
    else:
        moves = list(family["prefix"])
        trace = []
        used = [0, 0]
        state = dict(baseline=D64_PARENT, parent_id=family["parent_id"], moves=moves,
                     telemetry=trace, used=used, status="in_progress")
        atomic_json(partial, state)
    board = shogi.Board()
    for move in moves:
        word = shogi.Move.from_usi(move)
        if word not in board.legal_moves:
            raise ValueError("illegal history checkpoint")
        board.push(word)
    history = Position(board, model)
    end = None
    count = 0
    with NativeSelector() as native:
        native.register(model)
        while len(moves) < MAX_PLIES:
            legal = list(board.legal_moves)
            end = terminal(history, legal)
            if end:
                break
            if count >= search_limit:
                return
            player = int(board.turn)
            long = need_long(board, len(moves), legal, used[player])
            ms = LONG_MS if long else NORMAL_MS
            row = native.select(board, model, movetime_ms=ms, max_depth=64,
                                history=history, quiescence=0)
            if not row.get("move"):
                raise ValueError("native search returned no move in nonterminal position")
            if row.get("research_mate_nodes") is None:
                raise ValueError("D58 selfplay MUST use official D49 PVS+mate search, old alpha-beta rejected")
            # Exploratory moves are limited to the first 24 newly searched moves.
            explore_rng = random.Random(SEED + 100001 * lane + 4099 * game_side +
                                        len(moves) * 7151)
            explore = (len(moves) < len(family["prefix"]) + 24 and
                       explore_rng.random() < 0.05)
            if explore:
                word = explore_rng.choice(legal).usi()
                decision = "exploration"
            else:
                word = row["move"]
                decision = "nnue_search"
            move = shogi.Move.from_usi(word)
            if move not in legal:
                raise ValueError("illegal selected move")
            if long:
                used[player] += 1
            trace.append({"ply": len(moves) + 1, "side": player, "budget_ms": ms,
                          "longthink": long, "decision_source": decision,
                          **row, "move": word})
            history.push(move)
            moves.append(word)
            count += 1
            atomic_json(partial, state)
        end = end or terminal(history, list(board.legal_moves)) or {
            "status": "move_limit_unresolved", "winner": None,
        }
    game = {
        "start_sfen": shogi.STARTING_SFEN, "moves": moves, **end,
        "source": "selfplay", "role": "train", "parent_id": family["parent_id"],
        "seed": SEED + 100001 * lane + 4099 * game_side,
        "models": [D64_PARENT, D64_PARENT], "learning_start_ply": len(family["prefix"]),
        "telemetry": trace,
        "conditions": {"movetime_ms": NORMAL_MS, "longthink_ms": LONG_MS,
            "max_longthink_per_side": MAX_LONG_PER_SIDE,
            "longthink_used_per_side": used, "max_additional_plies": MAX_PLIES - len(family["prefix"]),
            "backend_by_side": ["cpp_nnue", "cpp_nnue"], "quiescence_by_side": [0, 0],
            "exploration_probability": .05, "exploration_additional_plies": 24,
            "knowledge": False, "value": "nnue_only", "formal_search":"d48_pvs_mate"},
        "lane": lane, "game_side": game_side,
    }
    validate_game(game, model, training=True)
    if len(trace) != len(moves) - len(family["prefix"]) or max(used) > 10:
        raise ValueError("telemetry or longthink count mismatch")
    atomic_json(complete, game)
    partial.unlink(missing_ok=True)

