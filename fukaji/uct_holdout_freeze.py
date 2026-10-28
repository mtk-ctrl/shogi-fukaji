"""Read-only frozen holdout audit; no UCT timing, learning, or model adoption."""
import argparse
import hashlib
import json
from pathlib import Path
import shogi
from .search_method_probe import load_source, build_positions, _board_from_prefix
from .uct_generalization_audit import validate_parent_split

MANIFEST = Path("docs/UCT_HOLDOUT_12_PARENTS_20261029.json")
SOURCE_SHA = "60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf"
MODEL_SHA = "18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492"

def verify(games_path, report_path, manifest_path=MANIFEST):
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if (m.get("schema_version") != 1 or m.get("measurement_started") is not False
            or m.get("unresolved_used_as_teacher") is not False
            or m.get("source_sha256") != SOURCE_SHA or m.get("model_sha256") != MODEL_SHA
            or m.get("source_run") != 40637486947 or m.get("model_run") != 40636602223):
        raise ValueError("holdout frozen provenance mismatch")
    games, sha = load_source(games_path, report_path)
    if sha != SOURCE_SHA:
        raise ValueError("source bytes SHA mismatch")
    dev = build_positions(games)
    if len(dev) != 12:
        raise ValueError("development selection changed")
    for i, (record, pos) in enumerate(zip(m["development"], dev)):
        if (record["pair"] != pos["pair"] or record["ply"] != pos["target_ply"]
                or record["sfen"] != pos["sfen"] or record["seed"] != 20279028+i):
            raise ValueError("development source changed")
    for i, record in enumerate(m["holdout"]):
        pair, ply = 12+i, (12,50,110)[i//4]
        if (record["pair"] != pair or record["game_row_index"] != 2*pair
                or record["ply"] != ply or record["seed"] != 20279128+i
                or record["g8_black"] is not True):
            raise ValueError("holdout selection changed")
        game = games[2*pair]
        if game["pair"] != pair or game["g8_black"] is not True:
            raise ValueError("holdout game identity mismatch")
        sequence = game["opening"] + game["moves"]
        if len(sequence) <= ply:
            raise ValueError("holdout source too short")
        board = _board_from_prefix(sequence[:ply])
        if board.sfen() != record["sfen"] or board.is_checkmate() or board.is_fourfold_repetition():
            raise ValueError("holdout SFEN/terminal mismatch")
        if not list(board.legal_moves):
            raise ValueError("holdout has no legal move")
    result = validate_parent_split(m["development"], m["holdout"], source_sha256=SOURCE_SHA)
    result.update({"status":"verified_frozen_holdout_no_timing_or_training",
                   "legality_replayed":True,
                   "source_games_legally_replayed":len(games),
                   "holdout_prefixes_legally_replayed":12,
                   "model_bytes_checked":False,"measurement_started":False})
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = verify(args.games,args.report,args.manifest)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
if __name__ == "__main__":
    main()
