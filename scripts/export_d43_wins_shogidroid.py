"""Export only the D43 (post-1000 training) wins from the frozen 200-game audit.

Public GitHub Release is the source. Produce one KIF and one CSA per winning
game for ShogiDroid, independent of the ongoing D45 training workflow.
The original game data, winner side, legality and KIF/CSA round trips are checked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path
import zipfile

import shogi
import shogi.CSA
import shogi.KIF

SOURCE_SHA = "c08ccab3f5e9ef90c288e8ccfc5f3951eb188dfc30876af775c2ba3cfb03f510"
BEFORE = "170d6c1ee41d77b69f0b75722ece3deb85f3eea189f70d9789db477a38266db1"
AFTER = "bf8c26f9f8f6adff8a87c46ab04fcf5cd76a7915be241da11b99bb4ae2ac8b9f"
LABEL = {BEFORE: "深路D37（学習前）", AFTER: "深路D43（学習後）"}
EXPECTED_WINS = 150

def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def verified_members(source: Path):
    if digest(source.read_bytes()) != SOURCE_SHA:
        raise ValueError("Wrong D43 original game Release SHA256")
    with tarfile.open(source, "r:gz") as archive:
        report = json.load(archive.extractfile("report.json"))
        plan = json.load(archive.extractfile("plan.json"))
        if (report.get("schema") != "d43-paired200-result-v1" or
            report.get("before") != BEFORE or report.get("after") != AFTER or
            report.get("games") != 200 or report.get("opening_pairs") != 100 or
            report.get("after_wins") != 150 or report.get("before_wins") != 50 or
            report.get("draws") != 0 or report.get("unresolved") != 0 or
            plan.get("schema") != "d43-paired200-v1"):
            raise ValueError("Frozen D43 pairing report does not match adopted result")
        hashes = report.get("game_sha256", {})
        if len(hashes) != 200:
            raise ValueError("Not exactly 200 individual source game hashes")
        all_games = []
        for filename, expected in sorted(hashes.items()):
            if "/" in filename or not filename.endswith(".json"):
                raise ValueError("Unsafe source filename")
            raw = archive.extractfile("games/" + filename).read()
            if digest(raw) != expected:
                raise ValueError("Individual original game SHA mismatch: " + filename)
            game = json.loads(raw)
            if (game.get("role") != "audit" or
                game.get("source") != "selfplay" or
                game.get("models") not in ([BEFORE, AFTER], [AFTER, BEFORE]) or
                game.get("candidate_side") not in (0, 1) or
                game.get("candidate_black") not in (0, 1) or
                game.get("winner") not in (0, 1) or
                game.get("family_index") not in range(100) or
                game.get("moves", [])[:6] !=
                    plan["families"][game["family_index"]]["prefix"] or
                game.get("parent_id") !=
                    plan["families"][game["family_index"]]["parent_id"]):
                raise ValueError("Wrong D43 original winner/model/role or parent provenance")
            winner = game["winner"]
            if game["models"][winner] !=
                    (AFTER if winner == game["candidate_side"] else BEFORE)):
                raise ValueError("Winner and trained candidate side do not match")
            if game["status"] not in ("checkmate", "perpetual_check"):
                raise ValueError("Nonstandard decisive game, review before export")
            all_games.append((filename, game))
        if len(all_games) != 200:
            raise ValueError("Missing a frozen game")
        counts = {AFTER: 0, BEFORE: 0}
        for _, game in all_games:
            counts[game["models"][game["winner"]]] += 1
        if counts != {AFTER: 150, BEFORE: 50}:
            raise ValueError("Original individual winners do not aggregate to 150-50")
        return all_games, report

def render_game(game):
    names = [LABEL[x] for x in game["models"]]
    winner = game["winner"]
    moves = game["moves"]
    if game.get("start_sfen") != shogi.STARTING_SFEN:
        raise ValueError("Unexpected nonstandard initial SFEN")
    board = shogi.Board()
    lines = [
        "開始日時：",
        "棋戦：深路D43 学習前後200局比較",
        "手合割：平手",
        "先手：" + names[0],
        "後手：" + names[1],
        "手数----指手---------消費時間--",
    ]
    csa = ["V2.2", "N+" + names[0], "N-" + names[1], "PI", "+"]
    for i, word in enumerate(moves, 1):
        move = shogi.Move.from_usi(word)
        if not board.is_legal(move):
            raise ValueError(f"Illegal source move {i}: {word}")
        kifmove = shogi.KIF.Exporter.kif_move_from(word, board)
        side_char = "+" if board.turn == shogi.BLACK else "-"
        origin = "00" if move.from_square is None else shogi.CSA.SQUARE_NAMES[move.from_square]
        destination = shogi.CSA.SQUARE_NAMES[move.to_square]
        board.push(move)
        piece = board.piece_at(move.to_square)
        csa_code = shogi.CSA.PIECE_SYMBOLS[piece.piece_type]
        csa.append(side_char + origin + destination + csa_code)
        lines.append(f"{i:4d} {kifmove}")
    if game["status"] == "checkmate":
        if not board.is_checkmate() or int(board.turn) == winner:
            raise ValueError("Original checkmate terminal and winner disagree")
        # "詰み" is a KIF terminal event, not a fictitious resignation.
        lines.append(f"{len(moves)+1:4d} 詰み")
        csa.append("%TSUMI")
    else:
        # The result line supplies the real winner; do not falsify a resignation.
        lines.append("* 終局：連続王手による千日手の判定")
        csa.append("%SENNICHITE")
    lines.append(f"まで{len(moves)}手で{'先手' if winner == shogi.BLACK else '後手'}の勝ち")
    kif_text = "\n".join(lines) + "\n"
    csa_text = "\n".join(csa) + "\n"
    # ShogiDroid accepts KIF and CSA; CP932 provides legacy KIF compatibility.
    kif_encoded = kif_text.encode("cp932")
    csa_encoded = csa_text.encode("cp932")
    parsed_kif = shogi.KIF.Parser.parse_str(kif_text)[0]
    if (parsed_kif["moves"] != moves or
        parsed_kif["names"] != names or
        parsed_kif["win"] != ("b" if winner == shogi.BLACK else "w")):
        raise ValueError("KIF roundtrip (moves, names, winner) failed")
    parsed_csa = shogi.CSA.Parser.parse_str(csa_text)[0]
    if parsed_csa["moves"] != moves:
        raise ValueError("CSA roundtrip failed")
    return kif_encoded, csa_encoded, kif_text

def export(source: Path, out: Path):
    games, report = verified_members(source)
    out.mkdir(parents=True, exist_ok=True)
    chosen = [(name, game) for name, game in games
              if game["models"][game["winner"]] == AFTER]
    if len(chosen) != EXPECTED_WINS:
        raise ValueError("Not exactly 150 games won by the trained D43")
    entries = []
    all_files = {}
    for rank, (original, game) in enumerate(chosen, 1):
        fam = game["family_index"]
        color = "sente" if game["candidate_side"] == 0 else "gote"
        stem = f"D43_after_win_{rank:03d}_pair{fam:03d}_{color}"
        kif, csa, utf8 = render_game(game)
        all_files[stem + ".kif"] = kif
        all_files[stem + ".csa"] = csa
        if rank == 1:
            (out / "D43_after_win_example.kif").write_bytes(kif)
            (out / "D43_after_win_example_utf8.kif").write_text(utf8, encoding="utf-8")
        entries.append({
            "number": rank, "filename_kif": stem + ".kif",
            "filename_csa": stem + ".csa",
            "source_json": original, "opening_pair": fam,
            "winner_model": "D43 (post-training)",
            "winner_side": "先手" if game["winner"] == 0 else "後手",
            "moves": len(game["moves"]), "status": game["status"],
            "sha256_kif": digest(kif), "sha256_csa": digest(csa),
        })
    index = io.StringIO()
    writer = csv.DictWriter(index, fieldnames=list(entries[0]))
    writer.writeheader()
    writer.writerows(entries)
    info = (
        "深路D43（学習後）勝局150局 / ShogiDroid用棋譜\n"
        "元データ：D43 vs D37 200局比較、GitHub Actions 42292240321\n"
        "学習後D43：150勝 / 学習前D37：50勝 / 引分0 / 未決着0\n"
        "各棋譜：KIF(Shift_JIS/CP932) と CSA(Shift_JIS/CP932) の両方\n"
        "ファイル名のpair番号は元の100独立序盤の番号です。\n"
        "対局者（先後）は原棋譜に厳密に一致。先後交替局は別ファイルです。\n"
        "棋譜の最終手までを収録。詰みの場合は詰みと記録し、投了を捏造しません。\n"
        "端末でZIPを展開し、ShogiDroidの「ファイルから棋譜を取り込む」等で.kifを開いてください。\n"
        "元の200局Release SHA256: " + SOURCE_SHA + "\n"
        "学習後NNUE SHA256: " + AFTER + "\n"
    )
    all_files["README.txt"] = info.encode("cp932")
    all_files["INDEX.csv"] = index.getvalue().encode("utf-8-sig")
    archive_path = out / "D43_after_wins_150_ShogiDroid.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, raw in sorted(all_files.items()):
            z.writestr(name, raw)
    with zipfile.ZipFile(archive_path) as z:
        if len(z.namelist()) != 302:
            raise ValueError("ZIP is not 150 KIF + 150 CSA + 2 guides")
        for name, raw in all_files.items():
            if z.read(name) != raw:
                raise ValueError("ZIP file extraction readback mismatch")
    manifest = {
        "schema": "fukaji-d43-shogidroid-winning-games-v1",
        "source_release_sha256": SOURCE_SHA, "source_games": 200,
        "before_wins": 50, "trained_after_wins": 150,
        "exported_games": len(entries), "kif_files": 150, "csa_files": 150,
        "format": "KIF+CSA", "encoding": "cp932",
        "individual_games_legally_replayed": len(entries),
        "kif_parser_roundtrips": len(entries),
        "csa_parser_roundtrips": len(entries),
        "archive": archive_path.name, "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": digest(archive_path.read_bytes()),
        "sample_sha256": digest((out / "D43_after_win_example.kif").read_bytes()),
        "entries": entries
    }
    (out / "D43_after_wins_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_verified": True, "won_by_after": len(chosen),
        "kif_roundtrip": 150, "csa_roundtrip": 150,
        "sha256_zip": manifest["archive_sha256"],
        "size_zip": manifest["archive_size_bytes"]
    }, ensure_ascii=False))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    export(a.source, a.out)
if __name__ == "__main__":
    main()
