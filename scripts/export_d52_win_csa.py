"""Export one *authentic*, verified D52 winning game from audited 200-game Release as ShogiDroid-compatible CSA."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import shogi

TRAINED="55ccc97db6c2d4ce9ff7c75f5e81d543fb6f12d169639c3ea6d24c89e95342ae"
BASE="641f36677b8d93c3237cfc4f0b6329a6beee55cff692302f529b3899ea74afd0"
ARCHIVE_SHA="5f05369e2ad7edec6aa4098a9393889c989697e83a607cdc062967049630a542"
CODES={1:"FU",2:"KY",3:"KE",4:"GI",5:"KI",6:"KA",7:"HI",8:"OU",
       9:"TO",10:"NY",11:"NK",12:"NG",13:"UM",14:"RY"}

def convert(archive: Path, output: Path, receipt: Path):
    raw=archive.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==ARCHIVE_SHA,"Reference Release checksum mismatch"
    with tarfile.open(archive,"r:gz") as t:
        member={m.name:m for m in t.getmembers() if m.isfile()}
        assert "report.json" in member and "plan.json" in member
        report=json.load(t.extractfile(member["report.json"]))
        assert report["games"]==200 and report["after_wins"]==156 and report["before_wins"]==44
        assert report["after"]==TRAINED and report["before"]==BASE
        assert report["unresolved"]==0 and report["draws"]==0
        hashes=report["game_sha256"]
        assert len(hashes)==200
        found=[]
        for name,expected in hashes.items():
            path="games/"+name
            assert path in member,"Audit game missing "+path
            game_bytes=t.extractfile(member[path]).read()
            assert hashlib.sha256(game_bytes).hexdigest()==expected,"Audit game SHA mismatch "+name
            g=json.loads(game_bytes)
            assert len(g["moves"])>=6 and g["moves"]==list(g["moves"])
            side=int(g["candidate_side"])
            assert side in (0,1) and g["candidate_black"] in (0,1)
            assert side== (0 if g["candidate_black"] else 1)
            assert g["models"]==([TRAINED,BASE] if side==0 else [BASE,TRAINED])
            if g["winner"]==side and g["status"]=="checkmate":
                found.append((name,g,expected))
    assert found,"No D52 winning checkmates in the verified 200 games"
    # Prefer a reasonably short complete game for comfortable viewing.
    found.sort(key=lambda x:(abs(len(x[1]["moves"])-90),x[1]["candidate_side"]!=0,len(x[1]["moves"]),x[0]))
    name,g,source_sha=found[0]
    side=int(g["candidate_side"])
    assert g["start_sfen"]==shogi.STARTING_SFEN
    names=["FUKAJI D52 Adam5606","FUKAJI D50 Adam4702"] if side==0 else ["FUKAJI D50 Adam4702","FUKAJI D52 Adam5606"]
    text=["V2.2","N+"+names[0],"N-"+names[1],
          "$EVENT:FUKAJI D52 vs D50 paired200 audit",
          "$SITE:GitHub Actions 42401026786",
          "' Original complete audited game: "+name,
          "' D52 learned model SHA256: "+TRAINED,
          "' D50 baseline SHA256: "+BASE,
          "' Original game JSON SHA256: "+source_sha,
          "PI","+"]
    board=shogi.Board()
    for i,word in enumerate(g["moves"],1):
        turn=int(board.turn)
        move=shogi.Move.from_usi(word)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal recorded move {i}: {word}")
        target=word.split("*",1)[-1][:2] if "*" in word else word[2:4]
        origin="00" if "*" in word else word[:2]
        def csa_square(usi_square):
            assert len(usi_square)==2 and usi_square[0] in "123456789" and usi_square[1] in "abcdefghi"
            return usi_square[0]+str(ord(usi_square[1])-96)
        board.push(move)
        piece=board.piece_at(move.to_square)
        if piece is None or piece.piece_type not in CODES:
            raise ValueError("Invalid after-move piece: "+word)
        text.append(("+" if turn==0 else "-")+csa_square(origin) if origin!="00" else ("+" if turn==0 else "-")+"00")
        text[-1]+=csa_square(target)+CODES[piece.piece_type]
    assert board.is_checkmate(),"Audit declared checkmate but reproduced final board is not checkmate"
    assert int(g["winner"])==1-int(board.turn),"Winner does not match mate on board"
    text.append("%TSUMI")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(("\n".join(text)+"\n").encode("ascii"))
    receipt.parent.mkdir(parents=True,exist_ok=True)
    metadata={
        "schema":"fukaji-d52-shogidroid-sample-v1",
        "original_run":42401026786,"source_release_tag":"fukaji-d52-d50-before-after-200games-42401026786",
        "source_archive_sha256":ARCHIVE_SHA,
        "source_game":name,"source_game_json_sha256":source_sha,
        "d52_snapshot_sha256":TRAINED,"d50_snapshot_sha256":BASE,
        "winner":"FUKAJI D52",
        "d52_side":"sente" if side==0 else "gote",
        "plies":len(g["moves"]),"last_status":g["status"],
        "csa_sha256":hashlib.sha256(output.read_bytes()).hexdigest()
    }
    receipt.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(metadata,ensure_ascii=False))
    return metadata

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    x=p.parse_args()
    convert(x.archive,x.output,x.receipt)
