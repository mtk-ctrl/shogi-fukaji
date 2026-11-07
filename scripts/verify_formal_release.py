"""Deployment checks with the real approved model; not a strength experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import queue
import random
import subprocess
import sys
import threading
import time
import copy
import shogi

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fukaji.formal_model import load
from fukaji.formal_usi import StopToken
from fukaji.native_nnue import NativeSelector


class Client:
    def __init__(self):
        self.p=subprocess.Popen([sys.executable,'-m','fukaji.engine'],cwd=ROOT,stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
        self.q=queue.Queue();self.lines=[]
        def reader():
            for line in self.p.stdout:self.q.put(line.strip())
        self.t=threading.Thread(target=reader,daemon=True);self.t.start()
    def send(self,line):self.p.stdin.write(line+'\n');self.p.stdin.flush()
    def until(self,prefix,timeout=15):
        deadline=time.monotonic()+timeout
        while True:
            line=self.q.get(timeout=max(.001,deadline-time.monotonic()));self.lines.append(line)
            if 'ERROR' in line:raise AssertionError(line)
            if line.startswith(prefix):return line
    def close(self):
        if self.p.poll() is None:self.send('quit')
        self.p.wait(5);self.t.join(2)
        assert self.p.returncode==0,self.p.stderr.read()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();m,spec=load();rng=random.Random(2026110232);b=shogi.Board();boards=[copy.deepcopy(b)]
    for ply in range(90):
        legal=list(b.legal_moves)
        if not legal or b.is_fourfold_repetition():break
        b.push(rng.choice(legal))
        if ply%3==0:boards.append(copy.deepcopy(b))
    boards=boards[:30]
    boards.extend([shogi.Board('4k4/9/9/9/9/9/9/9/4K4 b R2P2p 1'),
                   shogi.Board('4k4/9/4R4/9/9/9/9/9/4K4 w - 1')])
    for board in boards:
        with NativeSelector() as n:
            a=n.select(board,m,max_depth=2,fixed_depth=True);token=StopToken(n.lib)
            try:c=n.select(board,m,max_depth=2,fixed_depth=True,cancel=token)
            finally:token.close()
            for key in ('move','score','nodes','evaluations','depth'):assert a[key]==c[key],(key,a,c)
    # D49 formal contract: the deployed C++ must actually execute the D48
    # proof path, not merely advertise a new version string.
    mate_probe_verified=False
    if spec.get('search_backend_variant')=='d48_pvs_mate':
        mate_board=shogi.Board('8k/9/7R1/7L1/9/9/9/9/8K b - 1')
        with NativeSelector() as native:
            proof=native.select(mate_board,m,movetime_ms=1000)
        assert proof.get('research_mate_proven') is True,proof
        assert proof.get('research_mate_plies')==1,proof
        assert proof['move'] in {'2c1c','2c1c+','2c2a+','2c2b+'},proof
        mate_probe_verified=True
    c=Client();plies=0;elapsed=[]
    try:
        c.send('usi');c.until('usiok');c.send('isready');c.until('readyok',90)
        assert any(spec['weights_sha256'] in s for s in c.lines)
        for game in range(2):
            board=shogi.Board();moves=[];c.send('usinewgame')
            for ply in range(30):
                c.send('position startpos'+(' moves '+' '.join(moves) if moves else ''))
                start=time.monotonic();c.send('go movetime 20');line=c.until('bestmove ')
                elapsed.append((time.monotonic()-start)*1000);word=line.split()[1]
                assert word!='resign';move=shogi.Move.from_usi(word);assert move in board.legal_moves
                board.push(move);moves.append(word);plies+=1
                if board.is_fourfold_repetition() or board.is_game_over():break
            c.send('gameover draw')
        if spec.get('search_backend_variant')=='d48_pvs_mate':
            c.send('usinewgame')
            c.send('position sfen 8k/9/7R1/7L1/9/9/9/9/8K b - 1')
            c.send('go movetime 1000')
            mate_line=c.until('bestmove ',20)
            assert mate_line.split()[1] in {'2c1c','2c1c+','2c2a+','2c2b+'},mate_line
        c.send('usinewgame');c.send('position startpos');c.send('go infinite');time.sleep(.03)
        start=time.monotonic();c.send('isready');c.until('readyok');c.send('stop');line=c.until('bestmove ')
        stop_ms=(time.monotonic()-start)*1000
        assert stop_ms<2000;assert shogi.Move.from_usi(line.split()[1]) in shogi.Board().legal_moves
        c.send('go infinite');time.sleep(.02);c.send('position startpos moves 7g7f');c.send('go depth 1')
        line=c.until('bestmove ');board=shogi.Board();board.push_usi('7g7f')
        assert shogi.Move.from_usi(line.split()[1]) in board.legal_moves
        c.send('isready');c.until('readyok')
        assert c.q.empty(),'unexpected duplicate bestmove'
    finally:c.close()
    bundle=ROOT/'BUNDLE.json'
    if bundle.exists():
        for name,expected in json.loads(bundle.read_text())['files'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    result=dict(version=spec['version'],weights_sha256=spec['weights_sha256'],
                identical_depth2_positions=len(boards),usi_games=2,legal_usi_plies=plies,
                movetime_ms=20,observed_max_ms=max(elapsed),stop_ms=stop_ms,
                bundle_member_hashes_verified=bundle.exists(),
                research_mate_proof_verified=mate_probe_verified,
                search_backend_variant=spec.get('search_backend_variant','basic_alpha_beta'),
                purpose='deployment correctness, not strength')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
