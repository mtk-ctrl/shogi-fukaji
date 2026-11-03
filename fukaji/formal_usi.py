"""Official NNUE USI front end, with atomic stop and exact game histories."""
import copy
import ctypes as ct
import sys
import threading
import shogi
from . import formal_model
from .native_nnue import NativeSelector

VERSION='FUKAJI v0.3.0'


class StopToken:
    def __init__(self,lib):
        self.lib=lib
        lib.fukaji_nnue_stop_open.restype=ct.c_void_p
        lib.fukaji_nnue_stop_signal.argtypes=[ct.c_void_p]
        lib.fukaji_nnue_stop_close.argtypes=[ct.c_void_p]
        self.handle=lib.fukaji_nnue_stop_open()
        if not self.handle:raise MemoryError('native stop token allocation failed')
    def signal(self):self.lib.fukaji_nnue_stop_signal(self.handle)
    def close(self):
        if self.handle:self.lib.fukaji_nnue_stop_close(self.handle);self.handle=None


def limits(tokens,turn,default=100,depth=64):
    values={};i=1
    while i<len(tokens):
        key=tokens[i]
        if key=='infinite':values[key]=True;i+=1;continue
        if key not in ('movetime','btime','wtime','byoyomi','binc','winc','depth'):
            raise ValueError('unsupported go limit: '+key)
        if i+1>=len(tokens):raise ValueError('missing go limit')
        values[key]=int(tokens[i+1]);i+=2
    if any(v<0 for v in values.values()):raise ValueError('negative go limit')
    d=max(1,min(64,values.get('depth',depth)))
    fixed='depth' in values and not any(k in values for k in ('movetime','btime','wtime','infinite'))
    if values.get('infinite'):ms=10000
    elif 'movetime' in values:ms=values['movetime']
    elif 'btime' in values or 'wtime' in values:
        remaining=values.get('btime' if turn==0 else 'wtime',0)
        byo=values.get('byoyomi',0);inc=values.get('binc' if turn==0 else 'winc',0)
        ms=min(remaining+byo,max(0,remaining//30)+byo+inc)
        if remaining+byo>10:ms=min(ms,remaining+byo-10)
    else:ms=default
    return max(1,min(10000,ms)),d,fixed,bool(values.get('infinite'))


class Engine:
    def __init__(self,*,specification=None,output=None):
        self.spec=specification or formal_model.manifest();self.emit=output or (lambda s:print(s,flush=True))
        self.path=None;self.model=None;self.selector=None;self.board=shogi.Board();self.base=shogi.STARTING_SFEN
        self.depth=64;self.movetime=100;self.thread=None;self.cancel=None;self.stop_event=threading.Event()
        self.suppress=False;self.failed=False

    def finish(self,*,cancel=True,suppress=False):
        if self.thread is not None:
            self.suppress=suppress
            if cancel or self.infinite:
                self.stop_event.set();self.cancel.signal()
            self.thread.join();self.thread=None
            self.cancel.close();self.cancel=None

    def reset_selector(self):
        if self.selector is not None:self.selector.close();self.selector=None

    def ready(self,announce=True):
        if self.model is None:
            self.model,_=formal_model.load(self.path,specification=self.spec)
            self.emit('info string NNUE sha256 '+self.spec['weights_sha256']+
                      ' source '+self.spec['research_snapshot_sha256'])
        if self.selector is None:
            self.selector=NativeSelector();self.selector.register(self.model)
        if announce:self.emit('readyok')

    def position(self,tokens):
        self.finish(suppress=True)
        if len(tokens)<2:raise ValueError('missing position')
        if tokens[1]=='startpos':base=shogi.STARTING_SFEN;start=2
        elif tokens[1]=='sfen' and len(tokens)>=6:base=' '.join(tokens[2:6]);start=6
        else:raise ValueError('invalid position')
        words=[]
        if len(tokens)>start:
            if tokens[start]!='moves':raise ValueError('invalid position suffix')
            words=tokens[start+1:]
        old=[m.usi() for m in self.board.move_stack]
        append=base==self.base and words[:len(old)]==old
        b=copy.deepcopy(self.board) if append else shogi.Board(base)
        if any(k is None for k in b.king_squares):raise ValueError('both kings required')
        for word in words[len(old):] if append else words:
            move=shogi.Move.from_usi(word)
            if b.is_fourfold_repetition() or move not in b.legal_moves or b.piece_type_at(move.to_square)==shogi.KING:
                raise ValueError('illegal recorded move or move after terminal')
            b.push(move)
        if not append:self.reset_selector()
        self.board=b;self.base=base

    def go(self,tokens):
        self.finish(suppress=True)
        if self.failed:raise RuntimeError('previous search failed')
        if self.model is None or self.selector is None:self.ready(announce=False)
        ms,depth,fixed,self.infinite=limits(tokens,self.board.turn,self.movetime,self.depth)
        self.stop_event.clear();self.suppress=False;self.cancel=StopToken(self.selector.lib)
        board=copy.deepcopy(self.board)
        def work():
            try:
                result=self.selector.select(board,self.model,movetime_ms=ms,max_depth=depth,
                                            fixed_depth=fixed,cancel=self.cancel)
                score=result.get('score');info=f"info depth {result['depth']} nodes {result['nodes']}"
                if score is not None:
                    if abs(score)>999000:
                        mate=max(1,int(round(1000000-abs(score))));info+=f" score mate {mate if score>0 else -mate}"
                    else:info+=f' score cp {int(round(score))}'
                info+=f" time {int(round(result['elapsed_ms']))}"
                if self.infinite:self.stop_event.wait()
                if not self.suppress:
                    self.emit(info)
                    if result.get('terminal'):self.emit('info string terminal '+result['terminal']['status'])
                    self.emit('bestmove '+(result['move'] or 'resign'))
            except Exception as e:
                self.failed=True
                if not self.suppress:self.emit('info string ERROR NNUE search: '+str(e));self.emit('bestmove resign')
        self.thread=threading.Thread(target=work,name='fukaji-nnue-search');self.thread.start()

    def command(self,line):
        t=line.strip().split()
        if not t:return True
        cmd=t[0]
        if cmd=='usi':
            for s in ['id name '+VERSION,'id author mtk-ctrl',
                'option name NNUEFile type string default '+self.spec['weights_file'],
                'option name SearchDepth type spin default 64 min 1 max 64',
                'option name DefaultMoveTime type spin default 100 min 1 max 10000',
                'option name USI_Ponder type check default false','usiok']:self.emit(s)
        elif cmd=='isready':self.ready()
        elif cmd=='setoption':
            if 'name' not in t:raise ValueError('missing option name')
            i=t.index('value') if 'value' in t else len(t);name=' '.join(t[t.index('name')+1:i]);value=' '.join(t[i+1:])
            if name=='NNUEFile':
                self.finish(suppress=True);self.reset_selector();self.path=value or None;self.model=None
            elif name=='SearchDepth':self.depth=max(1,min(64,int(value)))
            elif name=='DefaultMoveTime':self.movetime=max(1,min(10000,int(value)))
            elif name=='USI_Ponder' and value.lower()=='false':pass
            else:raise ValueError('unsupported option: '+name)
        elif cmd=='position':self.position(t)
        elif cmd=='go':
            if len(t)>1 and t[1]=='mate':self.emit('checkmate notimplemented')
            else:self.go(t)
        elif cmd=='stop':self.finish()
        elif cmd in ('usinewgame','gameover'):
            self.finish(suppress=True)
            if cmd=='usinewgame':self.reset_selector();self.board=shogi.Board();self.base=shogi.STARTING_SFEN
        elif cmd=='quit':self.finish(suppress=True);return False
        return True

    def close(self):self.finish(suppress=True);self.reset_selector()


def command_loop():
    engine=Engine()
    try:
        for line in sys.stdin:
            try:
                if not engine.command(line):break
            except ValueError as e:
                if line.strip().startswith(('isready','go')):raise
                engine.emit('info string rejected command: '+str(e))
        else:engine.finish(cancel=False)
    except Exception as e:
        engine.emit('info string ERROR approved NNUE unavailable: '+str(e));raise SystemExit(2)
    finally:engine.close()
