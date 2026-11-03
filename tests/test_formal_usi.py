import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import numpy as np
import shogi
from fukaji.formal_model import load
from fukaji.formal_usi import Engine,StopToken,limits
from fukaji.native_nnue import NativeSelector
from fukaji.nnue_research import SmallNNUE


def fixture(folder):
    m=SmallNNUE.synthetic_test_model(41);p=Path(folder)/'model.npz'
    np.savez_compressed(p,w=m.hidden_weights,b=m.hidden_bias,v=m.output_weights,o=np.array([m.output_bias]))
    return dict(weights_file=str(p),weights_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                research_snapshot_sha256='a'*64,architecture=[2548,32,1])


class FormalModelTests(unittest.TestCase):
    def test_pinned_inference_model_and_corruption_refusal(self):
        with tempfile.TemporaryDirectory() as t:
            spec=fixture(t);m,_=load(specification=spec);self.assertEqual(m.inputs,2548)
            Path(spec['weights_file']).write_bytes(b'wrong model')
            with self.assertRaisesRegex(ValueError,'SHA256'):load(specification=spec)

    def test_cli_default_and_fail_closed_missing_model(self):
        p=subprocess.run([sys.executable,'-m','fukaji.engine'],input='usi\nquit\n',text=True,capture_output=True,timeout=10)
        self.assertEqual(p.returncode,0);self.assertIn('id name FUKAJI v0.3.0',p.stdout)
        self.assertIn('option name NNUEFile',p.stdout);self.assertNotIn('ValueModelMix',p.stdout)
        p=subprocess.run([sys.executable,'-m','fukaji.engine'],
            input='setoption name NNUEFile value /nonexistent/fukaji.npz\nisready\nquit\n',
            text=True,capture_output=True,timeout=10)
        self.assertNotEqual(p.returncode,0);self.assertNotIn('readyok',p.stdout)
        self.assertIn('ERROR approved NNUE unavailable',p.stdout)

    def test_clock_and_explicit_depth_budgets(self):
        self.assertEqual(limits(['go','movetime','100'],0),(100,64,False,False))
        self.assertEqual(limits(['go','depth','2'],0),(100,2,True,False))
        self.assertEqual(limits(['go','btime','0','wtime','1000','byoyomi','500'],0)[0],490)
        self.assertEqual(limits(['go','btime','30000','wtime','60000','binc','100'],0)[0],1100)
        self.assertTrue(limits(['go','infinite'],0)[3])
        with self.assertRaises(ValueError):limits(['go','movetime','-1'],0)


@unittest.skipUnless(shutil.which('g++'),'native compiler required')
class FormalRuntimeTests(unittest.TestCase):
    def test_interruptible_search_matches_original_and_stops_without_board_mutation(self):
        with tempfile.TemporaryDirectory() as t:
            m,_=load(specification=fixture(t));b=shogi.Board();sfen=b.sfen()
            with NativeSelector() as n:
                a=n.select(b,m,max_depth=2,fixed_depth=True);token=StopToken(n.lib)
                try:
                    c=n.select(b,m,max_depth=2,fixed_depth=True,cancel=token)
                    for key in ('move','score','nodes','evaluations','depth'):self.assertEqual(a[key],c[key])
                finally:token.close()
                token=StopToken(n.lib);result=[]
                th=threading.Thread(target=lambda:result.append(n.select(b,m,movetime_ms=10000,cancel=token)))
                th.start();time.sleep(.03);start=time.monotonic();token.signal();th.join(2)
                self.assertFalse(th.is_alive());self.assertLess(time.monotonic()-start,2)
                token.close();self.assertIn(shogi.Move.from_usi(result[0]['move']),b.legal_moves)
                self.assertEqual(b.sfen(),sfen)

    def test_usi_history_reset_transaction_and_stop_lifecycle(self):
        with tempfile.TemporaryDirectory() as t:
            out=[];e=Engine(specification=fixture(t),output=out.append)
            try:
                e.command('usi');e.command('isready');e.command('position startpos moves 7g7f 3c3d')
                before=e.board.sfen();e.command('go depth 1');e.finish(cancel=False)
                move=[s.split()[1] for s in out if s.startswith('bestmove ')][-1]
                self.assertIn(shogi.Move.from_usi(move),e.board.legal_moves);self.assertEqual(e.board.sfen(),before)
                with self.assertRaises(ValueError):e.command('position startpos moves 7g7f 7g7f')
                self.assertEqual(e.board.sfen(),before)
                e.command('go infinite');e.command('isready');e.command('stop')
                self.assertIsNone(e.thread)
                count=sum(s.startswith('bestmove ') for s in out)
                e.command('go infinite');time.sleep(.01);e.command('position startpos')
                self.assertEqual(sum(s.startswith('bestmove ') for s in out),count)
                e.command('usinewgame');e.command('go depth 1');e.finish(cancel=False)
                self.assertEqual(sum(s=='readyok' for s in out),2)
            finally:e.close()

    def test_terminal_repetition_uses_full_history(self):
        with tempfile.TemporaryDirectory() as t:
            out=[];e=Engine(specification=fixture(t),output=out.append)
            try:
                e.command('isready');cycle=['5i4i','5a4a','4i5i','4a5a']*3
                e.command('position sfen 4k4/9/9/9/9/9/9/9/4K4 b - 1 moves '+' '.join(cycle))
                e.command('go depth 1');e.finish(cancel=False)
                self.assertIn('info string terminal repetition_draw',out)
            finally:e.close()
