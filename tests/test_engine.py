import unittest
import shogi
from fukaji.engine import select_move, evaluate, ordered_moves, negamax, INF
import time

class EngineTests(unittest.TestCase):
    def test_search_returns_legal_move(self):
        b=shogi.Board()
        move, depth, nodes=select_move(b,30,max_depth=2)
        self.assertIn(shogi.Move.from_usi(move),b.legal_moves)
        self.assertGreaterEqual(depth,0)
    def test_side_to_move_reversal(self):
        b=shogi.Board()
        self.assertEqual(evaluate(b),0)
        b.push_usi("7g7f")
        self.assertLess(abs(evaluate(b)),100)
    def test_search_penalizes_visited_position(self):
        b=shogi.Board()
        # Real, legal four-ply cycle: both kings return to original squares.
        for move in ("5i5h","5a5b","5h5i","5b5a"):
            b.push_usi(move)
        self.assertEqual(b.transpositions[b.zobrist_hash()],2)
        score=negamax(b,0,-INF,INF,time.monotonic()+2,[0])
        self.assertEqual(score,-1000)

    def test_root_move_remains_legal_after_cycle(self):
        b=shogi.Board()
        for move in ("5i5h","5a5b","5h5i","5b5a"):
            b.push_usi(move)
        move,depth,nodes=select_move(b,20,max_depth=1)
        self.assertIn(shogi.Move.from_usi(move),b.legal_moves)

    def test_complete_depth_ignores_short_wall_clock_and_finishes(self):
        b=shogi.Board()
        one=select_move(b,1,max_depth=1,complete_depth=True)
        twenty=select_move(b,20,max_depth=1,complete_depth=True)
        self.assertEqual(one[0],twenty[0])
        self.assertEqual(one[1],1)
        self.assertEqual(twenty[1],1)
        self.assertEqual(one[2],twenty[2])
        self.assertIn(shogi.Move.from_usi(one[0]),b.legal_moves)

    def test_usi_protocol(self):
        import subprocess,sys
        process=subprocess.run([sys.executable,"-m","fukaji.engine"],
            input="usi\nisready\nposition startpos\ngo movetime 30\nquit\n",
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=15)
        self.assertEqual(process.returncode,0,process.stderr)
        self.assertIn("usiok",process.stdout)
        self.assertIn("readyok",process.stdout)
        self.assertIn("bestmove ",process.stdout)

if __name__=="__main__": unittest.main()
