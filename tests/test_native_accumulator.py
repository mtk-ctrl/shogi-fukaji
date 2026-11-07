import random
import shutil
import unittest

import shogi

from fukaji.native_accumulator import AccumulatorSelector,build_accumulator
from fukaji.native_nnue import NativeSelector,build_native
from fukaji.online_learning import fresh_model


@unittest.skipUnless(shutil.which('g++'),'g++ required')
class AccumulatorBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base=build_native();cls.fast=build_accumulator()

    def test_fixed_depth_capture_promotion_drop_and_random_tree_match(self):
        model=fresh_model(818)
        sfens=[shogi.STARTING_SFEN,'4k4/9/4P4/9/9/9/9/9/4K4 b P 1',
            '4k4/9/3+p5/4S4/9/9/9/9/4K4 b - 1',
            '3pkp3/3p1p3/4G4/9/9/9/9/9/4K4 b P 1',
            '7lk/8p/7G1/9/9/9/9/9/4K3R b - 1']
        b=shogi.Board();rng=random.Random(118)
        for _ in range(8):
            for _ in range(6):
                moves=list(b.legal_moves)
                if not moves:break
                b.push(rng.choice(moves))
            sfens.append(b.sfen())
        for sfen in sfens:
            with self.subTest(sfen=sfen),NativeSelector(self.base) as base,AccumulatorSelector(self.fast) as fast:
                board=shogi.Board(sfen);counts=+board.transpositions
                a=base.select(board,model,max_depth=2,fixed_depth=True)
                c=fast.select(board,model,max_depth=2,fixed_depth=True)
                self.assertEqual(a['move'],c['move'])
                self.assertGreater(a['nodes'],0)
                self.assertGreater(c['nodes'],0)
                self.assertGreater(a['evaluations'],0)
                self.assertGreater(c['evaluations'],0)
                self.assertAlmostEqual(a['score'],c['score'],places=8)
                self.assertEqual(c['full_accumulator_builds'],2)
                self.assertEqual(board.sfen(),sfen);self.assertEqual(+board.transpositions,counts)
                fast.state.assert_equal(board,full=True)

    def test_timeout_undo_and_both_perspectives_remain_exact(self):
        model=fresh_model(14);board=shogi.Board('4k4/9/3+p5/4S4/9/9/9/9/4K4 b RBGSNLP 1')
        with NativeSelector(self.base) as base,AccumulatorSelector(self.fast) as fast:
            before=[base.value(board,model,p) for p in (0,1)]
            for _ in range(4):
                got=fast.select(board,model,movetime_ms=1)
                self.assertIn(shogi.Move.from_usi(got['move']),board.legal_moves)
                for p in (0,1):self.assertAlmostEqual(fast.value(board,model,p),before[p],places=11)
                fast.state.assert_equal(board,full=True)

    def test_repetition_and_continuous_check_terminal_history(self):
        model=fresh_model(17)
        for sfen,words,reason,winner in [
            ('4k4/9/9/9/9/9/9/9/4K4 b - 1',['5i4i','5a4a','4i5i','4a5a'],'repetition_draw',None),
            ('4k4/9/5R3/9/9/9/9/9/4K4 b - 1',['4c5c','5a4a','5c4c','4a5a'],'perpetual_check',1)]:
            board=shogi.Board(sfen)
            for word in words*3:board.push(shogi.Move.from_usi(word))
            with AccumulatorSelector(self.fast) as fast:
                result=fast.select(board,model,movetime_ms=1)
                self.assertIsNone(result['move'])
                self.assertEqual(result['terminal'],{'status':reason,'winner':winner})
                fast.state.assert_equal(board,full=True)
