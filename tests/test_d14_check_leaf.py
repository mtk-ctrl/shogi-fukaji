"""D14 check-evasion single-factor research regression, NOT official adoption."""
import inspect
import time
import unittest
import shogi
from fukaji.engine import negamax, select_move, evaluate, INF, fast_legal_moves
from fukaji.d14_check_leaf import score_checked_leaf
from fukaji.kumoji_loss_study import snapshot


class D14CheckLeafTests(unittest.TestCase):
    def setUp(self):
        self.checked = shogi.Board("4r3k/9/9/9/9/9/9/9/4K4 b - 1")
        self.assertTrue(self.checked.is_check())
        self.assertFalse(self.checked.is_checkmate())

    def test_leaf_check_evasion_matches_all_legal_static_reply_values(self):
        b=self.checked
        before=snapshot(b)
        choices=list(b.legal_moves)
        self.assertTrue(choices)
        expected=-INF
        for move in choices:
            b.push(move)
            try:
                candidate=-evaluate(b)
            finally:
                b.pop()
            expected=max(expected,candidate)
        self.assertEqual(snapshot(b),before)
        nodes=[0]
        got=negamax(b,0,-INF,INF,time.monotonic()+30,nodes,
                    fast_legal_guard=True,leaf_check_evasion=True)
        self.assertEqual(got,expected)
        self.assertEqual(nodes[0],1+len(choices))
        self.assertEqual(snapshot(b),before)

    def test_quiet_leaf_is_baseline_and_root_history_is_unchanged(self):
        b=shogi.Board()
        before=snapshot(b)
        old=negamax(b,0,-INF,INF,time.monotonic()+10,[0])
        guarded=negamax(b,0,-INF,INF,time.monotonic()+10,[0],leaf_check_evasion=True)
        self.assertEqual(old,guarded)
        self.assertEqual(snapshot(b),before)
        selected=select_move(b,200,1,complete_depth=True,
                             fast_legal_guard=True,leaf_check_evasion=True)
        self.assertIn(shogi.Move.from_usi(selected[0]),b.legal_moves)
        self.assertEqual(snapshot(b),before)

    def test_timeout_never_mutates_repetition_or_board(self):
        b=self.checked
        before=snapshot(b)
        with self.assertRaises(TimeoutError):
            negamax(b,0,-INF,INF,time.monotonic()-1,[0],
                    fast_legal_guard=True,leaf_check_evasion=True,
                    time_check_each_node=True)
        self.assertEqual(snapshot(b),before)

    def test_legacy_interface_default_off_usi_default_on(self):
        for fn in (negamax,select_move):
            self.assertIs(inspect.signature(fn).parameters["leaf_check_evasion"].default,False)
        from fukaji import engine
        import subprocess,sys
        result=subprocess.run([sys.executable,"-m","fukaji.engine"],
            input="usi\nquit\n",text=True,capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("FastLegalGuard type check default true",result.stdout)
        self.assertNotIn("LeafCheckEvasion type",result.stdout)

if __name__=="__main__":
    unittest.main()
