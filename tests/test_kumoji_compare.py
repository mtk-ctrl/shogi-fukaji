import unittest
from fukaji.kumoji_compare import expected_request,validate_request,verify,empty_stats
class FrozenKumojiTests(unittest.TestCase):
    def test_request_rejects_drift(self):
        req=expected_request()
        self.assertEqual(validate_request(req),req)
        for key,value in (("training_allowed",True),("kumoji_ponder",False),
                          ("g7_root_alpha_beta",False),("kumoji_ref","main")):
            with self.assertRaises(ValueError):validate_request(dict(req,**{key:value}))
    def test_no_false_mate_or_unresolved_win(self):
        row=dict(opening=["1i1h","2c2d","6i6h","3a3b","6h7h","5a4b"],
                 moves=[],status="checkmate",winner=0,stats=empty_stats())
        with self.assertRaises(ValueError):verify(row)
        row["status"]="move_limit_unresolved";row["winner"]=None
        with self.assertRaises(ValueError):verify(row)
if __name__=="__main__":unittest.main()
