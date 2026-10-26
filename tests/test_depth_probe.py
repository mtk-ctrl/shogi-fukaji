import unittest
from fukaji.depth_probe import validate_request, sample_positions, MODEL_SHA, MODEL_RUN, SOURCE_RUN

class DepthProbeTests(unittest.TestCase):
    def cfg(self):
        return {"research_only": True, "source_run_id": SOURCE_RUN,
                "model_run_id": MODEL_RUN, "model_sha256": MODEL_SHA,
                "source_seed": 20271227, "source_pairs": 16,
                "max_samples": 12, "time_ms": [200,1000],
                "max_depths": [2,3], "mix_percent": 25,
                "training_allowed": False,
                "games_sha256": "a"*64, "report_sha256": "b"*64}
    def test_fail_closed(self):
        self.assertEqual(validate_request(self.cfg())["source_pairs"], 16)
        for key, value in (("source_run_id",0),("model_sha256","bad"),
                           ("max_samples",24),("time_ms",[100,1000]),
                           ("training_allowed",True),("report_sha256","bad")):
            c=self.cfg()
            c[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_request(c)

if __name__ == "__main__":
    unittest.main()
