"""Check that workflow request settings are used, not silently ignored."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

class ExperimentRequestTests(unittest.TestCase):
    def test_request_controls_are_applied_and_reported(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            config={"attempts":1,"movetime_ms":1,"depth":1,"max_plies":50,
                    "opening_plies":0,"seed":777777,"arena_pairs":7,
                    "arena_seed":999999,"branch_max_parents":2,
                    "branch_max_plies":30,"branch_variants":6}
            request=root/"request.json"
            request.write_text(json.dumps(config),encoding="utf-8")
            out=root/"results"
            proc=subprocess.run([sys.executable,"-m","fukaji.experiment",
                                 "--out",str(out),"--request",str(request)],
                                capture_output=True,text=True,timeout=20)
            self.assertEqual(proc.returncode,0,proc.stderr)
            report=json.loads((out/"experiment.json").read_text())
            for key in ("seed","arena_pairs","arena_seed","branch_max_parents","branch_max_plies"):
                self.assertEqual(report[key],config[key])
            self.assertEqual(report["requested_games"],1)
            if report["nondecisive_samples"]:
                self.assertEqual(report["nondecisive_samples"][0]["seed"],777777)

    def test_rejects_misleading_branch_variants(self):
        with tempfile.TemporaryDirectory() as d:
            request=Path(d)/"request.json"
            request.write_text(json.dumps({"branch_variants":5}))
            proc=subprocess.run([sys.executable,"-m","fukaji.experiment",
                                 "--request",str(request)],capture_output=True,
                                text=True,timeout=10)
            self.assertNotEqual(proc.returncode,0)
            self.assertIn("branch_variants",proc.stderr)

if __name__=="__main__":
    unittest.main()
