"""Guard D3 push triggers; a request must actually launch its gated workflow."""
from pathlib import Path
import unittest


class D3WorkflowGateTests(unittest.TestCase):
    def test_both_strict_request_files_trigger_push(self):
        text = Path(".github/workflows/d3-quiescence-ci.yml").read_text()
        self.assertIn("\n  push:\n", text)
        push = text.split("\n  push:\n", 1)[1].split("\npermissions:", 1)[0]
        for name in ("fukaji-d3-tactical-request.json",
                     "fukaji-d3-fast-tactical-request.json"):
            self.assertIn(name, push)

    def test_explicit_enabled_guard_and_single_runner(self):
        text = Path(".github/workflows/d3-quiescence-ci.yml").read_text()
        self.assertIn("if: steps.protocol.outputs.enabled == 'true'", text)
        self.assertIn("if baseline and fast:", text)
        self.assertIn("runs-on: ubuntu-latest", text)
        self.assertNotIn("max-parallel: 20", text)


if __name__ == "__main__":
    unittest.main()
