"""Frozen read-only legal guard protocol tests."""
import copy
import unittest
from fukaji.g7_fast_legal_guard_diagnosis import expected_request, validate_request

class GuardDiagnosisTests(unittest.TestCase):
    def test_exact_request(self):
        self.assertEqual(validate_request(expected_request()), expected_request())
        self.assertEqual(expected_request()["samples"], 32)
        self.assertFalse(expected_request()["adoption_allowed"])

    def test_reject_tampering(self):
        for key, value in (("training_allowed", True), ("adoption_allowed", True),
                           ("model_sha256", "0"*64), ("samples", 8),
                           ("fast_move_list", False), ("variants", ["guard"])):
            request = copy.deepcopy(expected_request())
            request[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_request(request)
        request = expected_request()
        request["unexpected"] = True
        with self.assertRaises(ValueError):
            validate_request(request)

if __name__ == "__main__":
    unittest.main()
