import copy
import unittest
from fukaji.g7_depth_diagnosis import expected_request, validate_request

class G7DepthDiagnosisTests(unittest.TestCase):
    def test_fixed_request(self):
        self.assertEqual(validate_request(expected_request()), expected_request())

    def test_reject_training_or_model_tamper(self):
        for field, value in (("training_allowed", True),
                             ("model_sha256", "0"*64),
                             ("source_run", 0),
                             ("max_depths", [2, 6]),
                             ("movetime_ms", 1000)):
            request = copy.deepcopy(expected_request())
            request[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_request(request)

    def test_reject_extra_keys(self):
        request = expected_request()
        request["unapproved"] = True
        with self.assertRaises(ValueError):
            validate_request(request)

if __name__ == "__main__":
    unittest.main()
