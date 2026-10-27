import copy
import unittest
from fukaji.g7_fast_move_list_diagnosis import expected_request, validate_request

class FastMoveListDiagnosisTests(unittest.TestCase):
    def test_frozen_request(self):
        self.assertEqual(validate_request(expected_request()), expected_request())

    def test_reject_training_and_tampering(self):
        for field, value in (("training_allowed", True), ("adoption_allowed", True),
                             ("source_run", 0), ("model_sha256", "0"*64),
                             ("movetime_ms", 1000), ("variants", ["fast_list"])):
            config = copy.deepcopy(expected_request())
            config[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_request(config)

    def test_reject_extra_keys(self):
        config = expected_request()
        config["unapproved"] = True
        with self.assertRaises(ValueError):
            validate_request(config)

if __name__ == "__main__":
    unittest.main()
