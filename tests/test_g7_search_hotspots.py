import copy
import unittest
from fukaji.g7_search_hotspots import expected_request, validate_request

class G7HotspotProtocolTests(unittest.TestCase):
    def test_frozen_request(self):
        self.assertEqual(validate_request(expected_request()), expected_request())

    def test_reject_training_and_unapproved_depth(self):
        for field, value in (("training_allowed", True),
                             ("adoption_allowed", True),
                             ("model_sha256", "0"*64),
                             ("source_run", 0),
                             ("samples", 128),
                             ("fixed_depth", 3),
                             ("instrumented", False)):
            c = copy.deepcopy(expected_request())
            c[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_request(c)

    def test_profile_is_bounded_and_read_only(self):
        c = expected_request()
        self.assertEqual(c['samples'], 16)
        self.assertEqual(c['fixed_depth'], 2)
        self.assertTrue(c['instrumented'])
        self.assertFalse(c['training_allowed'])
        self.assertFalse(c['adoption_allowed'])

    def test_reject_extra_key(self):
        c = expected_request()
        c["unapproved"] = True
        with self.assertRaises(ValueError):
            validate_request(c)

if __name__ == "__main__":
    unittest.main()
