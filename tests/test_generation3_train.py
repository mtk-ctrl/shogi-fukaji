import unittest
from fukaji.generation3_train import frozen_assignments, validate_request, group_sort


class Generation3TrainTest(unittest.TestCase):
    def valid(self):
        return {"research_only": True, "source_runs": [40404298089, 40414249980],
                "parent_model_runs": [40359123574, 40404769521],
                "training_allowed": True, "model_adoption_allowed": False,
                "preserve_existing_test": True}

    def test_frozen_request_rejects_promotion(self):
        self.assertEqual(validate_request(self.valid()), self.valid())
        cfg = self.valid()
        cfg["model_adoption_allowed"] = True
        with self.assertRaises(ValueError):
            validate_request(cfg)

    def test_existing_test_groups_are_never_reassigned(self):
        old = {f"{i:064x}" for i in range(1, 26)}
        new = {f"{i:064x}" for i in range(26, 32)}
        mapping = frozen_assignments(old, new)
        ordered_old = sorted(old, key=group_sort)
        self.assertTrue(all(mapping[g] == "train" for g in ordered_old[:21]))
        self.assertTrue(all(mapping[g] == "validation" for g in ordered_old[21:23]))
        self.assertTrue(all(mapping[g] == "test" for g in ordered_old[23:]))
        self.assertEqual({k: list(mapping.values()).count(k)
                          for k in ("train", "validation", "test")},
                         {"train": 25, "validation": 3, "test": 3})

    def test_insufficient_independent_groups_rejected(self):
        with self.assertRaisesRegex(ValueError, "group counts"):
            frozen_assignments({"a" * 64}, {"b" * 64})

    def test_existing_group_never_becomes_new_group(self):
        old = {f"{i:064x}" for i in range(1, 26)}
        new = {f"{i:064x}" for i in range(25, 31)}
        with self.assertRaises(ValueError):
            frozen_assignments(old, new)


if __name__ == "__main__":
    unittest.main()
