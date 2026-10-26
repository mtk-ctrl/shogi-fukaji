import unittest
from fukaji.generation4_train import expected_request, validate_request, frozen_assignments, group_sort

class Generation4TrainTest(unittest.TestCase):
    def sample(self):
        old = {f"{i:064x}" for i in range(1, 32)}
        ordered = sorted(old, key=group_sort)
        mapping = {k: "train" if i < 25 else "validation" if i < 28 else "test"
                   for i, k in enumerate(ordered)}
        new = {f"{i:064x}" for i in range(32, 104)}
        return mapping, old, new

    def test_rejects_promotion_or_test_tuning(self):
        self.assertEqual(validate_request(expected_request()), expected_request())
        for key, value in (("model_adoption_allowed", True),
                           ("old_test_for_tuning", True),
                           ("reuse_frozen_optimizer", False),
                           ("new_mates_sha256", "0"*64)):
            with self.subTest(key=key):
                cfg = expected_request()
                cfg[key] = value
                with self.assertRaises(ValueError):
                    validate_request(cfg)

    def test_all_old_heldouts_frozen_and_new_groups_disjoint(self):
        mapping, old, new = self.sample()
        result = frozen_assignments(mapping, old, new)
        self.assertEqual({k: result[k] for k in old}, mapping)
        self.assertEqual(len(result), 103)
        self.assertEqual({k: list(result.values()).count(k)
                          for k in ("train", "validation", "test")},
                         {"train": 83, "validation": 10, "test": 10})

    def test_reassigned_old_test_rejected(self):
        mapping, old, new = self.sample()
        mapping[next(k for k, v in mapping.items() if v == "test")] = "train"
        with self.assertRaisesRegex(ValueError, "prior heldout"):
            frozen_assignments(mapping, old, new)

    def test_new_parent_collision_rejected(self):
        mapping, old, new = self.sample()
        new.remove(next(iter(new)))
        new.add(next(iter(old)))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            frozen_assignments(mapping, old, new)

    def test_missing_parent_rejected(self):
        mapping, old, new = self.sample()
        new.pop()
        with self.assertRaisesRegex(ValueError, "independent"):
            frozen_assignments(mapping, old, new)

if __name__ == "__main__":
    unittest.main()
