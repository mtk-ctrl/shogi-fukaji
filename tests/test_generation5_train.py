import unittest
from fukaji.generation5_train import expected_request, validate_request, frozen_assignments, group_sort

class Generation5TrainTest(unittest.TestCase):
    def sample(self):
        old = {f"{i:064x}" for i in range(1, 104)}
        ordered = sorted(old, key=group_sort)
        mapping = {k: "train" if i < 83 else "validation" if i < 93 else "test"
                   for i, k in enumerate(ordered)}
        new = {f"{i:064x}" for i in range(104, 208)}
        return mapping, old, new

    def test_request_rejects_promotion_or_test_tuning(self):
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

    def test_parent_disjoint_and_old_splits_unchanged(self):
        mapping, old, new = self.sample()
        result = frozen_assignments(mapping, old, new)
        self.assertEqual({k: result[k] for k in old}, mapping)
        self.assertEqual(len(result), 207)
        self.assertEqual({k: list(result.values()).count(k)
                          for k in ("train", "validation", "test")},
                         {"train": 167, "validation": 20, "test": 20})

    def test_rejects_reassignment_and_overlap(self):
        mapping, old, new = self.sample()
        bad = dict(mapping)
        bad[next(k for k, v in mapping.items() if v == "test")] = "train"
        with self.assertRaisesRegex(ValueError, "prior heldout"):
            frozen_assignments(bad, old, new)
        overlap = set(new)
        overlap.pop()
        overlap.add(next(iter(old)))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            frozen_assignments(mapping, old, overlap)

    def test_rejects_missing_parent(self):
        mapping, old, new = self.sample()
        new.pop()
        with self.assertRaisesRegex(ValueError, "independent"):
            frozen_assignments(mapping, old, new)

if __name__ == "__main__":
    unittest.main()
