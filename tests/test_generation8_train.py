"""G8 research request and parent split isolation regression tests."""
import unittest
from fukaji.generation8_train import expected_request, validate_request, validate_splits


def fake_mate(parent, suffix):
    return {"start_sfen": "start", "moves": [suffix], "winner": 0,
            "source": "selfplay", "parent_id": parent}


class Generation8TrainTests(unittest.TestCase):
    def test_disabled_by_default_and_no_adoption(self):
        self.assertFalse(validate_request(expected_request(False)))
        self.assertTrue(validate_request(expected_request(True)))
        self.assertFalse(expected_request(True)["model_adoption_allowed"])
        self.assertFalse(expected_request(True)["unresolved_used_as_teacher"])

    def test_reject_unapproved_training_settings(self):
        for key, value in (("epochs", 9), ("fixed_update_shrinkage", 1.0),
                           ("holdout_used_for_training", True),
                           ("longthink_used_as_teacher", True),
                           ("model_adoption_allowed", True)):
            with self.subTest(key=key):
                req = expected_request(True)
                req[key] = value
                with self.assertRaises(ValueError):
                    validate_request(req)

    def test_exact_parent_splits_and_holdout_isolation(self):
        old = {f"old-{i}": ("train" if i < 447 else "validation" if i < 502 else "test")
               for i in range(557)}
        new = {f"new-{i}": ("train" if i < 320 else "validation" if i < 360 else "test")
               for i in range(400)}
        old_mates = [fake_mate(f"old-{i}", f"o{i}") for i in range(557)]
        old_mates += [fake_mate(f"old-{i}", f"o{i}-extra") for i in range(15)]
        new_mates = [fake_mate(f"new-{i}", f"n{i}") for i in
                     list(range(206)) + list(range(320, 346)) + list(range(360, 386))]
        all_games = [{"parent_id": f"new-{i}",
                      "status": ("checkmate" if i in
                                 set(list(range(206)) + list(range(320, 346)) +
                                     list(range(360, 386)))
                                 else "move_limit_unresolved")}
                     for i in range(400)]
        result = validate_splits(old, new, old_mates, new_mates, all_games)
        self.assertEqual(len(result), 957)
        self.assertEqual(result["old-447"], "validation")
        self.assertEqual(result["new-360"], "test")
        with self.assertRaises(ValueError):
            changed = dict(new, **{"new-360": "train"})
            validate_splits(old, changed, old_mates, new_mates, all_games)
        with self.assertRaises(ValueError):
            overlap = dict(new)
            overlap["old-1"] = overlap.pop("new-399")
            validate_splits(old, overlap, old_mates, new_mates, all_games)
        with self.assertRaises(ValueError):
            validate_splits(old, new, old_mates, new_mates[:-1], all_games)


if __name__ == "__main__":
    unittest.main()
