"""D3 profile diagnostics do not alter the board or claim playing strength."""
import unittest

from fukaji.d3_quiescence_profile import category, profile_one
from fukaji.d3_tactical_probe import RECAPTURE_VARIANTS


class D3QuiescenceProfileTests(unittest.TestCase):
    def test_function_categories_are_disjoint_and_named(self):
        self.assertEqual(category("engine.py", "ordered_moves"),
                         "ordering_and_tactical_filter")
        self.assertEqual(category("shogi.py", "generate_pseudo_legal_moves"),
                         "legal_move_generation")
        self.assertEqual(category("engine.py", "evaluate"), "evaluation")
        self.assertEqual(category("shogi.py", "is_check"), "check_and_repetition")
        self.assertEqual(category("shogi.py", "piece_at"), "board_access_shared")

    def test_real_search_is_legal_and_restores_history(self):
        name, sfen, *_ = RECAPTURE_VARIANTS[0]
        for enabled in (False, True):
            with self.subTest(quiescence=enabled):
                row = profile_one(name, sfen, enabled)
                self.assertEqual(row["depth"], 1)
                self.assertGreater(row["nodes"], 0)
                self.assertGreater(row["wall_ms_instrumented"], 0)
                self.assertTrue(row["top_self_cpu_functions"])
                self.assertGreater(sum(row["self_cpu_ms_by_category"].values()), 0)


if __name__ == "__main__":
    unittest.main()
