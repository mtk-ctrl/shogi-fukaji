"""Regression: G8 source validation must replay all 400, not only 100 candidates."""
import unittest
from unittest.mock import patch

from fukaji.generation8_selection import verify_all_games
from fukaji.generation7_selfplay import SEED


class FullG7ReplayTests(unittest.TestCase):
    def fixture(self):
        return [{
            "game_id": i, "parent_id": "id-" + str(SEED + i),
            "opening": ["7g7f"] * 6, "moves": ["2g2f"] * 234,
            "status": "move_limit_unresolved", "winner": None,
        } for i in range(400)]

    def test_every_game_is_replayed(self):
        games = self.fixture()
        with (patch("fukaji.arena.opening", return_value=["7g7f"] * 6),
              patch("fukaji.generation7_selfplay.parent_id",
                    side_effect=lambda seed, opening: "id-" + str(seed)),
              patch("fukaji.generation8_selection.replay_check") as replay):
            self.assertEqual(verify_all_games(games), 400 * 240)
            self.assertEqual(replay.call_count, 400)

    def test_missing_game_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "source game IDs"):
            verify_all_games(self.fixture()[:-1])

    def test_early_unresolved_fails_closed(self):
        games = self.fixture()
        games[0]["moves"].pop()
        with (patch("fukaji.arena.opening", return_value=["7g7f"] * 6),
              patch("fukaji.generation7_selfplay.parent_id",
                    side_effect=lambda seed, opening: "id-" + str(seed)),
              patch("fukaji.generation8_selection.replay_check")):
            with self.assertRaisesRegex(ValueError, "ply count"):
                verify_all_games(games)


if __name__ == "__main__":
    unittest.main()
