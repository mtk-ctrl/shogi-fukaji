"""Research-only search profiler input and sample integrity tests."""
import unittest

import shogi

from fukaji.search_profile import sample_positions, validate_request


class SearchProfileTests(unittest.TestCase):
    def config(self):
        return {
            "research_only": True,
            "paired_source_run_id": 40375047387,
            "relative_source_run_id": 40359123574,
            "paired_games_sha256": "a" * 64,
            "relative_model_sha256": "b" * 64,
            "max_samples": 12,
            "movetime_ms": 200,
            "max_depth": 2,
            "model_mixes": [0, 25, 100],
        }

    def games(self):
        result = []
        for pair in range(4):
            board = shogi.Board()
            moves = []
            for ply in range(22):
                legal = list(board.legal_moves)
                if not legal:
                    break
                move = legal[(pair * 19 + ply * 7) % len(legal)]
                moves.append(move.usi())
                board.push(move)
            self.assertGreaterEqual(len(moves), 18)
            for a_black in (True, False):
                result.append({
                    "arm": "equal_200ms", "pair": pair, "a_black": a_black,
                    "opening": moves[:6], "moves": moves,
                    "reason": "move_limit", "winner": None,
                })
        return result

    def test_fixed_profile_rejects_unsafe_changes(self):
        self.assertEqual(validate_request(self.config())["max_samples"], 12)
        for name, value in (
            ("research_only", False),
            ("paired_source_run_id", 1),
            ("relative_source_run_id", 1),
            ("max_samples", 100),
            ("movetime_ms", 10000),
            ("max_depth", 6),
            ("model_mixes", [25]),
            ("paired_games_sha256", "not_a_sha"),
        ):
            with self.subTest(name=name):
                config = self.config()
                config[name] = value
                with self.assertRaises(ValueError):
                    validate_request(config)

    def test_both_colors_and_independent_positions(self):
        samples = sample_positions(self.games(), 12)
        self.assertEqual(len(samples), 12)
        self.assertEqual(len({x["sfen"] for x in samples}), 12)
        self.assertEqual({x["pair"] for x in samples}, set(range(4)))
        self.assertEqual({x["a_black"] for x in samples}, {True, False})
        for row in samples:
            board = shogi.Board(row["sfen"])
            self.assertEqual(board.turn, shogi.BLACK if row["a_black"] else shogi.WHITE)

    def test_rejects_missing_color_or_opening_mismatch(self):
        games = self.games()
        with self.assertRaises(ValueError):
            sample_positions(games[:-1], 12)
        games = self.games()
        games[1]["opening"] = ["7g7f"]
        with self.assertRaises(ValueError):
            sample_positions(games, 12)


if __name__ == "__main__":
    unittest.main()
