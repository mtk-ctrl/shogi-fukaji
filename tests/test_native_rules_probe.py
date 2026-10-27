"""Stage-1 independent native rule probe; not an engine option or speed claim."""
from __future__ import annotations
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
import shogi

ROOT = Path(__file__).resolve().parents[1]

class NativeRulesProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("g++"):
            raise unittest.SkipTest("g++ is required for native equivalence")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.binary = str(Path(cls.tmp.name) / "rules_probe")
        p = subprocess.run(
            ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
             str(ROOT / "native" / "rules_probe.cpp"), "-o", cls.binary],
            text=True, capture_output=True, timeout=60)
        if p.returncode:
            raise AssertionError(p.stdout + p.stderr)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def probe(self, commands):
        p = subprocess.run([self.binary], input="\n".join(commands) + "\n",
                           text=True, capture_output=True, timeout=30)
        self.assertEqual(p.returncode, 0, p.stderr)
        output = p.stdout.splitlines()
        self.assertEqual(len(output), len(commands))
        return output

    @staticmethod
    def positions():
        positions = [
            shogi.Board(),
            shogi.Board("4r3k/9/9/9/9/9/9/4G4/4K4 b - 1"),
            shogi.Board("4r3k/9/9/9/9/9/9/9/4K4 b - 1"),
            shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b PGLNS 1"),
            shogi.Board("4k4/9/4R4/9/9/9/9/9/4K4 w - 1"),
            shogi.Board("3pkp3/3p1p3/4G4/9/9/9/9/9/4K4 b P 1"),
        ]
        rng = random.Random(20271027)
        for _ in range(12):
            b = shogi.Board()
            for _ in range(20):
                positions.append(b.copy())
                moves = [m for m in b.legal_moves]
                if not moves:
                    break
                b.push(rng.choice(moves))
        return positions

    def test_check_for_both_kings(self):
        positions = self.positions()
        commands, expected = [], []
        for b in positions:
            for color, code in ((shogi.BLACK, "b"), (shogi.WHITE, "w")):
                king = b.king_squares[color]
                if king is None:
                    continue
                commands.append("C " + code + " " + b.sfen())
                expected.append(str(int(b.is_attacked_by(not color, king))))
        self.assertEqual(self.probe(commands), expected)

    def test_legal_set_with_native_self_check_and_explicit_pawn_mate_fallback(self):
        # Python generates pseudo-legal candidates. Native independently checks
        # whether moving exposes the mover's king; pawn-drop mate is deliberately
        # delegated to python-shogi, not claimed as a native implementation.
        commands, expected, details = [], [], []
        positions = self.positions()
        for b in positions:
            legal = {m.usi() for m in b.legal_moves}
            before = b.sfen()
            for move in b.generate_pseudo_legal_moves():
                side = b.turn
                usi = move.usi()
                b.push(move)
                commands.append("C " + ("b" if side == shogi.BLACK else "w")
                                + " " + b.sfen())
                b.pop()
                is_pawn_drop = move.drop_piece_type == shogi.PAWN
                expected.append(usi in legal)
                details.append((before, usi, is_pawn_drop))
            self.assertEqual(before, b.sfen())
        output = self.probe(commands)
        for checked, should_be_legal, (sfen, usi, pawn_drop) in zip(
                output, expected, details):
            # For pawn-drop mate, retain the authoritative python-shogi guard.
            # This is a correctness prototype, not a full independent generator.
            native_hybrid_legal = checked == "0" and (
                not pawn_drop or should_be_legal)
            self.assertEqual(native_hybrid_legal, should_be_legal,
                             (sfen, usi, checked))

    def test_forbidden_pawn_drop_mate(self):
        b = shogi.Board("3pkp3/3p1p3/4G4/9/9/9/9/9/4K4 b P 1")
        pawn = shogi.Move.from_usi("P*5b")
        self.assertNotIn(pawn, b.legal_moves)
        # This remains a fallback case, NOT a native pawn-drop mate claim.
        self.assertEqual(self.probe(["C b " + b.sfen()]), ["0"])

    def test_fourfold_state_against_python_shogi(self):
        b = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        cycle = ("5i5h", "5a5b", "5h5i", "5b5a")
        commands, expected = ["Z", "R " + b.sfen()], ["OK", "0"]
        for _ in range(3):
            for usi in cycle:
                move = shogi.Move.from_usi(usi)
                self.assertIn(move, b.legal_moves)
                b.push(move)
                commands.append("R " + b.sfen())
                expected.append(str(int(b.is_fourfold_repetition())))
        self.assertEqual(expected[-1], "1")
        self.assertEqual(self.probe(commands), expected)

    def test_invalid_sfen_is_rejected(self):
        p = subprocess.run([self.binary], input="C b invalid\n",
                           text=True, capture_output=True, timeout=5)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("rejected input", p.stderr)

if __name__ == "__main__":
    unittest.main()
