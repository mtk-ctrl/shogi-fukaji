"""Independent native rule equivalence; not an engine option or speed claim."""
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
                positions.append(shogi.Board(b.sfen()))
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


    def test_native_pseudo_legal_generation_and_self_check_filter(self):
        # P is fully native pseudo-legal enumeration. H also filters self-check,
        # but intentionally does NOT forbid pawn-drop mate; that one rule is
        # still checked by python-shogi when comparing complete legal sets.
        positions = self.positions() + [
            shogi.Board("k8/4P4/9/9/9/9/9/9/4K4 b - 1"),
            shogi.Board("4k4/9/9/9/9/9/9/4P4/4K4 b 2P2L2N2S2G2B2R 1"),
            shogi.Board("4k4/9/9/9/9/9/9/9/4K4 w 2p2l2n2s2g2b2r 1"),
        ]
        commands = []
        for b in positions:
            commands.extend(("P " + b.sfen(), "H " + b.sfen()))
        outputs = self.probe(commands)
        self.assertEqual(len(outputs), len(positions) * 2)
        for b, pseudo_line, hybrid_line in zip(
                positions, outputs[::2], outputs[1::2]):
            pseudo = pseudo_line.split(",") if pseudo_line else []
            hybrid = hybrid_line.split(",") if hybrid_line else []
            expected_pseudo = {m.usi() for m in b.generate_pseudo_legal_moves()}
            expected_legal = {m.usi() for m in b.legal_moves}
            self.assertEqual(len(pseudo), len(set(pseudo)), b.sfen())
            self.assertEqual(len(hybrid), len(set(hybrid)), b.sfen())
            self.assertEqual(set(pseudo), expected_pseudo, b.sfen())
            self.assertEqual(set(hybrid), expected_legal, b.sfen())
            self.assertTrue(set(hybrid).issubset(set(pseudo)), b.sfen())

    def test_native_pawn_drop_mate_and_nonmate(self):
        # 5b is immediately in front of the opposing king in both positions.
        # The first pawn is protected and every king escape is blocked;
        # in the second the king can capture it, so the drop is legal.
        mate = shogi.Board("3pkp3/3p1p3/4G4/9/9/9/9/9/4K4 b P 1")
        nonmate = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b P 1")
        for b, allowed in ((mate, False), (nonmate, True)):
            pawn = "P*5b"
            self.assertEqual(pawn in {m.usi() for m in b.legal_moves},
                             allowed)
            pseudo, legal = self.probe(["P " + b.sfen(), "H " + b.sfen()])
            self.assertIn(pawn, pseudo.split(","))
            self.assertEqual(pawn in legal.split(","), allowed)
            self.assertEqual(set(legal.split(",")) if legal else set(),
                             {m.usi() for m in b.legal_moves})

    def test_native_legal_set_in_later_reachable_positions(self):
        # Include captures and hand drops beyond the first 20 plies.
        rng = random.Random(20271028)
        positions = []
        for _ in range(8):
            b = shogi.Board()
            for ply in range(100):
                if ply % 5 == 0:
                    positions.append(shogi.Board(b.sfen()))
                moves = list(b.legal_moves)
                if not moves:
                    break
                b.push(rng.choice(moves))
        output = self.probe(["H " + b.sfen() for b in positions])
        for b, line in zip(positions, output):
            self.assertEqual(set(line.split(",")) if line else set(),
                             {m.usi() for m in b.legal_moves}, b.sfen())


    def test_native_stateful_move_and_undo(self):
        # Each move is checked against python-shogi's exact SFEN, including
        # capture-to-hand, promotion, drops, side to move and move number.
        cases = [
            ("4k4/9/9/9/9/9/4p4/4R4/4K4 b - 1", "5h5g"),
            ("4k4/9/4P4/9/9/9/9/9/4K4 b - 1", "5c5b+"),
            ("4k4/9/9/9/9/9/9/9/4K4 b P 1", "P*5e"),
        ]
        for sfen, usi in cases:
            b = shogi.Board(sfen)
            move = shogi.Move.from_usi(usi)
            self.assertIn(move, b.legal_moves, (sfen, usi))
            b.push(move)
            self.assertEqual(self.probe(["B "+sfen, "M "+usi, "S", "U"]),
                             [sfen, b.sfen(), b.sfen(), sfen])
        rng = random.Random(20271029)
        for _ in range(6):
            b = shogi.Board()
            commands = ["B "+b.sfen()]
            expected = [b.sfen()]
            n = 0
            for _ in range(70):
                legal = [m for m in b.legal_moves]
                if not legal:
                    break
                move = rng.choice(legal)
                b.push(move)
                commands.append("M "+move.usi())
                expected.append(b.sfen())
                n += 1
            for _ in range(n):
                b.pop()
                commands.append("U")
                expected.append(b.sfen())
            self.assertEqual(self.probe(commands), expected)
            self.assertEqual(b.sfen(), shogi.Board().sfen())

    def test_native_stateful_repetition_and_undo(self):
        b = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        commands = ["B "+b.sfen(), "F"]
        expected = [b.sfen(), "0"]
        for _ in range(3):
            for usi in ("5i5h", "5a5b", "5h5i", "5b5a"):
                move = shogi.Move.from_usi(usi)
                self.assertIn(move, b.legal_moves)
                b.push(move)
                commands.append("M "+usi)
                expected.append(b.sfen())
            commands.append("F")
            expected.append(str(int(b.is_fourfold_repetition())))
        b.pop()
        commands.extend(("U", "F"))
        expected.extend((b.sfen(), "0"))
        self.assertEqual(self.probe(commands), expected)

    def test_native_stateful_legal_sets_every_ply_and_undo(self):
        # The native board must generate legal moves from its own updated state,
        # not from a newly parsed SFEN supplied by Python.
        rng = random.Random(20271030)
        for _ in range(6):
            b = shogi.Board()
            commands = ["B " + b.sfen(), "L"]
            expected = [b.sfen(), ",".join(sorted(m.usi() for m in b.legal_moves))]
            count = 0
            for _ in range(60):
                moves = list(b.legal_moves)
                if not moves:
                    break
                move = rng.choice(moves)
                b.push(move)
                commands.extend(("M " + move.usi(), "L"))
                expected.extend((b.sfen(),
                                 ",".join(sorted(m.usi() for m in b.legal_moves))))
                count += 1
            for _ in range(count):
                b.pop()
                commands.extend(("U", "L"))
                expected.extend((b.sfen(),
                                 ",".join(sorted(m.usi() for m in b.legal_moves))))
            self.assertEqual(self.probe(commands), expected)

    def test_native_continuous_check_repetition_and_undo(self):
        # python-shogi 1.1.1 exposes fourfold repetition but not the
        # continuous-check loser. Independently assert the shogi rule while
        # checking every move and repetition against python-shogi.
        cases = (
            ("4k4/9/5R3/9/9/9/9/9/4K4 b - 1",
             ("4c5c", "5a4a", "5c4c", "4a5a"), "black_loses"),
            ("4k4/9/9/9/9/9/5r3/9/4K4 w - 1",
             ("4g5g", "5i4i", "5g4g", "4i5i"), "white_loses"),
            ("4k4/9/9/9/9/9/9/9/4K4 b - 1",
             ("5i5h", "5a5b", "5h5i", "5b5a"), "draw"),
        )
        for sfen, cycle, verdict in cases:
            with self.subTest(verdict=verdict):
                b = shogi.Board(sfen)
                commands = ["B " + sfen, "F", "T"]
                expected = [sfen, "0", "none"]
                for repeat in range(3):
                    for index, usi in enumerate(cycle):
                        move = shogi.Move.from_usi(usi)
                        self.assertIn(move, b.legal_moves, (verdict, b.sfen(), usi))
                        b.push(move)
                        self.assertEqual(b.is_check(),
                                         verdict != "draw" and index % 2 == 0)
                        commands.append("M " + usi)
                        expected.append(b.sfen())
                    commands.extend(("F", "T"))
                    expected.extend(("1" if b.is_fourfold_repetition() else "0",
                                     verdict if repeat == 2 else "none"))
                self.assertTrue(b.is_fourfold_repetition())
                b.pop()
                commands.extend(("U", "F", "T"))
                expected.extend((b.sfen(), "0", "none"))
                self.assertEqual(self.probe(commands), expected)

    def test_invalid_sfen_is_rejected(self):
        p = subprocess.run([self.binary], input="C b invalid\n",
                           text=True, capture_output=True, timeout=5)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("rejected input", p.stderr)

if __name__ == "__main__":
    unittest.main()
