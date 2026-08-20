#!/usr/bin/env python3
"""Unit tests for tools/master2rasi.py.

Run with:  python3 tools/test_master2rasi.py [-v]

Every test here exists because the corresponding input used to be accepted
silently.  The last test is the one that matters most: a valid deck must
translate to exactly the same bytes it did before any of this hardening.
"""

import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import master2rasi as m2r


# --------------------------------------------------------------------------
# A minimal but complete deck.  Tests mutate one card at a time.
# --------------------------------------------------------------------------

MINIMAL = """\
%GEN_DIM
        3       3       2       1       1
        3       1       0       2       2
        2
%GEN_GEO
        21.5 100.0
        50 50
%GEN_SYM
        0       0       0
        0       0       0
%GEN_THD
        520.0
        295   24.5   306.85  155
%GEN_FDB
off off off
%GEN_PIN
        1       2       -17     -248
%LPD_BCH
       A1  A1  A1
       A1  A1  A1
       A1  A1  A1
%LPD_B&C
        A1     2*1
%LPD_C&X
        1      FA_A1           0     248
%EXE_STD
        keff tr tr 1.0
%EDT_OPT
        1       2      2       0
/
%EXE_DEP
     10.0   0
/
END
"""


def translate(deck_text, argv_extra=(), tfuel=900.0):
    """Run the full front end on *deck_text*; return (rc, stdout, stderr)."""
    with tempfile.TemporaryDirectory() as tmp:
        deck = os.path.join(tmp, "deck.inp")
        with open(deck, "w") as fh:
            fh.write(deck_text)
        out = os.path.join(tmp, "case.json")
        argv = [deck, "--xs", "lib.h5", "--no-verify-xs", "-o", out]
        if tfuel is not None:
            argv += ["--tfuel", str(tfuel)]
        argv += list(argv_extra)

        so, se = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        try:
            rc = m2r.main(argv)
            captured_out = sys.stdout.getvalue()
            captured_err = sys.stderr.getvalue()
        finally:
            sys.stdout, sys.stderr = so, se
        payload = None
        if os.path.exists(out):
            with open(out) as fh:
                payload = fh.read()
        return rc, captured_out, captured_err, payload


def swap(card, new_body):
    """Replace the body of *card* in MINIMAL with *new_body*."""
    lines = MINIMAL.splitlines(keepends=True)
    out, i = [], 0
    while i < len(lines):
        out.append(lines[i])
        if lines[i].strip() == card:
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("%") \
                    and lines[i].strip() not in ("/", "END"):
                i += 1
            out.append(new_body if new_body.endswith("\n") else new_body + "\n")
            continue
        i += 1
    return "".join(out)


class Baseline(unittest.TestCase):
    def test_minimal_deck_translates(self):
        rc, _, err, payload = translate(MINIMAL)
        self.assertEqual(rc, 0, err)
        case = json.loads(payload)
        self.assertEqual(case["geometry"]["dimensions"]["nz"], 2)
        self.assertEqual(case["TH"]["rated power"], 520.0)


class CardNameBoundary(unittest.TestCase):
    """A1: a suffixed card name must not match a shorter card."""

    def test_suffix_typo_is_refused(self):
        bad = MINIMAL.replace("%GEN_DIM\n", "%GEN_DIM1\n", 1)
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("unrecognised card syntax", err)
        self.assertIn("%GEN_DIM1", err)

    def test_suffix_typo_does_not_shift_fields(self):
        """The old regex ate '%GEN_DIM' and fed the stray '1' in as nx."""
        with self.assertRaises(m2r.DeckError):
            m2r.parse_deck("%GEN_DIM1 7 7 2 1 1 3 1 0 2 2 2\n")
        # ...and the correctly spelled card still yields the right first token.
        items, _ = m2r.parse_deck("%GEN_DIM 7 7 2 1 1 3 1 0 2 2 2\n")
        self.assertEqual(items[0].tokens[0], "7")

    def test_card_name_followed_by_data_still_parses(self):
        inline = MINIMAL.replace(
            "%GEN_FDB\noff off off\n", "%GEN_FDB off off off\n", 1)
        rc, _, err, _ = translate(inline)
        self.assertEqual(rc, 0, err)


class ExactFieldCounts(unittest.TestCase):
    """A1: fixed-width cards must not carry unread trailing data."""

    def test_gen_dim_extra_field(self):
        bad = swap("%GEN_DIM", "        3 3 2 1 1 3 1 0 2 2 2 99")
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("expected exactly 11", err)
        self.assertIn("99", err)

    def test_gen_sym_extra_field(self):
        bad = swap("%GEN_SYM", "        0 0 0 0 0 0 0")
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("expected exactly 6", err)

    def test_gen_pin_extra_field(self):
        bad = swap("%GEN_PIN", "        1 2 -17 -248 5")
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("expected exactly 4", err)

    def test_exe_std_extra_field(self):
        bad = MINIMAL.replace("        keff tr tr 1.0\n",
                              "        keff tr tr 1.0 0.5\n", 1)
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("expected exactly 4", err)

    def test_gen_thd_accepts_five(self):
        rc, _, err, _ = translate(MINIMAL)
        self.assertEqual(rc, 0, err)

    def test_gen_thd_accepts_seven(self):
        ok = swap("%GEN_THD", "        520.0 295 24.5 306.85 155\n        3600 1.0")
        rc, _, err, _ = translate(ok)
        self.assertEqual(rc, 0, err)

    def test_gen_thd_rejects_six(self):
        bad = swap("%GEN_THD", "        520.0 295 24.5 306.85 155 3600")
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("expected 5 or 7", err)


class NgeoFoldConflict(unittest.TestCase):
    """A2: --fold quarter on a deck that is already a partial core."""

    def test_partial_core_refuses_fold(self):
        bad = swap("%GEN_DIM", "        3 3 2 1 1 3 4 0 2 2 2")
        rc, _, err, _ = translate(bad, ["--fold", "quarter"])
        self.assertEqual(rc, 1)
        self.assertIn("ngeo=4", err)
        self.assertIn("full-core deck", err)

    def test_partial_core_without_fold_no_longer_falls_through(self):
        """The silent 4x error: ngeo=4 used to translate as a 360-degree core.

        This deck's %GEN_SYM leaves the cut faces at 0 (vacuum), so it does not
        describe a quarter core at all and is refused rather than emitted with
        the full-core symmetry block.  AlreadyFoldedCore covers the deck that
        does declare them.
        """
        bad = swap("%GEN_DIM", "        3 3 2 1 1 3 4 0 2 2 2")
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("ngeo=4", err)

    def test_check_reports_it_as_blocking(self):
        bad = swap("%GEN_DIM", "        3 3 2 1 1 3 4 0 2 2 2")
        rc, out, _, _ = translate(bad, ["--fold", "quarter", "--check"])
        self.assertEqual(rc, 1)
        self.assertIn("BLOCKED", out)


class AlreadyFoldedCore(unittest.TestCase):
    """The third geometry mode: %GEN_DIM ngeo != 1 with --fold none.

    This used to fall through to the full-core branch, declaring a quarter map
    as a 360-degree core with the full-core rated power -- a 4x power-density
    error that runs to completion.
    """

    # A 2x2 quarter map, diagonally symmetric, with the west/north faces
    # declared as symmetry planes through the middle of an assembly.
    QUARTER = (MINIMAL
               .replace("        3       3       2       1       1\n"
                        "        3       1       0       2       2\n",
                        "        2       2       2       1       1\n"
                        "        3       4       1       2       2\n", 1)
               .replace("       A1  A1  A1\n       A1  A1  A1\n       A1  A1  A1\n",
                        "       A1  A1\n       A1  A1\n", 1)
               .replace("        0       0       0\n        0       0       0\n",
                        "        -1      -1      0\n        0       0       0\n", 1))

    def test_quarter_deck_gets_the_quarter_symmetry_and_power(self):
        rc, _, err, payload = translate(self.QUARTER)
        self.assertEqual(rc, 0, err)
        case = json.loads(payload)
        self.assertEqual(case["geometry"]["symmetry"],
                         {"angle": 90, "mirror": True,
                          "center assembly divided": True})
        self.assertEqual(case["TH"]["rated power"], 520.0 / 4)
        self.assertEqual(case["geometry"]["albedo"]["west"], 0.0)
        self.assertEqual(case["geometry"]["albedo"]["north"], 0.0)

    def test_plane_on_assembly_boundary_is_not_divided(self):
        deck = self.QUARTER.replace("        -1      -1      0\n",
                                    "        1       1       0\n", 1)
        rc, _, err, payload = translate(deck)
        self.assertEqual(rc, 0, err)
        self.assertFalse(
            json.loads(payload)["geometry"]["symmetry"]["center assembly divided"])

    def test_undeclared_cut_face_is_refused(self):
        deck = self.QUARTER.replace("        -1      -1      0\n",
                                    "        0       -1      0\n", 1)
        rc, _, err, _ = translate(deck)
        self.assertEqual(rc, 1)
        self.assertIn("isymlx=0", err)
        self.assertIn("symmetry planes", err)

    def test_disagreeing_cut_faces_are_refused(self):
        deck = self.QUARTER.replace("        -1      -1      0\n",
                                    "        -1      1       0\n", 1)
        rc, _, err, _ = translate(deck)
        self.assertEqual(rc, 1)
        self.assertIn("single flag for both axes", err)

    def test_missing_gen_sym_is_refused_not_assumed(self):
        deck = self.QUARTER.replace(
            "%GEN_SYM\n        -1      -1      0\n        0       0       0\n", "", 1)
        rc, _, err, _ = translate(deck)
        self.assertEqual(rc, 1)
        self.assertIn("no %GEN_SYM card", err)

    def test_asymmetric_quarter_map_is_refused(self):
        deck = self.QUARTER.replace(
            "%LPD_B&C\n        A1     2*1\n",
            "%LPD_B&C\n        A1     2*1\n        A2     2*1\n", 1)
        deck = deck.replace("       A1  A1\n       A1  A1\n",
                            "       A1  A2\n       A1  A1\n", 1)
        deck = deck.replace("%GEN_DIM\n        2       2       2       1       1\n",
                            "%GEN_DIM\n        2       2       2       2       1\n", 1)
        rc, _, err, _ = translate(deck)
        self.assertEqual(rc, 1)
        self.assertIn("not symmetric about its main diagonal", err)

    def test_unsupported_sector_is_refused(self):
        deck = self.QUARTER.replace("        3       4       1       2       2\n",
                                    "        3       8       1       2       2\n", 1)
        rc, _, err, _ = translate(deck)
        self.assertEqual(rc, 1)
        self.assertIn("1/8 sector", err)

    def test_full_core_deck_is_untouched(self):
        """ngeo=1 must keep the old 360-degree behaviour exactly."""
        rc, _, err, payload = translate(MINIMAL)
        self.assertEqual(rc, 0, err)
        case = json.loads(payload)
        self.assertEqual(case["geometry"]["symmetry"],
                         {"angle": 360, "mirror": False,
                          "center assembly divided": False})
        self.assertEqual(case["TH"]["rated power"], 520.0)


class XenonPerState(unittest.TestCase):
    """%EXE_STD sets xenon per state; RASBERY carries one for the whole case."""

    MIXED = MINIMAL.replace(
        "%EXE_DEP\n     10.0   0\n/\nEND\n",
        "%EXE_DEP\n     10.0   0\n/\n%EXE_STD\n        keff eq eq 1.0\n/\nEND\n", 1)

    def test_mixed_modes_are_refused(self):
        rc, _, err, _ = translate(self.MIXED)
        self.assertEqual(rc, 1)
        self.assertIn("more than one xenon mode", err)
        self.assertIn("retroactively", err)

    def test_explicit_choice_resolves_it(self):
        rc, _, err, payload = translate(self.MIXED, ["--xenon", "equilibrium"])
        self.assertEqual(rc, 0, err)
        self.assertEqual(
            json.loads(payload)["default parameters"]["xenon"], "equilibrium")

    def test_explicit_choice_the_deck_never_asks_for_is_refused(self):
        rc, _, err, _ = translate(MINIMAL, ["--xenon", "equilibrium"])
        self.assertEqual(rc, 1)
        self.assertIn("no %EXE_STD in the deck", err)

    def test_single_mode_deck_is_untouched(self):
        rc, _, err, payload = translate(MINIMAL)
        self.assertEqual(rc, 0, err)
        self.assertEqual(
            json.loads(payload)["default parameters"]["xenon"], "transient")

    def test_last_wins_is_gone(self):
        """The old code emitted the LAST card's mode for the whole case."""
        rc, _, err, _ = translate(self.MIXED)
        self.assertNotEqual(rc, 0, "a mixed deck must not translate silently")


class ThermalHydraulicTail(unittest.TestCase):
    """%GEN_THD mflow/hgfl were parsed and then dropped without a word."""

    SEVEN = swap("%GEN_THD",
                 "        520.0 295 24.5 306.85 155\n        3600 1.0")

    def test_dropped_flow_is_an_assumption_when_feedback_is_on(self):
        deck = self.SEVEN.replace("off off off", "on on on", 1)
        rc, _, err, _ = translate(deck)
        self.assertEqual(rc, 3)
        self.assertIn("mflow=3600", err)
        self.assertIn("ASSUME", err)

    def test_dropped_flow_is_only_a_note_when_feedback_is_off(self):
        rc, _, err, _ = translate(self.SEVEN)
        self.assertEqual(rc, 0, err)
        self.assertIn("mflow=3600", err)
        self.assertNotIn("ASSUME", err)

    def test_five_field_card_says_nothing(self):
        rc, _, err, _ = translate(MINIMAL)
        self.assertEqual(rc, 0, err)
        self.assertNotIn("mflow", err)


class NonFiniteNumbers(unittest.TestCase):
    """A5: 'nan' and 'inf' are valid float() input and must not survive."""

    def test_nan_pressure(self):
        bad = swap("%GEN_THD", "        520.0 295 24.5 306.85 nan")
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("not a finite number", err)

    def test_inf_axial_mesh(self):
        bad = swap("%GEN_GEO", "        21.5 100.0\n        50 inf")
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("not a finite number", err)

    def test_nan_tfuel_from_cli(self):
        rc, _, err, _ = translate(MINIMAL, tfuel=float("nan"))
        self.assertEqual(rc, 1)
        self.assertIn("non-finite", err)


class HexagonalCore(unittest.TestCase):
    """A3: %JOB_HEX was listed as 'checked, then inert' with nothing checking."""

    def test_hexagonal_is_refused(self):
        bad = MINIMAL.replace("%GEN_DIM\n", "%JOB_HEX\n        1\n%GEN_DIM\n", 1)
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("hexagonal", err)

    def test_rectangular_flag_passes(self):
        ok = MINIMAL.replace("%GEN_DIM\n", "%JOB_HEX\n        0\n%GEN_DIM\n", 1)
        rc, _, err, _ = translate(ok)
        self.assertEqual(rc, 0, err)

    def test_card_is_reported_as_checked(self):
        ok = MINIMAL.replace("%GEN_DIM\n", "%JOB_HEX\n        0\n%GEN_DIM\n", 1)
        rc, out, _, _ = translate(ok, ["--coverage"])
        self.assertEqual(rc, 0)
        self.assertIn("%JOB_HEX", out)
        self.assertIn("checked (no output)", out)


class AlbedoBoundarySource(unittest.TestCase):
    """A4: %GEN_MTH ibndc=1/2 replaces the %GEN_SYM flags this translator reads."""

    MTH = "%GEN_MTH\n        9       1       {}       3       1\n"

    def test_ibndc_one_is_refused(self):
        bad = MINIMAL.replace("%GEN_DIM\n", self.MTH.format(1) + "%GEN_DIM\n", 1)
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("ibndc=1", err)

    def test_ibndc_two_is_refused(self):
        bad = MINIMAL.replace("%GEN_DIM\n", self.MTH.format(2) + "%GEN_DIM\n", 1)
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("ibndc=2", err)

    def test_ibndc_out_of_range_is_refused(self):
        bad = MINIMAL.replace("%GEN_DIM\n", self.MTH.format(7) + "%GEN_DIM\n", 1)
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("documented range", err)

    def test_ibndc_zero_passes(self):
        ok = MINIMAL.replace("%GEN_DIM\n", self.MTH.format(0) + "%GEN_DIM\n", 1)
        rc, _, err, _ = translate(ok)
        self.assertEqual(rc, 0, err)


class ThermalHydraulicPlausibility(unittest.TestCase):
    """A6."""

    def test_zero_pressure(self):
        bad = swap("%GEN_THD", "        520.0 295 24.5 306.85 0.0")
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("not positive", err)

    def test_negative_trise(self):
        bad = swap("%GEN_THD", "        520.0 295 -5.0 306.85 155")
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("negative", err)

    def test_sub_absolute_zero_inlet(self):
        bad = swap("%GEN_THD", "        520.0 -400 24.5 306.85 155")
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("absolute zero", err)


class SymmetryFlagWhitelist(unittest.TestCase):
    """A7: the z faces used to take any non-zero flag as reflective."""

    def test_bad_z_flag_is_refused(self):
        bad = swap("%GEN_SYM", "        0 0 2\n        0 0 0")
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("isymlz=2", err)

    def test_bad_x_flag_is_refused(self):
        bad = swap("%GEN_SYM", "        3 0 0\n        0 0 0")
        rc, _, err, _ = translate(bad)
        self.assertEqual(rc, 1)
        self.assertIn("isymlx=3", err)

    def test_reflective_z_flag_passes(self):
        ok = swap("%GEN_SYM", "        0 0 -1\n        0 0 1")
        rc, _, err, payload = translate(ok)
        self.assertEqual(rc, 0, err)
        albedo = json.loads(payload)["geometry"]["albedo"]
        self.assertEqual(albedo["bottom"], 0.0)
        self.assertEqual(albedo["up"], 0.0)


class ReflectorNameInvariant(unittest.TestCase):
    """A7: RASBERY splits fuel from reflector on the first letter."""

    def test_ref_rule_yielding_non_r_name_is_refused(self):
        cache = {}
        with self.assertRaises(m2r.DeckError) as ctx:
            m2r.resolve_model("REF_AXIAL_B", {"AXIAL_B"}, {}, cache)
        self.assertIn("does not start with", str(ctx.exception))

    def test_ref_rule_yielding_r_name_is_accepted(self):
        cache = {}
        self.assertEqual(m2r.resolve_model("REF_AXIAL_B", {"RB"}, {}, cache), "RB")
        self.assertEqual(m2r.resolve_model("REF_R1", {"R1"}, {}, cache), "R1")

    def test_fa_rule_is_unaffected(self):
        cache = {}
        self.assertEqual(m2r.resolve_model("FA_A1", {"A1"}, {}, cache), "A1")

    def test_explicit_alias_bypasses_the_rule(self):
        cache = {}
        self.assertEqual(
            m2r.resolve_model("REF_AXIAL_B", {"AXIAL_B"}, {"REF_AXIAL_B": "AXIAL_B"},
                              cache),
            "AXIAL_B")


class Assumptions(unittest.TestCase):
    """A8: a translation that guessed must not report success."""

    def test_missing_gen_sym_exits_three(self):
        no_sym = MINIMAL.replace(
            "%GEN_SYM\n        0       0       0\n        0       0       0\n", "", 1)
        rc, _, err, payload = translate(no_sym)
        self.assertEqual(rc, 3)
        self.assertIn("ASSUME", err)
        self.assertIsNotNone(payload, "the JSON must still be written")

    def test_allow_assumptions_returns_zero(self):
        no_sym = MINIMAL.replace(
            "%GEN_SYM\n        0       0       0\n        0       0       0\n", "", 1)
        rc, _, _, _ = translate(no_sym, ["--allow-assumptions"])
        self.assertEqual(rc, 0)

    def test_clean_deck_has_no_assumptions(self):
        rc, _, err, _ = translate(MINIMAL)
        self.assertEqual(rc, 0)
        self.assertNotIn("ASSUME", err)

    def test_quiet_still_prints_the_summary(self):
        no_sym = MINIMAL.replace(
            "%GEN_SYM\n        0       0       0\n        0       0       0\n", "", 1)
        rc, _, err, _ = translate(no_sym, ["--quiet"])
        self.assertEqual(rc, 3)
        self.assertIn("assumption(s)", err)


class BoronOption(unittest.TestCase):
    """A9: --boron explicitness is read off the parsed options, not argv."""

    def test_default_is_zero_and_not_explicit(self):
        rc, _, err, payload = translate(MINIMAL)
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads(payload)["default parameters"]["boron_ppm"], 0.0)

    def test_explicit_boron_is_carried(self):
        rc, _, err, payload = translate(MINIMAL, ["--boron", "600"])
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads(payload)["default parameters"]["boron_ppm"], 600.0)

    def test_abbreviated_option_counts_as_explicit(self):
        """argparse accepts '--bor 600'; the old argv sniff did not see it."""
        deck = MINIMAL.replace("%EXE_DEP\n     10.0   0\n",
                               "%EXE_DEP\n     10.0   1\n        -1.0  0.0\n", 1)
        rc, _, err, _ = translate(deck, ["--bor", "600"])
        self.assertEqual(rc, 1, "the deck sets 0 ppm; --bor 600 must conflict")
        self.assertIn("refusing to pick one", err)


class StripComment(unittest.TestCase):
    def test_hash_and_bang_both_cut(self):
        self.assertEqual(m2r.strip_comment("a b # c"), "a b ")
        self.assertEqual(m2r.strip_comment("a b ! c"), "a b ")
        self.assertEqual(m2r.strip_comment("a ! b # c"), "a ")
        self.assertEqual(m2r.strip_comment("a # b ! c"), "a ")
        self.assertEqual(m2r.strip_comment("plain"), "plain")


class ReferenceDeckByteIdentity(unittest.TestCase):
    """The whole point: a valid deck must not move.

    Skipped unless MASI_REF_DECK and MASI_REF_JSON point at a deck and the
    translation it produced before this work.
    """

    def run_deck(self, deck, extra=()):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "case.json")
            cwd = os.getcwd()
            os.chdir(os.path.dirname(os.path.abspath(deck)))
            so, se = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
            try:
                rc = m2r.main([os.path.basename(deck), "-o", out] + list(extra))
            finally:
                sys.stdout, sys.stderr = so, se
                os.chdir(cwd)
            with open(out, "rb") as fh:
                return rc, fh.read()

    def test_reference_translation_is_unchanged(self):
        deck = os.environ.get("MASI_REF_DECK")
        ref = os.environ.get("MASI_REF_JSON")
        if not deck or not ref:
            self.skipTest("set MASI_REF_DECK and MASI_REF_JSON to run")
        rc, payload = self.run_deck(deck)
        self.assertEqual(rc, 0)
        with open(ref, "rb") as fh:
            self.assertEqual(payload, fh.read())

    def test_partial_core_deck_matches_its_hand_corrected_case(self):
        """An already-quarter deck must now need no hand correction.

        MASI_PARTIAL_DECK is a deck with %GEN_DIM ngeo=4; MASI_PARTIAL_JSON is
        the case a human produced from it by fixing the symmetry block, the
        rated power and the xenon mode by hand.  The translator must reproduce
        it exactly.  MASI_PARTIAL_ARGS carries the deck-specific options.
        """
        deck = os.environ.get("MASI_PARTIAL_DECK")
        ref = os.environ.get("MASI_PARTIAL_JSON")
        if not deck or not ref:
            self.skipTest("set MASI_PARTIAL_DECK and MASI_PARTIAL_JSON to run")
        extra = os.environ.get("MASI_PARTIAL_ARGS", "").split()
        rc, payload = self.run_deck(deck, extra + ["--allow-assumptions"])
        self.assertEqual(rc, 0)
        with open(ref) as fh:
            self.assertEqual(json.loads(payload), json.load(fh))


if __name__ == "__main__":
    unittest.main()
