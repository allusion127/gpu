#!/usr/bin/env python3
"""Guard tests for the CHIFFON reflector input surface.

Drives the real binary against crafted library-build inputs and asserts that
each malformed spec is refused *loudly*, with a message that names the mistake.
Every case here used to be accepted: a mistyped neighbour name resolved to
"there is a node and it is nothing", a short "fuel sides" list left the trailing
nodes with no discontinuity factor, and "left/right neighbor" on a paired axial
reflector described a boundary treatment nothing implements.

Run with:
    RASBERY_BIN=build/RASBERY python3 tools/test_chiffon_input_guards.py

The HGC inputs come from test/CrossSections/2_i-SMR_Validation; the whole module
skips if either the binary or that directory is missing.
"""

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ASSETS = os.path.join(REPO, "test", "CrossSections", "2_i-SMR_Validation")
REF_JSON = os.path.join(ASSETS, "2_i-SMR_Validation.json")


def find_binary():
    env = os.environ.get("RASBERY_BIN")
    if env and os.access(env, os.X_OK):
        return env
    for cand in ("build/RASBERY", "build-nocuda/RASBERY", "build-Release/RASBERY"):
        path = os.path.join(REPO, cand)
        if os.access(path, os.X_OK):
            return path
    return None


BINARY = find_binary()


@unittest.skipIf(BINARY is None, "no RASBERY binary (set RASBERY_BIN)")
@unittest.skipIf(not os.path.exists(REF_JSON), "2_i-SMR_Validation assets absent")
class ReflectorSpecGuards(unittest.TestCase):
    """Each case builds a library and expects a specific refusal."""

    @classmethod
    def setUpClass(cls):
        with open(REF_JSON) as fh:
            cls.reference = json.load(fh)

    def build(self, mutate):
        """Write a mutated input into a scratch dir and run the library build.

        Returns (returncode, combined output).
        """
        spec = copy.deepcopy(self.reference)
        mutate(spec)
        with tempfile.TemporaryDirectory() as tmp:
            for name in os.listdir(ASSETS):
                if name.endswith(".HGC"):
                    os.symlink(os.path.join(ASSETS, name), os.path.join(tmp, name))
            inp = os.path.join(tmp, "case.json")
            with open(inp, "w") as fh:
                json.dump(spec, fh, indent=1)
            proc = subprocess.run(
                [BINARY, "--chiffoni", "./case.json", "--chiffono", "./out.h5"],
                cwd=tmp, capture_output=True, text=True, timeout=1800)
            return proc.returncode, proc.stdout + proc.stderr

    # -- B11 -------------------------------------------------------------
    def test_dead_neighbor_field_is_refused(self):
        def mutate(spec):
            spec["reflectors"]["neighbors"]["RB"]["left neighbor"] = "VACUUM"
        rc, out = self.build(mutate)
        self.assertNotEqual(rc, 0)
        self.assertIn("not implemented for reflector type", out)
        self.assertIn("left neighbor", out)

    def test_dead_right_neighbor_field_is_refused(self):
        def mutate(spec):
            spec["reflectors"]["neighbors"]["RT"]["right neighbor"] = "AX0805"
        rc, out = self.build(mutate)
        self.assertNotEqual(rc, 0)
        self.assertIn("not implemented for reflector type", out)

    # -- B12 -------------------------------------------------------------
    def test_unknown_neighbor_name_throws(self):
        """A typo in the node name used to become nullptr + boundary=NODE.

        Reached by asking ParseOptionalReflectorNeighbor directly -- which the
        B11 guard now shadows for shipped types -- so this asserts the pair of
        messages: whichever fires, the run must stop and name the field.
        """
        def mutate(spec):
            spec["reflectors"]["neighbors"]["RB"]["left neighbor"] = "AX9999"
        rc, out = self.build(mutate)
        self.assertNotEqual(rc, 0)
        self.assertTrue(
            "not implemented for reflector type" in out
            or "unknown reflector node 'AX9999'" in out,
            out[-2000:])

    def test_unknown_left_node_still_throws(self):
        """The pre-existing sibling behaviour B12 was aligned with."""
        def mutate(spec):
            spec["reflectors"]["neighbors"]["RB"]["left node"] = "AX9999"
        rc, out = self.build(mutate)
        self.assertNotEqual(rc, 0)
        self.assertIn("unknown reflector node 'AX9999'", out)

    # -- B15 -------------------------------------------------------------
    def test_short_fuel_sides_list_is_refused(self):
        def mutate(spec):
            spec["reflectors"]["neighbors"]["R1"]["fuel sides"] = ["west"]
        rc, out = self.build(mutate)
        self.assertNotEqual(rc, 0)
        self.assertIn("'fuel sides'", out)
        self.assertIn("positional", out)

    def test_long_fuel_sides_list_is_refused(self):
        def mutate(spec):
            nodes = spec["reflectors"]["neighbors"]["R2"]["nodes"]
            spec["reflectors"]["neighbors"]["R2"]["fuel sides"] = \
                ["west"] * (len(nodes) + 1)
        rc, out = self.build(mutate)
        self.assertNotEqual(rc, 0)
        self.assertIn("'fuel sides'", out)

    def test_matching_fuel_sides_list_is_accepted(self):
        def mutate(spec):
            nodes = spec["reflectors"]["neighbors"]["R3"]["nodes"]
            spec["reflectors"]["neighbors"]["R3"]["fuel sides"] = \
                ["west"] * len(nodes)
        rc, out = self.build(mutate)
        self.assertEqual(rc, 0, out[-3000:])

    def test_bad_fuel_side_value_is_refused(self):
        def mutate(spec):
            spec["reflectors"]["neighbors"]["R2"]["fuel side"] = "starboard"
        rc, out = self.build(mutate)
        self.assertNotEqual(rc, 0)
        self.assertIn("bad 'fuel side' value", out)

    # -- B13 -------------------------------------------------------------
    def test_declared_side_is_reported(self):
        def mutate(spec):
            spec["reflectors"]["neighbors"]["R2"]["fuel side"] = "west"
        rc, out = self.build(mutate)
        self.assertEqual(rc, 0, out[-3000:])
        self.assertIn("[CHIFFON][refl]", out)
        self.assertIn("fuel-facing side west", out)
        self.assertIn("usable window", out)

    def test_reference_input_builds_clean(self):
        rc, out = self.build(lambda spec: None)
        self.assertEqual(rc, 0, out[-3000:])
        self.assertNotIn("not implemented", out)

    # -- B14 -------------------------------------------------------------
    def test_library_records_its_dropped_branches(self):
        """A clean library must still say so, not stay silent."""
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py not available")
        spec = copy.deepcopy(self.reference)
        with tempfile.TemporaryDirectory() as tmp:
            for name in os.listdir(ASSETS):
                if name.endswith(".HGC"):
                    os.symlink(os.path.join(ASSETS, name), os.path.join(tmp, name))
            with open(os.path.join(tmp, "case.json"), "w") as fh:
                json.dump(spec, fh, indent=1)
            proc = subprocess.run(
                [BINARY, "--chiffoni", "./case.json", "--chiffono", "./out.h5"],
                cwd=tmp, capture_output=True, text=True, timeout=1800)
            self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
            with h5py.File(os.path.join(tmp, "out.h5")) as f:
                meta = f["Metadata"]
                self.assertIn("degenerate_branches", meta)
                self.assertIn("degenerate_branch_count", meta)
                payload = meta["degenerate_branches"][()]
                count = int(meta["degenerate_branch_count"][()])
                text = payload.decode() if isinstance(payload, bytes) else str(payload)
                self.assertEqual(count, 0 if not text else len(text.split("\n")))

    # -- B10 -------------------------------------------------------------
    def test_rodded_refit_reports_the_degenerate_counter(self):
        """The rodded refit must account for degenerate-axis drops.

        This library's branch axes all move, so the count is 0; what is being
        asserted is that the guard is on the path at all -- before this change
        the rodded refit had no degeneracy test and no way to report one.
        """
        with tempfile.TemporaryDirectory() as tmp:
            for name in os.listdir(ASSETS):
                if name.endswith(".HGC"):
                    os.symlink(os.path.join(ASSETS, name), os.path.join(tmp, name))
            shutil.copy(REF_JSON, os.path.join(tmp, "case.json"))
            env = dict(os.environ, CHIFFON_PROBE_RODBRTAB="1")
            proc = subprocess.run(
                [BINARY, "--chiffoni", "./case.json", "--chiffono", "./out.h5"],
                cwd=tmp, capture_output=True, text=True, timeout=1800, env=env)
            self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
            out = proc.stdout + proc.stderr
            self.assertIn("[rodbrtab]", out)
            self.assertIn("degenerate axis", out)


if __name__ == "__main__":
    unittest.main()
