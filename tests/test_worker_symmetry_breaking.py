"""
tests/test_worker_symmetry_breaking.py

Regression tests for the broken-symmetry initial guess.

The old code called ``scf.uhf.mulliken_meta`` / ``dft.uks.mulliken_meta`` to
"mix alpha/beta" and passed the result to ``mf.kernel(dm0=...)``. Those are
population-analysis routines: they return ``(pop, charges)``, not a density
matrix. Every call therefore raised inside the try block, the except branch
logged a warning and ran a plain ``mf.kernel()`` -- after the log had already
announced "Applying Symmetry Breaking". A user asking for a broken-symmetry
solution (singlet diradical, antiferromagnetic coupling) silently got the
spin-restricted one.

The trigger was also inverted: it fired only when spin_2s > 0, where alpha and
beta occupations already differ and there is no symmetry left to break.
"""

import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import MagicMock

import numpy as np


# ---------------------------------------------------------------------------
# Stub the Qt / rdkit surface so worker.py imports headlessly
# ---------------------------------------------------------------------------


def _install_stubs():
    pyqt6 = types.ModuleType("PyQt6")
    qt_core = types.ModuleType("PyQt6.QtCore")

    class _QThread:
        def __init__(self, *a, **k):
            pass

    qt_core.QThread = _QThread
    qt_core.pyqtSignal = lambda *a, **k: MagicMock()
    pyqt6.QtCore = qt_core

    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtCore"] = qt_core
    for name in ("rdkit", "rdkit.Chem", "rdkit.Chem.rdMolTransforms"):
        sys.modules.setdefault(name, MagicMock())


def _load_worker():
    _install_stubs()
    sys.modules["pyscf"] = MagicMock()
    for sub in ("pyscf.gto", "pyscf.scf", "pyscf.dft", "pyscf.solvent"):
        sys.modules[sub] = MagicMock()
    src = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "pyscf_calculator", "worker.py")
    )
    spec = importlib.util.spec_from_file_location("_worker_symbreak", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_worker_symbreak"] = mod
    spec.loader.exec_module(mod)
    return mod


WORKER_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "pyscf_calculator", "worker.py")
)


class _FakeMol:
    """Two atoms, 2 basis functions each."""

    def __init__(self, slices=((0, 0, 0, 2), (0, 0, 2, 4))):
        self._slices = slices

    def aoslice_by_atom(self):
        return np.array(self._slices)


def _make_mf(nao=4):
    """Mean-field stub whose guess is spin-restricted (alpha == beta)."""
    block = np.arange(nao * nao, dtype=float).reshape(nao, nao) + 1.0
    mf = MagicMock()
    mf.get_init_guess.return_value = np.array([block.copy(), block.copy()])
    return mf, block


class TestBrokenSymmetryGuess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_worker()

    def _worker(self):
        w = self.mod.PySCFWorker.__new__(self.mod.PySCFWorker)
        self.mod.PySCFWorker.__init__(
            w, "H 0 0 0\nH 0 0 0.74", {"method": "UHF", "spin": "1"}
        )
        w.log_signal = MagicMock()
        return w

    def test_returns_a_spin_resolved_density_matrix(self):
        mf, _ = _make_mf()
        dm = self._worker()._broken_symmetry_guess(mf, _FakeMol())
        self.assertEqual(dm.shape, (2, 4, 4))

    def test_beta_block_of_the_first_atom_is_zeroed(self):
        mf, _ = _make_mf()
        dm = self._worker()._broken_symmetry_guess(mf, _FakeMol())
        np.testing.assert_array_equal(dm[1][0:2, 0:2], np.zeros((2, 2)))

    def test_alpha_density_is_untouched(self):
        mf, block = _make_mf()
        dm = self._worker()._broken_symmetry_guess(mf, _FakeMol())
        np.testing.assert_array_equal(dm[0], block)

    def test_alpha_and_beta_actually_differ(self):
        """The whole point: a restricted guess collapses UHF back onto RHF."""
        mf, _ = _make_mf()
        dm = self._worker()._broken_symmetry_guess(mf, _FakeMol())
        self.assertFalse(np.allclose(dm[0], dm[1]))

    def test_the_original_guess_is_not_mutated(self):
        mf, block = _make_mf()
        original = mf.get_init_guess.return_value
        self._worker()._broken_symmetry_guess(mf, _FakeMol())
        np.testing.assert_array_equal(original[1], block)

    def test_rest_of_the_beta_density_survives(self):
        mf, block = _make_mf()
        dm = self._worker()._broken_symmetry_guess(mf, _FakeMol())
        np.testing.assert_array_equal(dm[1][2:4, 2:4], block[2:4, 2:4])

    def test_restricted_guess_is_rejected(self):
        mf = MagicMock()
        mf.get_init_guess.return_value = np.ones((4, 4))  # not spin-resolved
        with self.assertRaises(ValueError):
            self._worker()._broken_symmetry_guess(mf, _FakeMol())

    def test_atom_without_basis_functions_is_rejected(self):
        mf, _ = _make_mf()
        empty_first_atom = _FakeMol(slices=((0, 0, 0, 0), (0, 0, 0, 4)))
        with self.assertRaises(ValueError):
            self._worker()._broken_symmetry_guess(mf, empty_first_atom)


class TestSymmetryBreakingWiring(unittest.TestCase):
    """Guard the two defects at the call site, which needs a full SCF run to
    exercise directly."""

    def setUp(self):
        with open(WORKER_SRC, "r", encoding="utf-8") as fh:
            self.src = fh.read()

    def test_population_analysis_is_no_longer_used_as_a_density_matrix(self):
        self.assertNotIn("mulliken_meta", self.src)

    def test_symmetry_breaking_triggers_on_the_restricted_case(self):
        """spin_2s == 0 is where a restricted guess needs breaking; the old
        condition was spin_2s > 0."""
        # Behaviour is covered in test_worker_optimization_coverage
        # (TestSymmetryBreaking); this only pins the condition's direction.
        self.assertIn("and self._parse_spin_2s() == 0", self.src)
        self.assertNotIn("_parse_spin_2s() > 0", self.src)

    def test_the_helper_is_used_at_the_call_site(self):
        self.assertIn("_broken_symmetry_guess(mf, mol)", self.src)


class TestNumericHessianUnits(unittest.TestCase):
    def setUp(self):
        with open(WORKER_SRC, "r", encoding="utf-8") as fh:
            self.src = fh.read()

    def test_atom_coords_fallback_does_not_double_convert(self):
        """mol.atom_coords() is already Bohr; the fallback used to multiply by
        the Angstrom->Bohr factor, inflating every displacement by 1.89x."""
        self.assertNotIn("atom_coords() * 1.8897259886", self.src)


if __name__ == "__main__":
    unittest.main()
