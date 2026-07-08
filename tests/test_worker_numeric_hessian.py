"""
tests/test_worker_numeric_hessian.py

Tests for PySCFWorker.compute_numeric_hessian() — a pure-numpy finite-
difference routine that is exercisable with mocked mol/mf objects.

Coverage targets:
  - Cooperative stop at start of atom loop (lines 212-214)
  - Normal computation path: symmetrized hessian shape and values
  - Fallback path when mf.nuc_grad_method().as_scanner() raises (lines 183-191)
  - Progress log emitted every 3 steps (line 240)
"""

import os
import sys
import types
import unittest
import importlib.util
import numpy as np
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Qt / pyscf stubs
# ---------------------------------------------------------------------------


def _install_stubs(force=False):
    qt_core = types.ModuleType("PyQt6.QtCore")

    class _QThread:
        def __init__(self):
            pass

    qt_core.QThread = _QThread
    qt_core.pyqtSignal = lambda *a, **kw: MagicMock()
    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core

    def _set(k, v):
        if force:
            sys.modules[k] = v
        else:
            sys.modules.setdefault(k, v)

    _set("PyQt6", pyqt6)
    _set("PyQt6.QtCore", qt_core)
    _set("rdkit", MagicMock())
    _set("rdkit.Chem", MagicMock())
    _set("rdkit.Chem.rdMolTransforms", MagicMock())


_install_stubs()


def _load_worker_mod(pyscf_mock):
    _install_stubs(force=True)
    module_name = f"_worker_nhess_{id(pyscf_mock)}"
    sys.modules["pyscf"] = pyscf_mock
    if pyscf_mock is not None:
        for sub in ("pyscf.gto", "pyscf.scf", "pyscf.dft", "pyscf.solvent"):
            sys.modules[sub] = MagicMock()
    src = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "pyscf_calculator", "worker.py")
    )
    spec = importlib.util.spec_from_file_location(module_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_worker(mod, xyz="H 0 0 0\nH 0 0 0.74"):
    pyscf_mock = MagicMock()
    mod.pyscf = pyscf_mock
    w = mod.PySCFWorker.__new__(mod.PySCFWorker)
    cfg = {
        "job_type": "Single Point",
        "method": "RHF",
        "basis": "sto-3g",
        "charge": 0,
        "spin": "1",
        "threads": 0,
        "memory": 4000,
    }
    mod.PySCFWorker.__init__(w, xyz, cfg)
    w.log_signal = MagicMock()
    w.error_signal = MagicMock()
    w.finished_signal = MagicMock()
    w.result_signal = MagicMock()
    return w


def _make_mock_mol(n_atoms=2):
    """Build a mock PySCF mol with *n_atoms* atoms at origin + axis positions."""
    mol = MagicMock()
    mol.natm = n_atoms
    # coords: n_atoms × 3 array in Bohr
    coords = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        coords[i, 0] = float(i)
    mol.copy.return_value = mol
    mol.atom_coords.return_value = coords.copy()
    mol.set_geom_.return_value = None
    return mol


def _make_mock_mf(n_atoms=2):
    """Build a mock mf that returns zero gradients."""
    mf = MagicMock()
    # g_scanner returns (energy, grad) where grad is n_atoms×3 zeros
    g_zero = np.zeros((n_atoms, 3))
    scanner = MagicMock(return_value=(0.0, g_zero))
    mf.nuc_grad_method.return_value.as_scanner.return_value = scanner
    return mf


# ===========================================================================
# Cooperative stop
# ===========================================================================


class TestNumericHessianStop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_worker_mod(MagicMock())

    def test_stop_requested_raises_interrupted_error(self):
        w = _make_worker(self.mod)
        w._stop_requested = True
        mol = _make_mock_mol(n_atoms=2)
        mf = _make_mock_mf(n_atoms=2)

        with self.assertRaises(InterruptedError):
            w.compute_numeric_hessian(mf, mol)

    def test_stop_emits_log_message(self):
        w = _make_worker(self.mod)
        w._stop_requested = True
        mol = _make_mock_mol(n_atoms=2)
        mf = _make_mock_mf(n_atoms=2)

        try:
            w.compute_numeric_hessian(mf, mol)
        except InterruptedError:
            pass

        # log must contain the stop message
        all_msgs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("stopped", all_msgs.lower())


# ===========================================================================
# Normal computation path
# ===========================================================================


class TestNumericHessianCompute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_worker_mod(MagicMock())

    def test_hessian_shape(self):
        """Output must be (n_atoms, 3, n_atoms, 3)."""
        w = _make_worker(self.mod)
        n = 2
        mol = _make_mock_mol(n_atoms=n)
        mf = _make_mock_mf(n_atoms=n)

        hess = w.compute_numeric_hessian(mf, mol)
        self.assertEqual(hess.shape, (n, 3, n, 3))

    def test_hessian_is_symmetric(self):
        """Symmetrized hessian must equal its transpose."""
        w = _make_worker(self.mod)
        n = 2
        mol = _make_mock_mol(n_atoms=n)
        mf = _make_mock_mf(n_atoms=n)

        hess = w.compute_numeric_hessian(mf, mol)
        # Symmetry: H[i,j,k,l] == H[k,l,i,j]
        np.testing.assert_allclose(hess, hess.transpose(2, 3, 0, 1), atol=1e-12)

    def test_hessian_zeros_when_grad_is_zero(self):
        """Zero gradients everywhere → hessian must be all zeros."""
        w = _make_worker(self.mod)
        n = 2
        mol = _make_mock_mol(n_atoms=n)
        mf = _make_mock_mf(n_atoms=n)

        hess = w.compute_numeric_hessian(mf, mol)
        np.testing.assert_allclose(hess, np.zeros((n, 3, n, 3)), atol=1e-12)

    def test_progress_log_emitted_per_atom(self):
        """A progress message must be emitted after each atom (every 3 steps)."""
        w = _make_worker(self.mod)
        n = 2
        mol = _make_mock_mol(n_atoms=n)
        mf = _make_mock_mf(n_atoms=n)

        w.compute_numeric_hessian(mf, mol)

        # 2 atoms × 3 coords = 6 steps; every 3rd step emits a progress message
        all_msgs = [str(c) for c in w.log_signal.emit.call_args_list]
        atom_msgs = [m for m in all_msgs if "Atom" in m]
        self.assertEqual(len(atom_msgs), n)

    def test_start_log_emitted(self):
        w = _make_worker(self.mod)
        n = 1
        mol = _make_mock_mol(n_atoms=n)
        mf = _make_mock_mf(n_atoms=n)

        w.compute_numeric_hessian(mf, mol)

        first_call = str(w.log_signal.emit.call_args_list[0])
        self.assertIn("Finite Difference", first_call)


# ===========================================================================
# Fallback path when as_scanner raises
# ===========================================================================


class TestNumericHessianFallback(unittest.TestCase):
    """When mf.nuc_grad_method().as_scanner() raises, a manual fallback is used."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_worker_mod(MagicMock())

    def test_fallback_still_returns_valid_hessian(self):
        w = _make_worker(self.mod)
        n = 1
        mol = _make_mock_mol(n_atoms=n)

        # Make as_scanner raise so the fallback branch is exercised
        mf = MagicMock()
        mf.nuc_grad_method.return_value.as_scanner.side_effect = RuntimeError(
            "no scanner"
        )

        # Fallback uses mf.copy().reset(m).kernel() and grad_method(mf_scan).kernel()
        g_zero = np.zeros((n, 3))
        mf.copy.return_value.kernel.return_value = 0.0
        mf.nuc_grad_method.return_value.return_value.kernel.return_value = g_zero

        hess = w.compute_numeric_hessian(mf, mol)
        self.assertEqual(hess.shape, (n, 3, n, 3))


# ===========================================================================
# atom_coords(unit='Bohr') fallback (lines 204-206)
# ===========================================================================


class TestNumericHessianAtomCoordsFallback(unittest.TestCase):
    """
    If mol.atom_coords(unit='Bohr') raises (older PySCF), the code falls back
    to mol.atom_coords() * 1.8897... (lines 204-206).
    The hessian must still have valid shape.
    """

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_worker_mod(MagicMock())

    def test_fallback_when_unit_bohr_raises(self):
        w = _make_worker(self.mod)
        n = 2
        mol = _make_mock_mol(n_atoms=n)
        mf = _make_mock_mf(n_atoms=n)

        # Make atom_coords raise only when called with unit='Bohr'
        coords = np.zeros((n, 3))
        coords[1, 0] = 1.0

        def _atom_coords_side_effect(*args, **kwargs):
            if kwargs.get("unit") == "Bohr":
                raise TypeError("unit kwarg not supported")
            return coords

        mol.atom_coords.side_effect = _atom_coords_side_effect

        hess = w.compute_numeric_hessian(mf, mol)
        self.assertEqual(hess.shape, (n, 3, n, 3))

    def test_fallback_hessian_is_symmetric(self):
        w = _make_worker(self.mod)
        n = 2
        mol = _make_mock_mol(n_atoms=n)
        mf = _make_mock_mf(n_atoms=n)

        coords = np.zeros((n, 3))

        def _raise_on_bohr(*args, **kwargs):
            if kwargs.get("unit") == "Bohr":
                raise TypeError("no unit kwarg")
            return coords

        mol.atom_coords.side_effect = _raise_on_bohr

        hess = w.compute_numeric_hessian(mf, mol)
        np.testing.assert_allclose(hess, hess.transpose(2, 3, 0, 1), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
