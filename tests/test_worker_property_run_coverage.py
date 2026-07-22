"""
tests/test_worker_property_run_coverage.py

Coverage for PropertyWorker.run()'s task-dispatch loop (worker.py
~1671-1911): ESP, SpinDensity (UHF and ROKS variants + skip), and MO
orbital-index parsing (HOMO/LUMO relative labels, "MO n", "#n", spin
suffixes, out-of-range guard, parse-error guard).

This loop hits `from .utils import get_unique_path`, a package-relative
import. worker.py is loaded standalone (like the other worker test files)
so it has no real parent package; we give it one by registering a
throwaway fake package name as `__package__` and pre-loading the real
utils.py under that name — this avoids importing the real
`pyscf_calculator` package (which pulls in gui.py/vis_tab.py/pyvista and
would be heavy/unsafe under these stubs), and avoids polluting the real
`pyscf_calculator` entry in sys.modules for other test files.
"""

import os
import sys
import types
import tempfile
import unittest
import importlib.util
import numpy as np
from unittest.mock import MagicMock, patch


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

_FAKE_PKG = "_fake_pyscf_calculator_pkg_for_property_worker_test"


def _load_worker_mod():
    _install_stubs(force=True)
    pyscf_mock = MagicMock()
    sys.modules["pyscf"] = pyscf_mock
    for sub in ("pyscf.gto", "pyscf.scf", "pyscf.dft", "pyscf.solvent"):
        sys.modules[sub] = MagicMock()

    base_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "pyscf_calculator")
    )

    # Lightweight fake parent package (NOT the real "pyscf_calculator" name)
    # so `from .utils import get_unique_path` resolves without importing
    # the real package's heavy __init__.py (gui.py -> vis_tab.py -> pyvista).
    pkg_mod = types.ModuleType(_FAKE_PKG)
    pkg_mod.__path__ = [base_dir]
    sys.modules[_FAKE_PKG] = pkg_mod

    utils_src = os.path.join(base_dir, "utils.py")
    uspec = importlib.util.spec_from_file_location(f"{_FAKE_PKG}.utils", utils_src)
    umod = importlib.util.module_from_spec(uspec)
    sys.modules[f"{_FAKE_PKG}.utils"] = umod
    uspec.loader.exec_module(umod)

    worker_src = os.path.join(base_dir, "worker.py")
    module_name = f"{_FAKE_PKG}.worker"
    wspec = importlib.util.spec_from_file_location(module_name, worker_src)
    wmod = importlib.util.module_from_spec(wspec)
    sys.modules[module_name] = wmod
    wspec.loader.exec_module(wmod)
    return wmod


_mod = _load_worker_mod()


def _make_property_worker(tasks, out_dir, chkfile="/fake/run.chk"):
    pw = _mod.PropertyWorker.__new__(_mod.PropertyWorker)
    _mod.PropertyWorker.__init__(pw, chkfile, tasks, out_dir)
    pw.log_signal = MagicMock()
    pw.error_signal = MagicMock()
    pw.finished_signal = MagicMock()
    pw.result_signal = MagicMock()
    return pw


def _mock_pyscf_checkpoint(mo_occ, mo_coeff=None):
    if mo_coeff is None:
        mo_coeff = np.eye(4)
    mock_mol = MagicMock()
    mock_mol.output = None
    mock_mol.verbose = 4
    _mod.pyscf.lib.chkfile.load_mol.return_value = mock_mol
    _mod.pyscf.scf.chkfile.load.return_value = {"mo_coeff": mo_coeff, "mo_occ": mo_occ}
    return mock_mol


def _run_with_tasks(tasks, mo_occ, mo_coeff=None, out_dir=None):
    if out_dir is None:
        out_dir = tempfile.mkdtemp()
    _mock_pyscf_checkpoint(mo_occ, mo_coeff)
    pw = _make_property_worker(tasks, out_dir)

    # PropertyWorker.run() does `from pyscf import lib, scf, tools` at call
    # time — a *runtime* lookup against sys.modules, independent of the
    # `pyscf` name this module bound at import time. Other worker test files
    # reassign sys.modules["pyscf"] in their own setUpClass; re-pin it here
    # right before run() so the checkpoint mocks configured above are the
    # ones actually seen (regression for a cross-test-file mock leak).
    sys.modules["pyscf"] = _mod.pyscf

    with patch.object(_mod, "CaptureStdOut") as mock_cap:
        mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_cap.return_value.__exit__ = MagicMock(return_value=False)
        pw.run()
    return pw, out_dir


# ===========================================================================
# 1. ESP task
# ===========================================================================


class TestEspTask(unittest.TestCase):
    def test_rhf_esp_generates_two_cube_files(self):
        mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
        pw, out_dir = _run_with_tasks(["ESP"], mo_occ)
        pw.finished_signal.emit.assert_called_once()
        pw.error_signal.emit.assert_not_called()
        result = pw.result_signal.emit.call_args[0][0]
        self.assertEqual(len(result["files"]), 2)
        self.assertTrue(any("esp" in f for f in result["files"]))
        self.assertTrue(any("density" in f for f in result["files"]))

    def test_uhf_esp_uses_uhf_make_rdm1(self):
        alpha = np.array([1.0, 0.0])
        beta = np.array([1.0, 0.0])
        mo_coeff = (np.eye(2), np.eye(2))
        pw, out_dir = _run_with_tasks(["ESP"], (alpha, beta), mo_coeff=mo_coeff)
        pw.finished_signal.emit.assert_called_once()
        pw.error_signal.emit.assert_not_called()
        result = pw.result_signal.emit.call_args[0][0]
        self.assertEqual(len(result["files"]), 2)


# ===========================================================================
# 2. SpinDensity task
# ===========================================================================


class TestSpinDensityTask(unittest.TestCase):
    def test_uhf_spin_density_generates_one_cube(self):
        alpha = np.array([1.0, 0.0])
        beta = np.array([1.0, 0.0])
        mo_coeff = (np.eye(2), np.eye(2))
        pw, out_dir = _run_with_tasks(["SpinDensity"], (alpha, beta), mo_coeff=mo_coeff)
        pw.finished_signal.emit.assert_called_once()
        result = pw.result_signal.emit.call_args[0][0]
        self.assertEqual(len(result["files"]), 1)
        self.assertIn("spin_density", result["files"][0])

    def test_roks_spin_density_generates_one_cube(self):
        mo_occ = np.array([[2.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
        pw, out_dir = _run_with_tasks(["SpinDensity"], mo_occ)
        pw.finished_signal.emit.assert_called_once()
        result = pw.result_signal.emit.call_args[0][0]
        self.assertEqual(len(result["files"]), 1)

    def test_rhf_spin_density_skipped(self):
        """Closed-shell RHF has no spin density -> skip with log message."""
        mo_occ = np.array([2.0, 2.0, 0.0])
        pw, out_dir = _run_with_tasks(["SpinDensity"], mo_occ)
        pw.finished_signal.emit.assert_called_once()
        result = pw.result_signal.emit.call_args[0][0]
        self.assertEqual(result["files"], [])
        all_logs = " ".join(str(c) for c in pw.log_signal.emit.call_args_list)
        self.assertIn("Skipping Spin Density", all_logs)


# ===========================================================================
# 3. MO orbital index parsing
# ===========================================================================


class TestMoOrbitalParsing(unittest.TestCase):
    def test_homo_label(self):
        mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
        pw, out_dir = _run_with_tasks(["HOMO"], mo_occ)
        pw.finished_signal.emit.assert_called_once()
        result = pw.result_signal.emit.call_args[0][0]
        self.assertEqual(len(result["files"]), 1)
        self.assertIn("HOMO", result["files"][0])

    def test_lumo_plus_offset(self):
        mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
        pw, out_dir = _run_with_tasks(["LUMO+1"], mo_occ)
        result = pw.result_signal.emit.call_args[0][0]
        self.assertEqual(len(result["files"]), 1)

    def test_homo_minus_offset(self):
        mo_occ = np.array([2.0, 2.0, 2.0, 0.0])
        pw, out_dir = _run_with_tasks(["HOMO-1"], mo_occ)
        result = pw.result_signal.emit.call_args[0][0]
        self.assertEqual(len(result["files"]), 1)

    def test_explicit_mo_number_one_based(self):
        mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
        pw, out_dir = _run_with_tasks(["MO 2"], mo_occ)
        result = pw.result_signal.emit.call_args[0][0]
        self.assertEqual(len(result["files"]), 1)

    def test_hash_index_zero_based(self):
        mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
        pw, out_dir = _run_with_tasks(["#1"], mo_occ)
        result = pw.result_signal.emit.call_args[0][0]
        self.assertEqual(len(result["files"]), 1)

    def test_mo_label_with_index_regex(self):
        mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
        pw, out_dir = _run_with_tasks(["MO 2_HOMO"], mo_occ)
        result = pw.result_signal.emit.call_args[0][0]
        self.assertEqual(len(result["files"]), 1)

    def test_out_of_range_index_logged_and_skipped(self):
        mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
        pw, out_dir = _run_with_tasks(["MO 99"], mo_occ)
        pw.finished_signal.emit.assert_called_once()
        result = pw.result_signal.emit.call_args[0][0]
        self.assertEqual(result["files"], [])
        all_logs = " ".join(str(c) for c in pw.log_signal.emit.call_args_list)
        self.assertIn("out of bounds", all_logs)

    def test_unparseable_task_logged_and_skipped(self):
        mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
        pw, out_dir = _run_with_tasks(["???"], mo_occ)
        pw.finished_signal.emit.assert_called_once()
        result = pw.result_signal.emit.call_args[0][0]
        self.assertEqual(result["files"], [])

    def test_uhf_alpha_beta_suffix(self):
        alpha = np.array([1.0, 0.0])
        beta = np.array([1.0, 0.0])
        mo_coeff = (np.eye(2), np.eye(2))
        pw, out_dir = _run_with_tasks(["HOMO_B"], (alpha, beta), mo_coeff=mo_coeff)
        result = pw.result_signal.emit.call_args[0][0]
        self.assertEqual(len(result["files"]), 1)
        self.assertIn("b_", result["files"][0])

    def test_multiple_tasks_all_processed(self):
        """Sanity check: two valid tasks in one run both produce files."""
        mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
        pw, out_dir = _run_with_tasks(["HOMO", "LUMO"], mo_occ)
        pw.finished_signal.emit.assert_called_once()
        result = pw.result_signal.emit.call_args[0][0]
        self.assertEqual(len(result["files"]), 2)


if __name__ == "__main__":
    unittest.main()
