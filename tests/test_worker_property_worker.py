"""
tests/test_worker_property_worker.py

Tests for PropertyWorker.run() — specifically the HOMO/LUMO detection logic
and the tasks dispatch loop.  All PySCF calls are mocked so no real quantum
chemistry is performed.

Coverage targets (worker.py):
  - pyscf=None guard (line 1742-1744)
  - HOMO/LUMO detection: tuple (UHF), 2D ndarray (ROKS), 1D ndarray (RHF),
    list fallback (lines 1786-1820)
  - lumo_idx == -1 guard: set to homo_idx + 1 (line 1823)
  - Empty tasks list → finished_signal emitted with {"files": []}
  - stop_requested during task loop → breaks early
"""

import os
import sys
import types
import tempfile
import unittest
import importlib.util
import numpy as np
from unittest.mock import MagicMock, patch


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
    module_name = f"_worker_prop_{id(pyscf_mock)}"
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


def _make_property_worker(mod, tasks, out_dir, chkfile="/fake/run.chk"):
    pw = mod.PropertyWorker.__new__(mod.PropertyWorker)
    mod.PropertyWorker.__init__(pw, chkfile, tasks, out_dir)
    pw.log_signal = MagicMock()
    pw.error_signal = MagicMock()
    pw.finished_signal = MagicMock()
    pw.result_signal = MagicMock()
    return pw


def _mock_pyscf_checkpoint(mod, mo_occ, mo_coeff=None):
    """
    Configure mod.pyscf so that `from pyscf import lib, scf, tools` inside
    PropertyWorker.run() returns useful mocks with *mo_occ* / *mo_coeff*.
    """
    if mo_coeff is None:
        mo_coeff = np.eye(4)  # simple 4×4 identity as dummy

    mock_mol = MagicMock()
    mock_mol.output = None
    mock_mol.verbose = 4

    mod.pyscf.lib.chkfile.load_mol.return_value = mock_mol
    mod.pyscf.scf.chkfile.load.return_value = {
        "mo_coeff": mo_coeff,
        "mo_occ": mo_occ,
    }
    return mock_mol


def _run_with_tasks(mod, tasks, mo_occ, out_dir=None):
    """Helper: run PropertyWorker with given tasks and mo_occ, return worker."""
    if out_dir is None:
        out_dir = tempfile.mkdtemp()

    _mock_pyscf_checkpoint(mod, mo_occ)
    pw = _make_property_worker(mod, tasks, out_dir)

    with patch.object(mod, "CaptureStdOut") as mock_cap:
        mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_cap.return_value.__exit__ = MagicMock(return_value=False)
        pw.run()

    return pw


# ===========================================================================
# 1. pyscf=None guard
# ===========================================================================


class TestPropertyWorkerNoPySCF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_worker_mod(None)

    def test_error_when_pyscf_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pw = _make_property_worker(self.mod, [], tmpdir)
            pw.run()
            pw.error_signal.emit.assert_called_once()

    def test_finished_not_emitted_when_pyscf_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pw = _make_property_worker(self.mod, [], tmpdir)
            pw.run()
            pw.finished_signal.emit.assert_not_called()


# ===========================================================================
# 2. Empty tasks → finished with {"files": []}
# ===========================================================================


class TestPropertyWorkerEmptyTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_worker_mod(MagicMock())
        cls.mod.pyscf = cls.mod.pyscf or MagicMock()

    def test_empty_tasks_emits_finished(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
            pw = _run_with_tasks(self.mod, [], mo_occ, out_dir=tmpdir)
            pw.finished_signal.emit.assert_called_once()

    def test_empty_tasks_result_has_files_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
            pw = _run_with_tasks(self.mod, [], mo_occ, out_dir=tmpdir)
            # results dict is emitted via result_signal; finished_signal takes no args
            result = pw.result_signal.emit.call_args[0][0]
            self.assertIn("files", result)
            self.assertEqual(result["files"], [])

    def test_empty_tasks_no_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
            pw = _run_with_tasks(self.mod, [], mo_occ, out_dir=tmpdir)
            pw.error_signal.emit.assert_not_called()


# ===========================================================================
# 3. HOMO/LUMO detection for different mo_occ formats
# ===========================================================================


class TestHomoLumoDetection(unittest.TestCase):
    """
    The HOMO/LUMO detection side-effect is visible through completed task
    execution.  We verify it by checking that the run completes (finished
    emitted, no error) for each mo_occ format.
    """

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_worker_mod(MagicMock())

    def _assert_completes(self, mo_occ, mo_coeff=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            if mo_coeff is not None:
                _mock_pyscf_checkpoint(self.mod, mo_occ, mo_coeff)
            pw = _run_with_tasks(self.mod, [], mo_occ, out_dir=tmpdir)
            pw.finished_signal.emit.assert_called_once()
            pw.error_signal.emit.assert_not_called()

    def test_rhf_1d_ndarray(self):
        """RHF: 1D numpy array — standard path."""
        self._assert_completes(np.array([2.0, 2.0, 0.0, 0.0]))

    def test_uhf_tuple(self):
        """UHF: tuple of (alpha_occ, beta_occ) arrays."""
        alpha = np.array([1.0, 1.0, 0.0])
        beta = np.array([1.0, 0.0, 0.0])
        mo_coeff = (np.eye(3), np.eye(3))
        _mock_pyscf_checkpoint(self.mod, (alpha, beta), mo_coeff)
        with tempfile.TemporaryDirectory() as tmpdir:
            pw = _run_with_tasks(self.mod, [], (alpha, beta), out_dir=tmpdir)
            pw.finished_signal.emit.assert_called_once()

    def test_roks_2d_ndarray(self):
        """ROKS: 2D numpy array (2, N)."""
        mo_occ = np.array([[2.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
        self._assert_completes(mo_occ)

    def test_list_fallback(self):
        """Fallback: plain Python list of occupation numbers."""
        self._assert_completes([2.0, 2.0, 0.0, 0.0])

    def test_all_occupied_lumo_guard(self):
        """All occupied: lumo_idx stays -1 → guard sets it to homo_idx + 1."""
        # All orbitals occupied → lumo_idx never set → guard kicks in
        self._assert_completes(np.array([2.0, 2.0, 2.0]))


# ===========================================================================
# 4. stop_requested during task loop
# ===========================================================================


class TestPropertyWorkerStop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_worker_mod(MagicMock())

    def test_stop_before_tasks_breaks_loop(self):
        """If _stop_requested is True before any task runs, loop exits early."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mo_occ = np.array([2.0, 0.0])
            _mock_pyscf_checkpoint(self.mod, mo_occ)
            pw = _make_property_worker(self.mod, ["ESP"], tmpdir)
            pw._stop_requested = True

            with patch.object(self.mod, "CaptureStdOut") as mock_cap:
                mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
                mock_cap.return_value.__exit__ = MagicMock(return_value=False)
                pw.run()

            # finished_signal is still emitted (loop was just skipped)
            pw.finished_signal.emit.assert_called_once()
            # No error
            pw.error_signal.emit.assert_not_called()
            # The result must have empty files (ESP was skipped)
            result = pw.result_signal.emit.call_args[0][0]
            self.assertEqual(result["files"], [])


if __name__ == "__main__":
    unittest.main()
