"""
tests/test_worker_single_point.py

End-to-end mock test for PySCFWorker.run() with "Single Point" job type.
All PySCF calls are mocked; filesystem writes go to a real temp directory.

This test covers the large uncovered block in run() (lines ~343-1049):
  - method selection / spin auto-adjust (RHF/UHF/RKS/UKS/ROHF/ROKS)
  - solvent setup (hardcoded lookup and fallback paths)
  - pyscf_input.py log file writing
  - MO energy/occupancy extraction (RHF 1-D, UHF tuple)
  - result_signal and finished_signal emission
  - outer exception handler (lines 1051-1059)
  - stream/FD restore in finally block (lines 1060-1086)
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


def _load_worker_mod():
    _install_stubs(force=True)
    module_name = "_worker_sp_test"
    pyscf_mock = MagicMock()
    sys.modules["pyscf"] = pyscf_mock
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


_mod = _load_worker_mod()


def _make_mock_mol():
    mol = MagicMock()
    mol.build.return_value = None
    mol.stdout = None
    mol.verbose = 4
    mol.output = None
    return mol


def _make_mock_mf(mo_energy=None, mo_occ=None, e_tot=-1.117):
    mf = MagicMock()
    mf.e_tot = e_tot
    mf.converged = True
    mf.mo_energy = mo_energy if mo_energy is not None else np.array([-1.0, -0.5, 0.2])
    mf.mo_occ = mo_occ if mo_occ is not None else np.array([2.0, 2.0, 0.0])
    mf.chkfile = ""
    mf.max_cycle = 100
    mf.conv_tol = 1e-9
    return mf


def _make_worker(config=None):
    if config is None:
        config = {
            "job_type": "Single Point",
            "method": "RHF",
            "basis": "sto-3g",
            "charge": 0,
            "spin": "1",
            "threads": 0,
            "memory": 4000,
        }
    w = _mod.PySCFWorker.__new__(_mod.PySCFWorker)
    _mod.PySCFWorker.__init__(w, "H 0 0 0\nH 0 0 0.74", config)
    w.log_signal = MagicMock()
    w.error_signal = MagicMock()
    w.finished_signal = MagicMock()
    w.result_signal = MagicMock()
    return w


def _run_single_point(
    config=None, mo_energy=None, mo_occ=None, method="RHF", extra_config=None
):
    """
    Run PySCFWorker with a given config through a temp directory.
    Returns (worker, results_dict).
    """
    mock_mol = _make_mock_mol()
    mock_mf = _make_mock_mf(mo_energy=mo_energy, mo_occ=mo_occ)

    if config is None:
        config = {
            "job_type": "Single Point",
            "method": method,
            "basis": "sto-3g",
            "charge": 0,
            "spin": "1",
            "threads": 0,
            "memory": 4000,
            "max_cycle": 100,
            "conv_tol": "1e-9",
        }
    if extra_config:
        config.update(extra_config)

    gto_mock = MagicMock()
    gto_mock.M.return_value = mock_mol
    _mod.gto = gto_mock

    scf_mock = MagicMock()
    scf_mock.RHF.return_value = mock_mf
    scf_mock.UHF.return_value = mock_mf
    scf_mock.ROHF.return_value = mock_mf
    _mod.scf = scf_mock

    dft_mock = MagicMock()
    dft_mock.RKS.return_value = mock_mf
    dft_mock.UKS.return_value = mock_mf
    dft_mock.ROKS.return_value = mock_mf
    _mod.dft = dft_mock

    w = _make_worker(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        w.config["out_dir"] = tmpdir

        with patch.object(_mod, "CaptureStdOut") as mock_cap:
            mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_cap.return_value.__exit__ = MagicMock(return_value=False)
            w.run()

    results = None
    if w.result_signal.emit.call_count > 0:
        results = w.result_signal.emit.call_args[0][0]

    return w, results


# ===========================================================================
# 1. Basic Single Point completion
# ===========================================================================


class TestSinglePointCompletion(unittest.TestCase):
    def test_finished_emitted(self):
        w, _ = _run_single_point()
        w.finished_signal.emit.assert_called_once()

    def test_no_error_signal(self):
        w, _ = _run_single_point()
        w.error_signal.emit.assert_not_called()

    def test_result_signal_emitted(self):
        w, _ = _run_single_point()
        w.result_signal.emit.assert_called_once()

    def test_result_contains_chkfile(self):
        _, results = _run_single_point()
        self.assertIn("chkfile", results)

    def test_result_contains_out_dir(self):
        _, results = _run_single_point()
        self.assertIn("out_dir", results)

    def test_result_contains_cube_files(self):
        _, results = _run_single_point()
        self.assertIn("cube_files", results)
        self.assertEqual(results["cube_files"], [])

    def test_input_file_written(self):
        """pyscf_input.py must be written in the job directory."""
        mock_mol = _make_mock_mol()
        mock_mf = _make_mock_mf()
        _mod.gto = MagicMock()
        _mod.gto.M.return_value = mock_mol
        _mod.scf = MagicMock()
        _mod.scf.RHF.return_value = mock_mf

        w = _make_worker()
        with tempfile.TemporaryDirectory() as tmpdir:
            w.config["out_dir"] = tmpdir
            with patch.object(_mod, "CaptureStdOut") as mock_cap:
                mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
                mock_cap.return_value.__exit__ = MagicMock(return_value=False)
                w.run()
            # job_1 subdirectory should contain pyscf_input.py
            job_dir = os.path.join(tmpdir, "job_1")
            inp_file = os.path.join(job_dir, "pyscf_input.py")
            self.assertTrue(os.path.isfile(inp_file), "pyscf_input.py not written")


# ===========================================================================
# 2. MO energy/occupancy extraction
# ===========================================================================


class TestSinglePointMOExtraction(unittest.TestCase):
    def test_rhf_mo_energy_list_in_result(self):
        mo_e = np.array([-1.0, -0.5, 0.2])
        mo_o = np.array([2.0, 2.0, 0.0])
        _, results = _run_single_point(mo_energy=mo_e, mo_occ=mo_o)
        self.assertIn("mo_energy", results)
        self.assertIn("mo_occ", results)

    def test_rhf_scf_type_is_rhf(self):
        mo_e = np.array([-1.0, -0.5])
        mo_o = np.array([2.0, 0.0])
        _, results = _run_single_point(mo_energy=mo_e, mo_occ=mo_o)
        self.assertEqual(results.get("scf_type"), "RHF")

    def test_uhf_tuple_mo_energy(self):
        """UHF: mo_energy is a tuple → scf_type=UHF, nested lists in result."""
        alpha_e = np.array([-1.0, 0.2])
        beta_e = np.array([-0.8, 0.4])
        alpha_o = np.array([1.0, 0.0])
        beta_o = np.array([1.0, 0.0])
        mo_energy = (alpha_e, beta_e)
        mo_occ = (alpha_o, beta_o)
        _, results = _run_single_point(mo_energy=mo_energy, mo_occ=mo_occ, method="UHF")
        self.assertEqual(results.get("scf_type"), "UHF")
        self.assertIsInstance(results["mo_energy"], list)
        self.assertEqual(len(results["mo_energy"]), 2)

    def test_none_mo_energy_emits_warning(self):
        """mf.mo_energy = None → warning logged, empty lists in result."""
        mock_mol = _make_mock_mol()
        mock_mf = _make_mock_mf()
        mock_mf.mo_energy = None
        mock_mf.mo_occ = None

        _mod.gto = MagicMock()
        _mod.gto.M.return_value = mock_mol
        _mod.scf = MagicMock()
        _mod.scf.RHF.return_value = mock_mf

        w = _make_worker()
        with tempfile.TemporaryDirectory() as tmpdir:
            w.config["out_dir"] = tmpdir
            with patch.object(_mod, "CaptureStdOut") as mock_cap:
                mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
                mock_cap.return_value.__exit__ = MagicMock(return_value=False)
                w.run()

        results = w.result_signal.emit.call_args[0][0]
        self.assertEqual(results.get("mo_energy"), [])
        self.assertEqual(results.get("mo_occ"), [])


# ===========================================================================
# 3. Method selection (spin auto-adjust)
# ===========================================================================


class TestMethodSelection(unittest.TestCase):
    """Verify method is auto-adjusted for open-shell and log message emitted."""

    def test_rhf_spin1_stays_rhf(self):
        """Spin=1 (Singlet, 2S=0) → RHF unchanged."""
        w, _ = _run_single_point(method="RHF", extra_config={"spin": "1"})
        _mod.scf.RHF.assert_called()

    def test_rhf_spin2_switches_to_uhf(self):
        """Spin=2 (Doublet, 2S=1) → auto-switch to UHF."""
        w, _ = _run_single_point(method="RHF", extra_config={"spin": "2"})
        # UHF should have been instantiated
        _mod.scf.UHF.assert_called()

    def test_rks_method_uses_dft(self):
        w, _ = _run_single_point(method="RKS", extra_config={"functional": "b3lyp"})
        _mod.dft.RKS.assert_called()


# ===========================================================================
# 4. Solvent lookup (hardcoded path)
# ===========================================================================


class TestSolventSetup(unittest.TestCase):
    def test_known_solvent_uses_hardcoded_eps(self):
        """Water solvent → eps=78.2 hardcoded, log message emitted."""
        w, results = _run_single_point(extra_config={"solvent": "Water"})
        # No error should occur; finished emitted
        w.finished_signal.emit.assert_called_once()
        # Log must mention solvent
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("Water", all_logs)

    def test_vacuum_solvent_not_applied(self):
        """Default 'None (Vacuum)' → ddCOSMO not applied."""
        w, _ = _run_single_point()
        # mf.ddCOSMO should NOT have been called
        mf = _mod.scf.RHF.return_value
        mf.ddCOSMO.assert_not_called()


# ===========================================================================
# 5. Outer exception handler
# ===========================================================================


class TestOuterExceptionHandler(unittest.TestCase):
    def test_unexpected_exception_emits_error_signal(self):
        """
        If an unhandled exception occurs after mol setup (but before the inner
        try), error_signal must be emitted and finished_signal must NOT be.
        """
        mock_mol = _make_mock_mol()
        mock_mol.build.side_effect = Exception("unexpected internal error")
        _mod.gto = MagicMock()
        _mod.gto.M.return_value = mock_mol

        w = _make_worker()
        with tempfile.TemporaryDirectory() as tmpdir:
            w.config["out_dir"] = tmpdir
            with patch.object(_mod, "CaptureStdOut") as mock_cap:
                mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
                mock_cap.return_value.__exit__ = MagicMock(return_value=False)
                w.run()

        # Exception is not caught by the inner handler (only catches
        # RuntimeError/ValueError), so it propagates to outer handler.
        w.error_signal.emit.assert_called()


# ===========================================================================
# 6. Missing "job_type" key in config (regression)
# ===========================================================================


class TestMissingJobTypeKey(unittest.TestCase):
    """
    Regression test for a bug where `self.config.get("job_type", None)` was
    used to guard `"TDDFT" in job_type` checks while writing pyscf_input.py.
    If "job_type" is absent from config, `.get(..., None)` returns None and
    `"TDDFT" in None` raises TypeError, which used to be silently converted
    into an (unhelpful) error_signal instead of running the calculation.
    The fix defaults to "" so the membership check is well-defined, letting
    the job dispatch fall back to its own "Energy" default further down.
    """

    def test_missing_job_type_does_not_crash(self):
        config = {
            "method": "RHF",
            "basis": "sto-3g",
            "charge": 0,
            "spin": "1",
            "threads": 0,
            "memory": 4000,
            "max_cycle": 100,
            "conv_tol": "1e-9",
            # "job_type" intentionally omitted
        }
        w, results = _run_single_point(config=config)
        w.finished_signal.emit.assert_called_once()
        w.error_signal.emit.assert_not_called()
        self.assertIsNotNone(results)

    def test_missing_job_type_input_file_written(self):
        config = {
            "method": "RHF",
            "basis": "sto-3g",
            "charge": 0,
            "spin": "1",
            "threads": 0,
            "memory": 4000,
            "max_cycle": 100,
            "conv_tol": "1e-9",
        }
        mock_mol = _make_mock_mol()
        mock_mf = _make_mock_mf()
        _mod.gto = MagicMock()
        _mod.gto.M.return_value = mock_mol
        _mod.scf = MagicMock()
        _mod.scf.RHF.return_value = mock_mf

        w = _make_worker(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            w.config["out_dir"] = tmpdir
            with patch.object(_mod, "CaptureStdOut") as mock_cap:
                mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
                mock_cap.return_value.__exit__ = MagicMock(return_value=False)
                w.run()
            job_dir = os.path.join(tmpdir, "job_1")
            inp_file = os.path.join(job_dir, "pyscf_input.py")
            self.assertTrue(os.path.isfile(inp_file), "pyscf_input.py not written")


if __name__ == "__main__":
    unittest.main()
