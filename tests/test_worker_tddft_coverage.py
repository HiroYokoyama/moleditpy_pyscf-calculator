"""
tests/test_worker_tddft_coverage.py

Coverage for PySCFWorker.run() "TDDFT" job type (worker.py ~888-1030):
  - RKS/UKS dispatch -> tdscf.TDDFT; RHF/UHF -> tdscf.TDHF
  - Excitation energy / wavelength / oscillator strength extraction
  - oscillator_strength() raising -> fallback to zeros
  - Text + JSON result persistence (success and save-failure warnings)
  - SCF-not-converged warning path
"""

import os
import sys
import types
import tempfile
import unittest
import importlib.util
import json
import numpy as np
from unittest.mock import MagicMock, patch


def _install_stubs(force=False):
    qt_core = types.ModuleType("PyQt6.QtCore")

    class _QThread:
        def __init__(self):
            pass

        @staticmethod
        def msleep(ms):
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
    module_name = "_worker_tddft_test"
    pyscf_mock = MagicMock()
    sys.modules["pyscf"] = pyscf_mock
    for sub in ("pyscf.gto", "pyscf.scf", "pyscf.dft", "pyscf.solvent", "pyscf.tdscf"):
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


class FakeMF:
    def __init__(self, e_tot=-1.0, converged=True):
        self.e_tot = e_tot
        self.converged = converged
        self.mo_energy = np.array([-1.0, -0.5, 0.2])
        self.mo_occ = np.array([2.0, 2.0, 0.0])
        self.chkfile = ""
        self.max_cycle = 100
        self.conv_tol = 1e-9
        self.xc = None
        self.grids = MagicMock()
        self.kernel_calls = []

    def kernel(self, dm0=None):
        self.kernel_calls.append(dm0)
        self.e_tot = -1.0
        return self.e_tot


def _make_mock_mol():
    mol = MagicMock()
    mol.build.return_value = None
    mol.stdout = None
    mol.verbose = 4
    mol.output = None
    mol.natm = 2
    return mol


def _base_config(method="RKS", extra=None):
    config = {
        "job_type": "TDDFT",
        "method": method,
        "functional": "b3lyp",
        "basis": "sto-3g",
        "charge": 0,
        "spin": "1",
        "threads": 0,
        "memory": 4000,
        "max_cycle": 100,
        "conv_tol": "1e-9",
        "nstates": 3,
    }
    if extra:
        config.update(extra)
    return config


def _make_worker(config):
    w = _mod.PySCFWorker.__new__(_mod.PySCFWorker)
    _mod.PySCFWorker.__init__(w, "H 0 0 0\nH 0 0 0.74", config)
    w.log_signal = MagicMock()
    w.error_signal = MagicMock()
    w.finished_signal = MagicMock()
    w.result_signal = MagicMock()
    return w


def _run(config, fake_mf, td_obj=None):
    mock_mol = _make_mock_mol()

    gto_mock = MagicMock()
    gto_mock.M.return_value = mock_mol
    _mod.gto = gto_mock

    scf_mock = MagicMock()
    scf_mock.RHF.return_value = fake_mf
    scf_mock.UHF.return_value = fake_mf
    _mod.scf = scf_mock

    dft_mock = MagicMock()
    dft_mock.RKS.return_value = fake_mf
    dft_mock.UKS.return_value = fake_mf
    _mod.dft = dft_mock

    tdscf_mock = MagicMock()
    if td_obj is not None:
        tdscf_mock.TDDFT.return_value = td_obj
        tdscf_mock.TDHF.return_value = td_obj
    sys.modules["pyscf.tdscf"] = tdscf_mock
    # `from pyscf import tdscf` resolves via getattr() on the pyscf module
    # first; since pyscf is itself a MagicMock, its attribute auto-vivifies,
    # so it must be pointed at our mock explicitly.
    sys.modules["pyscf"].tdscf = tdscf_mock

    w = _make_worker(config)

    tmpdir = tempfile.mkdtemp()
    w.config["out_dir"] = tmpdir
    with patch.object(_mod, "CaptureStdOut") as mock_cap:
        mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_cap.return_value.__exit__ = MagicMock(return_value=False)
        w.run()
    out_dir = os.path.join(tmpdir, "job_1")

    results = None
    if w.result_signal.emit.call_count > 0:
        results = w.result_signal.emit.call_args[0][0]
    return w, results, out_dir, tdscf_mock


def _make_td_obj(e_tot_excited, oscs=None, osc_raises=False):
    td = MagicMock()
    td.e_tot = np.array(e_tot_excited)
    if osc_raises:
        td.oscillator_strength.side_effect = RuntimeError("no osc")
    else:
        td.oscillator_strength.return_value = np.array(
            oscs if oscs is not None else [0.1] * len(e_tot_excited)
        )
    return td


class TestTddftDispatch(unittest.TestCase):
    def test_rks_uses_tddft_class(self):
        td = _make_td_obj([0.9, 0.8])
        w, results, out_dir, tdscf_mock = _run(_base_config(method="RKS"), FakeMF(), td)
        tdscf_mock.TDDFT.assert_called_once()
        tdscf_mock.TDHF.assert_not_called()
        w.finished_signal.emit.assert_called_once()
        w.error_signal.emit.assert_not_called()

    def test_rhf_uses_tdhf_class(self):
        td = _make_td_obj([0.9, 0.8])
        w, results, out_dir, tdscf_mock = _run(_base_config(method="RHF"), FakeMF(), td)
        tdscf_mock.TDHF.assert_called_once()
        tdscf_mock.TDDFT.assert_not_called()

    def test_scf_kernel_run_when_not_converged_yet(self):
        td = _make_td_obj([0.9])
        fake_mf = FakeMF(e_tot=None)
        w, results, out_dir, _ = _run(_base_config(), fake_mf, td)
        self.assertEqual(len(fake_mf.kernel_calls), 1)

    def test_not_converged_warning_logged(self):
        td = _make_td_obj([0.9])
        fake_mf = FakeMF(converged=False)
        w, results, out_dir, _ = _run(_base_config(), fake_mf, td)
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("did not converge", all_logs)


class TestTddftResults(unittest.TestCase):
    def test_results_contain_tddft_list(self):
        td = _make_td_obj([-0.9, -0.8], oscs=[0.05, 0.2])
        w, results, out_dir, _ = _run(_base_config(), FakeMF(e_tot=-1.0), td)
        self.assertIn("tddft_data", results)
        self.assertEqual(len(results["tddft_data"]), 2)
        first = results["tddft_data"][0]
        self.assertIn("excitation_energy_ev", first)
        self.assertIn("wavelength_nm", first)
        self.assertEqual(first["oscillator_strength"], 0.05)

    def test_oscillator_strength_exception_falls_back_to_zero(self):
        td = _make_td_obj([-0.9, -0.8], osc_raises=True)
        w, results, out_dir, _ = _run(_base_config(), FakeMF(e_tot=-1.0), td)
        for item in results["tddft_data"]:
            self.assertEqual(item["oscillator_strength"], 0.0)

    def test_near_zero_excitation_gives_infinite_wavelength(self):
        # e_ground == e_exc_tot -> exc_ev ~ 0 -> wavelength = inf
        td = _make_td_obj([-1.0], oscs=[0.0])
        w, results, out_dir, _ = _run(_base_config(), FakeMF(e_tot=-1.0), td)
        self.assertEqual(results["tddft_data"][0]["wavelength_nm"], float("inf"))

    def test_text_and_json_files_written(self):
        td = _make_td_obj([-0.9], oscs=[0.1])
        w, results, out_dir, _ = _run(_base_config(), FakeMF(e_tot=-1.0), td)
        txt_path = os.path.join(out_dir, "tddft_results.txt")
        json_path = os.path.join(out_dir, "tddft_results.json")
        self.assertTrue(os.path.isfile(txt_path))
        self.assertTrue(os.path.isfile(json_path))
        with open(json_path) as f:
            data = json.load(f)
        self.assertIn("tddft_data", data)

    def test_json_save_failure_logs_warning(self):
        td = _make_td_obj([-0.9], oscs=[0.1])
        with patch.object(_mod.json, "dump", side_effect=OSError("disk full")):
            w, results, out_dir, _ = _run(_base_config(), FakeMF(e_tot=-1.0), td)
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("Failed to save TDDFT JSON", all_logs)
        w.finished_signal.emit.assert_called_once()

    def test_kernel_exception_logged_not_fatal(self):
        td = MagicMock()
        td.kernel.side_effect = RuntimeError("TDDFT diverged")
        w, results, out_dir, _ = _run(_base_config(), FakeMF(e_tot=-1.0), td)
        w.finished_signal.emit.assert_called_once()
        w.error_signal.emit.assert_not_called()
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("TDDFT calculation failed", all_logs)
        self.assertNotIn("tddft_data", results or {})


if __name__ == "__main__":
    unittest.main()
