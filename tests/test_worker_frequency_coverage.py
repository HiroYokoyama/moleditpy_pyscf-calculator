"""
tests/test_worker_frequency_coverage.py

Coverage for PySCFWorker.run() "Frequency" job type (worker.py ~735-887):
  - Successful Hessian + harmonic_analysis + thermo, real-valued and complex
    (imaginary) frequencies, JSON persistence
  - Solvent-present skip path (no analytic/numeric fallback attempted)
  - Hessian raising an exception -> logged, calculation continues
  - Freq JSON save failure -> warning logged, no crash
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
    module_name = "_worker_freq_test"
    pyscf_mock = MagicMock()
    sys.modules["pyscf"] = pyscf_mock
    for sub in ("pyscf.gto", "pyscf.scf", "pyscf.dft", "pyscf.solvent"):
        sys.modules[sub] = MagicMock()
    sys.modules["pyscf.hessian"] = MagicMock()
    sys.modules["pyscf.hessian.thermo"] = MagicMock()
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
    def __init__(self, e_tot=-1.117, converged=True):
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
        self._hessian_obj = MagicMock()

    def kernel(self, dm0=None):
        self.kernel_calls.append(dm0)
        self.e_tot = -1.117
        return self.e_tot

    def Hessian(self):
        return self._hessian_obj


def _make_mock_mol():
    mol = MagicMock()
    mol.build.return_value = None
    mol.stdout = None
    mol.verbose = 4
    mol.output = None
    mol.natm = 2
    return mol


def _base_config(extra=None):
    config = {
        "job_type": "Frequency",
        "method": "RHF",
        "basis": "sto-3g",
        "charge": 0,
        "spin": "1",
        "threads": 0,
        "memory": 4000,
        "max_cycle": 100,
        "conv_tol": "1e-9",
        "temperature": 298.15,
        "pressure": 101325,
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


def _run(config, fake_mf, thermo_mock=None):
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

    if thermo_mock is not None:
        sys.modules["pyscf.hessian.thermo"] = thermo_mock
        # `from pyscf.hessian import thermo` resolves via getattr() on the
        # parent module first; since pyscf.hessian is itself a MagicMock its
        # attribute auto-vivifies, so it must be set explicitly too.
        sys.modules["pyscf.hessian"].thermo = thermo_mock

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
    return w, results, out_dir


def _make_thermo_mock(freqs, intensities=None):
    thermo_mock = MagicMock()
    thermo_mock.harmonic_analysis.return_value = {
        "freq_au": np.array([0.01] * len(freqs)),
        "freq_wavenumber": np.array(freqs) if not any(isinstance(f, complex) for f in freqs) else freqs,
        "norm_mode": np.zeros((len(freqs), 2, 3)),
        "infra_red_intensity": np.array(intensities) if intensities is not None else None,
    }
    thermo_mock.thermo.return_value = {
        "E_tot": (1.0, "Eh"),
        "H_tot": (1.1, "Eh"),
        "G_tot": (0.9, "Eh"),
    }
    return thermo_mock


class TestFrequencySuccess(unittest.TestCase):
    def test_real_frequencies_stored(self):
        thermo_mock = _make_thermo_mock([100.0, 200.0, 300.0], intensities=[1.0, 2.0, 3.0])
        w, results, out_dir = _run(_base_config(), FakeMF(), thermo_mock)
        w.finished_signal.emit.assert_called_once()
        self.assertIn("freq_data", results)
        self.assertEqual(results["freq_data"]["freqs"], [100.0, 200.0, 300.0])
        self.assertIn("thermo_data", results)

    def test_json_persisted_to_out_dir(self):
        thermo_mock = _make_thermo_mock([100.0, 200.0])
        w, results, out_dir = _run(_base_config(), FakeMF(), thermo_mock)
        json_path = os.path.join(out_dir, "freq_analysis.json")
        self.assertTrue(os.path.isfile(json_path))
        with open(json_path) as f:
            data = json.load(f)
        self.assertIn("freq_data", data)

    def test_complex_imaginary_frequency_becomes_negative_real(self):
        thermo_mock = _make_thermo_mock([complex(0, 50.0), 100.0])
        w, results, out_dir = _run(_base_config(), FakeMF(), thermo_mock)
        self.assertEqual(results["freq_data"]["freqs"][0], -50.0)

    def test_complex_zero_imag_uses_real_part(self):
        thermo_mock = _make_thermo_mock([complex(120.0, 0.0)])
        w, results, out_dir = _run(_base_config(), FakeMF(), thermo_mock)
        self.assertEqual(results["freq_data"]["freqs"][0], 120.0)

    def test_scf_runs_when_not_converged_yet(self):
        thermo_mock = _make_thermo_mock([100.0])
        fake_mf = FakeMF(e_tot=None)
        w, results, out_dir = _run(_base_config(), fake_mf, thermo_mock)
        self.assertEqual(len(fake_mf.kernel_calls), 1)

    def test_not_converged_warning_logged(self):
        thermo_mock = _make_thermo_mock([100.0])
        fake_mf = FakeMF(converged=False)
        w, results, out_dir = _run(_base_config(), fake_mf, thermo_mock)
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("did not converge", all_logs)


class TestFrequencySolventSkip(unittest.TestCase):
    def test_solvent_present_skips_frequency(self):
        thermo_mock = _make_thermo_mock([100.0])
        w, results, out_dir = _run(
            _base_config(extra={"solvent": "Water"}), FakeMF(), thermo_mock
        )
        w.finished_signal.emit.assert_called_once()
        w.error_signal.emit.assert_not_called()
        self.assertNotIn("freq_data", results or {})
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("Solvent Not Supported", all_logs)


class TestFrequencyHessianFailure(unittest.TestCase):
    def test_hessian_kernel_exception_logged_not_fatal(self):
        fake_mf = FakeMF()
        fake_mf._hessian_obj.kernel.side_effect = RuntimeError("hessian blew up")
        w, results, out_dir = _run(_base_config(), fake_mf)
        w.finished_signal.emit.assert_called_once()
        w.error_signal.emit.assert_not_called()
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("Frequency analysis failed", all_logs)
        self.assertNotIn("freq_data", results or {})


class TestFrequencyJsonSaveFailure(unittest.TestCase):
    def test_json_dump_failure_logged_as_warning(self):
        thermo_mock = _make_thermo_mock([100.0])
        with patch.object(_mod.json, "dump", side_effect=OSError("disk full")):
            w, results, out_dir = _run(_base_config(), FakeMF(), thermo_mock)
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("Failed to save frequency JSON", all_logs)
        w.finished_signal.emit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
