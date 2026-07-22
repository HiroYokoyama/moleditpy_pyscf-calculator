"""
tests/test_worker_load_worker_full_coverage.py

Coverage for LoadWorker.run()'s main checkpoint-loading path (worker.py
~2031-2197) — everything downstream of "Original checkpoint loading logic".
tests/test_worker_load_worker.py already covers the pyscf=None guard and the
auxiliary-only path (scan/tddft/freq present but no .chk file); this file
covers RHF/UHF/ROKS type detection from a loaded checkpoint, the post-process
merge of freq/scan/tddft side files, and the two stop_requested checkpoints.
"""

import json
import os
import sys
import types
import unittest
import tempfile
import importlib.util
import numpy as np
from unittest.mock import MagicMock


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
    module_name = "_worker_load_full_test"
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


def _make_load_worker(chkfile):
    lw = _mod.LoadWorker.__new__(_mod.LoadWorker)
    _mod.LoadWorker.__init__(lw, chkfile)
    lw.finished_signal = MagicMock()
    lw.error_signal = MagicMock()
    return lw


def _make_mol(natm=2):
    mol = MagicMock()
    mol.natm = natm
    mol.atom_coords.return_value = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    mol.atom_symbol.side_effect = lambda i: ["H", "H"][i]
    return mol


def _run_load(chkfile, mo_energy, mo_occ):
    mol = _make_mol()
    # `from pyscf import lib, scf` in LoadWorker.run() is a runtime lookup
    # against sys.modules — re-pin it to our configured mock right before
    # run() (other worker test files reassign sys.modules["pyscf"] too).
    sys.modules["pyscf"] = _mod.pyscf
    _mod.pyscf.lib.chkfile.load_mol.return_value = mol
    _mod.pyscf.scf.chkfile.load.return_value = {
        "mo_energy": mo_energy,
        "mo_occ": mo_occ,
    }
    lw = _make_load_worker(chkfile)
    lw.run()
    results = None
    if lw.finished_signal.emit.call_count > 0:
        results = lw.finished_signal.emit.call_args[0][0]
    return lw, results


class TestCheckpointScfTypeDetection(unittest.TestCase):
    def test_rhf_1d_arrays(self):
        chkfile = os.path.join(tempfile.mkdtemp(), "pyscf.chk")
        mo_e = np.array([-1.0, -0.5, 0.2])
        mo_o = np.array([2.0, 2.0, 0.0])
        lw, results = _run_load(chkfile, mo_e, mo_o)
        lw.error_signal.emit.assert_not_called()
        self.assertEqual(results["scf_type"], "RHF")
        self.assertIsInstance(results["mo_energy"], list)
        self.assertIn("loaded_xyz", results)
        self.assertIn("chkfile", results)

    def test_uhf_tuple_mo_energy(self):
        chkfile = os.path.join(tempfile.mkdtemp(), "pyscf.chk")
        alpha_e = np.array([-1.0, 0.2])
        beta_e = np.array([-0.8, 0.4])
        alpha_o = np.array([1.0, 0.0])
        beta_o = np.array([1.0, 0.0])
        lw, results = _run_load(chkfile, (alpha_e, beta_e), (alpha_o, beta_o))
        self.assertEqual(results["scf_type"], "UHF")
        self.assertIsInstance(results["mo_energy"], list)
        self.assertEqual(len(results["mo_energy"]), 2)
        self.assertIsInstance(results["mo_energy"][0], list)

    def test_uhf_list_of_ndarrays(self):
        """is_uhf via the `isinstance(mo_energy, list) and len==2` branch."""
        chkfile = os.path.join(tempfile.mkdtemp(), "pyscf.chk")
        mo_e = [np.array([-1.0, 0.2]), np.array([-0.8, 0.4])]
        mo_o = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
        lw, results = _run_load(chkfile, mo_e, mo_o)
        self.assertEqual(results["scf_type"], "UHF")

    def test_roks_2d_mo_occ_partial_occupancy(self):
        chkfile = os.path.join(tempfile.mkdtemp(), "pyscf.chk")
        mo_e = np.array([-1.0, -0.5, 0.2])
        mo_o = np.array([[2.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
        lw, results = _run_load(chkfile, mo_e, mo_o)
        self.assertEqual(results["scf_type"], "ROKS")

    def test_roks_list_of_lists_partial_occupancy(self):
        chkfile = os.path.join(tempfile.mkdtemp(), "pyscf.chk")
        mo_e = np.array([-1.0, -0.5, 0.2])
        mo_o = [[2.0, 1.0, 0.0], [2.0, 0.0, 0.0]]
        lw, results = _run_load(chkfile, mo_e, mo_o)
        self.assertEqual(results["scf_type"], "ROKS")

    def test_no_partial_occupancy_stays_rhf(self):
        """2D mo_occ but no value in (0.5, 1.5) -> stays RHF (no ROKS signature)."""
        chkfile = os.path.join(tempfile.mkdtemp(), "pyscf.chk")
        mo_e = np.array([-1.0, -0.5, 0.2])
        mo_o = np.array([[2.0, 2.0, 0.0], [2.0, 2.0, 0.0]])
        lw, results = _run_load(chkfile, mo_e, mo_o)
        self.assertEqual(results["scf_type"], "RHF")


class TestCheckpointPostProcessMerge(unittest.TestCase):
    def test_freq_scan_tddft_all_merged(self):
        base_dir = tempfile.mkdtemp()
        chkfile = os.path.join(base_dir, "pyscf.chk")
        # Must exist on disk: if absent, the has_scan/has_tddft/has_freq guard
        # at the top of run() diverts to the aux-only path instead of the
        # "Original checkpoint loading logic" this test targets.
        with open(chkfile, "wb") as f:
            f.write(b"")

        with open(os.path.join(base_dir, "freq_analysis.json"), "w") as f:
            json.dump({"freq_data": {"freqs": [100.0]}, "thermo_data": {"E_tot": [1.0]}}, f)

        with open(os.path.join(base_dir, "scan_results.csv"), "w") as f:
            f.write("Step,Value,Energy\n1,0.70,-1.05\n")
        with open(os.path.join(base_dir, "scan_trajectory.xyz"), "w") as f:
            f.write("2\nStep 1\nH 0 0 0\nH 0 0 0.74")

        with open(os.path.join(base_dir, "tddft_results.json"), "w") as f:
            json.dump({"tddft_data": [{"state": 1}]}, f)

        mo_e = np.array([-1.0, -0.5])
        mo_o = np.array([2.0, 0.0])
        lw, results = _run_load(chkfile, mo_e, mo_o)

        self.assertIn("freq_data", results)
        self.assertIn("thermo_data", results)
        self.assertIn("scan_results", results)
        self.assertEqual(len(results["scan_results"]), 1)
        self.assertIn("scan_trajectory_path", results)
        self.assertIn("tddft_data", results)

    def test_malformed_freq_json_does_not_crash(self):
        base_dir = tempfile.mkdtemp()
        chkfile = os.path.join(base_dir, "pyscf.chk")
        with open(os.path.join(base_dir, "freq_analysis.json"), "w") as f:
            f.write("{not valid json")

        mo_e = np.array([-1.0, -0.5])
        mo_o = np.array([2.0, 0.0])
        lw, results = _run_load(chkfile, mo_e, mo_o)
        lw.error_signal.emit.assert_not_called()
        self.assertNotIn("freq_data", results)

    def test_malformed_tddft_json_does_not_crash(self):
        base_dir = tempfile.mkdtemp()
        chkfile = os.path.join(base_dir, "pyscf.chk")
        with open(os.path.join(base_dir, "tddft_results.json"), "w") as f:
            f.write("{not valid json")

        mo_e = np.array([-1.0, -0.5])
        mo_o = np.array([2.0, 0.0])
        lw, results = _run_load(chkfile, mo_e, mo_o)
        lw.error_signal.emit.assert_not_called()
        self.assertNotIn("tddft_data", results)


class TestCheckpointStopRequested(unittest.TestCase):
    def test_stop_after_load_mol_suppresses_finished(self):
        base_dir = tempfile.mkdtemp()
        chkfile = os.path.join(base_dir, "pyscf.chk")
        mol = _make_mol()
        sys.modules["pyscf"] = _mod.pyscf
        _mod.pyscf.lib.chkfile.load_mol.return_value = mol

        lw = _make_load_worker(chkfile)
        lw._stop_requested = True
        lw.run()
        lw.finished_signal.emit.assert_not_called()
        lw.error_signal.emit.assert_not_called()

    def test_stop_before_final_emit_suppresses_finished(self):
        base_dir = tempfile.mkdtemp()
        chkfile = os.path.join(base_dir, "pyscf.chk")
        mol = _make_mol()
        sys.modules["pyscf"] = _mod.pyscf
        _mod.pyscf.lib.chkfile.load_mol.return_value = mol
        _mod.pyscf.scf.chkfile.load.return_value = {
            "mo_energy": np.array([-1.0, -0.5]),
            "mo_occ": np.array([2.0, 0.0]),
        }

        lw = _make_load_worker(chkfile)

        # Flip the stop flag only once mol has loaded, simulating a stop
        # request that lands mid-way through the checkpoint-processing work.
        orig_load_mol = _mod.pyscf.lib.chkfile.load_mol

        def _load_and_stop(*a, **kw):
            lw._stop_requested = True
            return orig_load_mol(*a, **kw)

        _mod.pyscf.lib.chkfile.load_mol.side_effect = _load_and_stop
        try:
            lw.run()
        finally:
            _mod.pyscf.lib.chkfile.load_mol.side_effect = None

        lw.finished_signal.emit.assert_not_called()
        lw.error_signal.emit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
