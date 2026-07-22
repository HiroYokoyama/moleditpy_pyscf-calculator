"""
tests/test_worker_scan_dispatch_coverage.py

Coverage for PySCFWorker's "Scan" job-type dispatch and the
run_rigid_scan()/run_relaxed_scan() helper methods (worker.py ~578-595,
1138-1574). Owned exclusively by worker.py (not scan_dialog.py /
scan_results.py, which belong to other test suites).

All RDKit/PySCF calls are mocked; only real filesystem writes (CSV/XYZ) and
real numpy math (linalg.norm/arccos) are exercised.
"""

import os
import sys
import types
import builtins
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
    module_name = "_worker_scan_test"
    pyscf_mock = MagicMock()
    sys.modules["pyscf"] = pyscf_mock
    for sub in ("pyscf.gto", "pyscf.scf", "pyscf.dft", "pyscf.solvent"):
        sys.modules[sub] = MagicMock()
    for sub in list(sys.modules):
        if sub.startswith("pyscf.geomopt"):
            del sys.modules[sub]
    src = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "pyscf_calculator", "worker.py")
    )
    spec = importlib.util.spec_from_file_location(module_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_worker_mod()


class _BlockImport:
    def __init__(self, blocked_names):
        self.blocked = set(blocked_names)
        self._orig = builtins.__import__

    def __enter__(self):
        orig = self._orig

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in self.blocked:
                raise ImportError(f"blocked for test: {name}")
            return orig(name, globals, locals, fromlist, level)

        builtins.__import__ = fake_import
        return self

    def __exit__(self, *a):
        builtins.__import__ = self._orig


class FakeMF:
    """Copy-friendly mean-field stand-in for scan steps (copy.copy(mf))."""

    def __init__(self, e_tot=-1.0):
        self.e_tot = e_tot
        self.chkfile = ""
        self.verbose = 4
        self.max_cycle = 100
        self.conv_tol = 1e-9

    def reset(self, mol):
        return self

    def kernel(self):
        self.e_tot = -1.05
        return self.e_tot

    def ddCOSMO(self):
        self.with_solvent = MagicMock()
        return self

    def nuc_grad_method(self):
        return MagicMock()


def _make_mock_mol(natm=2, basis="sto-3g", charge=0, spin=0):
    mol = MagicMock()
    mol.build.return_value = None
    mol.stdout = None
    mol.verbose = 4
    mol.output = None
    mol.natm = natm
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.atom_coords.return_value = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    mol.atom_symbol.side_effect = lambda i: ["H", "H"][i]
    return mol


def _make_rd_stub(n_atoms=2, rd_mol_none=False):
    chem_mock = MagicMock()
    if rd_mol_none:
        chem_mock.MolFromXYZBlock.return_value = None
        return chem_mock, None, None

    rd_mol = MagicMock()
    chem_mock.MolFromXYZBlock.return_value = rd_mol

    rw_mol = MagicMock()
    rw_mol.GetBondBetweenAtoms.return_value = None
    chem_mock.RWMol.return_value = rw_mol
    chem_mock.BondType.SINGLE = "SINGLE"

    conf = MagicMock()
    pos = MagicMock()
    pos.x, pos.y, pos.z = 0.0, 0.0, 0.74
    conf.GetAtomPosition.return_value = pos
    rw_mol.GetConformer.return_value = conf

    atoms = []
    for _ in range(n_atoms):
        a = MagicMock()
        a.GetSymbol.return_value = "H"
        atoms.append(a)
    rw_mol.GetAtoms.return_value = atoms

    return chem_mock, rw_mol, conf


def _base_config(job_type, scan_params, method="RHF", extra=None):
    config = {
        "job_type": job_type,
        "method": method,
        "basis": "sto-3g",
        "charge": 0,
        "spin": "1",
        "threads": 0,
        "memory": 4000,
        "max_cycle": 100,
        "conv_tol": "1e-9",
        "scan_params": scan_params,
    }
    if extra:
        config.update(extra)
    return config


def _make_worker(config, xyz="H 0 0 0\nH 0 0 0.74"):
    w = _mod.PySCFWorker.__new__(_mod.PySCFWorker)
    _mod.PySCFWorker.__init__(w, xyz, config)
    w.log_signal = MagicMock()
    w.error_signal = MagicMock()
    w.finished_signal = MagicMock()
    w.result_signal = MagicMock()
    return w


def _run(config, fake_mf=None, chem_mock=None, rdmt_mock=None, block_imports=None):
    mock_mol = _make_mock_mol()

    gto_mock = MagicMock()
    gto_mock.M.return_value = mock_mol
    _mod.gto = gto_mock

    if fake_mf is None:
        fake_mf = FakeMF()

    scf_mock = MagicMock()
    scf_mock.RHF.return_value = fake_mf
    scf_mock.UHF.return_value = fake_mf
    _mod.scf = scf_mock

    dft_mock = MagicMock()
    dft_mock.RKS.return_value = fake_mf
    dft_mock.UKS.return_value = fake_mf
    _mod.dft = dft_mock

    # Always reset (rather than only-if-passed) so a previous test's
    # side_effect/raising mock never leaks into the next test in this file.
    _mod.Chem = chem_mock if chem_mock is not None else MagicMock()
    _mod.rdMolTransforms = rdmt_mock if rdmt_mock is not None else MagicMock()

    w = _make_worker(config)

    tmpdir = tempfile.mkdtemp()
    w.config["out_dir"] = tmpdir
    with patch.object(_mod, "CaptureStdOut") as mock_cap:
        mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_cap.return_value.__exit__ = MagicMock(return_value=False)
        if block_imports:
            with _BlockImport(block_imports):
                w.run()
        else:
            w.run()
    out_dir = os.path.join(tmpdir, "job_1")

    results = None
    if w.result_signal.emit.call_count > 0:
        results = w.result_signal.emit.call_args[0][0]
    return w, results, out_dir


# ===========================================================================
# 1. Scan dispatch guard: missing scan_params
# ===========================================================================


class TestScanParamsMissing(unittest.TestCase):
    def test_missing_scan_params_emits_error_and_returns(self):
        config = {
            "job_type": "Rigid Scan",
            "method": "RHF",
            "basis": "sto-3g",
            "charge": 0,
            "spin": "1",
            "threads": 0,
            "memory": 4000,
            "max_cycle": 100,
            "conv_tol": "1e-9",
        }
        w, results, out_dir = _run(config)
        w.error_signal.emit.assert_called_once()
        self.assertIn("Scan parameters missing", w.error_signal.emit.call_args[0][0])
        w.finished_signal.emit.assert_not_called()


# ===========================================================================
# 2. Rigid scan
# ===========================================================================


class TestRigidScan(unittest.TestCase):
    def test_rdkit_mol_none_still_emits_result(self):
        """run_rigid_scan bails out early, but run() still emits result/finished."""
        chem_mock, _, _ = _make_rd_stub(rd_mol_none=True)
        scan_params = {
            "type": "Dist",
            "atoms": [0, 1],
            "start": 0.7,
            "end": 0.9,
            "steps": 2,
        }
        w, results, out_dir = _run(
            _base_config("Rigid Scan", scan_params), chem_mock=chem_mock
        )
        w.error_signal.emit.assert_called_once()
        self.assertIn(
            "Failed to create RDKit molecule", w.error_signal.emit.call_args[0][0]
        )
        w.finished_signal.emit.assert_called_once()
        self.assertNotIn("scan_results", results)

    def test_dist_scan_success_writes_csv_and_trajectory(self):
        chem_mock, rw_mol, conf = _make_rd_stub()
        scan_params = {
            "type": "Dist",
            "atoms": [0, 1],
            "start": 0.7,
            "end": 0.9,
            "steps": 2,
        }
        w, results, out_dir = _run(
            _base_config("Rigid Scan", scan_params), chem_mock=chem_mock
        )
        w.finished_signal.emit.assert_called_once()
        w.error_signal.emit.assert_not_called()
        self.assertEqual(len(results["scan_results"]), 2)
        self.assertEqual(len(results["scan_trajectory"]), 2)
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "scan_results.csv")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "scan_trajectory.xyz")))

    def test_angle_scan_type(self):
        chem_mock, rw_mol, conf = _make_rd_stub()
        scan_params = {
            "type": "Angle",
            "atoms": [0, 1, 0],
            "start": 100.0,
            "end": 110.0,
            "steps": 2,
        }
        w, results, out_dir = _run(
            _base_config("Rigid Scan", scan_params), chem_mock=chem_mock
        )
        w.finished_signal.emit.assert_called_once()
        self.assertEqual(len(results["scan_results"]), 2)

    def test_dihedral_scan_type(self):
        chem_mock, rw_mol, conf = _make_rd_stub()
        scan_params = {
            "type": "Dihedral",
            "atoms": [0, 1, 0, 1],
            "start": 0.0,
            "end": 30.0,
            "steps": 2,
        }
        w, results, out_dir = _run(
            _base_config("Rigid Scan", scan_params), chem_mock=chem_mock
        )
        w.finished_signal.emit.assert_called_once()
        self.assertEqual(len(results["scan_results"]), 2)

    def test_stop_requested_breaks_loop_early(self):
        chem_mock, rw_mol, conf = _make_rd_stub()
        scan_params = {
            "type": "Dist",
            "atoms": [0, 1],
            "start": 0.7,
            "end": 0.9,
            "steps": 5,
        }

        config = _base_config("Rigid Scan", scan_params)
        mock_mol = _make_mock_mol()
        gto_mock = MagicMock()
        gto_mock.M.return_value = mock_mol
        _mod.gto = gto_mock
        fake_mf = FakeMF()
        scf_mock = MagicMock()
        scf_mock.RHF.return_value = fake_mf
        _mod.scf = scf_mock
        _mod.dft = MagicMock()
        _mod.Chem = chem_mock

        w = _make_worker(config)
        w._stop_requested = True
        tmpdir = tempfile.mkdtemp()
        w.config["out_dir"] = tmpdir
        with patch.object(_mod, "CaptureStdOut") as mock_cap:
            mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_cap.return_value.__exit__ = MagicMock(return_value=False)
            w.run()

        results = w.result_signal.emit.call_args[0][0]
        self.assertEqual(results["scan_results"], [])
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("stopped by user", all_logs)

    def test_sanitize_exception_falls_back_to_partial_update(self):
        chem_mock, rw_mol, conf = _make_rd_stub()
        chem_mock.SanitizeMol.side_effect = ValueError("bad valence")
        scan_params = {
            "type": "Dist",
            "atoms": [0, 1],
            "start": 0.7,
            "end": 0.9,
            "steps": 1,
        }
        w, results, out_dir = _run(
            _base_config("Rigid Scan", scan_params), chem_mock=chem_mock
        )
        w.finished_signal.emit.assert_called_once()
        rw_mol.UpdatePropertyCache.assert_called_once()
        chem_mock.GetSymmSSSR.assert_called_once()

    def test_geometry_set_exception_logs_and_continues(self):
        chem_mock, rw_mol, conf = _make_rd_stub()
        rdmt = MagicMock()
        rdmt.SetBondLength.side_effect = RuntimeError("bad geometry")
        scan_params = {
            "type": "Dist",
            "atoms": [0, 1],
            "start": 0.7,
            "end": 0.9,
            "steps": 2,
        }
        w, results, out_dir = _run(
            _base_config("Rigid Scan", scan_params), chem_mock=chem_mock, rdmt_mock=rdmt
        )
        w.finished_signal.emit.assert_called_once()
        # Both steps failed geometry set -> continue -> empty scan_results
        self.assertEqual(results["scan_results"], [])
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("Geometry set failed", all_logs)

    def test_solvent_applied_when_selected(self):
        chem_mock, rw_mol, conf = _make_rd_stub()
        scan_params = {
            "type": "Dist",
            "atoms": [0, 1],
            "start": 0.7,
            "end": 0.9,
            "steps": 1,
        }
        w, results, out_dir = _run(
            _base_config("Rigid Scan", scan_params, extra={"solvent": "Water"}),
            chem_mock=chem_mock,
        )
        w.finished_signal.emit.assert_called_once()
        self.assertEqual(len(results["scan_results"]), 1)


# ===========================================================================
# 3. Relaxed scan
# ===========================================================================


def _install_geometric(mol_eq):
    sys.modules.setdefault("pyscf.geomopt", types.ModuleType("pyscf.geomopt"))
    m = types.ModuleType("pyscf.geomopt.geometric_solver")
    m.optimize = MagicMock(return_value=mol_eq)
    sys.modules["pyscf.geomopt.geometric_solver"] = m
    return m.optimize


def _make_mol_eq(coords=None, symbols=("H", "H")):
    mol_eq = MagicMock()
    mol_eq.natm = len(symbols)
    if coords is None:
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]])
    mol_eq.atom_coords.return_value = coords
    mol_eq.atom_symbol.side_effect = lambda i: symbols[i]
    return mol_eq


class TestRelaxedScan(unittest.TestCase):
    def test_missing_geometric_emits_error(self):
        scan_params = {
            "type": "Dist",
            "atoms": [0, 1],
            "start": 0.7,
            "end": 0.9,
            "steps": 2,
        }
        w, results, out_dir = _run(
            _base_config("Relaxed Scan", scan_params),
            block_imports=["pyscf.geomopt.geometric_solver"],
        )
        w.error_signal.emit.assert_called_once()
        self.assertIn("geometric", w.error_signal.emit.call_args[0][0])
        w.finished_signal.emit.assert_called_once()  # run() still emits at the end
        self.assertNotIn("scan_results", results)

    def test_dist_relaxed_scan_success(self):
        mol_eq = _make_mol_eq()
        _install_geometric(mol_eq)
        step_mf = MagicMock()
        step_mf.kernel.return_value = -1.2
        scf_mock = MagicMock()
        scf_mock.RHF.return_value = step_mf
        scf_mock.UHF.return_value = step_mf
        _mod.scf = scf_mock
        _mod.dft = MagicMock()

        scan_params = {
            "type": "Dist",
            "atoms": [0, 1],
            "start": 0.7,
            "end": 0.9,
            "steps": 2,
        }
        w, results, out_dir = _run(
            _base_config("Relaxed Scan", scan_params), fake_mf=FakeMF()
        )
        w.finished_signal.emit.assert_called_once()
        self.assertEqual(len(results["scan_results"]), 2)
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "scan_results.csv")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "scan_trajectory.xyz")))

    def test_angle_relaxed_scan_success(self):
        mol_eq = _make_mol_eq(
            coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
            symbols=("H", "H", "H"),
        )
        _install_geometric(mol_eq)
        step_mf = MagicMock()
        step_mf.kernel.return_value = -1.2
        scf_mock = MagicMock()
        scf_mock.RHF.return_value = step_mf
        _mod.scf = scf_mock
        _mod.dft = MagicMock()

        scan_params = {
            "type": "Angle",
            "atoms": [0, 1, 2],
            "start": 90.0,
            "end": 100.0,
            "steps": 1,
        }
        w, results, out_dir = _run(
            _base_config("Relaxed Scan", scan_params), fake_mf=FakeMF()
        )
        w.finished_signal.emit.assert_called_once()
        self.assertEqual(len(results["scan_results"]), 1)

    def test_dihedral_relaxed_scan_success(self):
        mol_eq = _make_mol_eq(
            coords=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]
            ),
            symbols=("H", "H", "H", "H"),
        )
        _install_geometric(mol_eq)
        step_mf = MagicMock()
        step_mf.kernel.return_value = -1.2
        scf_mock = MagicMock()
        scf_mock.RHF.return_value = step_mf
        _mod.scf = scf_mock
        _mod.dft = MagicMock()

        chem_mock = MagicMock()
        rdmt = MagicMock()
        rdmt.GetDihedralDeg.return_value = 25.0

        scan_params = {
            "type": "Dihedral",
            "atoms": [0, 1, 2, 3],
            "start": 0.0,
            "end": 30.0,
            "steps": 1,
        }
        w, results, out_dir = _run(
            _base_config("Relaxed Scan", scan_params),
            fake_mf=FakeMF(),
            chem_mock=chem_mock,
            rdmt_mock=rdmt,
        )
        w.finished_signal.emit.assert_called_once()
        self.assertEqual(len(results["scan_results"]), 1)
        self.assertEqual(results["scan_results"][0]["value"], 25.0)

    def test_stop_requested_breaks_loop_early(self):
        mol_eq = _make_mol_eq()
        _install_geometric(mol_eq)
        step_mf = MagicMock()
        step_mf.kernel.return_value = -1.2
        scf_mock = MagicMock()
        scf_mock.RHF.return_value = step_mf
        _mod.scf = scf_mock
        _mod.dft = MagicMock()

        scan_params = {
            "type": "Dist",
            "atoms": [0, 1],
            "start": 0.7,
            "end": 0.9,
            "steps": 5,
        }
        config = _base_config("Relaxed Scan", scan_params)
        mock_mol = _make_mock_mol()
        gto_mock = MagicMock()
        gto_mock.M.return_value = mock_mol
        _mod.gto = gto_mock

        w = _make_worker(config)
        w._stop_requested = True
        tmpdir = tempfile.mkdtemp()
        w.config["out_dir"] = tmpdir
        with patch.object(_mod, "CaptureStdOut") as mock_cap:
            mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_cap.return_value.__exit__ = MagicMock(return_value=False)
            w.run()

        results = w.result_signal.emit.call_args[0][0]
        self.assertEqual(results["scan_results"], [])
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("stopped by user", all_logs)

    def test_step_optimization_exception_breaks_loop(self):
        mol_eq = _make_mol_eq()
        geo_optimize = _install_geometric(mol_eq)
        geo_optimize.side_effect = RuntimeError("optimizer diverged")

        scan_params = {
            "type": "Dist",
            "atoms": [0, 1],
            "start": 0.7,
            "end": 0.9,
            "steps": 3,
        }
        w, results, out_dir = _run(
            _base_config("Relaxed Scan", scan_params), fake_mf=FakeMF()
        )
        w.finished_signal.emit.assert_called_once()
        self.assertEqual(results["scan_results"], [])
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("Optimization step 1 failed", all_logs)


if __name__ == "__main__":
    unittest.main()
