"""
tests/test_worker_optimization_coverage.py

Coverage for PySCFWorker.run() "Optimization" job type dispatch and the
"Ensure Energy is calculated" block that follows it (worker.py ~597-733):
  - Regular / Transition-State geometry optimization via geometric-solver
  - ImportError fallback to Berny (regular optimization only, not TS)
  - Both optimizers missing -> error_signal
  - Symmetry-breaking initial guess for UHF/UKS (success + exception fallback)
  - break_symmetry=False skip path
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
    module_name = "_worker_opt_test"
    pyscf_mock = MagicMock()
    sys.modules["pyscf"] = pyscf_mock
    for sub in ("pyscf.gto", "pyscf.scf", "pyscf.dft", "pyscf.solvent"):
        sys.modules[sub] = MagicMock()
    # Drop any lingering geomopt stubs from previous runs so ImportError tests
    # start from a clean slate.
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
    """Force ImportError for specific dotted module names during `with`."""

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
    """Lightweight stand-in for a pyscf mean-field object.

    Unlike MagicMock, attribute access does NOT auto-vivify, so
    hasattr(mf, "with_solvent") correctly reflects whether ddCOSMO() was
    ever called — needed to exercise the solvent-reapplication branches.
    """

    def __init__(self, e_tot=None, converged=True):
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
        self.get_init_guess_raises = False

    def kernel(self, dm0=None):
        self.kernel_calls.append(dm0)
        self.e_tot = -1.117
        return self.e_tot

    def get_init_guess(self, key="minao"):
        if self.get_init_guess_raises:
            raise RuntimeError("init guess failed")
        # Spin-resolved (2, nao, nao) and non-zero, so that breaking symmetry
        # is observable in the returned matrix.
        return np.ones((2, 4, 4))

    def ddCOSMO(self):
        self.with_solvent = MagicMock()
        return self

    def nuc_grad_method(self):
        return MagicMock()

    def copy(self):
        return self


def _make_mock_mol(natm=2):
    mol = MagicMock()
    mol.build.return_value = None
    mol.stdout = None
    mol.verbose = 4
    mol.output = None
    mol.natm = natm
    # Two basis functions per atom, so the broken-symmetry guess has a real
    # AO slice to zero out.
    mol.aoslice_by_atom.return_value = np.array(
        [[0, 0, 2 * i, 2 * i + 2] for i in range(natm)]
    )
    return mol


def _make_mol_eq(symbols=("H", "H"), coords=None):
    mol_eq = MagicMock()
    mol_eq.natm = len(symbols)
    if coords is None:
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    mol_eq.atom_coords.return_value = coords
    mol_eq.atom_symbol.side_effect = lambda i: symbols[i]
    return mol_eq


def _install_geomopt(name, mol_eq):
    """Register a fake `pyscf.geomopt.<name>` module exposing `optimize`."""
    sys.modules.setdefault("pyscf.geomopt", types.ModuleType("pyscf.geomopt"))
    m = types.ModuleType(f"pyscf.geomopt.{name}")
    m.optimize = MagicMock(return_value=mol_eq)
    sys.modules[f"pyscf.geomopt.{name}"] = m
    return m.optimize


def _base_config(job_type="Geometry Optimization", method="RHF", extra=None):
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


def _run(config, fake_mf, block_imports=None):
    mock_mol = _make_mock_mol()

    gto_mock = MagicMock()
    gto_mock.M.return_value = mock_mol
    _mod.gto = gto_mock

    # A list hands out one fake per _build_mf call (pre- vs post-optimization).
    mf_kw = (
        {"side_effect": list(fake_mf)}
        if isinstance(fake_mf, list)
        else {"return_value": fake_mf}
    )
    scf_mock = MagicMock()
    dft_mock = MagicMock()
    for ctor in (scf_mock.RHF, scf_mock.UHF, scf_mock.ROHF):
        ctor.configure_mock(**mf_kw)
    for ctor in (dft_mock.RKS, dft_mock.UKS, dft_mock.ROKS):
        ctor.configure_mock(**mf_kw)
    _mod.scf = scf_mock
    _mod.dft = dft_mock

    w = _make_worker(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        w.config["out_dir"] = tmpdir
        with patch.object(_mod, "CaptureStdOut") as mock_cap:
            mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_cap.return_value.__exit__ = MagicMock(return_value=False)
            if block_imports:
                with _BlockImport(block_imports):
                    w.run()
            else:
                w.run()

    results = None
    if w.result_signal.emit.call_count > 0:
        results = w.result_signal.emit.call_args[0][0]
    return w, results


# ===========================================================================
# 1. Regular optimization via geometric-solver
# ===========================================================================


class TestGeometricOptimizationSuccess(unittest.TestCase):
    def test_optimized_xyz_in_results(self):
        mol_eq = _make_mol_eq()
        _install_geomopt("geometric_solver", mol_eq)
        w, results = _run(_base_config(), FakeMF())
        w.finished_signal.emit.assert_called_once()
        w.error_signal.emit.assert_not_called()
        self.assertIn("optimized_xyz", results)
        self.assertIn("Generated by PySCF Optimization", results["optimized_xyz"])

    def test_ts_optimization_log_message(self):
        mol_eq = _make_mol_eq()
        _install_geomopt("geometric_solver", mol_eq)
        w, results = _run(
            _base_config(job_type="Transition State Optimization"), FakeMF()
        )
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("Transition State Optimization", all_logs)
        self.assertIn("Generated by PySCF TS Optimization", results["optimized_xyz"])


# ===========================================================================
# 2. ImportError handling
# ===========================================================================


class TestOptimizationImportErrors(unittest.TestCase):
    def test_ts_geometric_missing_emits_error_no_berny(self):
        """TS optimization requires geometric; berny fallback is NOT attempted."""
        w, results = _run(
            _base_config(job_type="Transition State Optimization"),
            FakeMF(),
            block_imports=["pyscf.geomopt.geometric_solver"],
        )
        w.error_signal.emit.assert_called_once()
        msg = w.error_signal.emit.call_args[0][0]
        self.assertIn("geometric", msg)
        w.finished_signal.emit.assert_not_called()

    def test_regular_optimization_falls_back_to_berny(self):
        mol_eq = _make_mol_eq()
        _install_geomopt("berny_solver", mol_eq)
        w, results = _run(
            _base_config(),
            FakeMF(),
            block_imports=["pyscf.geomopt.geometric_solver"],
        )
        w.finished_signal.emit.assert_called_once()
        self.assertIn("(Berny)", results["optimized_xyz"])
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("geometric-lib not found", all_logs)

    def test_both_optimizers_missing_emits_error(self):
        w, results = _run(
            _base_config(),
            FakeMF(),
            block_imports=[
                "pyscf.geomopt.geometric_solver",
                "pyscf.geomopt.berny_solver",
            ],
        )
        w.error_signal.emit.assert_called_once()
        msg = w.error_signal.emit.call_args[0][0]
        self.assertIn("Neither", msg)
        w.finished_signal.emit.assert_not_called()


# ===========================================================================
# 3. Ensure-Energy block: mf.kernel() invoked when e_tot falsy
# ===========================================================================


class TestEnsureEnergyKernelCall(unittest.TestCase):
    def test_energy_job_type_calls_kernel_when_falsy(self):
        fake_mf = FakeMF(e_tot=None)
        w, results = _run(_base_config(job_type="Energy", method="RHF"), fake_mf)
        w.finished_signal.emit.assert_called_once()
        self.assertEqual(len(fake_mf.kernel_calls), 1)


# ===========================================================================
# 2b. The properties SCF runs at the optimized geometry, with user settings
# ===========================================================================


class TestPostOptimizationMF(unittest.TestCase):
    def _check(self, block_imports, solver):
        mol_eq = _make_mol_eq()
        _install_geomopt(solver, mol_eq)
        first, second = FakeMF(), FakeMF()
        w, _ = _run(
            _base_config(extra={"max_cycle": 321, "conv_tol": "1e-7"}),
            [first, second],
            block_imports=block_imports,
        )
        w.error_signal.emit.assert_not_called()
        # the final SCF ran on a fresh mf built at mol_eq ...
        self.assertIs(_mod.scf.RHF.call_args_list[-1][0][0], mol_eq)
        self.assertEqual(len(second.kernel_calls), 1)
        self.assertEqual(first.kernel_calls, [])
        # ... carrying the user's SCF settings
        self.assertEqual(second.max_cycle, 321)
        self.assertEqual(second.conv_tol, 1e-7)
        self.assertTrue(second.chkfile.endswith("pyscf.chk"))

    def test_geometric(self):
        self._check(None, "geometric_solver")

    def test_berny_fallback(self):
        """Berny used to leave mf on the starting geometry."""
        self._check(["pyscf.geomopt.geometric_solver"], "berny_solver")


# ===========================================================================
# 3b. Solvent reaches the mean-field object of every non-scan job
# ===========================================================================


class TestSolventApplied(unittest.TestCase):
    """The first mf used to be built without ddCOSMO, so Energy / TDDFT /
    Optimization ran in vacuum while the log announced a solvent."""

    def test_energy_job_is_solvated(self):
        fake_mf = FakeMF(e_tot=None)
        w, _ = _run(
            _base_config(job_type="Energy", extra={"solvent": "Water"}), fake_mf
        )
        w.error_signal.emit.assert_not_called()
        self.assertTrue(hasattr(fake_mf, "with_solvent"))
        self.assertEqual(fake_mf.with_solvent.eps, 78.2)

    def test_optimizer_receives_solvated_mf(self):
        mol_eq = _make_mol_eq()
        opt = _install_geomopt("geometric_solver", mol_eq)
        fake_mf = FakeMF()
        _run(_base_config(extra={"solvent": "Toluene"}), fake_mf)
        mf_passed = opt.call_args[0][0]
        self.assertTrue(hasattr(mf_passed, "with_solvent"))

    def test_vacuum_job_is_not_solvated(self):
        fake_mf = FakeMF(e_tot=None)
        _run(
            _base_config(job_type="Energy", extra={"solvent": "None (Vacuum)"}),
            fake_mf,
        )
        self.assertFalse(hasattr(fake_mf, "with_solvent"))


# ===========================================================================
# 4. Symmetry breaking (UHF/UKS, spin>0)
# ===========================================================================


class TestSymmetryBreaking(unittest.TestCase):
    """Symmetry breaking applies to the spin-RESTRICTED case (multiplicity 1),
    where UHF/UKS would otherwise relax straight back onto RHF/RKS. With
    spin_2s > 0 the alpha and beta occupations already differ, so there is
    nothing left to break -- the old code had this condition inverted."""

    def test_uhf_symmetry_breaking_success(self):
        fake_mf = FakeMF(e_tot=None)
        w, results = _run(
            _base_config(job_type="Energy", method="UHF", extra={"spin": "1"}),
            fake_mf,
        )
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("symmetry-broken initial guess", all_logs)
        # dm0 was passed on the (only) kernel call
        self.assertEqual(len(fake_mf.kernel_calls), 1)
        self.assertIsNotNone(fake_mf.kernel_calls[0])
        w.error_signal.emit.assert_not_called()

    def test_the_guess_handed_to_the_kernel_is_actually_asymmetric(self):
        fake_mf = FakeMF(e_tot=None)
        _run(
            _base_config(job_type="Energy", method="UHF", extra={"spin": "1"}),
            fake_mf,
        )
        dm0 = fake_mf.kernel_calls[0]
        self.assertFalse(np.allclose(dm0[0], dm0[1]))

    def test_uks_symmetry_breaking_success(self):
        fake_mf = FakeMF(e_tot=None)
        w, results = _run(
            _base_config(
                job_type="Energy",
                method="UKS",
                extra={"spin": "1", "functional": "b3lyp"},
            ),
            fake_mf,
        )
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("symmetry-broken initial guess", all_logs)
        self.assertEqual(len(fake_mf.kernel_calls), 1)

    def test_open_shell_does_not_need_breaking(self):
        fake_mf = FakeMF(e_tot=None)
        w, _results = _run(
            _base_config(job_type="Energy", method="UHF", extra={"spin": "3"}),
            fake_mf,
        )
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertNotIn("symmetry-broken initial guess", all_logs)
        self.assertIsNone(fake_mf.kernel_calls[0])

    def test_symmetry_breaking_exception_falls_back(self):
        fake_mf = FakeMF(e_tot=None)
        fake_mf.get_init_guess_raises = True
        w, results = _run(
            _base_config(job_type="Energy", method="UHF", extra={"spin": "1"}),
            fake_mf,
        )
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertIn("Symmetry breaking failed", all_logs)
        # Fallback path calls mf.kernel() with no dm0
        self.assertEqual(len(fake_mf.kernel_calls), 1)
        self.assertIsNone(fake_mf.kernel_calls[0])

    def test_break_symmetry_false_skips_mixing(self):
        fake_mf = FakeMF(e_tot=None)
        w, results = _run(
            _base_config(
                job_type="Energy",
                method="UHF",
                extra={"spin": "1", "break_symmetry": False},
            ),
            fake_mf,
        )
        all_logs = " ".join(str(c) for c in w.log_signal.emit.call_args_list)
        self.assertNotIn("symmetry-broken initial guess", all_logs)
        self.assertEqual(len(fake_mf.kernel_calls), 1)
        self.assertIsNone(fake_mf.kernel_calls[0])


if __name__ == "__main__":
    unittest.main()
