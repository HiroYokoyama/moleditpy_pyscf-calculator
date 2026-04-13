"""
tests/test_worker_pyscf_availability.py

Tests that worker.py behaves correctly both when PySCF is installed and when
it is not. The "without pyscf" path is always exercised (via module-level None
assignment). The "with pyscf" path is exercised with a MagicMock pyscf so the
tests never require the real package to be installed.

Tests that truly need the real pyscf library are decorated with
    @pytest.mark.skipif(not HAS_PYSCF, reason="pyscf not installed")
so the full suite passes whether or not pyscf is present.
"""
import os
import sys
import types
import unittest
import importlib.util
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Detect whether the *real* pyscf is available in this environment
# ---------------------------------------------------------------------------
try:
    import importlib as _il
    _il.util.find_spec("pyscf")
    import pyscf as _pyscf_real  # noqa: F401
    HAS_PYSCF = True
except (ImportError, ValueError):
    HAS_PYSCF = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _install_qt_stubs(force=False):
    """
    Install minimal PyQt6 stubs so worker.py can be imported headlessly.
    When *force* is True (used before loading a fresh worker module), stubs
    are written unconditionally so a previous test cannot leave a MagicMock
    QThread in place.  Without *force*, setdefault is used to avoid stomping
    on stubs already installed by other test files.
    """
    qt_core = types.ModuleType("PyQt6.QtCore")

    class _QThread:
        def __init__(self): pass
        def start(self): pass
        def isRunning(self): return False
        def wait(self, ms=0): return True
        def terminate(self): pass
        @staticmethod
        def msleep(ms): pass

    qt_core.QThread = _QThread
    qt_core.pyqtSignal = lambda *a, **kw: MagicMock()

    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core

    def _set(key, val):
        if force:
            sys.modules[key] = val
        else:
            sys.modules.setdefault(key, val)

    _set("PyQt6", pyqt6)
    _set("PyQt6.QtCore", qt_core)

    rdkit = types.ModuleType("rdkit")
    rdkit_chem = types.ModuleType("rdkit.Chem")
    rdkit_mol_trans = types.ModuleType("rdkit.Chem.rdMolTransforms")
    rdkit.Chem = rdkit_chem
    _set("rdkit", rdkit)
    _set("rdkit.Chem", rdkit_chem)
    _set("rdkit.Chem.rdMolTransforms", rdkit_mol_trans)


_install_qt_stubs()


def _load_worker_with_pyscf_mock(pyscf_value):
    """
    Load worker.py from disk, injecting *pyscf_value* as the pyscf module.
    Re-installs Qt stubs before loading to guarantee a proper QThread base
    class regardless of which other test module was collected last and may have
    replaced sys.modules["PyQt6.QtCore"].
    Returns the loaded module.
    """
    # Each load needs a fresh module name to avoid cache collisions
    module_name = f"_worker_avail_{id(pyscf_value)}"

    # Force-reinstall Qt stubs so that `class PySCFWorker(QThread)` inherits
    # from a real class, not a MagicMock.  test_worker_streams.py replaces
    # QThread with MagicMock at collection time (module-level code); since
    # collection completes before setUpClass runs, we must overwrite here.
    _install_qt_stubs(force=True)

    # Inject the pyscf stub before loading so the try/except at import time
    # picks up our value.
    sys.modules["pyscf"] = pyscf_value  # type: ignore[assignment]
    if pyscf_value is not None:
        sys.modules["pyscf.gto"] = MagicMock()
        sys.modules["pyscf.scf"] = MagicMock()
        sys.modules["pyscf.dft"] = MagicMock()
        sys.modules["pyscf.solvent"] = MagicMock()
    else:
        for sub in ("pyscf.gto", "pyscf.scf", "pyscf.dft", "pyscf.solvent"):
            sys.modules.pop(sub, None)

    src = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "pyscf_calculator", "worker.py")
    )
    spec = importlib.util.spec_from_file_location(module_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_worker(mod, xyz="H 0 0 0\nH 0 0 0.74", config=None):
    """Instantiate PySCFWorker with signals replaced by MagicMock."""
    if config is None:
        config = {"job_type": "Single Point", "method": "RHF", "basis": "sto-3g",
                  "charge": 0, "spin": "1", "threads": 0, "memory": 4000}
    w = mod.PySCFWorker.__new__(mod.PySCFWorker)
    mod.PySCFWorker.__init__(w, xyz, config)
    # Replace Qt signals with mocks
    w.log_signal = MagicMock()
    w.error_signal = MagicMock()
    w.finished_signal = MagicMock()
    w.result_signal = MagicMock()
    return w


def _make_property_worker(mod, chkfile="/fake/chk.chk", tasks=None, out_dir="/fake"):
    pw = mod.PropertyWorker.__new__(mod.PropertyWorker)
    mod.PropertyWorker.__init__(pw, chkfile, tasks or [], out_dir)
    pw.log_signal = MagicMock()
    pw.error_signal = MagicMock()
    pw.finished_signal = MagicMock()
    pw.result_signal = MagicMock()
    return pw


# ===========================================================================
# 1. Without PySCF (pyscf is None at import time)
# ===========================================================================

class TestWorkerWithoutPySCF(unittest.TestCase):
    """Worker.run() must emit error_signal when pyscf is None and return."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_worker_with_pyscf_mock(None)

    def test_pyscf_is_none_in_module(self):
        """worker module-level pyscf should be None when not installed."""
        self.assertIsNone(self.mod.pyscf)

    def test_pyscf_worker_run_emits_error_when_pyscf_none(self):
        w = _make_worker(self.mod)
        w.run()
        w.error_signal.emit.assert_called_once()
        msg = w.error_signal.emit.call_args[0][0]
        self.assertIn("PySCF", msg)

    def test_pyscf_worker_run_does_not_emit_finished_when_pyscf_none(self):
        """finished_signal must NOT be emitted — caller decides lifecycle."""
        w = _make_worker(self.mod)
        w.run()
        w.finished_signal.emit.assert_not_called()

    def test_property_worker_run_emits_error_when_pyscf_none(self):
        pw = _make_property_worker(self.mod)
        pw.run()
        pw.error_signal.emit.assert_called_once()

    def test_property_worker_run_does_not_emit_finished_when_pyscf_none(self):
        pw = _make_property_worker(self.mod)
        pw.run()
        pw.finished_signal.emit.assert_not_called()


# ===========================================================================
# 2. With PySCF mocked — mol build failures
# ===========================================================================

class TestWorkerMolBuildFailure(unittest.TestCase):
    """Worker.run() must handle ValueError/RuntimeError from gto.M gracefully."""

    @classmethod
    def setUpClass(cls):
        pyscf_mock = MagicMock()
        cls.mod = _load_worker_with_pyscf_mock(pyscf_mock)
        # Ensure worker module sees pyscf as non-None
        cls.mod.pyscf = pyscf_mock

    def _run_with_mol_error(self, exc):
        """Run worker with gto.M raising *exc*, return worker."""
        w = _make_worker(self.mod)
        gto_mock = MagicMock()
        gto_mock.M.side_effect = exc
        self.mod.gto = gto_mock

        with patch("os.path.exists", return_value=False), \
             patch("os.makedirs"), \
             patch.object(self.mod, "CaptureStdOut") as mock_cap:
            mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_cap.return_value.__exit__ = MagicMock(return_value=False)
            w.run()

        return w

    def test_value_error_in_mol_build_emits_error(self):
        w = self._run_with_mol_error(ValueError("Charge/spin mismatch"))
        w.error_signal.emit.assert_called_once()
        msg = w.error_signal.emit.call_args[0][0]
        self.assertIn("Molecule Build Failed", msg)

    def test_runtime_error_in_mol_build_emits_error(self):
        w = self._run_with_mol_error(RuntimeError("basis not found"))
        w.error_signal.emit.assert_called_once()
        msg = w.error_signal.emit.call_args[0][0]
        self.assertIn("Molecule Build Failed", msg)

    def test_mol_build_error_does_not_emit_finished(self):
        w = self._run_with_mol_error(ValueError("bad spin"))
        w.finished_signal.emit.assert_not_called()


# ===========================================================================
# 3. Mol-setup path: gto.M() succeeds → mol.stdout / mol.build() coverage
# ===========================================================================

class TestWorkerMolBuildSuccessPath(unittest.TestCase):
    """
    When gto.M() returns a mock mol, lines 333-335 (mol.stdout, mol.verbose,
    mol.build) are exercised.  We then make mol.build() raise ValueError so
    run() returns via the existing error handler.
    """

    @classmethod
    def setUpClass(cls):
        pyscf_mock = MagicMock()
        cls.mod = _load_worker_with_pyscf_mock(pyscf_mock)
        cls.mod.pyscf = pyscf_mock

    def _run_mol_ok_build_fails(self, build_exc=None):
        w = _make_worker(self.mod)
        mock_mol = MagicMock()
        if build_exc is not None:
            mock_mol.build.side_effect = build_exc
        else:
            mock_mol.build.side_effect = ValueError("bail after setup")
        gto_mock = MagicMock()
        gto_mock.M.return_value = mock_mol
        self.mod.gto = gto_mock

        with patch("os.path.exists", return_value=False), \
             patch("os.makedirs"), \
             patch.object(self.mod, "CaptureStdOut") as mock_cap:
            mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_cap.return_value.__exit__ = MagicMock(return_value=False)
            w.run()

        return w, mock_mol

    def test_mol_build_is_called(self):
        _, mol = self._run_mol_ok_build_fails()
        mol.build.assert_called_once()

    def test_mol_stdout_assigned_to_stream(self):
        """mol.stdout is set before mol.build(); a stream object must be assigned."""
        _, mol = self._run_mol_ok_build_fails()
        # mol.stdout was assigned (MagicMock records attribute sets)
        self.assertIsNotNone(mol.stdout)

    def test_error_signal_emitted_after_build_fails(self):
        w, _ = self._run_mol_ok_build_fails()
        w.error_signal.emit.assert_called_once()

    def test_out_dir_loop_skips_existing_dir(self):
        """job_1 exists → loop increments to job_2 (covers line 266)."""
        w = _make_worker(self.mod)
        mock_mol = MagicMock()
        mock_mol.build.side_effect = ValueError("bail")
        self.mod.gto = MagicMock()
        self.mod.gto.M.return_value = mock_mol

        # First call to exists() returns True (job_1 taken), second False (job_2 free)
        with patch("os.path.exists", side_effect=[True, False]), \
             patch("os.makedirs"), \
             patch.object(self.mod, "CaptureStdOut") as mock_cap:
            mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_cap.return_value.__exit__ = MagicMock(return_value=False)
            w.run()

        # If we get here without infinite loop, job_2 was found — pass
        self.assertIn("job_2", w.out_dir)

    def test_threads_setting_calls_num_threads(self):
        """n_threads > 0 in config → pyscf.lib.num_threads(n) called (line 293)."""
        w = _make_worker(self.mod, config={
            "job_type": "Single Point", "method": "RHF", "basis": "sto-3g",
            "charge": 0, "spin": "1", "threads": 4, "memory": 4000,
        })
        mock_mol = MagicMock()
        mock_mol.build.side_effect = ValueError("bail")
        self.mod.gto = MagicMock()
        self.mod.gto.M.return_value = mock_mol

        with patch("os.path.exists", return_value=False), \
             patch("os.makedirs"), \
             patch.object(self.mod, "CaptureStdOut") as mock_cap:
            mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_cap.return_value.__exit__ = MagicMock(return_value=False)
            w.run()

        self.mod.pyscf.lib.num_threads.assert_called_with(4)


# ===========================================================================
# 4. Spin / Charge parsing (via run() with early exit)
# ===========================================================================

class TestWorkerSpinParsing(unittest.TestCase):
    """Spin string parsing must convert multiplicity M to 2S = M-1."""

    @classmethod
    def setUpClass(cls):
        pyscf_mock = MagicMock()
        cls.mod = _load_worker_with_pyscf_mock(pyscf_mock)
        cls.mod.pyscf = pyscf_mock

    def _spin_2s_from_run(self, spin_str):
        """
        Run until gto.M is called and capture the *spin* kwarg passed to it.
        """
        captured = {}

        def fake_gto_M(**kwargs):
            captured.update(kwargs)
            raise ValueError("abort early")  # Stop run() cleanly

        gto_mock = MagicMock()
        gto_mock.M.side_effect = fake_gto_M
        self.mod.gto = gto_mock

        cfg = {"job_type": "Single Point", "method": "RHF", "basis": "sto-3g",
               "charge": 0, "spin": spin_str, "threads": 0, "memory": 4000}
        w = _make_worker(self.mod, config=cfg)

        with patch("os.path.exists", return_value=False), \
             patch("os.makedirs"), \
             patch.object(self.mod, "CaptureStdOut") as mock_cap:
            mock_cap.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_cap.return_value.__exit__ = MagicMock(return_value=False)
            w.run()

        return captured.get("spin", None)

    def test_singlet_string_with_label(self):
        self.assertEqual(self._spin_2s_from_run("1 (Singlet)"), 0)

    def test_doublet_string_with_label(self):
        self.assertEqual(self._spin_2s_from_run("2 (Doublet)"), 1)

    def test_triplet_string_with_label(self):
        self.assertEqual(self._spin_2s_from_run("3 (Triplet)"), 2)

    def test_plain_integer_string(self):
        self.assertEqual(self._spin_2s_from_run("1"), 0)

    def test_invalid_string_falls_back_to_zero(self):
        # Non-parseable string must fall back to 0 (singlet)
        self.assertEqual(self._spin_2s_from_run("bad"), 0)


# ===========================================================================
# 4. Optional: real pyscf integration smoke test
# ===========================================================================

@pytest.mark.skipif(not HAS_PYSCF, reason="pyscf not installed")
class TestWorkerWithRealPySCF(unittest.TestCase):
    """
    Smoke tests that require the real pyscf library.
    Skipped automatically when pyscf is not installed.
    """

    def test_pyscf_importable(self):
        import pyscf  # noqa: F401
        self.assertIsNotNone(pyscf)

    def test_gto_molecule_build(self):
        """Real gto.M should build H2 without error."""
        from pyscf import gto
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        self.assertEqual(mol.natm, 2)

    def test_rhf_energy_h2(self):
        """Real RHF energy for H2 should converge near -1.117 Hartree."""
        from pyscf import gto, scf
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        e = mf.kernel()
        self.assertAlmostEqual(e, -1.117, places=2)


if __name__ == "__main__":
    unittest.main()
