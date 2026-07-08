"""
tests/test_worker_load_worker.py

Tests for LoadWorker.run() — the auxiliary-data loading path that works
without a PySCF checkpoint file (scan CSV, TDDFT JSON, freq JSON).

Strategy:
  - pyscf is mocked so the initial guard passes and `from pyscf import lib, scf`
    succeeds without the real library.
  - Temporary directories with real CSV / JSON files exercise the pure-Python
    file-reading code paths in LoadWorker.run().
  - No real pyscf computation is performed.
"""

import csv
import json
import os
import sys
import types
import unittest
import tempfile
import importlib.util
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Qt and pyscf stubs (must be installed before worker.py is loaded)
# ---------------------------------------------------------------------------


def _install_stubs(force=False):
    qt_core = types.ModuleType("PyQt6.QtCore")

    class _QThread:
        def __init__(self):
            pass

        def start(self):
            pass

        def isRunning(self):
            return False

        def wait(self, ms=0):
            return True

        def terminate(self):
            pass

        @staticmethod
        def msleep(ms):
            pass

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
    _set("rdkit", MagicMock())
    _set("rdkit.Chem", MagicMock())
    _set("rdkit.Chem.rdMolTransforms", MagicMock())


_install_stubs()


def _load_worker_mod(pyscf_mock):
    """Load worker.py with a specific pyscf mock, force-refreshing Qt stubs."""
    _install_stubs(force=True)
    module_name = f"_worker_load_{id(pyscf_mock)}"
    sys.modules["pyscf"] = pyscf_mock
    if pyscf_mock is not None:
        sys.modules["pyscf.gto"] = MagicMock()
        sys.modules["pyscf.scf"] = MagicMock()
        sys.modules["pyscf.dft"] = MagicMock()
        sys.modules["pyscf.solvent"] = MagicMock()
    src = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "pyscf_calculator", "worker.py")
    )
    spec = importlib.util.spec_from_file_location(module_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_load_worker(mod, chkfile):
    lw = mod.LoadWorker.__new__(mod.LoadWorker)
    mod.LoadWorker.__init__(lw, chkfile)
    lw.finished_signal = MagicMock()
    lw.error_signal = MagicMock()
    return lw


# ===========================================================================
# 1. pyscf=None guard
# ===========================================================================


class TestLoadWorkerNoPySCF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_worker_mod(None)

    def test_emits_error_when_pyscf_none(self):
        lw = _make_load_worker(self.mod, "/fake/run.chk")
        lw.run()
        lw.error_signal.emit.assert_called_once()

    def test_does_not_emit_finished_when_pyscf_none(self):
        lw = _make_load_worker(self.mod, "/fake/run.chk")
        lw.run()
        lw.finished_signal.emit.assert_not_called()


# ===========================================================================
# 2. Auxiliary-only paths (no checkpoint file)
# ===========================================================================


class TestLoadWorkerAuxiliaryOnly(unittest.TestCase):
    """
    LoadWorker must load scan CSV, TDDFT JSON, and freq JSON when the
    checkpoint file is absent.  No pyscf computation needed for these paths.
    """

    @classmethod
    def setUpClass(cls):
        pyscf_mock = MagicMock()
        cls.mod = _load_worker_mod(pyscf_mock)
        cls.mod.pyscf = pyscf_mock

    def _run_with_files(self, files: dict):
        """
        Create a temp dir containing *files* (name→content string),
        run LoadWorker pointing at a nonexistent chkfile in that dir,
        and return the results dict passed to finished_signal.emit.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            for fname, content in files.items():
                fpath = os.path.join(tmpdir, fname)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)

            fake_chk = os.path.join(tmpdir, "nonexistent.chk")
            lw = _make_load_worker(self.mod, fake_chk)
            lw.run()

            lw.finished_signal.emit.assert_called_once()
            return lw.finished_signal.emit.call_args[0][0]

    def test_scan_csv_loaded(self):
        csv_content = "coord,energy\n1.0,-1.117\n2.0,-1.050\n"
        results = self._run_with_files({"scan_results.csv": csv_content})
        self.assertIn("scan_results", results)
        self.assertEqual(len(results["scan_results"]), 2)
        self.assertAlmostEqual(results["scan_results"][0]["energy"], -1.117)

    def test_scan_csv_coords_are_float(self):
        csv_content = "coord,energy\n1.5,-0.5\n"
        results = self._run_with_files({"scan_results.csv": csv_content})
        self.assertIsInstance(results["scan_results"][0]["coord"], float)

    def test_scan_trajectory_path_included(self):
        csv_content = "coord,energy\n1.0,-1.0\n"
        results = self._run_with_files(
            {
                "scan_results.csv": csv_content,
                "scan_trajectory.xyz": "2\nH2\nH 0 0 0\nH 0 0 0.74\n",
            }
        )
        self.assertIn("scan_trajectory_path", results)
        self.assertTrue(results["scan_trajectory_path"].endswith("scan_trajectory.xyz"))

    def test_tddft_json_loaded(self):
        tddft_data = {"tddft_data": [{"excitation": 3.5, "oscillator": 0.1}]}
        results = self._run_with_files({"tddft_results.json": json.dumps(tddft_data)})
        self.assertIn("tddft_data", results)
        self.assertEqual(results["tddft_data"][0]["excitation"], 3.5)

    def test_freq_json_loaded(self):
        freq_data = {"freq_data": {"freqs": [1000.0, 3000.0], "modes": []}}
        results = self._run_with_files({"freq_analysis.json": json.dumps(freq_data)})
        self.assertIn("freq_data", results)

    def test_all_three_aux_files_loaded(self):
        csv_content = "coord,energy\n1.0,-1.0\n"
        tddft = {"tddft_data": []}
        freq = {"freq_data": {}}
        results = self._run_with_files(
            {
                "scan_results.csv": csv_content,
                "tddft_results.json": json.dumps(tddft),
                "freq_analysis.json": json.dumps(freq),
            }
        )
        self.assertIn("scan_results", results)
        self.assertIn("tddft_data", results)
        self.assertIn("freq_data", results)

    def test_out_dir_in_results(self):
        csv_content = "coord,energy\n1.0,-1.0\n"
        results = self._run_with_files({"scan_results.csv": csv_content})
        self.assertIn("out_dir", results)

    def test_malformed_csv_does_not_raise(self):
        """A CSV that fails to parse must be skipped, not raise."""
        results = self._run_with_files({"scan_results.csv": "not,a,valid\n,,\n"})
        # The run must complete and emit finished (possibly without scan_results)
        # — we already asserted .assert_called_once() inside _run_with_files
        self.assertIsInstance(results, dict)


# ===========================================================================
# 3. Checkpoint present (no auxiliary files) → falls through to pyscf loading
# ===========================================================================


class TestLoadWorkerCheckpointPath(unittest.TestCase):
    """
    When the checkpoint file EXISTS, LoadWorker falls through to the
    `lib.chkfile.load_mol` path.  A mock pyscf that raises can be used to
    verify error_signal is emitted (since we can't call real pyscf).
    """

    @classmethod
    def setUpClass(cls):
        pyscf_mock = MagicMock()
        cls.mod = _load_worker_mod(pyscf_mock)
        cls.mod.pyscf = pyscf_mock

    def test_error_emitted_when_chkfile_loading_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".chk", delete=False) as f:
            f.write(b"dummy")
            chkpath = f.name

        try:
            lw = _make_load_worker(self.mod, chkpath)
            # `from pyscf import lib, scf` inside run() resolves to
            # sys.modules["pyscf"].lib — configure side_effect there.
            self.mod.pyscf.lib.chkfile.load_mol.side_effect = RuntimeError("bad chk")
            lw.run()
            lw.error_signal.emit.assert_called_once()
        finally:
            os.unlink(chkpath)
            # Reset side_effect so other tests in this class are unaffected
            self.mod.pyscf.lib.chkfile.load_mol.side_effect = None


# ===========================================================================
# 4. Cooperative stop: _stop_requested suppresses signal emission
# ===========================================================================


class TestLoadWorkerStopRequested(unittest.TestCase):
    """
    _safe_stop_worker() sets worker._stop_requested = True before waiting.
    run() must honor it: no finished_signal / error_signal after a stop, so
    the caller's wait(1500) succeeds instead of falling through to
    terminate().
    """

    @classmethod
    def setUpClass(cls):
        pyscf_mock = MagicMock()
        cls.mod = _load_worker_mod(pyscf_mock)
        cls.mod.pyscf = pyscf_mock

    def test_stop_suppresses_finished_on_aux_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, "scan_results.csv"), "w", encoding="utf-8"
            ) as f:
                f.write("coord,energy\n1.0,-1.0\n")
            lw = _make_load_worker(
                self.mod, os.path.join(tmpdir, "nonexistent.chk")
            )
            lw._stop_requested = True
            lw.run()
            lw.finished_signal.emit.assert_not_called()
            lw.error_signal.emit.assert_not_called()

    def test_stop_suppresses_error_on_chkfile_path(self):
        with tempfile.NamedTemporaryFile(suffix=".chk", delete=False) as f:
            f.write(b"dummy")
            chkpath = f.name
        try:
            lw = _make_load_worker(self.mod, chkpath)
            self.mod.pyscf.lib.chkfile.load_mol.side_effect = RuntimeError(
                "interrupted read"
            )
            lw._stop_requested = True
            lw.run()
            lw.error_signal.emit.assert_not_called()
            lw.finished_signal.emit.assert_not_called()
        finally:
            os.unlink(chkpath)
            self.mod.pyscf.lib.chkfile.load_mol.side_effect = None


if __name__ == "__main__":
    unittest.main()
