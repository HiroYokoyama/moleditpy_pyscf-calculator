"""
tests/test_worker_scan_integrity.py

Regression tests for the scan energy profile (v3.3.1).

Three defects, all of which silently corrupted a potential energy surface
rather than reporting a problem:

1. Scan points were recorded without checking SCF convergence, so an
   unconverged point entered the profile looking like a real feature.
2. When the final SCF of a relaxed-scan step failed outright, the energy was
   recorded as 0.0 Ha -- a ~76 Hartree spike next to real energies.
3. _load_scan_csv returned the CSV's capitalised headers ("Energy") while the
   viewer reads r["energy"], so reloading a saved scan raised KeyError.
"""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock


def _install_stubs():
    pyqt6 = types.ModuleType("PyQt6")
    qt_core = types.ModuleType("PyQt6.QtCore")

    class _QThread:
        def __init__(self, *a, **k):
            pass

        @staticmethod
        def msleep(_ms):
            return None

    qt_core.QThread = _QThread
    qt_core.pyqtSignal = lambda *a, **k: MagicMock()
    pyqt6.QtCore = qt_core
    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtCore"] = qt_core
    for name in ("rdkit", "rdkit.Chem", "rdkit.Chem.rdMolTransforms"):
        sys.modules.setdefault(name, MagicMock())


def _load_worker():
    _install_stubs()
    sys.modules["pyscf"] = MagicMock()
    for sub in ("pyscf.gto", "pyscf.scf", "pyscf.dft", "pyscf.solvent"):
        sys.modules[sub] = MagicMock()
    src = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "pyscf_calculator", "worker.py")
    )
    spec = importlib.util.spec_from_file_location("_worker_scan_integrity", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_worker_scan_integrity"] = mod
    spec.loader.exec_module(mod)
    return mod


WORKER_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "pyscf_calculator", "worker.py")
)


class TestLoadScanCsv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_worker()

    def _write_csv(self, rows, header="Step,Value,Energy,Converged"):
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        with open(path, "w", newline="") as fh:
            fh.write(header + "\n")
            for r in rows:
                fh.write(",".join(str(x) for x in r) + "\n")
        self.addCleanup(os.remove, path)
        return path

    def _load(self, path):
        loader = self.mod.LoadWorker._load_scan_csv
        return loader(path)

    def test_keys_are_lowercased_to_match_the_live_scan(self):
        rows = self._load(self._write_csv([[1, 1.5, -76.4, "yes"]]))
        self.assertIn("energy", rows[0])
        self.assertIn("value", rows[0])
        self.assertNotIn("Energy", rows[0])

    def test_the_viewers_accessor_works_on_a_reloaded_scan(self):
        """scan_results.py does `[r["energy"] for r in self.results]`."""
        rows = self._load(
            self._write_csv([[1, 1.5, -76.4, "yes"], [2, 1.6, -76.3, "yes"]])
        )
        energies = [r["energy"] for r in rows]
        self.assertEqual(energies, [-76.4, -76.3])

    def test_numeric_columns_are_floats(self):
        rows = self._load(self._write_csv([[1, 1.5, -76.4, "yes"]]))
        self.assertIsInstance(rows[0]["energy"], float)
        self.assertIsInstance(rows[0]["value"], float)

    def test_converged_column_becomes_a_bool(self):
        rows = self._load(
            self._write_csv([[1, 1.5, -76.4, "yes"], [2, 1.6, -76.3, "NO"]])
        )
        self.assertIs(rows[0]["converged"], True)
        self.assertIs(rows[1]["converged"], False)

    def test_legacy_csv_without_converged_column_still_loads(self):
        rows = self._load(
            self._write_csv([[1, 1.5, -76.4]], header="Step,Value,Energy")
        )
        self.assertEqual(rows[0]["energy"], -76.4)
        self.assertNotIn("converged", rows[0])

    def test_non_numeric_value_is_kept_verbatim(self):
        rows = self._load(self._write_csv([[1, "n/a", -76.4, "yes"]]))
        self.assertEqual(rows[0]["value"], "n/a")


class TestScanEnergyIntegrity(unittest.TestCase):
    """The failure modes live inside a full SCF loop, so pin them at source."""

    def setUp(self):
        with open(WORKER_SRC, "r", encoding="utf-8") as fh:
            self.src = fh.read()

    def test_a_failed_step_is_dropped_rather_than_recorded_as_zero(self):
        self.assertNotIn("e_tot = 0.0", self.src)

    def test_rigid_scan_records_convergence(self):
        self.assertIn('"converged": converged', self.src)

    def test_relaxed_scan_records_convergence(self):
        self.assertIn('"converged": step_converged', self.src)

    def test_rigid_scan_csv_has_a_converged_column(self):
        self.assertIn('"Step,Value,Energy,Converged"', self.src)

    def test_convergence_claim_is_conditional(self):
        """The relaxed scan logged a check mark and 'Converged' for every
        point, including ones where the SCF had just failed."""
        self.assertIn("if step_converged:", self.src)
        self.assertIn("** SCF NOT CONVERGED **", self.src)


if __name__ == "__main__":
    unittest.main()
