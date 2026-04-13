"""
tests/test_energy_diag_branches.py

Additional tests for EnergyDiagramDialog.__init__ covering branches not hit
by test_energy_diag.py:

  - safe_occ: list-of-lists occupations are flattened (line 78)
  - UHF fallback: energies not a 2-item list of lists (lines 90-93)
  - Empty energies: all_e is empty → default min/max/HOMO/LUMO (lines 104-106)
  - View range: current_min / current_max are centered on the HOMO-LUMO gap
  - Tiny gap: falls back to minimum span (lines 129-134)
"""
import os
import sys
import types
import unittest
import importlib.util
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Stubs (reuse same approach as test_energy_diag.py, but isolated module name)
# ---------------------------------------------------------------------------

def _install_stubs():
    qt_core = types.ModuleType("PyQt6.QtCore")

    class _Qt:
        class CursorShape:
            PointingHandCursor = 1
        class AlignmentFlag:
            AlignCenter = 2

    qt_core.Qt = _Qt

    qt_gui = types.ModuleType("PyQt6.QtGui")
    for name in ["QPainter", "QPen", "QColor", "QFont", "QAction"]:
        setattr(qt_gui, name, MagicMock)

    pyqt6 = sys.modules.get("PyQt6") or types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core
    pyqt6.QtGui = qt_gui
    sys.modules.setdefault("PyQt6", pyqt6)
    sys.modules.setdefault("PyQt6.QtCore", qt_core)
    sys.modules.setdefault("PyQt6.QtGui", qt_gui)

    class _QDialog:
        def __init__(self, parent=None): pass
        def setWindowTitle(self, *a): pass
        def resize(self, *a): pass
        def setMouseTracking(self, *a): pass

    qt_widgets = sys.modules.get("PyQt6.QtWidgets") or types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QDialog = _QDialog
    for name in ["QComboBox", "QFileDialog", "QMessageBox", "QMenu",
                 "QApplication", "QToolTip"]:
        setattr(qt_widgets, name, MagicMock)
    qt_widgets.QVBoxLayout = lambda *a, **k: MagicMock()
    qt_widgets.QHBoxLayout = lambda *a, **k: MagicMock()
    qt_widgets.QPushButton = lambda *a, **k: MagicMock()
    qt_widgets.QLabel = lambda *a, **k: MagicMock()
    pyqt6.QtWidgets = qt_widgets
    sys.modules.setdefault("PyQt6.QtWidgets", qt_widgets)


_install_stubs()


def _load_module_direct(relpath, module_name):
    src = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", relpath)
    )
    spec = importlib.util.spec_from_file_location(module_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_ed_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "energy_diag.py"),
    "pyscf_calculator_energy_diag_branches_under_test",
)
EnergyDiagramDialog = _ed_mod.EnergyDiagramDialog


def _make_dialog(mo_data):
    d = EnergyDiagramDialog.__new__(EnergyDiagramDialog)
    _ed_mod.EnergyDiagramDialog.__init__(d, mo_data)
    return d


# ===========================================================================
# safe_occ — list-of-lists flattening
# ===========================================================================

class TestSafeOccFlatten(unittest.TestCase):
    """Occupations given as list-of-lists must be flattened to scalars."""

    def test_list_of_lists_rhf(self):
        mo_data = {
            "type": "RHF",
            "energies": [-5.0, 2.0],
            "occupations": [[2.0], [0.0]],   # list-of-1-elem lists
        }
        d = _make_dialog(mo_data)
        self.assertEqual(d.occ_a, [2.0, 0.0])

    def test_list_of_tuples_rhf(self):
        mo_data = {
            "type": "RHF",
            "energies": [-5.0, 2.0],
            "occupations": [(2.0,), (0.0,)],  # list of tuples
        }
        d = _make_dialog(mo_data)
        self.assertEqual(d.occ_a, [2.0, 0.0])

    def test_homo_lumo_correct_after_flatten(self):
        mo_data = {
            "type": "RHF",
            "energies": [-5.0, 2.0],
            "occupations": [[2.0], [0.0]],
        }
        d = _make_dialog(mo_data)
        self.assertEqual(d.homo_energy, -5.0)
        self.assertEqual(d.lumo_energy, 2.0)


# ===========================================================================
# UHF fallback — energies not a 2-item list of lists
# ===========================================================================

class TestUHFFallback(unittest.TestCase):
    """When UHF energies are not [[alpha], [beta]], fall back gracefully."""

    def test_uhf_flat_energy_list_fallback(self):
        # energies is a flat list (not list of lists)
        mo_data = {
            "type": "UHF",
            "energies": [-10.0, -5.0, 2.0],   # flat, not [[alpha], [beta]]
            "occupations": [1.0, 1.0, 0.0],
        }
        d = _make_dialog(mo_data)
        self.assertTrue(d.is_uhf)
        # Fallback: energies_a = energies, energies_b = []
        self.assertEqual(d.energies_a, [-10.0, -5.0, 2.0])
        self.assertEqual(d.energies_b, [])
        self.assertEqual(d.occ_b, [])

    def test_uhf_single_list_fallback(self):
        mo_data = {
            "type": "UHF",
            "energies": [-3.0],
            "occupations": [1.0],
        }
        d = _make_dialog(mo_data)
        self.assertEqual(d.energies_a, [-3.0])
        self.assertEqual(d.energies_b, [])


# ===========================================================================
# Empty energies — uses default min/max
# ===========================================================================

class TestEmptyEnergies(unittest.TestCase):
    """When all_e is empty, defaults must be set (lines 104-106)."""

    def test_empty_rhf_energies(self):
        mo_data = {
            "type": "RHF",
            "energies": [],
            "occupations": [],
        }
        d = _make_dialog(mo_data)
        self.assertEqual(d.full_min, -1.0)
        self.assertEqual(d.full_max, 1.0)
        self.assertEqual(d.homo_energy, -0.5)
        self.assertEqual(d.lumo_energy, 0.5)

    def test_empty_uhf_energies(self):
        mo_data = {
            "type": "UHF",
            "energies": [[], []],
            "occupations": [[], []],
        }
        d = _make_dialog(mo_data)
        self.assertEqual(d.full_min, -1.0)
        self.assertEqual(d.full_max, 1.0)


# ===========================================================================
# View range — centered on gap
# ===========================================================================

class TestViewRange(unittest.TestCase):
    """current_min / current_max must be centered on the HOMO-LUMO gap × 3."""

    def test_view_range_centered(self):
        mo_data = {
            "type": "RHF",
            "energies": [-5.0, 2.0],
            "occupations": [2.0, 0.0],
        }
        d = _make_dialog(mo_data)
        gap = abs(2.0 - (-5.0))        # 7.0
        gap_center = (-5.0 + 2.0) / 2  # -1.5
        target_span = gap * 3.0         # 21.0
        expected_min = gap_center - target_span / 2.0
        expected_max = gap_center + target_span / 2.0

        self.assertAlmostEqual(d.current_min, expected_min, places=6)
        self.assertAlmostEqual(d.current_max, expected_max, places=6)

    def test_tiny_gap_uses_fallback_span(self):
        """Gap < 0.01 falls back to 0.05; span < 0.2 clamps to 0.2."""
        mo_data = {
            "type": "RHF",
            "energies": [-5.0, -5.0001],   # near-degenerate HOMO/LUMO
            "occupations": [2.0, 0.0],
        }
        d = _make_dialog(mo_data)
        # target_span must be approximately 0.2 (minimum clamp)
        actual_span = d.current_max - d.current_min
        self.assertAlmostEqual(actual_span, 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
