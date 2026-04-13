"""
tests/test_energy_diag.py
Unit tests for the Orbital Energy Diagram parser.
"""
import os
import sys
import types
import unittest
import importlib.util
from unittest.mock import MagicMock

def _load_module_direct(relpath, module_name):
    src = os.path.join(os.path.dirname(__file__), "..", relpath)
    src = os.path.normpath(src)
    spec = importlib.util.spec_from_file_location(module_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

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
        
    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core
    pyqt6.QtGui = qt_gui
    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtCore"] = qt_core
    sys.modules["PyQt6.QtGui"] = qt_gui

    class _QDialog:
        def __init__(self, parent=None): pass
        def setWindowTitle(self, *a): pass
        def resize(self, *a): pass
        def setMouseTracking(self, *a): pass

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QDialog = _QDialog

    for name in [
        "QComboBox", "QFileDialog", "QMessageBox", "QMenu",
        "QApplication", "QToolTip"
    ]:
        setattr(qt_widgets, name, MagicMock)
        
    qt_widgets.QVBoxLayout = lambda *a, **k: MagicMock()
    qt_widgets.QHBoxLayout = lambda *a, **k: MagicMock()
    qt_widgets.QPushButton = lambda *a, **k: MagicMock()
    qt_widgets.QLabel = lambda *a, **k: MagicMock()
        
    pyqt6.QtWidgets = qt_widgets
    sys.modules["PyQt6.QtWidgets"] = qt_widgets

_install_stubs()

ed_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "energy_diag.py"),
    "pyscf_calculator_energy_diag_under_test",
)
EnergyDiagramDialog = ed_mod.EnergyDiagramDialog

class TestEnergyDiagram(unittest.TestCase):
    
    def test_init_rhf_data(self):
        # RHF Data mock
        mo_data = {
            "type": "RHF",
            "energies": [-10.0, -5.0, 2.0, 5.0],
            "occupations": [2.0, 2.0, 0.0, 0.0]
        }
        
        dialog = EnergyDiagramDialog.__new__(EnergyDiagramDialog)
        ed_mod.EnergyDiagramDialog.__init__(dialog, mo_data)
        
        self.assertFalse(dialog.is_uhf)
        self.assertEqual(dialog.energies_a, [-10.0, -5.0, 2.0, 5.0])
        self.assertEqual(dialog.occ_a, [2.0, 2.0, 0.0, 0.0])
        self.assertEqual(dialog.energies_b, [])
        
        # Check gap centers
        self.assertEqual(dialog.homo_energy, -5.0)
        self.assertEqual(dialog.lumo_energy, 2.0)

    def test_init_uhf_data(self):
        # UHF Data mock
        mo_data = {
            "type": "UHF",
            "energies": [
                [-10.0, -5.0, 2.0], # Alpha
                [-9.5, -4.5, 3.0]   # Beta
            ],
            "occupations": [
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0]
            ]
        }
        
        dialog = EnergyDiagramDialog.__new__(EnergyDiagramDialog)
        ed_mod.EnergyDiagramDialog.__init__(dialog, mo_data)
        
        self.assertTrue(dialog.is_uhf)
        self.assertEqual(dialog.energies_a, [-10.0, -5.0, 2.0])
        self.assertEqual(dialog.energies_b, [-9.5, -4.5, 3.0])
        self.assertEqual(dialog.occ_a, [1.0, 1.0, 0.0])
        self.assertEqual(dialog.occ_b, [1.0, 0.0, 0.0])
        
        # Alpha HOMO is -5.0, Beta HOMO is -9.5. Max occupied is -5.0
        self.assertEqual(dialog.homo_energy, -5.0)
        # Alpha LUMO is 2.0, Beta LUMO is -4.5. Min virtual is -4.5
        self.assertEqual(dialog.lumo_energy, -4.5)

if __name__ == '__main__':
    unittest.main()
