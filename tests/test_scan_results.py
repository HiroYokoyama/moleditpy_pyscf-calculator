"""
tests/test_scan_results.py
Unit tests for the scan_results plotting and data conversion.
"""

import os
import sys
import types
import unittest
import importlib.util
from unittest.mock import MagicMock, patch, mock_open


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
    qt_core.pyqtSignal = lambda *a, **kw: MagicMock()

    class _Qt:
        class Orientation:
            Horizontal = 1

        class CursorShape:
            WaitCursor = 2
            ArrowCursor = 3

        class WindowModality:
            WindowModal = 4

    qt_core.Qt = _Qt

    class _QTimer:
        def __init__(self, *a):
            self.timeout = MagicMock()

        def start(self, ms):
            pass

        def stop(self):
            pass

        def isActive(self):
            return False

    qt_core.QTimer = _QTimer

    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core
    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtCore"] = qt_core

    class _QDialog:
        def __init__(self, parent=None):
            pass

        def setWindowTitle(self, *a):
            pass

        def resize(self, *a):
            pass

        def accept(self):
            pass

        def reject(self):
            pass

        def setCursor(self, *a):
            pass

    class _QSlider:
        def __init__(self, *a):
            self.valueChanged = MagicMock()

        def setRange(self, *a):
            pass

        def setMinimumWidth(self, *a):
            pass

        def setValue(self, *a):
            pass

        def blockSignals(self, *a):
            pass

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QDialog = _QDialog
    qt_widgets.QSlider = _QSlider

    for name in [
        "QVBoxLayout",
        "QHBoxLayout",
        "QLabel",
        "QLineEdit",
        "QPushButton",
        "QMessageBox",
        "QGroupBox",
        "QFormLayout",
        "QComboBox",
        "QSpinBox",
        "QCheckBox",
        "QDialogButtonBox",
        "QFileDialog",
        "QProgressDialog",
        "QApplication",
    ]:
        setattr(qt_widgets, name, MagicMock)

    pyqt6.QtWidgets = qt_widgets
    sys.modules["PyQt6.QtWidgets"] = qt_widgets

    sys.modules["rdkit"] = MagicMock()
    sys.modules["rdkit.Chem"] = MagicMock()
    sys.modules["rdkit.Chem.rdGeometry"] = MagicMock()

    # Mocks for matplotlib
    sys.modules["matplotlib"] = MagicMock()
    sys.modules["matplotlib.backends.backend_qtagg"] = MagicMock()
    sys.modules["matplotlib.figure"] = MagicMock()
    sys.modules["matplotlib.collections"] = MagicMock()
    sys.modules["PIL"] = MagicMock()

    # Needs to export FigureCanvasQTAgg class cleanly
    backend = types.ModuleType("matplotlib.backends.backend_qtagg")

    class _FigureCanvasQTAgg:
        def __init__(self, *a, **k):
            pass

    backend.FigureCanvasQTAgg = _FigureCanvasQTAgg
    sys.modules["matplotlib.backends.backend_qtagg"] = backend


_install_stubs()

sr_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "scan_results.py"),
    "pyscf_calculator_scan_results_under_test",
)
ScanResultDialog = sr_mod.ScanResultDialog


class TestScanResults(unittest.TestCase):
    def setUp(self):
        self.mock_results = [
            {"step": 0, "value": 1.0, "energy": -76.0},
            {"step": 1, "value": 1.1, "energy": -76.5},
            {"step": 2, "value": 1.2, "energy": -76.2},
        ]

        self.dialog = ScanResultDialog.__new__(ScanResultDialog)
        self.dialog.results = self.mock_results
        self.dialog.trajectory = None
        self.dialog.context = MagicMock()
        self.dialog.scan_type = "Dist"
        self.dialog.scan_result_dir = "/tmp"

        # Mock UI
        self.dialog.unit_combo = MagicMock()
        self.dialog.chk_relative = MagicMock()
        self.dialog.canvas = MagicMock()
        self.dialog.canvas.axes = MagicMock()
        self.dialog.canvas.axes.plot.return_value = [MagicMock()]

    def test_plot_data_relative_kjmol(self):
        self.dialog.unit_combo.currentText.return_value = "kJ/mol"
        self.dialog.chk_relative.isChecked.return_value = True

        self.dialog.plot_data()

        # Absolute y values: [-76.0, -76.5, -76.2]
        # Min = -76.5
        # Rel: [0.5, 0.0, 0.3]
        # converted (* 2625.5): [1312.75, 0, 787.65]

        plot_calls = self.dialog.canvas.axes.plot.call_args_list
        x, y = plot_calls[0][0][0], plot_calls[0][0][1]

        self.assertEqual(x, [1.0, 1.1, 1.2])
        self.assertAlmostEqual(y[0], 1312.75, places=2)
        self.assertAlmostEqual(y[1], 0.0, places=2)
        self.assertAlmostEqual(y[2], 787.65, places=2)

        # Test Labels
        self.dialog.canvas.axes.set_ylabel.assert_called_with(
            "Relative Energy (kJ/mol)"
        )

    def test_plot_data_absolute_hartree(self):
        self.dialog.unit_combo.currentText.return_value = "Hartree"
        self.dialog.chk_relative.isChecked.return_value = False

        self.dialog.plot_data()

        plot_calls = self.dialog.canvas.axes.plot.call_args_list
        x, y = plot_calls[0][0][0], plot_calls[0][0][1]

        self.assertEqual(x, [1.0, 1.1, 1.2])
        self.assertEqual(y, [-76.0, -76.5, -76.2])
        self.dialog.canvas.axes.set_ylabel.assert_called_with("Energy (Hartree)")

    @patch("builtins.open", new_callable=mock_open)
    def test_save_csv(self, mock_file):
        orig_file = getattr(sr_mod.QFileDialog, "getSaveFileName", None)
        orig_msg = getattr(sr_mod.QMessageBox, "information", None)
        sr_mod.QFileDialog.getSaveFileName = MagicMock(
            return_value=("/fake/path.csv", "")
        )
        sr_mod.QMessageBox.information = MagicMock()

        try:
            self.dialog.save_csv()
        finally:
            if orig_file is not None:
                sr_mod.QFileDialog.getSaveFileName = orig_file
            if orig_msg is not None:
                sr_mod.QMessageBox.information = orig_msg

        mock_file.assert_called_with("/fake/path.csv", "w", newline="")
        handle = mock_file()
        handle.write.assert_any_call("step,value,energy\r\n")
        handle.write.assert_any_call("0,1.0,-76.0\r\n")


class TestCreateBaseMoleculeMarksModified(unittest.TestCase):
    """
    Regression test: create_base_molecule() replaces context.current_molecule
    with a topology-reconstructed frame-0 molecule from the scan trajectory,
    but context.current_molecule's setter (see PluginContext.current_mol in
    the main app) only pushes the mol to the 3D view — it does NOT set the
    unsaved-changes/dirty flag. Without an explicit mark_project_modified()
    call, opening the Scan Results viewer silently swaps the document's
    molecule with no indication that the project now has unsaved changes.
    """

    def _make_dialog(self):
        dialog = ScanResultDialog.__new__(ScanResultDialog)
        dialog.trajectory = ["2\ncomment\nH 0.0 0.0 0.0\nH 0.0 0.0 1.0"]
        dialog.context = MagicMock()
        mw = MagicMock()
        dialog.context.get_main_window.return_value = mw
        return dialog

    def test_mark_project_modified_called(self):
        dialog = self._make_dialog()
        dialog.create_base_molecule()
        dialog.context.mark_project_modified.assert_called_once()

    def test_current_molecule_set_before_mark_modified(self):
        dialog = self._make_dialog()
        dialog.create_base_molecule()
        # current_molecule must be assigned (not left at whatever default)
        self.assertIsNotNone(dialog.context.current_molecule)

    def test_no_crash_without_context(self):
        dialog = ScanResultDialog.__new__(ScanResultDialog)
        dialog.trajectory = ["2\ncomment\nH 0.0 0.0 0.0\nH 0.0 0.0 1.0"]
        dialog.context = None
        try:
            dialog.create_base_molecule()
        except Exception as exc:
            self.fail(f"create_base_molecule() raised with context=None: {exc}")


if __name__ == "__main__":
    unittest.main()
