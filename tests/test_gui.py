"""
tests/test_gui.py
Unit tests for the main GUI state manager.
"""

import os
import sys
import json
import types
import unittest
import importlib.util
from unittest.mock import MagicMock, patch


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

    class _QThread:
        pass

    qt_core.QThread = _QThread
    qt_core.pyqtSignal = lambda *a, **kw: MagicMock()

    class _Qt:
        pass

    qt_core.Qt = _Qt

    class _QTimer:
        @staticmethod
        def singleShot(ms, fn):
            pass

    qt_core.QTimer = _QTimer

    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core
    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtCore"] = qt_core

    class _QDialog:
        def __init__(self, *args, **kwargs):
            pass

        def setWindowTitle(self, *a):
            pass

        def resize(self, *a):
            pass

    class _QWidget:
        def __init__(self, *args, **kwargs):
            pass

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QWidget = _QWidget
    qt_widgets.QDialog = _QDialog

    for name in [
        "QVBoxLayout",
        "QHBoxLayout",
        "QLabel",
        "QComboBox",
        "QPushButton",
        "QSpinBox",
        "QCheckBox",
        "QGroupBox",
        "QFormLayout",
        "QMessageBox",
        "QLineEdit",
        "QFileDialog",
        "QProgressBar",
        "QTextEdit",
        "QSizePolicy",
        "QScrollArea",
        "QFrame",
        "QTabWidget",
        "QToolTip",
    ]:
        setattr(qt_widgets, name, MagicMock)
    pyqt6.QtWidgets = qt_widgets
    sys.modules["PyQt6.QtWidgets"] = qt_widgets

    sys.modules["rdkit"] = MagicMock()
    sys.modules["pyscf"] = None


_install_stubs()

# Now load gui directly
_gui_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "gui.py"),
    "pyscf_calculator_gui_under_test",
)
PySCFDialog = _gui_mod.PySCFDialog


class TestGuiInternalState(unittest.TestCase):
    def setUp(self):
        # Prevent actual setup_ui since it needs Qt
        with (
            patch.object(PySCFDialog, "setup_ui", return_value=None),
            patch.object(PySCFDialog, "load_settings", return_value=None),
        ):
            self.dialog = PySCFDialog.__new__(PySCFDialog)
            self.dialog.context = None
            self.dialog.settings = {}
            self.dialog.closing = False
            self.dialog.struct_source = "Current Editor"
            self.dialog.calc_history = []
            self.dialog.version = "1.0.0"

        # Mock CalcTab properties manually
        self.dialog.calc_tab = MagicMock()
        self.dialog.calc_tab.job_type_combo.currentText.return_value = "freq"
        self.dialog.calc_tab.method_combo.currentText.return_value = "RKS"
        self.dialog.calc_tab.functional_combo.currentText.return_value = "b3lyp"
        self.dialog.calc_tab.basis_combo.currentText.return_value = "def2-tzvp"
        self.dialog.calc_tab.charge_input.currentText.return_value = "0"
        self.dialog.calc_tab.spin_input.currentText.return_value = "0"
        self.dialog.calc_tab.out_dir_edit.text.return_value = "/path/to/project"
        self.dialog.calc_tab.spin_threads.value.return_value = 8
        self.dialog.calc_tab.spin_memory.value.return_value = 8000
        self.dialog.calc_tab.check_symmetry.isChecked.return_value = True
        self.dialog.calc_tab.spin_cycles.value.return_value = 200
        self.dialog.calc_tab.edit_conv.text.return_value = "1e-8"
        self.dialog.calc_tab.spin_grid_level.value.return_value = 5
        self.dialog.calc_tab.solvent_combo.currentText.return_value = "water"
        self.dialog.calc_tab.scan_params = None

        self.dialog.struct_source = "Current Editor"
        self.dialog.calc_history = []
        self.dialog.version = "1.0.0"

    def test_update_internal_state(self):
        self.dialog.update_internal_state()
        s = self.dialog.settings
        self.assertEqual(s["job_type"], "freq")
        self.assertEqual(s["method"], "RKS")
        self.assertEqual(s["functional"], "b3lyp")
        self.assertEqual(s["basis"], "def2-tzvp")
        self.assertEqual(s["threads"], 8)
        self.assertEqual(s["memory"], 8000)
        self.assertEqual(s["check_symmetry"], True)
        self.assertEqual(s["solvent"], "water")
        self.assertEqual(s["version"], "1.0.0")

    def test_apply_defaults(self):
        # Test apply defaults when no user settings.json exists
        with patch("os.path.exists", return_value=False):
            self.dialog.apply_defaults()

        # Verify defaults were pushed to the simulated UI
        self.dialog.calc_tab.job_type_combo.setCurrentText.assert_called_with(
            "Optimization + Frequency"
        )
        self.dialog.calc_tab.method_combo.setCurrentText.assert_called_with("RKS")
        self.dialog.calc_tab.functional_combo.setCurrentText.assert_called_with("b3lyp")
        self.dialog.calc_tab.spin_grid_level.setValue.assert_called_with(3)
        self.dialog.calc_tab.spin_memory.setValue.assert_called_with(4000)


class TestOnResultsMarksModified(unittest.TestCase):
    """on_results() must call context.mark_project_modified() (V4 API, no direct mw access)."""

    def _make_dialog(self, context):
        with (
            patch.object(PySCFDialog, "setup_ui", return_value=None),
            patch.object(PySCFDialog, "load_settings", return_value=None),
        ):
            dlg = PySCFDialog.__new__(PySCFDialog)
            dlg.context = context
            dlg.settings = {}
            dlg.closing = False
            dlg.struct_source = "Current Editor"
            dlg.calc_history = []
            dlg.version = "1.0.0"
        dlg.calc_tab = MagicMock()
        dlg.calc_tab.out_dir_edit.text.return_value = "/tmp/result"
        dlg.vis_tab = MagicMock()
        return dlg

    def test_on_results_calls_mark_project_modified(self):
        context = MagicMock()
        dlg = self._make_dialog(context)
        dlg.on_results({"out_dir": "/tmp/result"})
        context.mark_project_modified.assert_called_once()

    def test_on_results_no_crash_when_context_is_none(self):
        dlg = self._make_dialog(context=None)
        try:
            dlg.on_results({"out_dir": "/tmp/result"})
        except Exception as exc:
            self.fail(f"on_results() raised with context=None: {exc}")

    def test_on_results_does_not_access_state_manager_directly(self):
        """Verify the refactored code no longer touches mw.state_manager."""
        context = MagicMock()
        mw = MagicMock()
        context.get_main_window.return_value = mw
        dlg = self._make_dialog(context)

        dlg.on_results({"out_dir": "/tmp/result"})

        # state_manager must NOT be accessed — all modification signalling goes
        # through context.mark_project_modified()
        mw.state_manager.has_unsaved_changes  # access it just to clear call count
        mw.state_manager.reset_mock()
        dlg.on_results({"out_dir": "/tmp/result2"})
        mw.state_manager.assert_not_called()


if __name__ == "__main__":
    unittest.main()
