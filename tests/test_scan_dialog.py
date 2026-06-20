"""
tests/test_scan_dialog.py
Unit tests for the surface scan setup dialog.
"""

import os
import sys
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
    qt_core.pyqtSignal = lambda *a, **kw: MagicMock()

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

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QDialog = _QDialog

    for name in [
        "QVBoxLayout",
        "QHBoxLayout",
        "QLabel",
        "QLineEdit",
        "QPushButton",
        "QMessageBox",
        "QGroupBox",
        "QFormLayout",
    ]:
        setattr(qt_widgets, name, MagicMock)

    pyqt6.QtWidgets = qt_widgets
    sys.modules["PyQt6.QtWidgets"] = qt_widgets
    sys.modules["rdkit"] = MagicMock()
    sys.modules["rdkit.Chem"] = MagicMock()


_install_stubs()

scan_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "scan_dialog.py"),
    "pyscf_calculator_scan_dialog_under_test",
)
ScanDialog = scan_mod.ScanDialog


class TestScanDialog(unittest.TestCase):
    def setUp(self):
        self.context = MagicMock()
        self.context.get_main_window.return_value = None
        self.dialog = ScanDialog.__new__(ScanDialog)

        # Manually construct fields simulating init
        self.dialog.context = self.context
        self.dialog.mw = None
        self.dialog.selected_atoms = []
        self.dialog.scan_params = {}

        # UI Mocks
        self.dialog.lbl_selection = MagicMock()
        self.dialog.grp_params = MagicMock()
        self.dialog.btn_ok = MagicMock()
        self.dialog.lbl_type = MagicMock()
        self.dialog.lbl_current = MagicMock()
        self.dialog.edit_start = MagicMock()
        self.dialog.edit_end = MagicMock()
        self.dialog.edit_steps = MagicMock()
        self.dialog.scan_configured = MagicMock()
        self.dialog.scan_type = "Dist"

    def test_update_ui_state_enables_group(self):
        self.dialog.selected_atoms = [0, 1]
        # Prevents calculate_current_value from crashing
        with patch.object(self.dialog, "calculate_current_value"):
            self.dialog.update_ui_state()
            self.dialog.grp_params.setEnabled.assert_called_with(True)
            self.dialog.btn_ok.setEnabled.assert_called_with(True)

    def test_update_ui_state_disables_group_for_invalid(self):
        self.dialog.selected_atoms = [0]
        self.dialog.update_ui_state()
        self.dialog.grp_params.setEnabled.assert_called_with(False)
        self.dialog.btn_ok.setEnabled.assert_called_with(False)

    def test_accept_scan_invalid_number(self):
        scan_mod.QMessageBox.warning = MagicMock()
        self.dialog.edit_start.text.return_value = "bad"
        self.dialog.accept_scan()
        scan_mod.QMessageBox.warning.assert_called_once()
        self.assertFalse(self.dialog.scan_configured.emit.called)

    def test_accept_scan_valid_emits_params(self):
        scan_mod.QMessageBox.warning = MagicMock()
        self.dialog.edit_start.text.return_value = "1.5"
        self.dialog.edit_end.text.return_value = "2.5"
        self.dialog.edit_steps.text.return_value = "10"
        self.dialog.selected_atoms = [0, 1]
        self.dialog.scan_type = "Dist"

        with patch.object(self.dialog, "accept"):
            self.dialog.accept_scan()
            self.dialog.scan_configured.emit.assert_called_once()

            # verify dictionary
            emitted = self.dialog.scan_configured.emit.call_args[0][0]
            self.assertEqual(emitted["type"], "Dist")
            self.assertEqual(emitted["start"], 1.5)
            self.assertEqual(emitted["end"], 2.5)
            self.assertEqual(emitted["steps"], 10)
            self.assertEqual(emitted["atoms"], [0, 1])


if __name__ == "__main__":
    unittest.main()
