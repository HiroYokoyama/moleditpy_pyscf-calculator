"""
tests/test_init_show_dialog.py

Tests for the show_dialog() closure in pyscf_calculator/__init__.py.
Covers lines 54-76: new dialog creation, raise/activate existing, replace
hidden dialog, and RuntimeError recovery when the C++ object was deleted.
"""

import os
import sys
import types
import unittest
import importlib.util
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Install Qt stubs so that gui.py can be imported in a headless environment.
# Must happen at module level (before any module that imports PyQt6 is loaded).
# ---------------------------------------------------------------------------
def _install_qt_stubs():
    qt_core = types.ModuleType("PyQt6.QtCore")

    class _QThread:
        def __init__(self):
            pass

    qt_core.QThread = _QThread
    qt_core.pyqtSignal = lambda *a, **kw: MagicMock()

    class _Qt:
        class AlignmentFlag:
            AlignRight = None

        class Orientation:
            Horizontal = None

        class CursorShape:
            PointingHandCursor = None

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

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    for name in [
        "QWidget",
        "QDialog",
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

    sys.modules.setdefault("rdkit", MagicMock())
    sys.modules.setdefault("rdkit.Chem", MagicMock())
    sys.modules.setdefault("pyscf", None)


_install_qt_stubs()


def _load_module_direct(relpath, module_name):
    src = os.path.join(os.path.dirname(__file__), "..", relpath)
    src = os.path.normpath(src)
    spec = importlib.util.spec_from_file_location(module_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# The __init__.py does `from .gui import PySCFDialog`.  When loaded under
# module name M (an __init__.py → treated as package), Python resolves the
# relative import as `M.gui`.  We therefore pre-populate BOTH the canonical
# package key AND the under-test key so the mock is found regardless of which
# path Python takes at runtime.
# ---------------------------------------------------------------------------
_INIT_MODULE_NAME = "pyscf_calculator_init_show_dialog_under_test"

_MockDialogClass = MagicMock(name="PySCFDialog")
_mock_gui_module = MagicMock()
_mock_gui_module.PySCFDialog = _MockDialogClass

# Cover both resolution paths for the relative import
sys.modules["pyscf_calculator.gui"] = _mock_gui_module
sys.modules[f"{_INIT_MODULE_NAME}.gui"] = _mock_gui_module

init_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "__init__.py"),
    _INIT_MODULE_NAME,
)


# ---------------------------------------------------------------------------
# Helper: run initialize() on a fresh context and extract show_dialog
# ---------------------------------------------------------------------------
def _fresh_show_dialog():
    init_mod.PLUGIN_SETTINGS.clear()
    context = MagicMock()
    # Wire register_window / get_window so they share state, mirroring
    # how the real PluginContext works.  Without this, get_window() returns
    # a truthy MagicMock by default, making show_dialog() always think an
    # existing dialog is present and never creating a new one.
    _windows = {}
    context.get_window.side_effect = lambda wid: _windows.get(wid)
    context.register_window.side_effect = lambda wid, win: _windows.update({wid: win})
    init_mod.initialize(context)
    show_dialog = context.add_menu_action.call_args[0][1]
    return show_dialog, context


# ===========================================================================
# New dialog creation
# ===========================================================================


class TestShowDialogNewInstance(unittest.TestCase):
    """show_dialog() must create and show a new dialog when none exists."""

    def setUp(self):
        _MockDialogClass.reset_mock()
        self.show_dialog, self.context = _fresh_show_dialog()

    def test_creates_dialog_when_none(self):
        self.show_dialog()
        _MockDialogClass.assert_called_once()

    def test_calls_show_on_new_dialog(self):
        self.show_dialog()
        _MockDialogClass.return_value.show.assert_called_once()

    def test_dialog_receives_main_window_as_first_arg(self):
        mw = MagicMock()
        self.context.get_main_window.return_value = mw
        self.show_dialog()
        args, _ = _MockDialogClass.call_args
        self.assertIs(args[0], mw)

    def test_dialog_receives_settings_dict(self):
        self.show_dialog()
        _, kwargs = _MockDialogClass.call_args
        self.assertIn("settings", kwargs)
        self.assertIs(kwargs["settings"], init_mod.PLUGIN_SETTINGS)


# ===========================================================================
# Existing visible dialog
# ===========================================================================


class TestShowDialogExistingVisible(unittest.TestCase):
    """show_dialog() must raise+activate an already-visible dialog."""

    def setUp(self):
        _MockDialogClass.reset_mock()
        self.show_dialog, self.context = _fresh_show_dialog()

    def test_raises_existing_visible_dialog(self):
        self.show_dialog()
        existing = _MockDialogClass.return_value
        existing.isVisible.return_value = True

        self.show_dialog()

        self.assertEqual(_MockDialogClass.call_count, 1)
        existing.raise_.assert_called_once()
        existing.activateWindow.assert_called_once()

    def test_does_not_call_show_again_for_visible_dialog(self):
        self.show_dialog()
        existing = _MockDialogClass.return_value
        existing.isVisible.return_value = True
        existing.show.reset_mock()

        self.show_dialog()
        existing.show.assert_not_called()


# ===========================================================================
# Replace hidden dialog
# ===========================================================================


class TestShowDialogReplaceHidden(unittest.TestCase):
    """show_dialog() must close + replace a hidden dialog."""

    def setUp(self):
        _MockDialogClass.reset_mock()
        self.show_dialog, self.context = _fresh_show_dialog()

    def test_replaces_hidden_dialog(self):
        self.show_dialog()
        first_instance = _MockDialogClass.return_value
        first_instance.isVisible.return_value = False

        second_instance = MagicMock()
        _MockDialogClass.return_value = second_instance

        self.show_dialog()

        first_instance.close.assert_called_once()
        first_instance.deleteLater.assert_called_once()
        second_instance.show.assert_called_once()

    def test_two_dialogs_created_in_total(self):
        self.show_dialog()
        first = _MockDialogClass.return_value
        first.isVisible.return_value = False
        _MockDialogClass.return_value = MagicMock()

        self.show_dialog()
        self.assertEqual(_MockDialogClass.call_count, 2)


# ===========================================================================
# RuntimeError recovery (C++ object deleted)
# ===========================================================================


class TestShowDialogRuntimeError(unittest.TestCase):
    """show_dialog() must handle RuntimeError from isVisible() gracefully."""

    def setUp(self):
        _MockDialogClass.reset_mock()
        self.show_dialog, self.context = _fresh_show_dialog()

    def test_runtime_error_creates_new_dialog(self):
        self.show_dialog()
        first_instance = _MockDialogClass.return_value
        first_instance.isVisible.side_effect = RuntimeError("C++ object deleted")

        fresh_instance = MagicMock()
        _MockDialogClass.return_value = fresh_instance

        self.show_dialog()  # must not raise

        fresh_instance.show.assert_called_once()

    def test_two_dialogs_after_runtime_error(self):
        self.show_dialog()
        first = _MockDialogClass.return_value
        first.isVisible.side_effect = RuntimeError("deleted")
        _MockDialogClass.return_value = MagicMock()

        self.show_dialog()
        self.assertEqual(_MockDialogClass.call_count, 2)


if __name__ == "__main__":
    unittest.main()
