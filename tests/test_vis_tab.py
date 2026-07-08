"""
tests/test_vis_tab.py

Unit tests for pyscf_calculator/vis_tab.py — specifically the pure-logic
parts of VisTab that don't require a live Qt event loop:

  - run_specific_analysis(): busy-guard against overlapping PropertyWorker
    runs (regression test)
  - PropertyWorker wiring (signals connected, worker started)

vis_tab.py was previously 0% covered (documented as Qt-widget-heavy in
tests/TESTS.md). These tests instantiate VisTab via __new__ (bypassing
setup_ui/__init__, which need a live QApplication) and manually populate
just the attributes each method under test touches.
"""

import os
import sys
import types
import unittest
import importlib.util
from unittest.mock import MagicMock, patch


def _load_module_direct(relpath, module_name):
    """Load a .py file as a module without going through the package __init__."""
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
        class ItemFlag:
            ItemIsUserCheckable = 1

        class CheckState:
            Unchecked = 0
            Checked = 2

        class ItemDataRole:
            UserRole = 256

        class DockWidgetArea:
            RightDockWidgetArea = 1

        class Orientation:
            Horizontal = 1

    qt_core.Qt = _Qt
    qt_core.pyqtSignal = lambda *a, **kw: MagicMock()

    class _QTimer:
        @staticmethod
        def singleShot(ms, fn):
            pass

    qt_core.QTimer = _QTimer

    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core
    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtCore"] = qt_core

    class _QWidget:
        def __init__(self, *args, **kwargs):
            pass

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QWidget = _QWidget

    for name in [
        "QVBoxLayout",
        "QHBoxLayout",
        "QLabel",
        "QPushButton",
        "QListWidget",
        "QListWidgetItem",
        "QGroupBox",
        "QLineEdit",
        "QDoubleSpinBox",
        "QSlider",
        "QComboBox",
        "QTableWidget",
        "QTableWidgetItem",
        "QHeaderView",
        "QDockWidget",
        "QFileDialog",
        "QDialog",
        "QColorDialog",
    ]:
        setattr(qt_widgets, name, MagicMock)
    # QMessageBox.warning/.information/.critical are called like static
    # methods, so it must be a Mock *instance* (auto-attrs), not the
    # MagicMock class itself.
    qt_widgets.QMessageBox = MagicMock()
    pyqt6.QtWidgets = qt_widgets
    sys.modules["PyQt6.QtWidgets"] = qt_widgets

    sys.modules["rdkit"] = MagicMock()
    sys.modules["pyscf"] = None


_install_stubs()

_vis_tab_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "vis_tab.py"),
    "pyscf_calculator_vis_tab_under_test",
)
VisTab = _vis_tab_mod.VisTab


def _make_vis_tab():
    vt = VisTab.__new__(VisTab)
    vt.parent_dialog = MagicMock()
    vt.context = MagicMock()
    vt.chkfile_path = "/fake/out/pyscf.chk"
    vt.last_out_dir = "/fake/out"
    vt.prop_worker = None
    vt.load_worker = None
    vt.btn_run_analysis = MagicMock()
    vt.orb_list = MagicMock()
    vt.log = MagicMock()
    return vt


class TestRunSpecificAnalysisBusyGuard(unittest.TestCase):
    """
    Regression test: run_specific_analysis() used to unconditionally create
    and start a new PropertyWorker, even if a previous one (from the
    button-triggered "Generate & Visualize Selected", or from double
    clicking an orbital in the Energy Diagram) was still running. That
    silently replaced self.prop_worker while the old QThread was still
    active, racing the finished/result signal handlers against each other.
    """

    def setUp(self):
        self.mock_worker_cls = MagicMock()
        self.mock_worker_instance = MagicMock()
        self.mock_worker_cls.return_value = self.mock_worker_instance
        _vis_tab_mod.PropertyWorker = self.mock_worker_cls

    def test_starts_worker_when_idle(self):
        vt = _make_vis_tab()
        with patch.object(_vis_tab_mod.os.path, "exists", return_value=True):
            vt.run_specific_analysis(["ESP"])
        self.mock_worker_instance.start.assert_called_once()
        self.assertIs(vt.prop_worker, self.mock_worker_instance)

    def test_blocks_when_worker_already_running(self):
        vt = _make_vis_tab()
        running_worker = MagicMock()
        running_worker.isRunning.return_value = True
        vt.prop_worker = running_worker

        with patch.object(_vis_tab_mod.os.path, "exists", return_value=True):
            vt.run_specific_analysis(["ESP"])

        # A second worker must NOT have been constructed/started.
        self.mock_worker_cls.assert_not_called()
        # The original (still running) worker reference must be preserved.
        self.assertIs(vt.prop_worker, running_worker)

    def test_allows_new_run_after_previous_worker_finished(self):
        vt = _make_vis_tab()
        finished_worker = MagicMock()
        finished_worker.isRunning.return_value = False
        vt.prop_worker = finished_worker

        with patch.object(_vis_tab_mod.os.path, "exists", return_value=True):
            vt.run_specific_analysis(["ESP"])

        self.mock_worker_cls.assert_called_once()
        self.assertIs(vt.prop_worker, self.mock_worker_instance)

    def test_missing_chkfile_path_returns_without_starting(self):
        vt = _make_vis_tab()
        vt.chkfile_path = None
        vt.run_specific_analysis(["ESP"])
        self.mock_worker_cls.assert_not_called()

    def test_button_disabled_and_progress_shown(self):
        vt = _make_vis_tab()
        with patch.object(_vis_tab_mod.os.path, "exists", return_value=True):
            vt.run_specific_analysis(["ESP"])
        vt.btn_run_analysis.setEnabled.assert_called_with(False)
        vt.parent_dialog.progress_bar.show.assert_called_once()


if __name__ == "__main__":
    unittest.main()
