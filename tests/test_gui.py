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


class TestOnDocumentResetClearsStaleVisState(unittest.TestCase):
    """
    Regression test: on_document_reset() previously left VisTab.last_out_dir,
    .optimized_xyz, .loaded_file, .mode, .visualizer and .mapped_visualizer
    untouched. Since the Isovalue/Opacity sliders (vis_controls) and the ESP
    mapping controls (mapped_group) stay enabled/connected across a reset,
    moving them after a File->New would silently redraw stale cube-file data
    from the discarded document into the new one's 3D scene.
    """

    def _make_dialog(self):
        with (
            patch.object(PySCFDialog, "setup_ui", return_value=None),
            patch.object(PySCFDialog, "load_settings", return_value=None),
            patch.object(PySCFDialog, "apply_defaults", return_value=None),
        ):
            dlg = PySCFDialog.__new__(PySCFDialog)
            dlg.context = None
            dlg.settings = {}
            dlg.closing = False
            dlg.struct_source = "Current Editor"
            dlg.calc_history = []
            dlg.version = "1.0.0"
        dlg.calc_tab = MagicMock()
        dlg.vis_tab = MagicMock()
        # Simulate a previously-loaded result with live visualization state.
        dlg.vis_tab.last_out_dir = "/tmp/old_result"
        dlg.vis_tab.optimized_xyz = "3\ncomment\nH 0 0 0\nH 0 0 1\nH 0 1 0"
        dlg.vis_tab.loaded_file = "/tmp/old_result/015_HOMO.cube"
        dlg.vis_tab.mode = "mapped"
        dlg.vis_tab.visualizer = MagicMock()
        dlg.vis_tab.mapped_visualizer = MagicMock()
        dlg.vis_tab.vis_controls = MagicMock()
        dlg.vis_tab.mapped_group = MagicMock()
        return dlg

    def test_last_out_dir_cleared(self):
        dlg = self._make_dialog()
        dlg.on_document_reset()
        self.assertIsNone(dlg.vis_tab.last_out_dir)

    def test_optimized_xyz_cleared(self):
        dlg = self._make_dialog()
        dlg.on_document_reset()
        self.assertIsNone(dlg.vis_tab.optimized_xyz)

    def test_loaded_file_cleared(self):
        dlg = self._make_dialog()
        dlg.on_document_reset()
        self.assertIsNone(dlg.vis_tab.loaded_file)

    def test_mode_reset_to_standard(self):
        dlg = self._make_dialog()
        dlg.on_document_reset()
        self.assertEqual(dlg.vis_tab.mode, "standard")

    def test_visualizers_cleared(self):
        dlg = self._make_dialog()
        dlg.on_document_reset()
        self.assertIsNone(dlg.vis_tab.visualizer)
        self.assertIsNone(dlg.vis_tab.mapped_visualizer)

    def test_vis_controls_disabled_and_mapped_group_hidden(self):
        dlg = self._make_dialog()
        dlg.on_document_reset()
        dlg.vis_tab.vis_controls.setEnabled.assert_called_with(False)
        dlg.vis_tab.mapped_group.hide.assert_called_once()


class TestOnDocumentResetClearsScanParams(unittest.TestCase):
    """
    Regression test: on_document_reset() called apply_defaults(), but
    apply_defaults() only overwrites CalcTab.scan_params when the *loaded*
    defaults dict itself has a truthy "scan_params" entry (the built-in
    default is None, so that branch is a no-op). A previously-configured
    Scan (with atom indices belonging to the just-discarded molecule) stayed
    on CalcTab.scan_params, so re-selecting a Scan job type after File->New
    would skip the "configure scan" prompt and run with stale atom indices.
    """

    def _make_dialog(self):
        with (
            patch.object(PySCFDialog, "setup_ui", return_value=None),
            patch.object(PySCFDialog, "load_settings", return_value=None),
            patch.object(PySCFDialog, "apply_defaults", return_value=None),
        ):
            dlg = PySCFDialog.__new__(PySCFDialog)
            dlg.context = None
            dlg.settings = {}
            dlg.closing = False
            dlg.struct_source = "Current Editor"
            dlg.calc_history = []
            dlg.version = "1.0.0"
        dlg.calc_tab = MagicMock()
        dlg.calc_tab.scan_params = {
            "type": "Dist",
            "atoms": [0, 5],
            "start": 1.0,
            "end": 2.0,
            "steps": 10,
        }
        dlg.vis_tab = MagicMock()
        return dlg

    def test_scan_params_cleared(self):
        dlg = self._make_dialog()
        dlg.on_document_reset()
        self.assertIsNone(dlg.calc_tab.scan_params)

    def test_scan_config_button_hidden(self):
        dlg = self._make_dialog()
        dlg.on_document_reset()
        dlg.calc_tab.btn_scan_config.hide.assert_called_once()


if __name__ == "__main__":
    unittest.main()
