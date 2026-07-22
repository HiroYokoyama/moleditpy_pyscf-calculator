"""
tests/test_gui_coverage.py

Additional coverage for pyscf_calculator/gui.py: setup_ui()/update_proxies(),
log(), on_results()/on_error(), _safe_stop_worker(), closeEvent(),
save_custom_defaults(), load_settings() history/auto-load branches,
update_internal_state() exception paths.

Not covered here: on_document_reset()/apply_defaults()/update_internal_state()
happy path already exercised by tests/test_gui.py (not duplicated).
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

        def closeEvent(self, event):
            pass

    class _QWidget:
        def __init__(self, *args, **kwargs):
            pass

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QWidget = _QWidget
    qt_widgets.QDialog = _QDialog

    def _widget_factory(*args, **kwargs):
        return MagicMock()

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
        setattr(qt_widgets, name, _widget_factory)
    pyqt6.QtWidgets = qt_widgets
    sys.modules["PyQt6.QtWidgets"] = qt_widgets

    sys.modules["rdkit"] = MagicMock()
    sys.modules["pyscf"] = None


_install_stubs()

_gui_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "gui.py"),
    "pyscf_calculator_gui_coverage_under_test",
)
PySCFDialog = _gui_mod.PySCFDialog

# QMessageBox/QToolTip are bare factory functions in the stub above (not
# objects exposing .critical/.showText); replace with real mock instances so
# calls like `QMessageBox.critical(...)` resolve.
_gui_mod.QMessageBox = MagicMock()
_gui_mod.QToolTip = MagicMock()


def _make_dialog_bare():
    """Construct a PySCFDialog without running setup_ui()/load_settings()."""
    with (
        patch.object(PySCFDialog, "setup_ui", return_value=None),
        patch.object(PySCFDialog, "load_settings", return_value=None),
    ):
        dlg = PySCFDialog.__new__(PySCFDialog)
        dlg.context = None
        dlg.settings = {}
        dlg.closing = False
        dlg.struct_source = "Current Editor"
        dlg.calc_history = []
        dlg.version = "1.0.0"
    return dlg


class TestInitVersionTitle(unittest.TestCase):
    def test_title_includes_version_when_provided(self):
        with (
            patch.object(PySCFDialog, "setup_ui", return_value=None),
            patch.object(PySCFDialog, "load_settings", return_value=None),
            patch.object(PySCFDialog, "setWindowTitle") as mock_set_title,
        ):
            PySCFDialog(parent=None, context=None, settings={}, version="9.9.9")
        mock_set_title.assert_called_once_with("PySCF Calculator v9.9.9")


class TestSetupUIRealPath(unittest.TestCase):
    """CalcTab/VisTab are None in this loading mode (relative imports fail
    since gui.py is loaded standalone, not as a package submodule), so
    setup_ui() exercises the "(Error)" placeholder-tab fallback branches."""

    def test_setup_ui_uses_error_placeholders_when_tabs_missing(self):
        self.assertIsNone(_gui_mod.CalcTab)
        self.assertIsNone(_gui_mod.VisTab)

        with patch.object(PySCFDialog, "load_settings", return_value=None):
            dlg = PySCFDialog(parent=None, context=MagicMock(), settings={})

        dlg.tabs.addTab.assert_any_call(
            dlg.tabs.addTab.call_args_list[0][0][0], "Calc (Error)"
        )
        self.assertFalse(hasattr(dlg, "calc_tab"))
        self.assertFalse(hasattr(dlg, "vis_tab"))

    def test_update_proxies_noop_when_tabs_absent(self):
        dlg = _make_dialog_bare()
        dlg.update_proxies()  # must not raise, and sets nothing
        self.assertFalse(hasattr(dlg, "out_dir_edit"))
        self.assertFalse(hasattr(dlg, "btn_load_geom"))

    def test_update_proxies_copies_calc_tab_and_vis_tab_attrs(self):
        dlg = _make_dialog_bare()
        dlg.calc_tab = MagicMock()
        dlg.vis_tab = MagicMock()
        dlg.update_proxies()
        self.assertIs(dlg.out_dir_edit, dlg.calc_tab.out_dir_edit)
        self.assertIs(dlg.progress_bar, dlg.calc_tab.progress_bar)
        self.assertIs(dlg.btn_load_geom, dlg.vis_tab.btn_load_geom)


class TestLog(unittest.TestCase):
    def test_log_noop_while_closing(self):
        dlg = _make_dialog_bare()
        dlg.closing = True
        dlg.calc_tab = MagicMock()
        dlg.log("hello")
        dlg.calc_tab.log.assert_not_called()

    def test_log_delegates_to_calc_tab(self):
        dlg = _make_dialog_bare()
        dlg.calc_tab = MagicMock()
        dlg.log("hello")
        dlg.calc_tab.log.assert_called_once_with("hello")

    def test_log_falls_back_to_logging_when_only_vis_tab(self):
        dlg = _make_dialog_bare()
        dlg.vis_tab = MagicMock()
        with patch.object(_gui_mod.logging, "warning") as mock_warn:
            dlg.log("hello")
        mock_warn.assert_called_once_with("%s", "hello")

    def test_log_noop_when_no_tabs(self):
        dlg = _make_dialog_bare()
        dlg.log("hello")  # must not raise


class TestOnResultsHistoryLimit(unittest.TestCase):
    def test_history_capped_at_ten(self):
        dlg = _make_dialog_bare()
        dlg.context = MagicMock()
        dlg.calc_history = [f"/r{i}" for i in range(10)]
        dlg.update_internal_state = MagicMock()
        dlg.on_results({"out_dir": "/r10"})
        self.assertEqual(len(dlg.calc_history), 10)
        self.assertNotIn("/r0", dlg.calc_history)
        self.assertIn("/r10", dlg.calc_history)

    def test_no_out_dir_skips_history_update(self):
        dlg = _make_dialog_bare()
        dlg.context = MagicMock()
        dlg.update_internal_state = MagicMock()
        dlg.on_results({})
        dlg.update_internal_state.assert_not_called()

    def test_delegates_to_vis_tab_when_present(self):
        dlg = _make_dialog_bare()
        dlg.context = MagicMock()
        dlg.update_internal_state = MagicMock()
        dlg.vis_tab = MagicMock()
        dlg.on_results({"out_dir": "/r"})
        dlg.vis_tab.on_calculation_finished.assert_called_once_with({"out_dir": "/r"})


class TestOnError(unittest.TestCase):
    def test_on_error_logs_shows_critical_and_cleans_up(self):
        dlg = _make_dialog_bare()
        dlg.calc_tab = MagicMock()
        with patch.object(_gui_mod.QMessageBox, "critical") as mock_crit:
            dlg.on_error("bad")
        mock_crit.assert_called_once()
        dlg.calc_tab.cleanup_ui_state.assert_called_once()

    def test_on_error_no_calc_tab_no_crash(self):
        dlg = _make_dialog_bare()
        with patch.object(_gui_mod.QMessageBox, "critical"):
            dlg.on_error("bad")  # must not raise


class TestSafeStopWorker(unittest.TestCase):
    def _make_dlg(self):
        return _make_dialog_bare()

    def test_worker_none_is_noop(self):
        dlg = self._make_dlg()
        dlg._safe_stop_worker(None)  # must not raise

    def test_worker_not_running_is_noop(self):
        dlg = self._make_dlg()
        worker = MagicMock()
        worker.isRunning.return_value = False
        dlg._safe_stop_worker(worker)
        worker.terminate.assert_not_called()

    def test_stream_close_exception_silenced(self):
        dlg = self._make_dlg()
        worker = MagicMock()
        worker.isRunning.return_value = True
        worker._stream.close.side_effect = RuntimeError("boom")
        worker.wait.return_value = True
        dlg._safe_stop_worker(worker)  # must not raise
        self.assertTrue(worker._stop_requested)

    def test_disconnect_exception_silenced(self):
        dlg = self._make_dlg()
        worker = MagicMock()
        worker.isRunning.return_value = True
        worker._stream = None
        worker.finished_signal.disconnect.side_effect = RuntimeError("boom")
        worker.wait.return_value = True
        dlg._safe_stop_worker(worker)  # must not raise

    def test_force_terminate_on_wait_timeout(self):
        dlg = self._make_dlg()
        worker = MagicMock()
        worker.isRunning.return_value = True
        worker._stream = None
        worker.wait.return_value = False
        dlg._safe_stop_worker(worker)
        worker.terminate.assert_called_once()

    def test_no_terminate_when_wait_succeeds(self):
        dlg = self._make_dlg()
        worker = MagicMock()
        worker.isRunning.return_value = True
        worker._stream = None
        worker.wait.return_value = True
        dlg._safe_stop_worker(worker)
        worker.terminate.assert_not_called()


class TestCloseEvent(unittest.TestCase):
    def test_close_event_stops_calc_and_vis_workers(self):
        dlg = _make_dialog_bare()
        dlg.calc_tab = MagicMock()
        dlg.vis_tab = MagicMock()
        dlg.vis_tab.freq_dock = MagicMock()
        dlg._safe_stop_worker = MagicMock()

        dlg.closeEvent(MagicMock())

        self.assertTrue(dlg.closing)
        dlg.calc_tab.stop_calculation.assert_called_once()
        dlg.vis_tab.clear_3d_actors.assert_called_once()
        self.assertEqual(dlg._safe_stop_worker.call_count, 2)
        dlg.vis_tab.freq_dock.close.assert_called_once()

    def test_close_event_no_freq_dock(self):
        dlg = _make_dialog_bare()
        dlg.calc_tab = MagicMock()
        dlg.vis_tab = MagicMock()
        dlg.vis_tab.freq_dock = None
        dlg._safe_stop_worker = MagicMock()

        dlg.closeEvent(MagicMock())  # must not raise

    def test_close_event_no_tabs(self):
        dlg = _make_dialog_bare()
        dlg.closeEvent(MagicMock())  # must not raise
        self.assertTrue(dlg.closing)


class TestSaveCustomDefaults(unittest.TestCase):
    def test_noop_when_no_calc_tab(self):
        dlg = _make_dialog_bare()
        dlg.save_custom_defaults()  # must not raise

    def test_writes_settings_json_and_shows_tooltip(self):
        dlg = _make_dialog_bare()
        dlg.calc_tab = MagicMock()
        dlg.calc_tab.out_dir_edit.text.return_value = "/out"
        dlg.calc_tab.spin_threads.value.return_value = 4
        dlg.calc_tab.spin_memory.value.return_value = 4000
        dlg.calc_tab.job_type_combo.currentText.return_value = "Energy"
        dlg.calc_tab.method_combo.currentText.return_value = "RKS"
        dlg.calc_tab.functional_combo.currentText.return_value = "b3lyp"
        dlg.calc_tab.basis_combo.currentText.return_value = "sto-3g"
        dlg.calc_tab.check_symmetry.isChecked.return_value = False
        dlg.calc_tab.spin_cycles.value.return_value = 100
        dlg.calc_tab.edit_conv.text.return_value = "1e-9"
        dlg.calc_tab.spin_grid_level.value.return_value = 3
        dlg.calc_tab.solvent_combo.currentText.return_value = "None (Vacuum)"
        dlg.calc_tab.scan_params = None
        dlg.log = MagicMock()
        dlg.cursor = MagicMock(return_value=MagicMock())

        m_open = unittest.mock.mock_open()
        with (
            patch("builtins.open", m_open),
            patch.object(_gui_mod.json, "dump") as mock_dump,
        ):
            dlg.save_custom_defaults()

        mock_dump.assert_called_once()
        dlg.log.assert_called_with("Default settings saved.")
        _gui_mod.QToolTip.showText.assert_called_once()

    def test_write_exception_logs_failure(self):
        dlg = _make_dialog_bare()
        dlg.calc_tab = MagicMock()
        dlg.calc_tab.out_dir_edit.text.return_value = "/out"
        dlg.calc_tab.spin_threads.value.return_value = 4
        dlg.calc_tab.spin_memory.value.return_value = 4000
        dlg.calc_tab.job_type_combo.currentText.return_value = "Energy"
        dlg.calc_tab.method_combo.currentText.return_value = "RKS"
        dlg.calc_tab.functional_combo.currentText.return_value = "b3lyp"
        dlg.calc_tab.basis_combo.currentText.return_value = "sto-3g"
        dlg.calc_tab.check_symmetry.isChecked.return_value = False
        dlg.calc_tab.spin_cycles.value.return_value = 100
        dlg.calc_tab.edit_conv.text.return_value = "1e-9"
        dlg.calc_tab.spin_grid_level.value.return_value = 3
        dlg.calc_tab.solvent_combo.currentText.return_value = "None (Vacuum)"
        dlg.calc_tab.scan_params = None
        dlg.log = MagicMock()

        with patch("builtins.open", side_effect=OSError("disk full")):
            dlg.save_custom_defaults()

        self.assertIn("Failed to save default settings", dlg.log.call_args[0][0])


class TestLoadSettingsHistoryAndAutoLoad(unittest.TestCase):
    def _make_dlg_for_load_settings(self):
        with patch.object(PySCFDialog, "apply_defaults", return_value=None):
            dlg = PySCFDialog.__new__(PySCFDialog)
        dlg.context = None
        dlg.settings = {}
        dlg.closing = False
        dlg.struct_source = "Current Editor"
        dlg.calc_history = []
        dlg.version = "1.0.0"
        dlg.calc_tab = None
        dlg.vis_tab = None
        return dlg

    def test_no_context_no_history_no_crash(self):
        dlg = self._make_dlg_for_load_settings()
        dlg.load_settings()
        self.assertEqual(dlg.calc_history, [])

    def test_context_exception_silenced_project_dir_lookup(self):
        dlg = self._make_dlg_for_load_settings()
        context = MagicMock()
        context.get_main_window.side_effect = RuntimeError("boom")
        dlg.context = context
        dlg.load_settings()  # must not raise

    def test_relative_history_resolved_against_project_dir(self):
        dlg = self._make_dlg_for_load_settings()
        context = MagicMock()
        mw = MagicMock()
        mw.init_manager.current_file_path = os.path.join("proj", "file.pmeprj")
        context.get_main_window.return_value = mw
        dlg.context = context
        dlg.settings = {"calc_history": ["rel_result"]}

        dlg.load_settings()

        self.assertEqual(len(dlg.calc_history), 1)
        self.assertTrue(
            os.path.isabs(dlg.calc_history[0]) or "proj" in dlg.calc_history[0]
        )

    def test_history_relpath_exception_keeps_raw_path(self):
        dlg = self._make_dlg_for_load_settings()
        context = MagicMock()
        mw = MagicMock()
        mw.init_manager.current_file_path = os.path.join("proj", "file.pmeprj")
        context.get_main_window.return_value = mw
        dlg.context = context
        dlg.settings = {"calc_history": ["rel_result"]}

        with patch.object(_gui_mod.os.path, "isabs", side_effect=RuntimeError("boom")):
            dlg.load_settings()  # must not raise

        self.assertEqual(dlg.calc_history, ["rel_result"])

    def test_struct_source_restored_from_settings(self):
        dlg = self._make_dlg_for_load_settings()
        dlg.vis_tab = MagicMock()
        dlg.settings = {"struct_source": "Loaded File"}
        dlg.load_settings()
        self.assertEqual(dlg.struct_source, "Loaded File")
        dlg.vis_tab.lbl_struct_source.setText.assert_called_with(
            "Structure Source: Loaded File"
        )

    def test_auto_load_latest_result_schedules_singleshot(self):
        dlg = self._make_dlg_for_load_settings()
        dlg.vis_tab = MagicMock()
        dlg.log = MagicMock()
        existing_dir = os.path.dirname(os.path.abspath(__file__))
        dlg.settings = {"calc_history": [existing_dir]}

        with patch.object(_gui_mod.QTimer, "singleShot") as mock_single_shot:
            dlg.load_settings()

        mock_single_shot.assert_called_once()
        dlg.log.assert_called_with(f"Auto-loading latest result: {existing_dir}")

    def test_last_path_not_a_dir_no_auto_load(self):
        dlg = self._make_dlg_for_load_settings()
        dlg.vis_tab = MagicMock()
        dlg.log = MagicMock()
        dlg.settings = {"calc_history": ["/nonexistent/path/xyz"]}

        with patch.object(_gui_mod.QTimer, "singleShot") as mock_single_shot:
            dlg.load_settings()

        mock_single_shot.assert_not_called()

    def test_calc_tab_settings_applied_when_present(self):
        dlg = self._make_dlg_for_load_settings()
        dlg.calc_tab = MagicMock()
        dlg.settings = {
            "job_type": "Energy",
            "method": "RKS",
            "functional": "b3lyp",
            "basis": "sto-3g",
            "charge": "0",
            "spin": "0",
            "out_dir": "/out",
            "threads": 4,
            "memory": 4000,
            "check_symmetry": True,
            "spin_cycles": 200,
            "conv_tol": "1e-8",
            "grid_level": 4,
            "scan_params": {"type": "Dist"},
            "solvent": "water",
        }
        dlg.load_settings()
        dlg.calc_tab.job_type_combo.setCurrentText.assert_called_with("Energy")
        dlg.calc_tab.solvent_combo.setCurrentText.assert_called_with("water")
        self.assertEqual(dlg.calc_tab.scan_params, {"type": "Dist"})


class TestUpdateInternalStateExceptionPaths(unittest.TestCase):
    def _make_dlg(self):
        dlg = _make_dialog_bare()
        dlg.calc_tab = MagicMock()
        dlg.calc_tab.job_type_combo.currentText.return_value = "Energy"
        dlg.calc_tab.method_combo.currentText.return_value = "RKS"
        dlg.calc_tab.functional_combo.currentText.return_value = "b3lyp"
        dlg.calc_tab.basis_combo.currentText.return_value = "sto-3g"
        dlg.calc_tab.charge_input.currentText.return_value = "0"
        dlg.calc_tab.spin_input.currentText.return_value = "0"
        dlg.calc_tab.out_dir_edit.text.return_value = "relative_out"
        dlg.calc_tab.spin_threads.value.return_value = 4
        dlg.calc_tab.spin_memory.value.return_value = 4000
        dlg.calc_tab.check_symmetry.isChecked.return_value = False
        dlg.calc_tab.spin_cycles.value.return_value = 100
        dlg.calc_tab.edit_conv.text.return_value = "1e-9"
        dlg.calc_tab.spin_grid_level.value.return_value = 3
        dlg.calc_tab.solvent_combo.currentText.return_value = "None (Vacuum)"
        return dlg

    def test_relpath_exception_keeps_absolute_history(self):
        dlg = self._make_dlg()
        dlg.calc_history = ["/abs/history/path"]
        mw = MagicMock()
        mw.init_manager.current_file_path = os.path.join("proj", "file.pmeprj")
        dlg.context = MagicMock()
        dlg.context.get_main_window.return_value = mw

        with patch.object(_gui_mod.os.path, "relpath", side_effect=ValueError("boom")):
            dlg.update_internal_state()

        self.assertEqual(dlg.settings["calc_history"], ["/abs/history/path"])

    def test_relative_out_dir_uses_relative_history(self):
        # NOTE: tests/test_calc_tab.py's helper permanently monkeypatches the
        # real os.path.isabs() to always return True (it does
        # `_calc_tab_mod.os.path = os.path` then rebinds `.isabs` on that
        # shared object). Force the correct "relative" evaluation here
        # regardless of prior test-order pollution.
        dlg = self._make_dlg()
        dlg.calc_history = [os.path.join("proj", "results", "run1")]
        mw = MagicMock()
        mw.init_manager.current_file_path = os.path.join("proj", "file.pmeprj")
        dlg.context = MagicMock()
        dlg.context.get_main_window.return_value = mw

        with patch.object(_gui_mod.os.path, "isabs", return_value=False):
            dlg.update_internal_state()

        self.assertEqual(
            dlg.settings["calc_history"], [os.path.join("results", "run1")]
        )

    def test_no_context_skips_relpath_logic(self):
        dlg = self._make_dlg()
        dlg.calc_history = ["some_history"]
        dlg.context = None
        dlg.update_internal_state()  # must not raise
        self.assertEqual(dlg.settings["calc_history"], ["some_history"])

    def test_associated_filename_exception_silenced(self):
        dlg = self._make_dlg()
        dlg.calc_history = []
        dlg.calc_tab.out_dir_edit.text.return_value = "/abs/out"
        context = MagicMock()
        context.get_main_window.side_effect = RuntimeError("boom")
        dlg.context = context

        dlg.update_internal_state()  # must not raise

    def test_associated_filename_set_when_present(self):
        dlg = self._make_dlg()
        dlg.calc_history = []
        dlg.calc_tab.out_dir_edit.text.return_value = "/abs/out"
        mw = MagicMock()
        mw.init_manager.current_file_path = os.path.join("proj", "file.pmeprj")
        dlg.context = MagicMock()
        dlg.context.get_main_window.return_value = mw

        dlg.update_internal_state()

        self.assertEqual(dlg.settings["associated_filename"], "file.pmeprj")


class TestApplyDefaultsUserOverrides(unittest.TestCase):
    def _make_dlg(self):
        with patch.object(PySCFDialog, "setup_ui", return_value=None):
            dlg = PySCFDialog.__new__(PySCFDialog)
        dlg.context = None
        dlg.settings = {}
        dlg.closing = False
        dlg.struct_source = "Current Editor"
        dlg.calc_history = []
        dlg.version = "1.0.0"
        dlg.calc_tab = MagicMock()
        return dlg

    def test_corrupt_settings_json_is_silenced(self):
        dlg = self._make_dlg()
        with (
            patch.object(_gui_mod.os.path, "exists", return_value=True),
            patch("builtins.open", unittest.mock.mock_open(read_data="{bad json")),
        ):
            dlg.apply_defaults()  # must not raise
        dlg.calc_tab.job_type_combo.setCurrentText.assert_called_with(
            "Optimization + Frequency"
        )

    def test_user_scan_params_and_scan_job_shows_config_button(self):
        dlg = self._make_dlg()
        dlg.calc_tab.btn_scan_config = MagicMock()
        user_defaults = {
            "job_type": "Rigid Surface Scan",
            "scan_params": {"type": "Dist", "steps": 5},
        }
        with (
            patch.object(_gui_mod.os.path, "exists", return_value=True),
            patch.object(_gui_mod.json, "load", return_value=user_defaults),
            patch("builtins.open", unittest.mock.mock_open()),
        ):
            dlg.apply_defaults()

        self.assertEqual(dlg.calc_tab.scan_params, {"type": "Dist", "steps": 5})
        dlg.calc_tab.btn_scan_config.show.assert_called_once()


class TestOnDocumentResetSettingsCleared(unittest.TestCase):
    def test_calc_history_and_associated_filename_removed_from_settings(self):
        with (
            patch.object(PySCFDialog, "setup_ui", return_value=None),
            patch.object(PySCFDialog, "apply_defaults", return_value=None),
        ):
            dlg = PySCFDialog.__new__(PySCFDialog)
        dlg.context = None
        dlg.settings = {
            "calc_history": ["/a", "/b"],
            "associated_filename": "old.pmeprj",
        }
        dlg.closing = False
        dlg.struct_source = "Current Editor"
        dlg.calc_history = []
        dlg.version = "1.0.0"
        dlg.calc_tab = None
        dlg.vis_tab = None
        dlg.log = MagicMock()

        dlg.on_document_reset()

        self.assertEqual(dlg.settings["calc_history"], [])
        self.assertNotIn("associated_filename", dlg.settings)


class TestSaveSettings(unittest.TestCase):
    def test_save_settings_delegates_to_update_internal_state(self):
        dlg = _make_dialog_bare()
        dlg.update_internal_state = MagicMock()
        dlg.save_settings()
        dlg.update_internal_state.assert_called_once()


if __name__ == "__main__":
    unittest.main()
