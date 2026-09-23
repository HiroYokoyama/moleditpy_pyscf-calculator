import json
import logging
import os
import traceback

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QMessageBox,
    QTabWidget,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# Local Imports
try:
    from .calc_tab import CalcTab
    from .scan_dialog import ScanDialog
    from .scan_results import ScanResultDialog
    from .vis_tab import VisTab
    from .worker import LoadWorker, PropertyWorker, PySCFWorker
except ImportError:
    traceback.print_exc()
    PySCFWorker = None
    LoadWorker = None
    PropertyWorker = None
    ScanDialog = None
    ScanResultDialog = None
    CalcTab = None
    VisTab = None


# Every persisted Calculation-tab setting, once: (settings key, CalcTab
# widget attribute, widget kind, default). Saving, restoring and "Save as
# Default" all walk this table, so a new option cannot be half-persisted
# (Break Initial Guess Symmetry once was missing from three of the four
# hand-written lists).
_FIELDS = (
    ("job_type", "job_type_combo", "combo", "Optimization + Frequency"),
    ("method", "method_combo", "combo", "RKS"),
    ("functional", "functional_combo", "combo", "b3lyp"),
    ("basis", "basis_combo", "combo", "sto-3g"),
    ("charge", "charge_input", "combo", "0"),
    ("spin", "spin_input", "combo", "1 (Singlet)"),
    ("threads", "spin_threads", "int", 0),
    ("memory", "spin_memory", "int", 4000),
    ("check_symmetry", "check_symmetry", "bool", False),
    ("break_symmetry", "check_break_sym", "bool", False),
    ("hessian", "hessian_combo", "combo", "Analytic"),
    ("dispersion", "dispersion_combo", "combo", "None"),
    ("temperature", "spin_temperature", "float", 298.15),
    ("pressure_atm", "spin_pressure", "float", 1.0),
    ("spin_cycles", "spin_cycles", "int", 100),
    ("conv_tol", "edit_conv", "line", "1e-9"),
    ("grid_level", "spin_grid_level", "int", 3),
    ("solvent", "solvent_combo", "combo", "None (Vacuum)"),
)
# Per-molecule, so not part of the user's global defaults.
_NOT_IN_DEFAULTS = ("charge", "spin")

_GETTERS = {
    "combo": lambda w: w.currentText(),
    "line": lambda w: w.text(),
    "int": lambda w: w.value(),
    "float": lambda w: w.value(),
    "bool": lambda w: w.isChecked(),
}
_SETTERS = {
    "combo": lambda w, v: w.setCurrentText(str(v)),
    "line": lambda w, v: w.setText(str(v)),
    "int": lambda w, v: w.setValue(int(v)),
    "float": lambda w, v: w.setValue(float(v)),
    "bool": lambda w, v: w.setChecked(bool(v)),
}


def _read_fields(calc_tab, keys=None):
    return {
        key: _GETTERS[kind](getattr(calc_tab, attr))
        for key, attr, kind, _ in _FIELDS
        if keys is None or key in keys
    }


def _write_fields(calc_tab, values):
    """Apply every known key present in values; a malformed stored value
    is logged and skipped instead of aborting the whole restore."""
    for key, attr, kind, _ in _FIELDS:
        if key not in values:
            continue
        try:
            _SETTERS[kind](getattr(calc_tab, attr), values[key])
        except (TypeError, ValueError) as exc:
            logger.warning("ignoring stored %s=%r: %s", key, values[key], exc)


def _defaults_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")


class PySCFDialog(QDialog):
    def __init__(self, parent=None, context=None, settings=None, version=None):
        super().__init__(parent)
        self.context = context
        self.settings = settings if settings is not None else {}
        self.closing = False
        self.struct_source = "Current Editor"
        self.calc_history = []

        title = "PySCF Calculator"
        self.version = version
        if version:
            title += f" v{version}"
        self.setWindowTitle(title)

        self.resize(600, 700)

        # Load Settings (Pre-UI) to ensure defaults logic
        # But we need UI to populate.

        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # === TAB 1: Calculation ===
        if CalcTab:
            self.calc_tab = CalcTab(self, self.context, self.settings)
            self.tabs.addTab(self.calc_tab, "Calculation")
        else:
            self.tabs.addTab(QWidget(), "Calc (Error)")

        # === TAB 2: Visualization ===
        if VisTab:
            self.vis_tab = VisTab(self, self.context)
            self.tabs.addTab(self.vis_tab, "Visualization")
        else:
            self.tabs.addTab(QWidget(), "Vis (Error)")

        # Exposing pointers for legacy access or inter-tab comms if needed
        # self.out_dir_edit is in calc_tab.
        # But vis_tab accesses parent_dialog.out_dir_edit...
        # I need to proxy or fix access.
        # FIX: VisTab uses self.parent_dialog.out_dir_edit.text() fallback.
        # I should expose properties or direct objects.

        self.update_proxies()

    def update_proxies(self):
        # Create proxies for properties that tabs might expect on parent
        # or that I want to expose for convenience.
        if getattr(self, "calc_tab", None) is not None:
            self.out_dir_edit = self.calc_tab.out_dir_edit
            self.progress_bar = self.calc_tab.progress_bar
            self.run_btn = self.calc_tab.run_btn
            self.stop_btn = self.calc_tab.stop_btn
            self.job_type_combo = self.calc_tab.job_type_combo
            self.method_combo = self.calc_tab.method_combo
            self.functional_combo = self.calc_tab.functional_combo
            self.basis_combo = self.calc_tab.basis_combo
            self.charge_input = self.calc_tab.charge_input
            self.spin_input = self.calc_tab.spin_input
            self.spin_threads = self.calc_tab.spin_threads
            self.spin_memory = self.calc_tab.spin_memory
            self.check_symmetry = self.calc_tab.check_symmetry
            self.check_break_sym = self.calc_tab.check_break_sym
            self.spin_cycles = self.calc_tab.spin_cycles
            self.edit_conv = self.calc_tab.edit_conv
            self.spin_grid_level = self.calc_tab.spin_grid_level

        if getattr(self, "vis_tab", None) is not None:
            self.btn_load_geom = self.vis_tab.btn_load_geom

    def log(self, message):
        if self.closing:
            return
        if getattr(self, "calc_tab", None) is not None:
            self.calc_tab.log(message)
        elif getattr(self, "vis_tab", None) is not None:
            logger.warning("%s", message)

    def on_results(self, result_data):
        # Called by CalcTab worker

        self.log("Processing results...")

        # Update History
        out_dir = result_data.get("out_dir", None)
        if out_dir:
            self.calc_history.append(out_dir)
            # Limit history
            if len(self.calc_history) > 10:
                self.calc_history.pop(0)
            self.update_internal_state()  # Save settings

        # Mark project as modified
        if self.context:
            self.context.mark_project_modified()
        # Delegate to VisTab
        if getattr(self, "vis_tab", None) is not None:
            self.vis_tab.on_calculation_finished(result_data)

    def on_error(self, err_msg):
        self.log(f"\nERROR: {err_msg}")
        QMessageBox.critical(self, "Error", err_msg)
        if getattr(self, "calc_tab", None) is not None:
            self.calc_tab.cleanup_ui_state()

    def _safe_stop_worker(self, worker):
        if worker and worker.isRunning():
            worker._stop_requested = True
            try:
                if getattr(worker, "_stream", None):
                    worker._stream.close()
            except Exception:
                pass

            try:
                if hasattr(worker, "finished_signal"):
                    worker.finished_signal.disconnect()
                if hasattr(worker, "error_signal"):
                    worker.error_signal.disconnect()
                if hasattr(worker, "log_signal"):
                    worker.log_signal.disconnect()
                if hasattr(worker, "result_signal"):
                    worker.result_signal.disconnect()
            except Exception:
                pass

            if not worker.wait(1500):
                worker.terminate()
                worker.wait(500)

    def closeEvent(self, event):
        self.closing = True

        # Stop CalcTab Worker
        if getattr(self, "calc_tab", None) is not None:
            self.calc_tab.stop_calculation()

        # Cleanup VisTab Actors/Workers
        if getattr(self, "vis_tab", None) is not None:
            self.vis_tab.clear_3d_actors()
            self._safe_stop_worker(self.vis_tab.load_worker)
            self._safe_stop_worker(self.vis_tab.prop_worker)

            # Close Dock
            if self.vis_tab.freq_dock:
                self.vis_tab.freq_dock.close()

        super().closeEvent(event)

    def on_document_reset(self):
        """Callback to reset plugin state when the document is reset (File -> New)."""
        # Abort workers
        if getattr(self, "calc_tab", None) is not None:
            self.calc_tab.stop_calculation()

            # Clear any Scan configuration from the discarded document —
            # atom indices in scan_params refer to the old molecule and
            # would silently be reused (without re-prompting the user) if a
            # Scan job type is selected again after File -> New.
            self.calc_tab.scan_params = None
            if getattr(self.calc_tab, "btn_scan_config", None) is not None:
                self.calc_tab.btn_scan_config.hide()

        if getattr(self, "vis_tab", None) is not None:
            self._safe_stop_worker(self.vis_tab.load_worker)
            self.vis_tab.load_worker = None

            self._safe_stop_worker(self.vis_tab.prop_worker)
            self.vis_tab.prop_worker = None

            self.vis_tab.clear_3d_actors()
            self.vis_tab.chkfile_path = None
            self.vis_tab.mo_data = None
            self.vis_tab.freq_data = None
            self.vis_tab.thermo_data = None
            self.vis_tab.orb_list.clear()
            self.vis_tab.file_list.clear()
            self.vis_tab.result_path_display.clear()
            self.vis_tab.btn_load_geom.setEnabled(False)
            self.vis_tab.btn_run_analysis.setEnabled(False)
            self.vis_tab.btn_show_diagram.setEnabled(False)
            self.vis_tab.btn_show_thermo.setEnabled(False)

            self.vis_tab.close_freq_window()

            # Clear stale visualization state so the Isovalue/Opacity/ESP
            # controls (which stay enabled and connected to their update
            # slots) don't silently redraw a cube file or grid belonging to
            # the document that was just discarded.
            self.vis_tab.last_out_dir = None
            self.vis_tab.optimized_xyz = None
            self.vis_tab.loaded_file = None
            self.vis_tab.mode = "standard"
            self.vis_tab.visualizer = None
            self.vis_tab.mapped_visualizer = None
            if getattr(self.vis_tab, "vis_controls", None) is not None:
                self.vis_tab.vis_controls.setEnabled(False)
            if getattr(self.vis_tab, "mapped_group", None) is not None:
                self.vis_tab.mapped_group.hide()

        # Clear internal state
        self.struct_source = "Current Editor"
        self.calc_history = []
        if "calc_history" in self.settings:
            self.settings["calc_history"] = []

        if "associated_filename" in self.settings:
            del self.settings["associated_filename"]

        # Reset Defaults
        self.apply_defaults()

        if getattr(self, "vis_tab", None) is not None:
            self.vis_tab.lbl_struct_source.setText("")

        self.log("Document reset: Plugin state cleared.")

    def save_custom_defaults(self):
        if getattr(self, "calc_tab", None) is None:
            return
        keys = [f[0] for f in _FIELDS if f[0] not in _NOT_IN_DEFAULTS]
        local_settings = _read_fields(self.calc_tab, keys)
        local_settings["root_path"] = self.calc_tab.out_dir_edit.text()
        local_settings["scan_params"] = getattr(self.calc_tab, "scan_params", None)
        try:
            with open(_defaults_path(), "w", encoding="utf-8") as f:
                json.dump(local_settings, f, indent=4)
            self.log("Default settings saved.")
            QToolTip.showText(self.cursor().pos(), "Defaults Saved!", self)
        except (OSError, TypeError) as e:
            self.log(f"Failed to save default settings: {e}")

    def apply_defaults(self):
        defaults = {key: default for key, _, _, default in _FIELDS}
        defaults["root_path"] = os.path.join(os.path.expanduser("~"), "PySCF_Results")
        defaults["scan_params"] = None

        # The user's "Save as Default" choices override the built-ins
        json_path = _defaults_path()
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    defaults.update(json.load(f))
            except (OSError, ValueError) as _e:
                logger.warning("user defaults not readable: %s", _e)

        if getattr(self, "calc_tab", None) is None:
            return
        _write_fields(self.calc_tab, defaults)
        self.calc_tab.out_dir_edit.setText(str(defaults["root_path"]))
        if defaults.get("scan_params"):
            self.calc_tab.scan_params = defaults["scan_params"]
            if "Scan" in str(defaults["job_type"]) and hasattr(
                self.calc_tab, "btn_scan_config"
            ):
                self.calc_tab.btn_scan_config.show()

    def load_settings(self):
        self.apply_defaults()

        s = self.settings
        if getattr(self, "calc_tab", None) is not None:
            _write_fields(self.calc_tab, s)
            if "out_dir" in s:
                self.calc_tab.out_dir_edit.setText(s["out_dir"])
            if "scan_params" in s:
                self.calc_tab.scan_params = s["scan_params"]

        raw_history = s.get("calc_history", [])
        self.calc_history = []

        project_dir = None
        if self.context:
            try:
                mw = self.context.get_main_window()
                current_path = getattr(
                    getattr(mw, "init_manager", None), "current_file_path", None
                )
                if current_path:
                    project_dir = os.path.dirname(current_path)
            except Exception as _e:
                logger.warning("load_settings project_dir silenced: %s", _e)

        for h_path in raw_history:
            final_path = h_path
            try:
                if not os.path.isabs(h_path) and project_dir:
                    final_path = os.path.normpath(os.path.join(project_dir, h_path))
            except Exception as _e:
                logger.warning("load_settings relpath silenced: %s", _e)
            self.calc_history.append(final_path)

        loaded_source = s.get("struct_source", None)
        if loaded_source:
            self.struct_source = loaded_source

        if getattr(self, "vis_tab", None) is not None and self.struct_source:
            self.vis_tab.lbl_struct_source.setText(
                f"Structure Source: {self.struct_source}"
            )

        if self.calc_history:
            last_path = self.calc_history[-1]
            if os.path.exists(last_path) and os.path.isdir(last_path):
                self.log(f"Auto-loading latest result: {last_path}")
                if getattr(self, "vis_tab", None) is not None:
                    QTimer.singleShot(
                        200,
                        lambda: self.vis_tab.load_result_folder(
                            last_path, update_structure=False
                        ),
                    )

    def update_internal_state(self):
        # Syncs UI to self.settings for saving project
        if getattr(self, "calc_tab", None) is not None:
            self.settings.update(_read_fields(self.calc_tab))
            self.settings["out_dir"] = self.calc_tab.out_dir_edit.text()
            self.settings["scan_params"] = getattr(self.calc_tab, "scan_params", None)

        self.settings["version"] = self.version

        # History
        history_to_save = self.calc_history
        if getattr(self, "calc_tab", None) is not None:
            out_dir_val = self.calc_tab.out_dir_edit.text().strip()
            is_relative_setting = not os.path.isabs(out_dir_val)
            if is_relative_setting:
                try:
                    mw = self.context.get_main_window()
                    current_path = getattr(
                        getattr(mw, "init_manager", None), "current_file_path", None
                    )
                    if current_path:
                        project_dir = os.path.dirname(current_path)
                        relative_history = []
                        for h_path in self.calc_history:
                            try:
                                rel = os.path.relpath(h_path, project_dir)
                                relative_history.append(rel)
                            except Exception:
                                relative_history.append(h_path)
                        history_to_save = relative_history
                except Exception as _e:
                    logger.warning("update_internal_state relpath silenced: %s", _e)

        self.settings["calc_history"] = history_to_save
        self.settings["struct_source"] = self.struct_source

        try:
            if self.context:
                mw = self.context.get_main_window()
                if hasattr(mw, "init_manager") and mw.init_manager.current_file_path:
                    self.settings["associated_filename"] = os.path.basename(
                        mw.init_manager.current_file_path
                    )
        except Exception as _e:
            logger.warning("update_internal_state associated_filename silenced: %s", _e)

    def save_settings(self):
        self.update_internal_state()
