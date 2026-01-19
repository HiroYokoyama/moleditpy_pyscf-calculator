import os
import json
import traceback

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QMessageBox, QWidget, QToolTip
)
from PyQt6.QtCore import Qt, QTimer

# Local Imports
try:
    from .worker import PySCFWorker, LoadWorker, PropertyWorker
    from .scan_dialog import ScanDialog
    from .scan_results import ScanResultDialog
    from .calc_tab import CalcTab
    from .vis_tab import VisTab
except ImportError:
    import traceback
    traceback.print_exc()
    PySCFWorker = None
    LoadWorker = None
    PropertyWorker = None
    ScanDialog = None
    ScanResultDialog = None
    CalcTab = None
    VisTab = None

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
        if hasattr(self, 'calc_tab'):
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
        
        if hasattr(self, 'vis_tab'):
            self.btn_load_geom = self.vis_tab.btn_load_geom

    def log(self, message):
        if self.closing: return
        if hasattr(self, 'calc_tab'):
            self.calc_tab.log(message)
        elif hasattr(self, 'vis_tab'):
             # Fallback log?
             print(message)

    def on_results(self, result_data):
        # Called by CalcTab worker
        
        self.log("Processing results...")
        
        # Update History
        out_dir = result_data.get("out_dir")
        if out_dir:
            self.calc_history.append(out_dir)
            # Limit history
            if len(self.calc_history) > 10:
                self.calc_history.pop(0)
            self.update_internal_state() # Save settings
        
        # Mark project as modified
        try:
            if self.context:
                mw = self.context.get_main_window()
                if hasattr(mw, 'has_unsaved_changes'):
                    mw.has_unsaved_changes = True
                    if hasattr(mw, 'update_window_title'):
                        mw.update_window_title()
        except: pass
        # Delegate to VisTab
        if hasattr(self, 'vis_tab'):
            self.vis_tab.on_calculation_finished(result_data)

    def on_error(self, err_msg):
        self.log(f"\nERROR: {err_msg}")
        QMessageBox.critical(self, "Error", err_msg)
        if hasattr(self, 'calc_tab'):
            self.calc_tab.cleanup_ui_state()

    def closeEvent(self, event):
        self.closing = True
        
        # Stop CalcTab Worker
        if hasattr(self, 'calc_tab'):
            self.calc_tab.stop_calculation()
            
        # Cleanup VisTab Actors/Workers
        if hasattr(self, 'vis_tab'):
            self.vis_tab.clear_3d_actors()
            if self.vis_tab.load_worker and self.vis_tab.load_worker.isRunning():
                 self.vis_tab.load_worker.terminate()
            if self.vis_tab.prop_worker and self.vis_tab.prop_worker.isRunning():
                 self.vis_tab.prop_worker.terminate()
            
            # Close Dock
            if self.vis_tab.freq_dock:
                 self.vis_tab.freq_dock.close()
            
        super().closeEvent(event)

    def on_document_reset(self):
        """Callback to reset plugin state when the document is reset (File -> New)."""
        # Abort workers
        if hasattr(self, 'calc_tab'):
             self.calc_tab.stop_calculation()
        
        if hasattr(self, 'vis_tab'):
             if self.vis_tab.load_worker and self.vis_tab.load_worker.isRunning():
                 self.vis_tab.load_worker.terminate()
             self.vis_tab.load_worker = None
             
             if self.vis_tab.prop_worker and self.vis_tab.prop_worker.isRunning():
                 self.vis_tab.prop_worker.terminate()
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

        # Clear internal state
        self.struct_source = "Current Editor"
        self.calc_history = []
        if "calc_history" in self.settings:
             self.settings["calc_history"] = []
        
        if "associated_filename" in self.settings:
            del self.settings["associated_filename"]
            
        # Reset Defaults
        self.apply_defaults()
        
        if hasattr(self, 'vis_tab'):
             self.vis_tab.lbl_struct_source.setText("")
        
        self.log("Document reset: Plugin state cleared.")

    def save_custom_defaults(self):
        # Proxy to CalcTab's settings mostly
        if not hasattr(self, 'calc_tab'): return
        
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
        local_settings = {
             "root_path": self.calc_tab.out_dir_edit.text(),
             "threads": self.calc_tab.spin_threads.value(),
             "memory": self.calc_tab.spin_memory.value(),
             # Calc Settings
             "job_type": self.calc_tab.job_type_combo.currentText(),
             "method": self.calc_tab.method_combo.currentText(),
             "functional": self.calc_tab.functional_combo.currentText(),
             "basis": self.calc_tab.basis_combo.currentText(),
             "check_symmetry": self.calc_tab.check_symmetry.isChecked(),
             "spin_cycles": self.calc_tab.spin_cycles.value(),
             "conv_tol": self.calc_tab.edit_conv.text(),
             "grid_level": self.calc_tab.spin_grid_level.value(),
             "solvent": self.calc_tab.solvent_combo.currentText(),
             "scan_params": getattr(self.calc_tab, 'scan_params', None)
        }
        try:
             with open(json_path, 'w') as f:
                 json.dump(local_settings, f, indent=4)
             
             self.log("Default settings saved.")
             QToolTip.showText(self.cursor().pos(), "Defaults Saved!", self)
             
        except Exception as e:
             self.log(f"Failed to save default settings: {e}")

    def apply_defaults(self):
        # Defaults dict
        defaults = {
            "job_type": "Optimization + Frequency",
            "method": "RKS",
            "functional": "b3lyp",
            "basis": "sto-3g",
            "charge": "0",
            "spin": "0",
            "root_path": os.path.join(os.path.expanduser("~"), "PySCF_Results"),
            "threads": 0,
            "memory": 4000,
            "check_symmetry": False,
            "spin_cycles": 100,
            "conv_tol": "1e-9",
            "grid_level": 3,
            "solvent": "None (Vacuum)",
            "scan_params": None
        }

        # Load User Defaults
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
        if os.path.exists(json_path):
             try:
                 with open(json_path, 'r') as f:
                     user_defaults = json.load(f)
                     defaults.update(user_defaults)
             except: pass
        
        if hasattr(self, 'calc_tab'):
            self.calc_tab.job_type_combo.setCurrentText(defaults["job_type"])
            self.calc_tab.method_combo.setCurrentText(defaults["method"])
            self.calc_tab.functional_combo.setCurrentText(defaults["functional"])
            self.calc_tab.basis_combo.setCurrentText(defaults["basis"])
            self.calc_tab.charge_input.setCurrentText(str(defaults["charge"]))
            self.calc_tab.spin_input.setCurrentText(str(defaults["spin"]))
            
            self.calc_tab.out_dir_edit.setText(defaults["root_path"])
            
            self.calc_tab.spin_threads.setValue(int(defaults["threads"]))
            self.calc_tab.spin_memory.setValue(int(defaults["memory"]))
            
            self.calc_tab.check_symmetry.setChecked(defaults["check_symmetry"])
            self.calc_tab.spin_cycles.setValue(int(defaults["spin_cycles"]))
            self.calc_tab.edit_conv.setText(defaults["conv_tol"])
            self.calc_tab.spin_grid_level.setValue(int(defaults["grid_level"]))
            
            if "solvent" in defaults:
                self.calc_tab.solvent_combo.setCurrentText(defaults["solvent"])
                
            if "scan_params" in defaults and defaults["scan_params"]:
                self.calc_tab.scan_params = defaults["scan_params"]
                if "Scan" in defaults["job_type"]:
                     if hasattr(self.calc_tab, 'btn_scan_config'):
                         self.calc_tab.btn_scan_config.show()

    def load_settings(self):
        self.apply_defaults()
        
        s = self.settings
        if hasattr(self, 'calc_tab'):
            if "job_type" in s: self.calc_tab.job_type_combo.setCurrentText(s["job_type"])
            if "method" in s: self.calc_tab.method_combo.setCurrentText(s["method"])
            if "functional" in s: self.calc_tab.functional_combo.setCurrentText(s["functional"])
            if "basis" in s: self.calc_tab.basis_combo.setCurrentText(s["basis"])
            if "charge" in s: self.calc_tab.charge_input.setCurrentText(s["charge"])
            if "spin" in s: self.calc_tab.spin_input.setCurrentText(s["spin"])
            if "out_dir" in s: self.calc_tab.out_dir_edit.setText(s["out_dir"])
            
            # Restore extended settings
            if "threads" in s: self.calc_tab.spin_threads.setValue(int(s["threads"]))
            if "memory" in s: self.calc_tab.spin_memory.setValue(int(s["memory"]))
            if "check_symmetry" in s: self.calc_tab.check_symmetry.setChecked(bool(s["check_symmetry"]))
            if "spin_cycles" in s: self.calc_tab.spin_cycles.setValue(int(s["spin_cycles"]))
            if "spin_cycles" in s: self.calc_tab.spin_cycles.setValue(int(s["spin_cycles"]))
            if "conv_tol" in s: self.calc_tab.edit_conv.setText(str(s["conv_tol"]))
            if "grid_level" in s: self.calc_tab.spin_grid_level.setValue(int(s["grid_level"]))
            if "scan_params" in s: self.calc_tab.scan_params = s["scan_params"]
            if "solvent" in s: self.calc_tab.solvent_combo.setCurrentText(s["solvent"])
        
        raw_history = s.get("calc_history", [])
        self.calc_history = []
        
        project_dir = None
        if self.context:
            try:
                mw = self.context.get_main_window()
                current_path = getattr(mw, 'current_file_path', None)
                if current_path:
                    project_dir = os.path.dirname(current_path)
            except: pass
        
        for h_path in raw_history:
             final_path = h_path
             try:
                 if not os.path.isabs(h_path) and project_dir:
                     final_path = os.path.normpath(os.path.join(project_dir, h_path))
             except: pass
             self.calc_history.append(final_path)

        loaded_source = s.get("struct_source", None)
        if loaded_source:
             self.struct_source = loaded_source
             
        if hasattr(self, 'vis_tab') and self.struct_source:
             self.vis_tab.lbl_struct_source.setText(f"Structure Source: {self.struct_source}")

        if self.calc_history:
             last_path = self.calc_history[-1]
             
             should_reset = False
             # Logic for reset based on file association could go here if re-implemented.
             # For now, we trust document_reset hook.
             
             if should_reset:
                 # Reset logic...
                 pass
             else:
                 # Auto-load logic
                 if os.path.exists(last_path) and os.path.isdir(last_path):
                     self.log(f"Auto-loading latest result: {last_path}")
                     if hasattr(self, 'vis_tab'):
                         QTimer.singleShot(200, lambda: self.vis_tab.load_result_folder(last_path, update_structure=False))

    def update_internal_state(self):
        # Syncs UI to self.settings for saving project
        if hasattr(self, 'calc_tab'):
            self.settings["job_type"] = self.calc_tab.job_type_combo.currentText()
            self.settings["method"] = self.calc_tab.method_combo.currentText()
            self.settings["functional"] = self.calc_tab.functional_combo.currentText()
            self.settings["basis"] = self.calc_tab.basis_combo.currentText()
            self.settings["charge"] = self.calc_tab.charge_input.currentText()
            self.settings["spin"] = self.calc_tab.spin_input.currentText()
            self.settings["out_dir"] = self.calc_tab.out_dir_edit.text()
            
            # Additional Settings ensuring project state match defaults
            self.settings["threads"] = self.calc_tab.spin_threads.value()
            self.settings["memory"] = self.calc_tab.spin_memory.value()
            self.settings["check_symmetry"] = self.calc_tab.check_symmetry.isChecked()
            self.settings["spin_cycles"] = self.calc_tab.spin_cycles.value()
            self.settings["spin_cycles"] = self.calc_tab.spin_cycles.value()
            self.settings["conv_tol"] = self.calc_tab.edit_conv.text()
            self.settings["grid_level"] = self.calc_tab.spin_grid_level.value()
            self.settings["scan_params"] = getattr(self.calc_tab, 'scan_params', None)
            self.settings["solvent"] = self.calc_tab.solvent_combo.currentText()
        
        self.settings["version"] = self.version
        
        # History
        history_to_save = self.calc_history
        if hasattr(self, 'calc_tab'):
            out_dir_val = self.calc_tab.out_dir_edit.text().strip()
            is_relative_setting = not os.path.isabs(out_dir_val)
            if is_relative_setting:
                 try:
                    mw = self.context.get_main_window()
                    current_path = getattr(mw, 'current_file_path', None)
                    if current_path:
                        project_dir = os.path.dirname(current_path)
                        relative_history = []
                        for h_path in self.calc_history:
                            try:
                                rel = os.path.relpath(h_path, project_dir)
                                relative_history.append(rel)
                            except: relative_history.append(h_path)
                        history_to_save = relative_history
                 except: pass
        
        self.settings["calc_history"] = history_to_save
        self.settings["struct_source"] = self.struct_source
        
        try:
            if self.context:
                 mw = self.context.get_main_window()
                 if hasattr(mw, 'current_file_path') and mw.current_file_path:
                     self.settings["associated_filename"] = os.path.basename(mw.current_file_path)
        except: pass

    def save_settings(self):
        self.update_internal_state()

