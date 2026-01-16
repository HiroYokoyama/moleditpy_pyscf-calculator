import os
import json
import glob
import traceback
from rdkit import Chem

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QTextEdit, QProgressBar, QCheckBox, QGroupBox,
    QFormLayout, QMessageBox, QFileDialog, QTabWidget, QWidget, QLineEdit,
    QSpinBox, QListWidget, QListWidgetItem, QDoubleSpinBox,
    QDockWidget, QApplication, QMenu, QToolTip,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QAction

try:
    from .worker import PySCFWorker, LoadWorker, PropertyWorker
    from .utils import update_molecule_from_xyz
except ImportError:
    import traceback
    traceback.print_exc()
    PySCFWorker = None
    LoadWorker = None
    PropertyWorker = None

try:
    from pyscf.data import nist
except ImportError:
    nist = None

try:
    from .freq_vis import FreqVisualizer
except ImportError:
    FreqVisualizer = None

class PySCFDialog(QDialog):
    def __init__(self, parent=None, context=None, settings=None, version=None):
        super().__init__(parent)
        self.context = context
        self.settings = settings if settings is not None else {}
        self.mo_data = None # Initialize to prevent AttributeError
        self.closing = False # Emergency flag to block updates during shutdown
        self.struct_source = None
        self.calc_history = []
        
        title = "PySCF Calculator"
        self.version = version
        if version:
            title += f" v{version}"
        self.setWindowTitle(title)
        
        self.resize(600, 700)
        self.worker = None
        self.setup_ui()
        self.load_settings()

    def on_document_reset(self):
        """Callback to reset plugin state when the document is reset (File -> New)."""
        # Abort pending workers to prevent crash or ghost updates
        for attr in ['worker', 'prop_worker', 'load_worker']:
            w = getattr(self, attr, None)
            if w:
                try: w.disconnect()
                except: pass
                if w.isRunning():
                     w.terminate()
                     w.wait(50)
                setattr(self, attr, None)

        # Clear data buffers
        self.mo_data = None
        self.freq_data = None
        self.thermo_data = None

        # Clear file association
        if "associated_filename" in self.settings:
            del self.settings["associated_filename"]
            
        # Clear internal state
        self.loaded_file = None
        self.struct_source = None
        self.calc_history = []
        if "calc_history" in self.settings:
             self.settings["calc_history"] = []

        
        # Clear result folder setting
        # Reset result folder to default/saved preference
        # Reset Settings (OutDir, Threads, Memory) to Default/Saved
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
        local_settings = {}
        if os.path.exists(json_path):
             try:
                 with open(json_path, 'r') as f:
                     local_settings = json.load(f)
             except: pass
        
        # 1. Output Dir
        if hasattr(self, 'out_dir_edit'):
             home_dir = os.path.expanduser("~")
             default_out = local_settings.get("root_path", os.path.join(home_dir, "PySCF_Results"))
             self.out_dir_edit.setText(default_out)
             
        # 2. Threads
        if hasattr(self, 'spin_threads'):
             self.spin_threads.setValue(local_settings.get("threads", 0))
             
        # 3. Memory
        if hasattr(self, 'spin_memory'):
             self.spin_memory.setValue(local_settings.get("memory", 4000))

        # 4. Calculation Settings Defaults
        if hasattr(self, 'job_type_combo'):
            self.job_type_combo.setCurrentText("Optimization + Frequency")
        if hasattr(self, 'method_combo'):
            self.method_combo.setCurrentText("RKS")
        if hasattr(self, 'functional_combo'):
            self.functional_combo.setCurrentText("b3lyp")
        if hasattr(self, 'basis_combo'):
            self.basis_combo.setCurrentText("sto-3g")
        if hasattr(self, 'charge_input'):
            self.charge_input.setCurrentText("0")
        if hasattr(self, 'spin_input'):
            self.spin_input.setCurrentIndex(0) # Singlet
        if hasattr(self, 'check_symmetry'):
            self.check_symmetry.setChecked(False)
        if hasattr(self, 'check_break_sym'):
            self.check_break_sym.setChecked(False)
        if hasattr(self, 'spin_cycles'):
            self.spin_cycles.setValue(100)
        if hasattr(self, 'edit_conv'):
            self.edit_conv.setText("1e-9")
        
        # Clear Visualizations
        if hasattr(self, 'clear_3d_actors'):
             self.clear_3d_actors()
             
        if hasattr(self, 'file_list'):
             self.file_list.clear()
        if hasattr(self, 'orb_list'):
             self.orb_list.clear()

        # Cleanup Frequency Dock
        if hasattr(self, 'freq_vis') and self.freq_vis:
            try: self.freq_vis.cleanup()
            except: pass
            self.freq_vis = None

        if hasattr(self, 'freq_dock') and self.freq_dock:
             mw = self.context.get_main_window()
             if mw:
                 try: mw.removeDockWidget(self.freq_dock)
                 except: pass
             self.freq_dock.close()
             self.freq_dock.deleteLater()
             self.freq_dock = None

        # Clear Checkpoint Path
        self.chkfile_path = None

        # Reset Result Path Display
        if hasattr(self, 'result_path_display'):
             self.result_path_display.clear()
        
        # Disable Buttons
        self.btn_load_geom.setEnabled(False)
        self.btn_run_analysis.setEnabled(False) 
        self.btn_show_diagram.setEnabled(False)
        self.btn_show_thermo.setEnabled(False)

        # Clear Structure Source Label
        if hasattr(self, 'lbl_struct_source'):
            self.lbl_struct_source.setText("")
             
        if hasattr(self, 'log_text'):
             self.log_text.clear()
        
        if hasattr(self, 'log'):
             self.log("Document reset: Plugin state cleared.")

        # User Request: Apply Saved Defaults on New File
        self.apply_defaults()

    def apply_defaults(self):
        """Load and apply default settings (User Defaults or Factory Defaults)."""
        # Factory Defaults
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
            "conv_tol": "1e-9"
        }

        # Load User Defaults (settings.json)
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
        if os.path.exists(json_path):
             try:
                 with open(json_path, 'r') as f:
                     user_defaults = json.load(f)
                     defaults.update(user_defaults)
             except: pass
        
        # Apply to UI
        if hasattr(self, 'job_type_combo'):
            self.job_type_combo.setCurrentText(defaults["job_type"])
            self.method_combo.setCurrentText(defaults["method"])
            self.functional_combo.setCurrentText(defaults["functional"])
            self.basis_combo.setCurrentText(defaults["basis"])
            self.charge_input.setCurrentText(str(defaults["charge"]))
            self.spin_input.setCurrentText(str(defaults["spin"]))
            
            self.out_dir_edit.setText(defaults["root_path"])
            
            self.spin_threads.setValue(int(defaults["threads"]))
            self.spin_memory.setValue(int(defaults["memory"]))
            
            self.check_symmetry.setChecked(defaults["check_symmetry"])
            self.spin_cycles.setValue(int(defaults["spin_cycles"]))
            self.edit_conv.setText(defaults["conv_tol"])

    def update_internal_state(self):
        """
        Update the internal 'self.settings' dictionary with current UI values.
        This provides the source of truth for 'on_save_project' (Project File Persistence).
        NOTE: This does NOT save to the global 'settings.json' file.
        """
        # Update shared dictionary
        self.settings["job_type"] = self.job_type_combo.currentText()
        self.settings["method"] = self.method_combo.currentText()
        self.settings["functional"] = self.functional_combo.currentText()
        self.settings["basis"] = self.basis_combo.currentText()
        self.settings["charge"] = self.charge_input.currentText()
        self.settings["spin"] = self.spin_input.currentText()
        self.settings["out_dir"] = self.out_dir_edit.text()
        self.settings["version"] = self.version # Save version to project file
        
        # Persist History (Handle Relative Paths)
        if hasattr(self, 'calc_history'):
            history_to_save = self.calc_history
            
            # Check if user is using relative path setting
            out_dir_val = self.out_dir_edit.text().strip()
            is_relative_setting = not os.path.isabs(out_dir_val)
            
            if is_relative_setting:
                try:
                    # Resolve project directory
                    mw = self.context.get_main_window()
                    current_path = getattr(mw, 'current_file_path', None)
                    if current_path:
                        project_dir = os.path.dirname(current_path)
                        # Convert history to relative
                        relative_history = []
                        for h_path in self.calc_history:
                            try:
                                # Only convert if on same drive/valid
                                rel = os.path.relpath(h_path, project_dir)
                                relative_history.append(rel)
                            except:
                                relative_history.append(h_path)
                        history_to_save = relative_history
                except:
                    pass
            
            self.settings["calc_history"] = history_to_save

        if hasattr(self, 'struct_source'):
             self.settings["struct_source"] = self.struct_source
        
        # Save Context Association (to prevent auto-load on new/different file)
        try:
            if self.context:
                 mw = self.context.get_main_window()
                 # Only update if we can reliably read the current file path
                 if hasattr(mw, 'current_file_path'):
                     path = mw.current_file_path
                     # If path exists -> basename. If None/Empty -> None.
                     name = os.path.basename(path) if path else None
                     self.settings["associated_filename"] = name
        except: pass

    def save_settings(self):
        """Legacy name kept for compatibility, but just updates internal state now."""
        self.update_internal_state()


    def save_custom_defaults(self):
        """Save current UI settings as default for future new projects."""
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
        local_settings = {
             "root_path": self.out_dir_edit.text(),
             "threads": self.spin_threads.value(),
             "memory": self.spin_memory.value(),
             # Calc Settings
             "job_type": self.job_type_combo.currentText(),
             "method": self.method_combo.currentText(),
             "functional": self.functional_combo.currentText(),
             "basis": self.basis_combo.currentText(),
             "check_symmetry": self.check_symmetry.isChecked(),
             "spin_cycles": self.spin_cycles.value(),
             "conv_tol": self.edit_conv.text()
        }
        try:
             with open(json_path, 'w') as f:
                 json.dump(local_settings, f, indent=4)
             
             # Feedback
             # self.statusBar() might not be available if not QMainWindow, but qdialog doesn't have it by default unless added?
             # gui.py inherits QDialog? 
             # Use log or tooltip or simple message box?
             # Simple transient message box or log is better.
             self.log("Default settings saved.")
             QToolTip.showText(self.cursor().pos(), "Defaults Saved!", self)
             
        except Exception as e:
             self.log(f"Failed to save default settings: {e}")

    def load_settings(self):
        # Apply Defaults First (User or Factory)
        self.apply_defaults()

        # Project Settings (Load Overrides)
        s = self.settings
        if "job_type" in s: self.job_type_combo.setCurrentText(s["job_type"])
        if "method" in s: self.method_combo.setCurrentText(s["method"])
        if "functional" in s: self.functional_combo.setCurrentText(s["functional"])
        if "basis" in s: self.basis_combo.setCurrentText(s["basis"])
        if "charge" in s: self.charge_input.setCurrentText(s["charge"])
        if "spin" in s: self.spin_input.setCurrentText(s["spin"])
        if "out_dir" in s: self.out_dir_edit.setText(s["out_dir"])
        
        
        # Load History & Source
        raw_history = s.get("calc_history", [])
        self.calc_history = []
        
        # Resolve History Paths
        try:
            # Attempt to resolve against current project path
            # (Context usually set before settings loaded, or we check if we can get it)
            project_dir = None
            if self.context:
                mw = self.context.get_main_window()
                current_path = getattr(mw, 'current_file_path', None)
                if current_path:
                    project_dir = os.path.dirname(current_path)

        except:
             # Fallback
             pass

        # FIX: Revert to overwrite history on load (User Request: "Delete it")
        self.calc_history = []
        
        for h_path in raw_history:
             # Resolve relative
             final_path = h_path
             try:
                 if not os.path.isabs(h_path) and project_dir:
                     final_path = os.path.normpath(os.path.join(project_dir, h_path))
             except: pass
             
             self.calc_history.append(final_path)

        # Structure Source: Only overwrite if stored one is valid, otherwise keep current
        # (User Request: "structure source need to be kept if structual change does not take place")
        loaded_source = s.get("struct_source", None)
        if loaded_source:
             self.struct_source = loaded_source
        # Update Label if UI ready (setup_ui called before load_settings)
        if hasattr(self, 'lbl_struct_source') and self.struct_source:
             self.lbl_struct_source.setText(f"Structure Source: {self.struct_source}")

        if self.calc_history:
             last_path = self.calc_history[-1]
             
             # Smart Reset Logic based on Associated File
             # (User Request: Removed filename dependency - rely on document_reset handlers)
             # current_filename = None
             # saved_filename = s.get("associated_filename", None)
             should_reset = False
             
             # if saved_filename:
             #     if current_filename and current_filename != saved_filename:
             #         # Case 1: Different named files (A.xyz -> B.xyz)
             #         should_reset = True
             #         self.log(f"File changed ({saved_filename} -> {current_filename}). Resetting settings.")
             #     elif not current_filename:
             #         # Case 2: Named file -> Untitled (A.xyz -> Untitled / New)
             #         should_reset = True
             #         self.log(f"File closed ({saved_filename} -> Untitled). Resetting settings.")
             
             if should_reset:
                 # Clear Source Label because we are in a fresh session for a DIFFERENT file
                 self.struct_source = None
                 if hasattr(self, 'lbl_struct_source'):
                     self.lbl_struct_source.setText(f"Structure Source: Current Editor ({current_filename})")
                 
                 # Reset History
                 self.calc_history = []
                 if "calc_history" in self.settings:
                     self.settings["calc_history"] = []
                     
                 # Reset All Calculation Settings to Defaults
                 self.log("Resetting calculation parameters for new file.")
                 
                 # Apply Defaults (Local or Hardcoded)
                 self.job_type_combo.setCurrentText(local_settings.get("job_type", "Optimization + Frequency"))
                 self.method_combo.setCurrentText(local_settings.get("method", "RKS"))
                 self.functional_combo.setCurrentText(local_settings.get("functional", "b3lyp")) 
                 self.basis_combo.setCurrentText(local_settings.get("basis", "sto-3g"))
                 self.charge_input.setCurrentText("0")
                 self.spin_input.setCurrentText("0")
                 
                 # Use Local Settings if available, else defaults
                 self.spin_threads.setValue(local_settings.get("threads", 0))
                 self.spin_memory.setValue(local_settings.get("memory", 4000))
                 if "root_path" in local_settings:
                     self.out_dir_edit.setText(local_settings["root_path"])
                     
                 self.check_symmetry.setChecked(local_settings.get("check_symmetry", False))
                 self.spin_cycles.setValue(local_settings.get("spin_cycles", 100))
                 self.edit_conv.setText(local_settings.get("conv_tol", "1e-9"))
                 
                 # Clear shared settings to prevent persistence from previous session
                 keys_to_clear = ["job_type", "method", "functional", "basis", 
                                  "charge", "spin", "threads", "memory", "struct_source", "associated_filename"]
                 for k in keys_to_clear:
                     if k in self.settings: del self.settings[k]
                     
             else:
                 # Attempt Auto-load
                 if os.path.exists(last_path) and os.path.isdir(last_path):
                     self.log(f"Auto-loading latest result from history: {last_path}")
                     # Pass update_structure=False to prevent overwriting current molecule
                     QTimer.singleShot(200, lambda: self.load_result_folder(last_path, update_structure=False))
                 else:
                     # Warn if missing
                     self.log(f"Warning: Last result folder not found: {last_path}")
                     QTimer.singleShot(500, lambda: QMessageBox.warning(self, "Result Not Found", f"Calculation result was not found:\n{last_path}"))

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # === TAB 1: Calculation ===
        self.calc_tab = QWidget()
        self.setup_calc_tab()
        self.tabs.addTab(self.calc_tab, "Calculation")

        # === TAB 2: Visualization ===
        self.vis_tab = QWidget()
        self.setup_vis_tab()
        self.tabs.addTab(self.vis_tab, "Visualization")

    def setup_calc_tab(self):
        layout = QVBoxLayout(self.calc_tab)

        # --- Configuration Section ---
        config_group = QGroupBox("Calculation Settings")
        form_layout = QFormLayout()

        self.job_type_combo = QComboBox()
        self.job_type_combo.addItems(["Energy", "Geometry Optimization", "Frequency", "Optimization + Frequency", "TDDFT"])
        self.job_type_combo.currentTextChanged.connect(self.update_options)
        form_layout.addRow("Job Type:", self.job_type_combo)

        # N States (TDDFT)
        self.lbl_nstates = QLabel("N States:")
        self.nstates_input = QSpinBox()
        self.nstates_input.setRange(1, 100)
        self.nstates_input.setValue(10)
        # Initially hidden
        self.lbl_nstates.setVisible(False)
        self.nstates_input.setVisible(False)
        form_layout.addRow(self.lbl_nstates, self.nstates_input)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["RKS", "RHF", "UKS", "UHF", "ROKS", "ROHF"])
        self.method_combo.currentTextChanged.connect(self.update_options)
        form_layout.addRow("Method:", self.method_combo)

        # Functional (only for DFT)
        self.functional_combo = QComboBox()
        self.functional_combo.addItems(["b3lyp", "pbe", "lda", "m062x"])
        self.functional_combo.setEnabled(True) # Default is RKS now
        form_layout.addRow("Functional:", self.functional_combo)

        self.basis_combo = QComboBox()
        self.basis_combo.addItems(["sto-3g", "3-21g", "6-31g", "6-31g*", "cc-pvdz", "def2-svp"])
        form_layout.addRow("Basis Set:", self.basis_combo)
        
        self.charge_input = QComboBox() 
        self.charge_input.addItems([str(i) for i in range(-5, 6)]) 
        self.charge_input.setCurrentText("0")
        self.charge_input.currentTextChanged.connect(self.validate_spin_settings)
        form_layout.addRow("Charge:", self.charge_input)

        self.spin_input = QComboBox()
        # User Request: Descriptive items
        spin_items = [
            "1 (Singlet)", 
            "2 (Doublet)", 
            "3 (Triplet)", 
            "4 (Quartet)", 
            "5 (Quintet)", 
            "6 (Sextet)"
        ]
        self.spin_input.addItems(spin_items)
        self.spin_input.setCurrentIndex(0) # Default Singlet
        self.spin_input.currentTextChanged.connect(self.validate_spin_settings)
        
        # Validation visual feedback (Red background)
        # HBox for Spin Input
        h_spin = QHBoxLayout()
        h_spin.addWidget(self.spin_input)
        
        form_layout.addRow("Spin Multiplicity (2S+1):", h_spin)
        
        # Auto Detect Button
        self.btn_auto_detect = QPushButton("Auto Detect")
        self.btn_auto_detect.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_auto_detect.clicked.connect(self.auto_detect_charge_spin)
        self.btn_auto_detect.setStyleSheet("padding: 2px 8px; margin-left: 5px;")
        
        form_layout.addRow("", self.btn_auto_detect)
        
        # User Request: Auto-detect on launch once
        # Using a QTimer to run it after UI is settled, or just direct call if context exists.
        # But setup_calc_tab is called in __init__ -> setup_ui.
        # It's safer to queue it.
        if self.context and self.context.current_molecule:
             QTimer.singleShot(100, self.auto_detect_charge_spin)
        
        # Threads
        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(0, 64)
        self.spin_threads.setValue(0)
        self.spin_threads.setToolTip("0 = Auto (All available cores)")
        form_layout.addRow("Max Threads (0=Auto):", self.spin_threads)
        
        # Memory
        self.spin_memory = QSpinBox()
        self.spin_memory.setRange(500, 256000)
        self.spin_memory.setValue(4000)
        self.spin_memory.setSingleStep(1000)
        self.spin_memory.setSuffix(" MB")
        form_layout.addRow("Max Memory:", self.spin_memory)

        # Advanced Settings
        self.check_symmetry = QCheckBox("Enable Symmetry")
        self.check_symmetry.setChecked(False)
        self.check_symmetry.setToolTip("Detect and use point group symmetry to speed up calculation")
        form_layout.addRow(self.check_symmetry) # Add to form layout properly
        
        # User Request: Option for Symmetry Breaking (Default Disabled)
        self.check_break_sym = QCheckBox("Break Initial Guess Symmetry")
        self.check_break_sym.setChecked(False)
        self.check_break_sym.setToolTip("For UKS/UHF: Mix Alpha/Beta densities in initial guess to encourage spin polarization.")
        form_layout.addRow(self.check_break_sym)
        # No empty row needed for check_symmetry here, handled above.
        
        self.spin_cycles = QSpinBox()
        self.spin_cycles.setRange(1, 2000)
        self.spin_cycles.setValue(100)
        self.spin_cycles.setToolTip("Maximum number of SCF iterations")
        form_layout.addRow("Max SCF Cycles:", self.spin_cycles)
        
        self.edit_conv = QLineEdit()
        self.edit_conv.setText("1e-9")
        self.edit_conv.setToolTip("SCF Convergence Tolerance (e.g. 1e-9)")
        form_layout.addRow("Conv. Tolerance:", self.edit_conv)


        # Output Directory
        self.out_dir_edit = QLineEdit()
        # Default to Home Directory/PySCF_Results
        home_dir = os.path.expanduser("~")
        self.out_dir_edit.setText(os.path.join(home_dir, "PySCF_Results"))
        self.out_dir_edit.setToolTip("Path to save results. Relative paths (e.g. 'results') will be resolved relative to the current file location or default to the home directory.")
        
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_out_dir)
        
        h_box = QHBoxLayout()
        h_box.addWidget(self.out_dir_edit)
        h_box.addWidget(btn_browse)
        form_layout.addRow("Output Dir:", h_box)

        # User Request: "Save as Default" button (Inside Config Group, Right Aligned)
        h_default = QHBoxLayout()
        h_default.addStretch()
        self.btn_save_default = QPushButton("Save as Default")
        self.btn_save_default.setStyleSheet("padding: 5px;") 
        self.btn_save_default.clicked.connect(self.save_custom_defaults)
        self.btn_save_default.setToolTip("Save current settings (Job, Method, Path, etc.) as default for new projects.")
        h_default.addWidget(self.btn_save_default)
        
        form_layout.addRow(h_default)

        config_group.setLayout(form_layout)
        layout.addWidget(config_group)

        # --- Actions ---
        
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Calculation")
        self.run_btn.clicked.connect(self.run_calculation)
        self.run_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        
        btn_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_calculation)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        
        layout.addLayout(btn_layout)
        
        # --- Progress & Log ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Set Requests Defaults
        self.job_type_combo.setCurrentText("Optimization + Frequency")

    def auto_detect_charge_spin(self):
        """Auto-detect appropriate charge and spin multiplicity based on molecule."""
        if not self.context or not self.context.current_molecule:
            QMessageBox.warning(self, "Warning", "No molecule loaded.")
            return

        try:
            mol = self.context.current_molecule
            
            # Simple Total Electron Count using RDKit
            # Note: Explicit valence or implicit H should be handled by RDKit correctly if molecule is valid.
            total_electrons = 0
            has_transition_metal = False
            
            # Transition Metals (Sc-Zn, Y-Cd, Hf-Hg) roughly
            # Atomic nums: 21-30, 39-48, 72-80
            tm_nums = set(list(range(21, 31)) + list(range(39, 49)) + list(range(72, 81)))
            
            for atom in mol.GetAtoms():
                an = atom.GetAtomicNum()
                total_electrons += an
                if an in tm_nums:
                    has_transition_metal = True
            
            # Adjust for current charge if we want to detect FROM structure?
            # Usually RDKit mol has formal charges.
            charge = Chem.GetFormalCharge(mol)
            
            # Net electrons
            # We assume the user wants to simulate the neutral species by default 
            # OR the species as defined in RDKit.
            # If RDKit formal charge is 0, we assume 0.
            
            net_electrons = total_electrons - charge
            
            # Parity Rule
            # Even -> Singlet (1)
            # Odd -> Doublet (2)
            if net_electrons % 2 == 0:
                suggested_spin = 1
            else:
                suggested_spin = 2
                
            # Update UI
            # Charge
            idx_c = self.charge_input.findText(str(charge))
            if idx_c >= 0: self.charge_input.setCurrentIndex(idx_c)
            else: self.charge_input.setCurrentText(str(charge)) # Fallback
            
            # Spin
            # Now items are "1 (Singlet)", "2 (Doublet)" etc.
            # We need to match the integer part.
            
            target_str = str(suggested_spin)
            # Search logic
            found = False
            for i in range(self.spin_input.count()):
                text = self.spin_input.itemText(i)
                if text.startswith(target_str + " "):
                    self.spin_input.setCurrentIndex(i)
                    found = True
                    break
            
            if not found:
                # If high spin (e.g. 7), maybe not in list?
                # Just warn or leave as is?
                pass
            
            # User Request: Automatically switch method (RHF<->UHF, RKS<->UKS)
            current_method = self.method_combo.currentText()
            new_method = current_method
            
            # Logic:
            # Singlet (1) -> Restricted (R...)
            # Multiplet (>1) -> Unrestricted (U...)
            
            if suggested_spin == 1:
                # Switch U -> R
                if current_method == "UHF": new_method = "RHF"
                elif current_method == "UKS": new_method = "RKS"
            else:
                # Switch R -> U
                if current_method == "RHF": new_method = "UHF"
                elif current_method == "RKS": new_method = "UKS"
            
            if new_method != current_method:
                self.method_combo.setCurrentText(new_method)
                self.log(f"Auto-Detect: Switched method to {new_method} based on spin.")
            
            # Warning for TM
            if has_transition_metal:
                self.log("Auto-Detect: Transition metal detected. High spin states may be possible.")
                QMessageBox.information(self, "Info", 
                    "Transition metal detected.\nCharge and Spin have been set to standard values (Low Spin),\nbut you may need to adjust Multiplicity manually for High Spin states.")
            else:
                self.log(f"Auto-Detect: Set Charge={charge}, Mult={suggested_spin} (Electrons={net_electrons})")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Auto-detection failed: {e}")
            
    def validate_spin_settings(self):
        """Check consistency of charge and spin."""
        try:
            if not self.context or not self.context.current_molecule: return
            
            mol = self.context.current_molecule
            total_protons = sum(atom.GetAtomicNum() for atom in mol.GetAtoms())
            
            try:
                charge = int(self.charge_input.currentText())
                # Parse spin from "1 (Singlet)"
                spin_txt = self.spin_input.currentText()
                if " " in spin_txt:
                    mult = int(spin_txt.split(" ")[0])
                else:
                    mult = int(spin_txt)
            except:
                return
                
            electrons = total_protons - charge
            
            # Logic:
            # Multiplicity M = 2S + 1
            # Unpaired e- = M - 1
            # Remaining e- = Total - Unpaired
            # Remaining must be even (paired)
            
            unpaired = mult - 1
            remaining = electrons - unpaired
            
            is_valid = (remaining >= 0) and (remaining % 2 == 0)
            
            if is_valid:
                self.spin_input.setStyleSheet("")
                self.spin_input.setToolTip("")
                self.charge_input.setStyleSheet("")
                self.charge_input.setToolTip("")
            else:
                # User Request: Red background for invalid
                # "Don't clash with HOMO LUMO colors" -> Use a distinct soft red/pink
                style = "background-color: #ffcccc; color: black;"
                self.spin_input.setStyleSheet(style) 
                self.charge_input.setStyleSheet(style)
                
                # Determine specific message
                if remaining < 0:
                    msg = f"Invalid: More unpaired electrons ({unpaired}) than total electrons ({electrons})!"
                else:
                    msg = f"Invalid: {electrons} electrons cannot have multiplicity {mult} (requires odd/even mismatch)."
                
                self.spin_input.setToolTip(msg)
                self.charge_input.setToolTip(msg)
                    
        except:
            pass

    def browse_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.out_dir_edit.setText(d)
        
    def setup_vis_tab(self):
        from PyQt6.QtWidgets import QListWidget, QSlider, QDoubleSpinBox, QColorDialog
        from .vis import CubeVisualizer

        layout = QVBoxLayout(self.vis_tab)
        
        # --- Result Path Section ---
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Result Folder:"))
        self.result_path_display = QLineEdit()
        self.result_path_display.setReadOnly(True)
        self.result_path_display.setPlaceholderText("No result loaded")
        path_layout.addWidget(self.result_path_display)
        
        self.btn_load_result = QPushButton("Load Result Folder...")
        self.btn_load_result.clicked.connect(self.load_result_folder)
        path_layout.addWidget(self.btn_load_result)
        
        layout.addLayout(path_layout)
        
        # Result Actions
        self.btn_load_geom = QPushButton("Load Optimized Structure")
        self.btn_load_geom.clicked.connect(self.load_optimized_geometry)
        self.btn_load_geom.setEnabled(False) 
        layout.addWidget(self.btn_load_geom)
        
        # Structure Source Label
        self.lbl_struct_source = QLabel("")
        self.lbl_struct_source.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.lbl_struct_source)
        
        # --- Analysis Selection ---
        analysis_group = QGroupBox("Post-Calculation Analysis")
        self.analysis_layout = QVBoxLayout(analysis_group)
        
        # Orbital/Property List
        self.orb_list = QListWidget()
        self.orb_list.setFixedHeight(120)
        self.analysis_layout.addWidget(self.orb_list)
        
        # Manual MO Input
        mo_input_layout = QHBoxLayout()
        self.mo_input = QLineEdit()
        self.mo_input.setPlaceholderText("MO Index (e.g. 15)")
        self.btn_add_mo = QPushButton("Add")
        self.btn_add_mo.setFixedWidth(60)
        self.btn_add_mo.clicked.connect(self.add_custom_mo)
        mo_input_layout.addWidget(self.mo_input)
        mo_input_layout.addWidget(self.btn_add_mo)
        self.analysis_layout.addLayout(mo_input_layout)
        
        # Populate with default range (can be done here or after calc)
        self.populate_analysis_options()
        
        # Button to run analysis
        # Button to run analysis
        self.btn_run_analysis = QPushButton("Generate & Visualize Selected")
        self.btn_run_analysis.clicked.connect(self.run_selected_analysis)
        self.btn_run_analysis.setEnabled(False)
        self.analysis_layout.addWidget(self.btn_run_analysis)
        
        # Energy Diagram Button
        self.btn_show_diagram = QPushButton("Show Orbital Energy Diagram")
        self.btn_show_diagram.clicked.connect(self.show_energy_diagram)
        self.btn_show_diagram.setEnabled(False)
        self.analysis_layout.addWidget(self.btn_show_diagram)
        
        # Thermo Button
        self.btn_show_thermo = QPushButton("Show Thermodynamic Properties")
        self.btn_show_thermo.clicked.connect(self.show_thermo_data)
        self.btn_show_thermo.setEnabled(False)
        self.analysis_layout.addWidget(self.btn_show_thermo)
        
        layout.addWidget(analysis_group)
        
        # File List (Results)
        layout.addWidget(QLabel("Visualization Files:"))
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        # Enable Arrow Key Navigation
        self.file_list.currentItemChanged.connect(self.on_file_selected)
        layout.addWidget(self.file_list)
        

        
        # Controls Group
        self.vis_controls = QGroupBox("Visualization Controls")
        self.vis_controls.setEnabled(False)
        v_layout = QVBoxLayout(self.vis_controls)
        
        # Isovalue
        iso_layout = QHBoxLayout()
        iso_layout.addWidget(QLabel("Isovalue:"))
        self.iso_spin = QDoubleSpinBox()
        self.iso_spin.setRange(0.0001, 10.0)
        self.iso_spin.setDecimals(4) # Allow precision for 0.004
        self.iso_spin.setSingleStep(0.001)
        self.iso_spin.setValue(0.04)
        self.iso_spin.valueChanged.connect(self.update_visualization)
        iso_layout.addWidget(self.iso_spin)
        v_layout.addLayout(iso_layout)

        # Colors
        color_layout = QHBoxLayout()
        self.btn_color_p = QPushButton("Positive")
        self.btn_color_p.setStyleSheet("background-color: blue; color: white;")
        self.btn_color_p.clicked.connect(lambda: self.choose_color('p'))
        color_layout.addWidget(self.btn_color_p)
        
        self.btn_color_n = QPushButton("Negative")
        self.btn_color_n.setStyleSheet("background-color: red; color: white;")
        self.btn_color_n.clicked.connect(lambda: self.choose_color('n'))
        color_layout.addWidget(self.btn_color_n)
        v_layout.addLayout(color_layout)
        
        self.color_p = "blue"
        self.color_n = "red"

        # Opacity
        op_layout = QHBoxLayout()
        op_layout.addWidget(QLabel("Opacity:"))
        self.op_slider = QSlider(Qt.Orientation.Horizontal)
        self.op_slider.setRange(0, 100)
        self.op_slider.setValue(40)
        self.op_slider.valueChanged.connect(self.update_visualization)
        op_layout.addWidget(self.op_slider)
        v_layout.addLayout(op_layout)
        
        # --- Mapped Controls (Hidden by default) ---
        self.mapped_group = QGroupBox("ESP Mapping Controls")
        self.mapped_group.hide()
        m_layout = QVBoxLayout(self.mapped_group)
        
        # Surface Iso
        m_iso_layout = QHBoxLayout()
        m_iso_layout.addWidget(QLabel("Surface Iso:"))
        self.m_iso_spin = QDoubleSpinBox()
        self.m_iso_spin.setRange(0.0001, 10.0)
        self.m_iso_spin.setDecimals(4)
        self.m_iso_spin.setSingleStep(0.001)
        self.m_iso_spin.setValue(0.004) # Standard density iso
        self.m_iso_spin.valueChanged.connect(self.update_mapped_vis)
        m_iso_layout.addWidget(self.m_iso_spin)
        m_layout.addLayout(m_iso_layout)
        
        # Min/Max Controls
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Min/Max:"))
        
        self.m_min_spin = QDoubleSpinBox()
        self.m_min_spin.setRange(-10.0, 10.0)
        self.m_min_spin.setDecimals(4)
        self.m_min_spin.setSingleStep(0.001)
        self.m_min_spin.setValue(-0.05)
        self.m_min_spin.valueChanged.connect(self.update_mapped_vis)
        range_layout.addWidget(self.m_min_spin)
        
        self.m_max_spin = QDoubleSpinBox()
        self.m_max_spin.setRange(-10.0, 10.0)
        self.m_max_spin.setDecimals(4)
        self.m_max_spin.setSingleStep(0.001)
        self.m_max_spin.setValue(0.05)
        self.m_max_spin.valueChanged.connect(self.update_mapped_vis)
        range_layout.addWidget(self.m_max_spin)
        
        self.btn_fit_range = QPushButton("Fit")
        self.btn_fit_range.clicked.connect(self.fit_mapped_range)
        self.btn_fit_range.setToolTip("Auto-fit Color Range to Surface Values")
        range_layout.addWidget(self.btn_fit_range)
        
        m_layout.addLayout(range_layout)
        
        # Colormap
        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([
            "jet", "jet_r", 
            "bwr", "bwr_r", 
            "seismic", "seismic_r", 
            "coolwarm", "coolwarm_r", 
            "viridis", "viridis_r"
        ])
        self.cmap_combo.setCurrentText("jet_r")
        self.cmap_combo.currentTextChanged.connect(self.update_mapped_vis)
        cmap_layout.addWidget(self.cmap_combo)
        m_layout.addLayout(cmap_layout)
        
        # Opacity for ESP Mapping
        m_op_layout = QHBoxLayout()
        m_op_layout.addWidget(QLabel("Opacity:"))
        self.m_op_slider = QSlider(Qt.Orientation.Horizontal)
        self.m_op_slider.setRange(0, 100)
        self.m_op_slider.setValue(40)
        self.m_op_slider.valueChanged.connect(self.update_mapped_vis)
        m_op_layout.addWidget(self.m_op_slider)
        m_layout.addLayout(m_op_layout)


        layout.addWidget(self.mapped_group)
        layout.addWidget(self.vis_controls)
        
        # Helper
        self.visualizer = CubeVisualizer(self.context.get_main_window())
        self.mapped_visualizer = None # Lazy init
        self.mode = "standard" # or "mapped"
        self.loaded_file = None

    def choose_color(self, mode):
        # reuse standard QColorDialog
        from PyQt6.QtWidgets import QColorDialog
        c = QColorDialog.getColor()
        if c.isValid():
            name = c.name()
            if mode == 'p':
                self.color_p = name
                self.btn_color_p.setStyleSheet(f"background-color: {name}; color: white;")
            else:
                self.color_n = name
                self.btn_color_n.setStyleSheet(f"background-color: {name}; color: white;")
            self.update_visualization()

    def on_file_selected(self, item, previous=None):
        if not item: return # currentItemChanged might pass None if cleared
        self.clear_3d_actors()
        path = item.toolTip() 
        if not path or not os.path.exists(path):
             return

        # Check for ESP Pair (heuristic: if file is esp.cube, look for density.cube in same dir)
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        
        is_esp_pair = False
        if basename.lower() == "esp.cube":
             # Try case-insensitive density check
             density_path = os.path.join(dirname, "density.cube")
             if os.path.exists(density_path):
                is_esp_pair = True
                surf_file = density_path
                prop_file = path
        
        try:
            if is_esp_pair:
                self.switch_to_mapped_mode(surf_file, prop_file)
            else:
                self.switch_to_standard_mode(path)
        except Exception as e:
            self.log(f"Error loading visualization file: {e}")
            self.switch_to_standard_mode(path) # Fallback attempt

    def switch_to_standard_mode(self, path):
        self.mode = "standard"
        self.vis_controls.show()
        self.vis_controls.setEnabled(True)
        self.mapped_group.hide()
        
        # Clean mapped
        if self.mapped_visualizer: 
             try: self.mapped_visualizer.clear_actors()
             except: pass
        
        self.loaded_file = path
        
        # Lazy Init Visualizer if cleaned
        if not hasattr(self, 'visualizer') or self.visualizer is None:
            from .vis import CubeVisualizer
            self.visualizer = CubeVisualizer(self.context.get_main_window())

        if self.visualizer.load_file(path):
            self.iso_spin.blockSignals(True)
            try:
                # Defensive range set
                new_max = max(10.0, self.visualizer.data_max)
                if new_max <= 0: new_max = 1.0
                self.iso_spin.setRange(0.0001, new_max)
                
                fname = os.path.basename(path).lower()
                if "density" in fname:
                     self.iso_spin.setValue(0.04)
                else:
                     self.iso_spin.setValue(0.04)
            except Exception as e:
                print(f"Error setting spin range: {e}")
            finally:
                self.iso_spin.blockSignals(False)
            
            self.update_visualization()

    def switch_to_mapped_mode(self, surf_file, prop_file):
        self.mode = "mapped"
        self.vis_controls.hide()
        self.mapped_group.show()
        
        # Clean standard
        if hasattr(self, 'visualizer') and self.visualizer: 
            try: self.visualizer.clear_actors()
            except: pass
        
        from .vis import MappedVisualizer
        if not self.mapped_visualizer:
            self.mapped_visualizer = MappedVisualizer(self.context.get_main_window())
            
        if self.mapped_visualizer.load_files(surf_file, prop_file):
             # Auto settings
             self.m_iso_spin.blockSignals(True)
             self.m_iso_spin.setValue(0.004)
             self.m_iso_spin.blockSignals(False)
             
             # Use Mapped Range from Surface Sample
             p_min, p_max = self.mapped_visualizer.get_mapped_range(0.004)
             
             if p_max - p_min < 1e-9:
                p_max += 0.05
                p_min -= 0.05
             
             ui_min = min(-0.1, p_min)
             ui_max = max(0.1, p_max)
             
             self.m_min_spin.setRange(ui_min * 10, ui_max * 10)
             self.m_max_spin.setRange(ui_min * 10, ui_max * 10)
             
             self.m_min_spin.setValue(p_min)
             self.m_max_spin.setValue(p_max)
             
             self.update_mapped_vis()

    def load_optimized_geometry(self):
        if hasattr(self, 'optimized_xyz') and self.optimized_xyz:
            self.update_geometry(self.optimized_xyz)
            QMessageBox.information(self, "Success", "Geometry updated with optimized structure.")

    def update_mapped_vis(self):
        # Guard against missing mapped visualizer
        if not hasattr(self, 'mapped_visualizer') or self.mapped_visualizer is None:
            return
        
        if not self.mapped_visualizer: return
        iso = self.m_iso_spin.value()
        val_min = self.m_min_spin.value()
        val_max = self.m_max_spin.value()
        cmap = self.cmap_combo.currentText()
        # Use dedicated ESP mapping opacity slider
        opacity = self.m_op_slider.value() / 100.0
        
        self.mapped_visualizer.update_mesh(
            iso, opacity, cmap=cmap, clim=[val_min, val_max]
        )

    def fit_mapped_range(self):
        if not self.mapped_visualizer: return
        iso = self.m_iso_spin.value()
        
        # Get range from actual data on isosurface
        p_min, p_max = self.mapped_visualizer.get_mapped_range(iso)
        
        if p_max - p_min < 1e-9:
            p_max += 0.05
            p_min -= 0.05
            
        # Update UI ranges if needed to accommodate values
        cur_min_limit = self.m_min_spin.minimum()
        cur_max_limit = self.m_max_spin.maximum()
        
        if p_min < cur_min_limit: self.m_min_spin.setMinimum(p_min - 1.0)
        if p_max > cur_max_limit: self.m_max_spin.setMaximum(p_max + 1.0)
        
        self.m_min_spin.setValue(p_min)
        self.m_max_spin.setValue(p_max)
        
        # Apply the new range
        self.update_mapped_vis()

    def update_visualization(self):
        if self.mode == "mapped":
            self.update_mapped_vis()
            return

        # Guard against missing visualizer
        if not hasattr(self, 'visualizer') or self.visualizer is None:
            return

        if not self.loaded_file: return
        
        val = self.iso_spin.value()
        opacity = self.op_slider.value() / 100.0
        
        self.visualizer.update_iso(val, self.color_p, self.color_n, opacity)

    def closeEvent(self, event):
        self.closing = True # BLOCK ALL UI UPDATES IMMEDIATELY
        # self.save_settings() # Removed: Global settings only saved by explicit button
        
        # Safe Thread Cleanup
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.log("Closing dialog: Stopping active calculation...")
            # Try graceful stop first
            if hasattr(self.worker, 'stop'):
                 self.worker.stop()
            
            # Wait briefly then force if needed
            if not self.worker.wait(2000):
                 self.log("Force stopping worker...")
                 self.worker.terminate()
                 self.worker.wait()
            
            # Disconnect signals to prevent updates to dead UI
            try:
                self.worker.log_signal.disconnect()
                self.worker.finished_signal.disconnect()
                self.worker.error_signal.disconnect()
                self.worker.result_signal.disconnect()
            except:
                pass
            self.worker = None

        if hasattr(self, 'prop_worker') and self.prop_worker and self.prop_worker.isRunning():
             if not self.prop_worker.wait(1000):
                 self.prop_worker.terminate()
                 self.prop_worker.wait()
             self.prop_worker = None
        
        # Cleanup Visualizers
        self.clear_3d_actors() # This now safely checks for existence
        
        # Close Dock
        if hasattr(self, 'freq_dock') and self.freq_dock:
             mw = self.context.get_main_window()
             if mw:
                 try: mw.removeDockWidget(self.freq_dock)
                 except: pass
             self.freq_dock.close()
             self.freq_dock.deleteLater()
             self.freq_dock = None
             
        # Cleanup Freq Vis
        if hasattr(self, 'freq_vis') and self.freq_vis:
            try: self.freq_vis.cleanup()
            except: pass
            self.freq_vis = None
            
        super().closeEvent(event)

    def update_options(self, text=None):
        method = self.method_combo.currentText()
        is_dft = "KS" in method
        self.functional_combo.setEnabled(is_dft)
        
        # User Request: Disable Symmetry Breaking option for non-UKS/UHF
        is_unrestricted = method in ["UKS", "UHF"]
        if hasattr(self, 'check_break_sym'):
             self.check_break_sym.setEnabled(is_unrestricted)
             # User Request: "起動時にFalseになってる" -> Do not force uncheck when disabled.
             # Keep previous state (Default True) so when user selects UKS it is ready.
        
        job = self.job_type_combo.currentText()
        # Toggle TDDFT N States
        if hasattr(self, 'lbl_nstates') and hasattr(self, 'nstates_input'):
             is_tddft = (job == "TDDFT")
             self.lbl_nstates.setVisible(is_tddft)
             self.nstates_input.setVisible(is_tddft)

    def get_spin_value(self):
        """Safely parse spin multiplicity from GUI."""
        try:
            txt = self.spin_input.currentText()
            # Handle "1 (Singlet)" format
            if " " in txt:
                return int(txt.split(" ")[0])
            return int(txt)
        except:
            return 1 # Default

    def run_calculation(self):
        if not self.context or not self.context.current_molecule:
            msg = "Error: No molecule loaded. Please load a molecule in the main window."
            self.log(msg)
            QMessageBox.warning(self, "No Molecule", "Please load a molecule first.")
            return

        # Prepare configuration
        # Helper to resolve output directory
        raw_out_dir = self.out_dir_edit.text().strip()
        final_out_dir = raw_out_dir
        
        if not os.path.isabs(raw_out_dir):
            # Check context for current file
            mw = self.context.get_main_window()
            current_path = getattr(mw, 'current_file_path', None)
            
            # If unsaved, prompt user
            if not current_path:
                fallback_base = os.path.expanduser("~")
                fallback_path = os.path.abspath(os.path.join(fallback_base, raw_out_dir))
                
                msg = (f"You are using a relative path ('{raw_out_dir}') with an unsaved project.\n"
                       "Do you want to save the project first to establish a base directory?\n\n"
                       f"Clicking 'No' will save results to: {fallback_path}")
                
                reply = QMessageBox.question(self, "Unsaved Project", msg, 
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                
                if reply == QMessageBox.StandardButton.Yes:
                    # Trigger save
                    if hasattr(mw, 'save_project'):
                        mw.save_project()
                    # Re-check path
                    current_path = getattr(mw, 'current_file_path', None)
            
            # Resolve path
            if current_path:
                base_dir = os.path.dirname(current_path)
                final_out_dir = os.path.join(base_dir, raw_out_dir)
            else:
                # Fallback to home
                final_out_dir = os.path.join(os.path.expanduser("~"), raw_out_dir)

        config = {
            "job_type": self.job_type_combo.currentText(),
            "method": self.method_combo.currentText(),
            "functional": self.functional_combo.currentText(),
            "basis": self.basis_combo.currentText(),
            "charge": int(self.charge_input.currentText()),
            "spin": self.get_spin_value(),
            "nstates": self.nstates_input.value(),
            "threads": self.spin_threads.value(),
            "memory": self.spin_memory.value(),
            "symmetry": self.check_symmetry.isChecked(),
            "break_symmetry": self.check_break_sym.isChecked(),
            "max_cycle": self.spin_cycles.value(),
            "conv_tol": self.edit_conv.text(),
            "out_dir": os.path.abspath(final_out_dir)
        }
        
        # Ensure dir exists
        try:
            os.makedirs(config["out_dir"], exist_ok=True)
        except Exception as e:
            self.log(f"Error creating output directory: {e}")
            return

        # Reset states
        self.btn_load_geom.setEnabled(False)
        self.optimized_xyz = None # store result here

        # Setup Worker
        if PySCFWorker is None:
            self.log("Error: Could not import PySCFWorker. Check installation.")
            return

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.show()
        self.log_text.clear()
        self.log("Starting PySCF Calculation...\n---------------------------------")

        # Create worker
        # We pass the RDKit molecule as an XYZ string to ensure thread safety
        from .utils import rdkit_to_xyz
        xyz_str = rdkit_to_xyz(self.context.current_molecule)
        
        self.worker = PySCFWorker(xyz_str, config)
        self.worker.log_signal.connect(self.log_append)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.result_signal.connect(self.on_results)
        
        self.worker.start()

    def stop_calculation(self):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.log("\nStopping calculation...")
            
            # Disconnect signals immediately to stop UI updates
            try:
                self.worker.log_signal.disconnect()
                self.worker.finished_signal.disconnect()
                self.worker.error_signal.disconnect()
                self.worker.result_signal.disconnect()
            except:
                pass

            # Graceful stop attempt (if supported by backend)
            # PySCF doesn't have a simple interrupt, so we rely on thread termination
            # But we can try to wait a bit
            if not self.worker.wait(500):
                self.worker.terminate()
                self.worker.wait()
            
            self.log("Calculation stopped.")
            self.cleanup_ui_state()

    def log(self, message):
        if self.closing: return
        self.log_text.append(message)
        # Scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

    def log_append(self, text):
        if self.closing: return
        # Insert without adding extra newlines if chunks are small
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_text.setTextCursor(cursor)

    def on_finished(self):
        if self.closing: return
        
        # Capture current state for potential project save
        self.update_internal_state()
        
        # Mark Project as Unsaved
        if self.context:
             mw = self.context.get_main_window()
             if mw:
                 mw.has_unsaved_changes = True
                 mw.update_window_title()

        self.log("\n---------------------------------\nCalculation Finished.")
        self.cleanup_ui_state()

    def on_error(self, err_msg):
        if self.closing: return
        self.log(f"\nERROR: {err_msg}")
        QMessageBox.critical(self, "Calculation Error", err_msg)
        self.cleanup_ui_state()

    def clear_3d_actors(self):
        """Safely remove all PySCF-related actors from the main window plotter."""
        try:
             if hasattr(self, 'context') and self.context:
                 mw = self.context.get_main_window()
                 # Strict check: exist, not None
                 if not mw or not hasattr(mw, 'plotter') or mw.plotter is None:
                     return
                     
                 # Direct removals (if any legacy names linger)
                 try: mw.plotter.remove_actor("pyscf_iso_p")
                 except: pass
                 
                 try: mw.plotter.remove_actor("pyscf_iso_n")
                 except: pass
                 
                 try: mw.plotter.remove_actor("pyscf_mapped")
                 except: pass
                 
                 # Also defer to Visualizer classes if they hold references
                 if hasattr(self, 'visualizer') and self.visualizer:
                     try: self.visualizer.clear_actors()
                     except: pass
                     self.visualizer = None

                 if hasattr(self, 'mapped_visualizer') and self.mapped_visualizer:
                     try: self.mapped_visualizer.clear_actors()
                     except: pass
                     self.mapped_visualizer = None
                     
                 # Render update
                 # CRITICAL: Do NOT render if we are closing (Segfault risk)
                 if not getattr(self, 'closing', False):
                     try: mw.plotter.render()
                     except: pass
                 
        except Exception as e:
            # Do not crash on cleanup
            pass
        
        # Clear Freq Vis vectors (Separate Safety)
        if hasattr(self, 'freq_vis') and self.freq_vis:
            try:
                self.freq_vis.cleanup()
            except: pass

    def cleanup_ui_state(self):
        if self.closing: return
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.hide()
        self.worker = None # Release worker reference
        self.worker = None

    def add_custom_mo(self):
        text = self.mo_input.text().strip()
        if not text: return
        
        import re
        from PyQt6.QtWidgets import QMessageBox, QListWidgetItem
        from PyQt6.QtCore import Qt
        # Validate: Allow integer or HOMO/LUMO syntax
        is_rel = re.match(r"^(HOMO|LUMO)([-+]\d+)?$", text, re.IGNORECASE)
        is_digit = text.isdigit()
        
        if not (is_digit or is_rel):
             QMessageBox.warning(self, "Invalid Input", "Please enter a valid integer index (e.g. 15) or relative label (e.g. HOMO-1).")
             return
             
        display_label = text
        task_data = text
        
        # User Request: Support "10a", "10b", "10A", "10B"
        import re
        is_digit = text.isdigit()
        is_ab_suffix = False
        parsed_idx = -1
        suffix_char = ""
        
        # Check patterns like "10a" or "10b"
        match = re.match(r"^(\d+)([aAbB])$", text)
        if match:
             parsed_idx = int(match.group(1))
             suffix_char = match.group(2).lower()
             is_ab_suffix = True
        
        if is_digit or is_ab_suffix:
             if is_ab_suffix:
                 idx = parsed_idx
                 spin = "_A" if suffix_char == "a" else "_B"
                 task_data = f"MO {idx}{spin}"
                 display_label = f"MO {idx} ({'Alpha' if spin=='_A' else 'Beta'})"
             else:
                 idx = int(text)
                 # Worker expects "MO <n>" for absolute indices
                 task_data = f"MO {idx}"
                 display_label = f"MO {idx}"
             
             # Attempt to resolve relative label for better UI
             if hasattr(self, 'mo_data') and self.mo_data:
                 try:
                     occ = self.mo_data["occ"]
                     # Handle simple RHF case or Alpha of UHF
                     target_occ = occ
                     if isinstance(occ, list) and len(occ) > 0 and isinstance(occ[0], list):
                         target_occ = occ[0] 
                     
                     homo_i = -1
                     for i, o in enumerate(target_occ):
                         if o > 0.5: # Threshold for occupancy
                             homo_i = i
                         else:
                             break
                     
                     
                     # Check if input is 1-based (from user text)
                     # Convert to 0-based for comparison
                     comp_idx = idx - 1
                     
                     if comp_idx <= homo_i:
                         diff = homo_i - comp_idx
                         lb = "HOMO" if diff == 0 else f"HOMO-{diff}"
                         display_label = f"{lb} (Index {idx})"
                     else:
                         # LUMO
                         lumo_i = homo_i + 1
                         diff = comp_idx - lumo_i
                         lb = "LUMO" if diff == 0 else f"LUMO+{diff}"
                         display_label = f"{lb} (Index {idx})"
                 except Exception as e:
                     print(f"Error resolving labels: {e}")

        else:
             # Normalize to Uppercase (HOMO-1)
             task_data = text.upper().replace(" ", "")
             display_label = task_data
             
        # Check duplicates (check against data, not label)
        for i in range(self.orb_list.count()):
             if self.orb_list.item(i).data(Qt.ItemDataRole.UserRole) == task_data:
                 return # Exists

        item = QListWidgetItem(display_label)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        item.setData(Qt.ItemDataRole.UserRole, task_data)
        self.orb_list.addItem(item)
        self.mo_input.clear()

    def populate_analysis_options(self):
        self.orb_list.clear()
        
        # Add Standard Options
        from PyQt6.QtWidgets import QListWidgetItem
        
        # ESP
        item_esp = QListWidgetItem("ESP (Electrostatic Potential + Density)")
        item_esp.setFlags(item_esp.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item_esp.setCheckState(Qt.CheckState.Unchecked)
        item_esp.setData(Qt.ItemDataRole.UserRole, "ESP")
        self.orb_list.addItem(item_esp)
        
        # Determine calculation type and extract occupancies
        is_uhf = False
        is_roks = False
        occ_a = []
        occ_b = []
        
        # Check calculation type from loaded data
        scf_type = self.mo_data.get("type", "RHF") if self.mo_data else "RHF"
        
        if scf_type in ["UHF", "UKS"]:
            is_uhf = True
            
            # Add Spin Density Option
            item_sd = QListWidgetItem("Spin Density")
            item_sd.setFlags(item_sd.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item_sd.setCheckState(Qt.CheckState.Unchecked)
            item_sd.setData(Qt.ItemDataRole.UserRole, "SpinDensity")
            self.orb_list.addItem(item_sd)

            try:
                # Extract occupancies for checking SOMO
                energies = self.mo_data.get("energies", [])
                occupations = self.mo_data.get("occupations", [])
                
                # Safety flatten helper (same as in EnergyDiagram)
                # Safety flatten helper (robust version)
                def safe_occ(occ_list):
                    try:
                        if occ_list is None: return []
                        # Ensure it's a list/tuple
                        if not isinstance(occ_list, (list, tuple)):
                             return [occ_list]
                        if not occ_list: return []
                        
                        # Check first element to detect nesting
                        first = occ_list[0]
                        if isinstance(first, (list, tuple)):
                             # Nested column vector [[1], [2]] -> [1, 2]
                             res = []
                             for x in occ_list:
                                 if isinstance(x, (list, tuple)) and len(x) > 0:
                                     res.append(x[0])
                                 else:
                                     res.append(0)
                             return res
                             
                        # Flat list
                        return occ_list
                    except:
                        return []

                if len(energies) >= 2 and isinstance(energies[0], list):
                    # Standard UKS format: [[alpha], [beta]]
                    occ_a = safe_occ(occupations[0])
                    occ_b = safe_occ(occupations[1])
                elif len(energies) == 2 and isinstance(energies, list):
                     # Possible flat list fallback
                     if len(occupations) >= 2:
                         occ_a = safe_occ(occupations[0])
                         occ_b = safe_occ(occupations[1])
                     else:
                         occ_a = safe_occ(occupations)
                         occ_b = []
                else:
                    # Fallback or RHF structure in UHF type?
                    occ_a = safe_occ(occupations)
                    occ_b = []
            except:
                pass
        
        elif scf_type in ["ROKS", "ROHF"]:
            # ROKS: Restricted Open-shell
            # Has single set of MO energies but 2D occupancy array
            is_roks = True
            
            try:
                occupations = self.mo_data.get("occupations", [])
                if not occupations:
                    pass  # Empty, skip
                # ROKS occupancy is 2D: [alpha_occ, beta_occ]
                elif isinstance(occupations, list) and len(occupations) >= 2:
                    # Safely extract first element
                    if isinstance(occupations[0], list):
                        occ_a = occupations[0]
                    elif hasattr(occupations[0], 'tolist'):
                        occ_a = occupations[0].tolist()
                    else:
                        occ_a = list(occupations[0]) if occupations[0] else []
                elif isinstance(occupations, list):
                    # Fallback: single array
                    occ_a = occupations
                elif hasattr(occupations, 'tolist'):
                    # NumPy array
                    occ_a = occupations.tolist()
            except Exception as e:
                # Safe fallback
                occ_a = []
        
        else:
            # RHF / Standard RKS
            occs = self.mo_data.get("occupations", []) if self.mo_data else []
            
            # Robust extraction logic
            if occs and isinstance(occs, list) and len(occs) > 0:
                 # Check for nested structure (e.g. [[...]] or [[],[]])
                 if isinstance(occs[0], (list, tuple)):
                      # Case: Single row vector [[1, 2, 3]]
                      if len(occs) == 1:
                           occ_a = list(occs[0])
                      else:
                           # Case: Column vector or list of lists [[1], [2], ...]
                           occ_a = []
                           for x in occs:
                               if isinstance(x, (list, tuple)) and len(x) > 0:
                                   occ_a.append(x[0])
                               else:
                                   occ_a.append(x) # Fallback if mixed
                 else:
                      # Already 1D flat list [1, 2, 3]
                      occ_a = occs
            else:
                 occ_a = []

        def add_orb_items(suffix="", label_suffix="", range_lumo=range(0, 5), range_homo=range(0, 5), check_somo=False):
             # LUMOs
            for i in reversed(range_lumo):
                if i == 0: label = "LUMO"
                else: label = f"LUMO+{i}"
                
                full_label = f"{label}{label_suffix}"
                task_str = f"{label}{suffix}"
                
                item = QListWidgetItem(full_label)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setData(Qt.ItemDataRole.UserRole, task_str)
                self.orb_list.addItem(item)

            # HOMOs
            # Find HOMO index for this spin channel
            if "Beta" in label_suffix:
                current_occ = occ_b
            else:
                current_occ = occ_a
            # Wait, this logic for finding HOMO index inside the loop is tricky 
            # because 'i' is offset from HOMO.
            # We need the absolute HOMO index first.
            # Use threshold 0.1 to match worker.py and avoid numerical precision errors
            occ_threshold = 0.1
            my_homo_idx = -1
            if current_occ:
                for idx, o in enumerate(current_occ):
                    if o > occ_threshold: my_homo_idx = idx
            
            for i in range_homo:
                target_idx = my_homo_idx - i
                if target_idx < 0: continue
                
                # Robust Bounds Check
                if current_occ and target_idx >= len(current_occ):
                     continue

                label = "HOMO"
                
                # Check ROKS SOMO
                # Condition: Not UHF, and occupancy is ~1.0
                is_roks_somo = False
                if not is_uhf and current_occ:
                    # current_occ is available, check value
                    val = current_occ[target_idx]
                    if abs(val - 1.0) < 0.1:
                        is_roks_somo = True
                
                # Construct Display Label matching Diagram Logic
                if is_roks_somo:
                    label = "SOMO"  # Display label for user
                elif i == 0:
                    label = "HOMO"
                else:
                    label = f"HOMO-{i}"
                
                full_label = f"{label}{label_suffix}"
                
                # Task String for Worker
                # IMPORTANT: Always use HOMO for worker (no SOMO special case)
                # Worker standardizes on HOMO/LUMO + Alpha/Beta convention
                # e.g. "MO 15_HOMO_A" (even if display shows "SOMO")
                worker_label = "HOMO" if i == 0 else f"HOMO-{i}"
                task_str = f"MO {target_idx+1}_{worker_label}{suffix}"
                
                item = QListWidgetItem(full_label)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setData(Qt.ItemDataRole.UserRole, task_str)
                self.orb_list.addItem(item)

        if is_uhf:
            # Add Alpha (Check SOMO)
            add_orb_items(suffix="_A", label_suffix=" (Alpha)", check_somo=True)
            # Add Beta (No SOMO)
            add_orb_items(suffix="_B", label_suffix=" (Beta)", check_somo=False)
        elif is_roks:
            # ROKS: Restricted open-shell (no Alpha/Beta separation, but check for SOMO)
            add_orb_items(check_somo=True)
        else:
            # RHF: Restricted closed-shell
            add_orb_items()
    
        # User Request: "remove checkbox orrf spin density in the buttom"
        # The functionality is already provided by the "Spin Density" item in the list above.
        self.check_spin = None

    def disable_existing_analysis_items(self, cube_files):
        """
        Disable checkboxes for analysis items that already have cube files.
        
        Args:
            cube_files: List of cube file paths
        """
        if not cube_files:
            return
        
        # Extract basenames for easier matching
        basenames = [os.path.basename(f).lower() for f in cube_files]
        
        # Check each item in the analysis list
        for i in range(self.orb_list.count()):
            item = self.orb_list.item(i)
            task_data = item.data(Qt.ItemDataRole.UserRole)
            
            should_disable = False
            
            # Check if ESP files exist
            if task_data == "ESP":
                if "esp.cube" in basenames and "density.cube" in basenames:
                    should_disable = True
            
            # Check if orbital files exist
            elif task_data and task_data != "ESP":
                # For orbital tasks, we need to check if a cube file with matching label exists
                # Cube files are named like: "15_HOMO.cube", "16_LUMO.cube", "17_LUMO+1.cube"
                
                # Normalize the task label
                search_label = task_data.upper().replace(" ", "")
                if search_label.startswith("MO"):
                    # For "MO 15" format, extract the index
                    try:
                        mo_idx = search_label.replace("MO", "").strip()
                        # Look for files starting with this index
                        padded_idx = f"{int(mo_idx):03d}"
                        for bn in basenames:
                            if (bn.startswith(f"{mo_idx}_") or bn.startswith(f"{padded_idx}_")) and bn.endswith(".cube"):
                                should_disable = True
                                break
                    except:
                        pass
                else:
                    # For HOMO/LUMO format, look for matching label in filename
                    # Files are named like "15_HOMO.cube" or "16_LUMO+1.cube"
                    for bn in basenames:
                        if ".cube" in bn:
                            # Extract the label part (after underscore, before .cube)
                            parts = bn.replace(".cube", "").split("_")
                            if len(parts) >= 2:
                                file_label = "_".join(parts[1:]).upper()
                                if file_label == search_label:
                                    should_disable = True
                                    break
            
            # Disable the item if cube file exists
            if should_disable:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setCheckState(Qt.CheckState.Unchecked)
            
    def run_selected_analysis(self):
        if not hasattr(self, 'chkfile_path') or not self.chkfile_path:
             QMessageBox.warning(self, "Error", "No checkpoint file found.")
             return
             
        # Gather selected tasks
        tasks = []
        for i in range(self.orb_list.count()):
            item = self.orb_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                tasks.append(item.data(Qt.ItemDataRole.UserRole))
        
        if not tasks:
            QMessageBox.information(self, "Info", "No analysis selected.")
            return
            
        self.run_specific_analysis(tasks)

    def generate_specific_orbital(self, task_or_index, label=None, spin_suffix=""):
        """
        Public method to generate a specific orbital.
        Can be called with a single task string (e.g. "SOMO_A", "#15_B") 
        OR with (index, label, spin_suffix) for legacy compatibility.
        """
        if isinstance(task_or_index, str) and (label is None and spin_suffix == ""):
            # Called with single string task
            task = task_or_index
        else:
            # Called with index, etc.
            index = task_or_index
            
            # User Request: If logic says it is SOMO, ensure backend names it SOMO
            # We append "_SOMO" which the backend now strips before parsing index
            if label == "SOMO":
                 task = f"#{index}_SOMO{spin_suffix}"
            else:
                 task = f"#{index}{spin_suffix}"
            # If label is special (SOMO/HOMO), prefer that formatting?
            # But the caller (Diagram) now handles task construction.
            # This path is for legacy calls if any.
        
        # We can also use label if index is not available, but index is safer.
        self.run_specific_analysis([task])

    def run_specific_analysis(self, tasks, out_d=None):
        if not hasattr(self, 'chkfile_path') or not self.chkfile_path:
             QMessageBox.warning(self, "Error", "No checkpoint file found.")
             return

        # Verification: Check if chkfile actually exists on disk
        if not os.path.exists(self.chkfile_path):
             QMessageBox.warning(self, "Error", f"Checkpoint file missing at: {self.chkfile_path}")
             return
             
        if not tasks:
             return

        self.log(f"Starting Analysis for: {', '.join(tasks)}...")
        
        # Determine Output Directory
        if not out_d:
            out_d = getattr(self, 'last_out_dir', None)
            if not out_d and getattr(self, 'chkfile_path', None):
                 out_d = os.path.dirname(self.chkfile_path)
                 
            if not out_d:
                 out_d = self.out_dir_edit.text()
                 if not out_d:
                     # Fallback to default output directory if not specified
                     out_d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        
        self.prop_worker = PropertyWorker(self.chkfile_path, tasks, out_d)
        self.prop_worker.log_signal.connect(self.log_append)
        self.prop_worker.finished_signal.connect(self.on_prop_finished)
        self.prop_worker.error_signal.connect(self.on_error) # Shared error handler
        self.prop_worker.result_signal.connect(self.on_prop_results)
        
        self.btn_run_analysis.setEnabled(False)
        self.progress_bar.show()
        self.prop_worker.start()

    def on_prop_finished(self):
        if self.closing: return
        self.log("\nAnalysis Finished.")
        self.btn_run_analysis.setEnabled(True)
        self.progress_bar.hide()
        self.prop_worker = None
        
        # Disable and Uncheck processed items
        for i in range(self.orb_list.count()):
            item = self.orb_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                item.setCheckState(Qt.CheckState.Unchecked)
                # Disable the item to indicate it's done
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)

    def on_prop_results(self, result_data):
        if self.closing: return
        
        # User Feedback
        new_files = result_data.get("files", [])
        if new_files:
            self.log(f"Generated {len(new_files)} new files.")
            
        # CRITICAL FIX: Always rescan directory to ensure list is in sync with disk
        # This handles cases where files are overwritten or existing files need to be shown
        self.file_list.clear()
        
        d = getattr(self, 'last_out_dir', None)
        if not d and hasattr(self, 'chkfile_path') and self.chkfile_path:
             d = os.path.dirname(self.chkfile_path)
             
        if d and os.path.exists(d):
            import glob
            cubes = glob.glob(os.path.join(d, "*.cube"))
            cubes.sort() # A-Z
            
            last_item = None
            target_item = None
            
            # Find the "most interesting" new file to select (ESP or last generated)
            target_path = None
            if new_files:
                # Prefer ESP if in new files
                for nf in new_files:
                    if "esp" in os.path.basename(nf).lower():
                        target_path = nf
                        break
                # Fallback to last new file
                if not target_path:
                    target_path = new_files[-1]

            for c in cubes:
                name = os.path.basename(c)
                item = QListWidgetItem(name)
                item.setToolTip(c)
                self.file_list.addItem(item)
                
                # Check match for selection
                if target_path and os.path.normpath(c) == os.path.normpath(target_path):
                    target_item = item
                
                last_item = item
            
            # Auto-select
            if target_item:
                 self.file_list.setCurrentItem(target_item)
                 self.file_list.scrollToItem(target_item)
                 self.on_file_selected(target_item)
            elif last_item and new_files: # If we made files but didn't specific match, select last
                 self.file_list.setCurrentItem(last_item)
                 self.file_list.scrollToItem(last_item)
                 self.on_file_selected(last_item)

    def on_results(self, result_data):
        # Handle post-processing, e.g., visualization
        self.log("Processing results...")
        
        # CRITICAL: Clean up existing 3D actors before processing new results
        # This prevents segfaults when calculation type changes (RKS->UKS, etc.)
        try:
            self.clear_3d_actors()
            
            # Force reset visualizers to prevent RKS⇔UKS segfault
            # Same fix as on_load_finished: destroy old visualizer instances
            # to prevent VTK mapper inconsistencies with new calculation types
            self.visualizer = None
            self.mapped_visualizer = None
            
        except Exception as cleanup_err:
            self.log(f"Warning during actor cleanup: {cleanup_err}")
        
        # Extract Output Directory first to ensure it's available for labels
        out_dir = result_data.get("out_dir")
        if out_dir:
            self.last_out_dir = out_dir
            
            # Add to History (Fix for "Latest only" issue)
            # Since PySCFWorker generates unique paths (job_1, job_2...),
            # we must explicitly add this new path to our history list.
            if not hasattr(self, 'calc_history'): self.calc_history = []
            
            # Ensure absolute path
            abs_out = os.path.abspath(out_dir)
            if abs_out not in self.calc_history:
                self.calc_history.append(abs_out)
                # Note: We don't save to file here (removed per user request), 
                # but update_internal_state will capture it for Project Save.
            
        if result_data.get("optimized_xyz"):
             self.optimized_xyz = result_data["optimized_xyz"]
             self.btn_load_geom.setEnabled(True)
             self.log("Optimization converged. Automatically updating geometry...")
             self.update_geometry(self.optimized_xyz)
             
             self.update_geometry(self.optimized_xyz)
             
             # Update Source Label ONLY if geometry changed
             src_name = "Result"
             src_path = ""
             if self.last_out_dir:
                 src_name = os.path.basename(self.last_out_dir)
                 src_path = self.last_out_dir
             
             self.struct_source = f"optimized from {src_name} ({src_path})"
             if hasattr(self, 'lbl_struct_source'):
                  self.lbl_struct_source.setText(f"Structure Source: {self.struct_source}")
                  # Force repaint to ensure UI updates immediately
                  self.lbl_struct_source.repaint()
        
        # Store Energy Data
        if result_data.get("mo_energy"):
             self.mo_data = {
                 "energies": result_data["mo_energy"],
                 "occupations": result_data["mo_occ"],
                 "type": result_data.get("scf_type", "RHF")
             }
             self.btn_show_diagram.setEnabled(True)
             
             # Auto-Show Diagram as requested
             # self.show_energy_diagram() # Disabled per user request
             
             # Populate Analysis Options (Crucial for SOMO/Orbital List update)
             self.log("Updating Analysis Options List based on new results...")
             self.populate_analysis_options()

        # Store chkfile & path
        if result_data.get("chkfile"):
            self.chkfile_path = result_data["chkfile"]
            self.last_out_dir = result_data.get("out_dir") # Store specific job folder
            self.btn_run_analysis.setEnabled(True)
            self.log(f"Checkpoint saved: {self.chkfile_path}")
            
            # --- Update Result Path Display ---
            if "out_dir" in result_data:
                self.result_path_display.setText(result_data["out_dir"])
                
                # Add to History
                d = result_data["out_dir"]
                if not hasattr(self, 'calc_history'): self.calc_history = []
                if d not in self.calc_history:
                    self.calc_history.append(d)
                
                # Redundant label update removed to respect correct format set earlier
            
            # Redundant label update removed to respect correct format set earlier

        # CRITICAL FIX: Robustly scan for cube files regardless of result_data
        # This ensures we show all files even if they were pre-existing or somehow not reported
        if self.last_out_dir and os.path.exists(self.last_out_dir):
            import glob
            cubes = glob.glob(os.path.join(self.last_out_dir, "*.cube"))
            cubes.sort()
            
            if result_data.get("cube_files"):
                 self.log(f"Generated Cube Files reported: {len(result_data['cube_files'])}")
            
            if cubes:
                self.log(f"Found {len(cubes)} visualization files in output directory.")
                self.file_list.clear()
                for c in cubes:
                    name = os.path.basename(c)
                    item = QListWidgetItem(name)
                    item.setToolTip(c)
                    self.file_list.addItem(item)
                
                # Auto-select the last one if we have new files reported
                if result_data.get("cube_files"):
                     # Try to select the last generated file
                     last_gen = result_data["cube_files"][-1]
                     for i in range(self.file_list.count()):
                         it = self.file_list.item(i)
                         if os.path.normpath(it.toolTip()) == os.path.normpath(last_gen):
                             self.file_list.setCurrentItem(it)
                             self.file_list.scrollToItem(it)
                             break
            else:
                 self.file_list.clear() # Clear if no cubes found
        
        # User Request: Auto-Switch to Visualization Tab when calculation completes
        self.tabs.setCurrentIndex(1)
            
        if result_data.get("freq_data"):
             self.log("Frequency Analysis available.")
             # Store data
             self.freq_data = result_data["freq_data"]
             
             # Create/Show Freq Visualizer
             try:
                 mol = self.context.current_molecule
                 
                 self.freq_vis = FreqVisualizer(
                     self.context.get_main_window(), 
                     mol, 
                     self.freq_data['freqs'], 
                     self.freq_data['modes'],
                     intensities=self.freq_data.get('intensities')
                 )
                 
                 mw = self.context.get_main_window()
                 
                 self.freq_dock = QDockWidget("PySCF Frequencies", mw)
                 self.freq_dock.setWidget(self.freq_vis)
                 self.freq_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
                 mw.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.freq_dock)
                 self.freq_dock.show()
                 self.freq_dock.raise_()
                 self.log("Frequency Visualizer opened in Dock.")
             except Exception as e:
                 self.log(f"Error opening Frequency Visualizer: {e}")
                 import traceback
                 self.log(traceback.format_exc())
                 QMessageBox.warning(self, "Visualizer Error", f"Failed to open Frequency Visualizer:\n{e}")

        if result_data.get("thermo_data"):
             self.thermo_data = result_data["thermo_data"]
             self.btn_show_thermo.setEnabled(True)
             self.log("Thermodynamic data captured.")
             
        # Auto-Switch to Visualization Tab when calculation completes
        self.tabs.setCurrentIndex(1)

    def load_result_folder(self, path=None, update_structure=True):
        self.loading_update_struct = update_structure
        d = path
        # If path is bool (from signal) or None, ask user
        if not d or isinstance(d, bool):
            d = QFileDialog.getExistingDirectory(self, "Select Result Directory")
        if not d: return
        
        # Ensure absolute path to prevent empty dirname issues
        d = os.path.abspath(d)
        
        chk_path = os.path.join(d, "pyscf.chk")
        if not os.path.exists(chk_path):
             # Try generic if not found?
             chk_path_alt = os.path.join(d, "checkpoint.chk")
             if os.path.exists(chk_path_alt):
                 chk_path = chk_path_alt
             else:
                 QMessageBox.warning(self, "Error", f"No checkpoint file (pyscf.chk) found in {d}")
                 return
        
        # Set Display
        self.result_path_display.setText(d)

        # Add to History & Save Immediately
        if not hasattr(self, 'calc_history'): self.calc_history = []
        
        if d not in self.calc_history:
            self.calc_history.append(d)
            self._history_changed = True
            
        self.update_internal_state()
            
        # Run LoadWorker
        if LoadWorker is None:
             QMessageBox.critical(self, "Error", "PySCF worker (LoadWorker) is not available.\nThis likely means PySCF is not installed or failed to import.")
             return

        # Guard against double-loading
        if hasattr(self, 'load_worker') and self.load_worker and self.load_worker.isRunning():
            QMessageBox.warning(self, "Busy", "A result is already loading. Please wait.")
            return

        self.load_worker = LoadWorker(chk_path)
        self.load_worker.finished_signal.connect(self.on_load_finished)
        self.load_worker.error_signal.connect(self.on_error)
        
        self.log(f"\nLoading result from: {d}...")
        self.progress_bar.show()
        self.load_worker.start()

    def on_load_finished(self, result_data):
         self.log("Result loaded successfully.")
         self.progress_bar.hide()
         
         # Mark Project as Unsaved (User Request)
         # Only if history was modified (i.e. new result folder loaded from disk that wasn't known)
         if getattr(self, '_history_changed', False):
             if self.context:
                 mw = self.context.get_main_window()
                 if mw:
                     mw.has_unsaved_changes = True
                     mw.update_window_title()
             self._history_changed = False
         
         # CRITICAL: Clean up existing state FIRST to prevent segfaults
         # This is essential when loading different calculation types (RKS->UKS, etc.)
         try:
             # 1. Stop and cleanup FreqVisualizer if active
             if hasattr(self, 'freq_vis') and self.freq_vis:
                 try:
                     self.freq_vis.cleanup()
                 except: pass
                 self.freq_vis = None
             
             # 2. Remove FreqVisualizer dock if exists
             if hasattr(self, 'freq_dock') and self.freq_dock:
                 try:
                     mw = self.context.get_main_window()
                     mw.removeDockWidget(self.freq_dock)
                     self.freq_dock.deleteLater()
                     self.freq_dock = None
                 except: pass
             
             # 3. Clear all 3D actors (orbitals, density, etc.)
             self.clear_3d_actors()
             
             # 4. Force reset visualizers to prevent RKS⇔UKS segfault
             # When switching between calculation types, old Visualizer instances
             # with incompatible VTK mappers can cause memory corruption.
             # Setting to None ensures fresh initialization on next render.
             self.visualizer = None
             self.mapped_visualizer = None
             
         except Exception as cleanup_err:
             self.log(f"Warning during initial cleanup: {cleanup_err}")
         
         # 1. Restore Checkpoint Path
         if result_data.get("chkfile"):
             self.chkfile_path = result_data["chkfile"]
             self.last_out_dir = os.path.dirname(self.chkfile_path)
             
         # 2. Restore Energy Data
         if result_data.get("mo_energy"):
             self.mo_data = {
                 "energies": result_data["mo_energy"],
                 "occupations": result_data["mo_occ"],
                 "type": result_data.get("scf_type", "RHF")
             }
             self.btn_show_diagram.setEnabled(True)
             self.btn_run_analysis.setEnabled(True)
             
         # 3. Restore Geometry (Optional)
         # Using 'loaded_xyz' from LoadWorker which always returns the chkfile geometry
         if result_data.get("loaded_xyz"):
             self.optimized_xyz = result_data["loaded_xyz"]
             self.btn_load_geom.setEnabled(True)
         elif result_data.get("optimized_xyz"):
             # Fallback for old compatibility or if manually added
             self.optimized_xyz = result_data["optimized_xyz"]
             self.btn_load_geom.setEnabled(True)
         
         # Populate Options List so we can disable existing ones
         try:
             self.populate_analysis_options()
         except Exception as e:
             self.log(f"ERROR populating analysis options: {e}")
             import traceback
             self.log(traceback.format_exc())

         # 4. Scan for Visualization Files
         self.file_list.clear()
         d = self.last_out_dir
         cubes = []
         if d and os.path.exists(d):
             # Cubes
             cubes = glob.glob(os.path.join(d, "*.cube"))
             cubes.sort() # Sort A-Z
             for c in cubes:
                 name = os.path.basename(c)
                 item = QListWidgetItem(name)
                 item.setToolTip(c)
                 self.file_list.addItem(item)
             self.log(f"Found {len(cubes)} existing visualization files.")
             
             # User Request: Do not auto-read cube files on load.
             # Just leave them in the list.
              
             # Disable checkboxes for existing cube files
             self.disable_existing_analysis_items(cubes)

         # 5. Restore Thermo Data
         if result_data.get("thermo_data"):
             self.thermo_data = result_data["thermo_data"]
             self.btn_show_thermo.setEnabled(True)
             self.log("Thermodynamic data restored.")
         
         self.save_settings()

         # 6. Auto-load Geometry & 7. Finalize (Linked)
         should_update_geom = getattr(self, 'loading_update_struct', True)
         if should_update_geom and hasattr(self, 'optimized_xyz') and self.optimized_xyz:
             # Add delay to prevent crash on startup/load
             # CRITICAL FIX: Chain the finalize_load AFTER geometry update
             def update_and_finalize():
                 # Update Source Label
                 # Check if we should overwrite. If checking explicit optimization result, 
                 # it might have been set in `on_results`.
                 # How to detect?
                 # `self.struct_source` might already contain "optimized from".
                 # If so, don't overwrite with "Loaded from".
                 current_src = getattr(self, 'struct_source', "")
                 is_optimization_result = current_src and "optimized from" in current_src
                 
                 if not is_optimization_result:
                     if hasattr(self, 'chkfile_path') and self.chkfile_path:
                         chk_abs = os.path.abspath(self.chkfile_path)
                         full_path = os.path.dirname(chk_abs)
                         basename = os.path.basename(full_path)
                         self.struct_source = f"Loaded from {basename} ({full_path})"
                     else:
                         self.struct_source = "Loaded from Result (Unknown)"

                 if hasattr(self, 'lbl_struct_source'):
                      self.lbl_struct_source.setText(f"Structure Source: {self.struct_source}")
                 
                 # Actually load the geometry into the 3D viewer
                 self.update_geometry(self.optimized_xyz)
                 
                 # Reset camera to view the loaded molecule
                 try:
                     mw = self.context.get_main_window()
                     if hasattr(mw, 'plotter'):
                         mw.plotter.reset_camera()
                     
                     # User Request: Enter 3D only mode ONLY for manual loads, not startup
                     # Check loading_update_struct to distinguish manual vs auto-load
                     is_manual_load = getattr(self, 'loading_update_struct', True)
                     if is_manual_load and hasattr(mw, 'minimize_2d_panel'):
                         mw.minimize_2d_panel()
                 except: pass

                 # Wrap finalize_load in try-except for robustness
                 try:
                     self.finalize_load(result_data, cubes)
                 except Exception as e:
                     self.log(f"Warning during finalize_load: {e}")
                     import traceback
                     self.log(traceback.format_exc())

             self.log("Optimized geometry loaded automatically.")
             QTimer.singleShot(100, update_and_finalize)
             
         else:
             # No geometry update needed, proceed immediately
             try:
                 self.finalize_load(result_data, cubes)
             except Exception as e:
                 self.log(f"Warning during finalize_load: {e}")
                 import traceback
                 self.log(traceback.format_exc())
         
         # User Request: Auto-switch to Visualization tab for ALL loads (startup and manual)
         # This must be outside the if-else to work for update_structure=False case
         QTimer.singleShot(150, lambda: self.tabs.setCurrentIndex(1))

    def finalize_load(self, result_data, cubes=None):
         # Note: Primary cleanup is done at start of on_load_finished.
         # This is a safety check in case finalize_load is called independently.
         if hasattr(self, 'freq_vis') and self.freq_vis:
             try: 
                 self.freq_vis.cleanup()
                 self.freq_vis = None
             except: pass

         # Remove old dock if it exists (safety check)
         if hasattr(self, 'freq_dock') and self.freq_dock:
             try:
                 mw = self.context.get_main_window()
                 mw.removeDockWidget(self.freq_dock)
                 self.freq_dock.deleteLater()
                 self.freq_dock = None
             except: pass

         # 7. Restore Frequency Data (AFTER geometry is guaranteed loaded)
         if result_data.get("freq_data"):
             self.log("Frequency data found in result.")
             self.freq_data = result_data["freq_data"]
             try:
                 mol = self.context.current_molecule
                 
                 if not mol:
                     self.log("Error: No molecule loaded for frequency visualizer.")
                     # Try to recover if possible? No, we need the molecule.
                 else:
                     self.log("Creating frequency visualizer...")
                     self.clear_3d_actors()
                     self.freq_vis = FreqVisualizer(
                         self.context.get_main_window(), 
                         mol, 
                         self.freq_data['freqs'], 
                         self.freq_data['modes'],
                         intensities=self.freq_data.get('intensities')
                     )
                     
                     from PyQt6.QtWidgets import QDockWidget
                     mw = self.context.get_main_window()
                     
                     # Dock cleanup already done at top of method
                     
                     self.freq_dock = QDockWidget("PySCF Frequencies", mw)
                     self.freq_dock.setWidget(self.freq_vis)
                     self.freq_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
                     mw.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.freq_dock)
                     self.freq_dock.show()
                     self.freq_dock.raise_()
                     self.log("Frequency Visualizer opened in Dock.")
             except Exception as e:
                 self.log(f"Error opening Frequency Visualizer: {e}")
                 import traceback
                 self.log(traceback.format_exc())
         else:
             self.log("No frequency data in loaded result.")

         # 9. Final State Update
         # CRITICAL: Do NOT un-conditionally enable items here.
         # Instead, ensure everything is valid, and RE-DISABLE existing files
         # to prevent the "reenable" bug.
         if cubes:
             self.disable_existing_analysis_items(cubes)

    def update_geometry(self, xyz_content):
        # Safely clear previous visualization
        self.clear_3d_actors()
        # Clean up Frequency Visualizer if active
        if hasattr(self, 'freq_vis') and self.freq_vis:
            try: self.freq_vis.cleanup()
            except: pass
            self.freq_vis = None
            
        # Use utils to update the specific molecule in context
        update_molecule_from_xyz(self.context, xyz_content)
        
        # Push Undo State
        # Push Undo State (Safe)
        try:
            mw = self.context.get_main_window()
            if hasattr(mw, 'push_undo_state'):
                mw.push_undo_state()
        except Exception as e:
            print(f"Undo push failed: {e}")
            
        self.log("Geometry updated.")

    def show_energy_diagram(self):
        if not hasattr(self, 'mo_data'): return
        
        # Modeless Check
        if hasattr(self, 'energy_dlg') and self.energy_dlg:
             self.energy_dlg.close()
             self.energy_dlg = None

        # Pass last_out_dir to allow loading cubes
        result_dir = getattr(self, 'last_out_dir', None)
        
        self.energy_dlg = EnergyDiagramDialog(self.mo_data, parent=self, result_dir=result_dir)
        self.energy_dlg.show()

    def load_file_by_path(self, path):
         # Helper to load a cube file from absolute path (used by Energy Diagram)
         if not os.path.exists(path): return
         
         norm_path = os.path.normpath(path)
         
         # 1. Update list selection if exists
         found = False
         for i in range(self.file_list.count()):
             item = self.file_list.item(i)
             item_path = os.path.normpath(item.toolTip())
             
             if item_path == norm_path:
                 self.file_list.setCurrentItem(item)
                 self.file_list.scrollToItem(item)
                 self.on_file_selected(item)
                 found = True
                 break
         
         # 2. If not in list (weird?), just force load
         if not found:
             # Just trigger standard logic
             self.switch_to_standard_mode(path)

    def show_thermo_data(self):
        if not hasattr(self, 'thermo_data'): 
            QMessageBox.information(self, "Info", "No thermodynamic data available.")
            return

        data = self.thermo_data
        
        # Helper function to flatten nested list structures
        def flatten_value(v):
            """
            Flatten nested list/tuple structures from PySCF thermo data.
            Examples:
            - [value, unit] -> (value, unit)
            - [[['value1'],['value2'],['value3']],[unit]] -> ("value1, value2, value3", unit)
            - [[value1, [value2], [value3], [Unit]]] -> flatten all numeric values
            - Simple value -> (value, None)
            """
            if v is None:
                return (None, None)
            
            # If it's a simple numeric type, return as-is
            if isinstance(v, (int, float, bool)):
                return (v, None)
            
            # If it's a string, return as-is
            if isinstance(v, str):
                return (v, None)
            
            # Handle list/tuple structures
            if isinstance(v, (list, tuple)):
                # Empty list
                if len(v) == 0:
                    return (None, None)
                
                # Single element
                if len(v) == 1:
                    return flatten_value(v[0])
                
                # Two elements: Check for special formats
                if len(v) == 2:
                    # Format: [[['value1'],['value2'],['value3']],[unit]]
                    # First element is list of values, second is unit string
                    if isinstance(v[1], str) and isinstance(v[0], (list, tuple)):
                        # Extract all values from nested structure
                        values_list = []
                        for item in v[0]:
                            val, _ = flatten_value(item)
                            if val is not None:
                                values_list.append(val)
                        
                        # Format multiple values as comma-separated string
                        if len(values_list) > 1:
                            formatted = ", ".join([f"{float(x):.6f}" if isinstance(x, (int, float)) else str(x) for x in values_list])
                            return (formatted, v[1])
                        elif len(values_list) == 1:
                            return (values_list[0], v[1])
                        else:
                            return (None, v[1])
                    
                    # Standard [value, unit] format
                    elif isinstance(v[1], str):
                        # Extract numeric value from first element
                        val, _ = flatten_value(v[0])
                        return (val, v[1])
                    else:
                        # Both are values, take first
                        val, unit = flatten_value(v[0])
                        return (val, unit)
                
                # More complex nested structure: [[value1, [value2], ...]]
                # Flatten recursively and collect all numeric values
                flat_values = []
                unit = None
                
                for item in v:
                    val, u = flatten_value(item)
                    if isinstance(val, (int, float)):
                        flat_values.append(val)
                    elif val is not None and not isinstance(val, str):
                        flat_values.append(val)
                    if u is not None:
                        unit = u
                
                if flat_values:
                    # If multiple values, format as comma-separated
                    if len(flat_values) > 1:
                        formatted = ", ".join([f"{float(x):.6f}" for x in flat_values])
                        return (formatted, unit)
                    else:
                        return (flat_values[0], unit)
                else:
                    return (None, unit)
            
            # Fallback: return as string
            return (str(v), None)
        
        # Create a dialog with table
        dlg = QDialog(self)
        dlg.setWindowTitle("Thermodynamic Properties")
        dlg.resize(600, 400)  
        
        layout = QVBoxLayout(dlg)
        
        # Add description label
        desc_label = QLabel("Thermodynamic Properties (standard conditions)")
        desc_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
        layout.addWidget(desc_label)
        
        # Create table
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Property", "Value", "Unit"])
        
        # Property order and labels
        order = ["E_tot", "H_tot", "G_tot", "ZPE", "S_tot", "Cv_tot"]
        labels = {
            "E_tot": "Total Energy (E0 + ZPE + corrections)",
            "H_tot": "Enthalpy (H)",
            "G_tot": "Gibbs Free Energy (G)",
            "S_tot": "Entropy (S)",
            "ZPE": "Zero Point Energy",
            "Cv_tot": "Heat Capacity (Cv)"
        }
        
        # Populate main properties
        row = 0
        for k in order:
            if k in data:
                v = data[k]
                label = labels.get(k, k)
                
                # Flatten nested structures
                value, unit_from_data = flatten_value(v)
                
                # Default unit if not provided in data
                unit = unit_from_data if unit_from_data else "Ha"
                
                # Format value
                if value is not None:
                    try:
                        vf = float(value)
                        value_str = f"{vf:.6f}"
                    except:
                        value_str = str(value)
                else:
                    value_str = "N/A"
                
                table.insertRow(row)
                table.setItem(row, 0, QTableWidgetItem(label))
                table.setItem(row, 1, QTableWidgetItem(value_str))
                table.setItem(row, 2, QTableWidgetItem(unit))
                row += 1
        
        # Add any other properties not in the main list
        for k, v in data.items():
            if k not in order:
                value, unit_from_data = flatten_value(v)
                
                # Format value
                if value is not None:
                    try:
                        vf = float(value)
                        value_str = f"{vf:.6f}"
                    except:
                        value_str = str(value)
                else:
                    value_str = str(v)
                
                table.insertRow(row)
                table.setItem(row, 0, QTableWidgetItem(k))
                table.setItem(row, 1, QTableWidgetItem(value_str))
                table.setItem(row, 2, QTableWidgetItem(unit_from_data if unit_from_data else ""))
                row += 1
        
        # Configure table appearance
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)  # Allow multi-select
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectItems)  # Select cells not rows
        table.setAlternatingRowColors(True)
        
        layout.addWidget(table)
        
        # Add button layout
        btn_layout = QHBoxLayout()
        
        # CSV Export button
        btn_export = QPushButton("Export CSV")
        btn_export.clicked.connect(lambda: self.export_thermo_csv(table))
        btn_layout.addWidget(btn_export)
        
        btn_layout.addStretch()
        
        # Close button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_layout.addWidget(btn_close)
        
        layout.addLayout(btn_layout)
        
        dlg.exec()

    def export_thermo_csv(self, table):
        """Export thermodynamic properties table to CSV file."""
        import csv
        
        fname, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Thermodynamic Data", 
            "thermodynamic_properties.csv", 
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not fname:
            return
        
        try:
            with open(fname, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                headers = []
                for col in range(table.columnCount()):
                    header_item = table.horizontalHeaderItem(col)
                    headers.append(header_item.text() if header_item else f"Column {col}")
                writer.writerow(headers)
                
                # Write data rows
                for row in range(table.rowCount()):
                    row_data = []
                    for col in range(table.columnCount()):
                        item = table.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)
            
            QMessageBox.information(self, "Export Successful", f"Data exported to:\n{fname}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CSV:\n{e}")

class EnergyDiagramDialog(QDialog):
    def __init__(self, mo_data, parent=None, result_dir=None):
        super().__init__(parent)
        self.result_dir = result_dir
        self.setWindowTitle("Orbital Energy Diagram")
        self.resize(450, 600)
        
        # Enable mouse tracking to receive hover events
        self.setMouseTracking(True)
        self.hit_zones = []
        
        # Add Save Button overlay
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 20, 20) # margins
        layout.addStretch()
        btn_layout = QHBoxLayout()
        # Unit Selection
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["eV", "Hartree"])
        self.unit_combo.currentTextChanged.connect(self.update_unit)
        self.lbl_unit = QLabel("Unit:")
        btn_layout.addWidget(self.lbl_unit)
        btn_layout.addWidget(self.unit_combo)
        
        btn_layout.addStretch() # Right align
        
        self.btn_save = QPushButton("Save PNG")
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #999;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)
        
        # Status Label (User Request: Bottom message)
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)
        
        self.data = mo_data
        self.is_uhf = (self.data["type"] == "UHF")
        
        # Extract energy levels
        self.energies = self.data["energies"]
        self.occupations = self.data["occupations"]

        # Safety: Flatten occupancy lists if they act weirdly (sometimes list of lists?)
        def safe_occ(occ_list):
            if not occ_list: return []
            # Check if first element is list
            if isinstance(occ_list[0], (list, tuple)):
                return [x[0] if len(x)>0 else 0 for x in occ_list]
            return occ_list

        if self.is_uhf:
             # UHF
             if len(self.energies) == 2 and isinstance(self.energies[0], list):
                 self.energies_a = self.energies[0]
                 self.energies_b = self.energies[1]
                 self.occ_a = safe_occ(self.occupations[0])
                 self.occ_b = safe_occ(self.occupations[1])
             else:
                 # Fallback
                 self.energies_a = self.energies
                 self.energies_b = []
                 self.occ_a = safe_occ(self.occupations)
                 self.occ_b = []
        else:
             # RHF
             self.energies_a = self.energies
             self.occ_a = safe_occ(self.occupations)
             self.energies_b = []
             self.occ_b = []
             
        all_e = self.energies_a + self.energies_b
            
        if not all_e:
            self.full_min = -1.0
            self.full_max = 1.0
            h_e, l_e = -0.5, 0.5
        else:
            self.full_min = min(all_e)
            self.full_max = max(all_e)
            
            # Find HOMO/LUMO for default center
            occupied = [e for i, e in enumerate(self.energies_a) if self.occ_a[i] > 0]
            virtual = [e for i, e in enumerate(self.energies_a) if self.occ_a[i] == 0]
            
            if self.is_uhf:
                occupied += [e for i, e in enumerate(self.energies_b) if self.occ_b[i] > 0]
                virtual += [e for i, e in enumerate(self.energies_b) if self.occ_b[i] == 0]
                
            h_e = max(occupied) if occupied else self.full_min
            l_e = min(virtual) if virtual else self.full_max
        
        # Store for double-click reset
        self.homo_energy = h_e
        self.lumo_energy = l_e
        gap_center = (h_e + l_e) / 2
        
        # User Request: Default view is 3x the HOMO-LUMO gap, centered on gap
        gap = abs(l_e - h_e)
        if gap < 0.01: gap = 0.05 # Fallback for near-degeneracy
        
        target_span = gap * 3.0
        
        # Ensure reasonable minimum view if gap is tiny
        if target_span < 0.2: target_span = 0.2
        
        self.current_min = gap_center - target_span / 2.0
        self.current_max = gap_center + target_span / 2.0
        
    def wheelEvent(self, event):
        # Optimized Pan Logic for Trackpads
        
        # 1. Try high-res pixel delta (Trackpads usually send this)
        pixel_delta = event.pixelDelta().y()
        angle_delta = event.angleDelta().y()
        
        # Pixels to Energy Scale estimate
        h = self.height()
        margin_top = 40
        margin_bottom = 40
        draw_h = h - margin_top - margin_bottom
        
        range_e = self.current_max - self.current_min
        if abs(range_e) < 1e-9: range_e = 1.0
        
        # How much energy per pixel?
        scale_per_pixel = range_e / draw_h if draw_h > 0 else 0.01
        
        change = 0.0
        
        if not event.pixelDelta().isNull() and pixel_delta != 0:
             # Trackpad Case: Direct 1:1 mapping feels natural
             # Scroll UP content (positive pixel_delta) -> View moves UP -> Energy increases
             # Sensitivity Factor can be tuned. 1.0 means 1 pixel scroll = 1 pixel shift
             sensitivity = 1.0 
             change = pixel_delta * scale_per_pixel * sensitivity
        elif angle_delta != 0:
             # Mouse Wheel Case: Fixed steps
             # 120 units = 1 notch usually.
             # Let's say 120 units = 10% shift
             
             fraction = angle_delta / 120.0
             change = (range_e * 0.1) * fraction
             
        # Apply Change
        # Scroll UP (Positive) usually means "Move content Down", so View moves UP.
        # Energy Y axis grows UP. 
        # So Positive Delta = Increase Min/Max
        
        self.current_min += change
        self.current_max += change
        self.update()
        
    def mouseDoubleClickEvent(self, event):
        # Reset to 3x HOMO-LUMO gap centered on the gap
        if hasattr(self, 'homo_energy') and hasattr(self, 'lumo_energy'):
            gap = abs(self.lumo_energy - self.homo_energy)
            center = (self.homo_energy + self.lumo_energy) / 2
            range_size = gap * 3
            self.current_min = center - range_size / 2
            self.current_max = center + range_size / 2
        else:
            # Fallback to full view if HOMO/LUMO not available
            self.current_min = self.full_min - 0.05 * (self.full_max - self.full_min)
            self.current_max = self.full_max + 0.05 * (self.full_max - self.full_min)
        self.update()

    def mousePressEvent(self, event):
         if event.button() == Qt.MouseButton.LeftButton:
             # Hit Testing with "Closest to Mouse" logic
             pos = event.position()
             point = pos.toPoint()
             y_click = point.y()
             
             best_hit = None
             min_dist = 1000.0
             
             if hasattr(self, 'hit_zones'):
                  for rect, index, label, spin_suffix in self.hit_zones:
                      # Check X-bounds first (strict)
                      if point.x() >= rect.left() and point.x() <= rect.right():
                          # Check Y-vicinity (e.g. +/- 10 pixels to allow slack)
                          center_y = rect.center().y()
                          dist = abs(y_click - center_y)
                          
                          if rect.contains(point):
                              if dist < min_dist:
                                  min_dist = dist
                                  best_hit = (index, label, spin_suffix)
                                  
             if best_hit:
                 # best_hit is (index, label, spin_suffix)
                 self.try_load_cube(best_hit[0], best_hit[1], best_hit[2])
                 return
             
             self.dragging = True
             self.last_mouse_y = event.position().y()

    def try_load_cube(self, index, label, spin_suffix=""):
        if not self.result_dir:
            # QMessageBox.information(self, "Info", "No result directory linked.")
            return
            
        import glob
        # Pattern matching
        # Files like: "15_HOMO.cube" or "16_LUMO.cube" or "15_MO_15.cube"
        
        # We need to construct likely filenames based on index (which is reliable)
        # Search pattern: "15_*.cube"
        
        # Try padded first (normalized sorting)
        # Check for 10a/10b convention if spin_suffix is present
        # spin_suffix is "_A" or "_B" from Diagram click
        
        patterns = []
        
        # New Convention: "15a_..." or "15b_..."
        # index is 0-based logic. Filename uses 1-based index.
        # Ensure we use 1-based index for finding file.
        idx_1b = int(index) + 1
        target_idx = idx_1b
        

        if spin_suffix == "_A":
             patterns.append(f"{target_idx:03d}a_*.cube") # Padded 010a
        elif spin_suffix == "_B":
             patterns.append(f"{target_idx:03d}b_*.cube") # Padded 010b
        else:
             # Standard Convention: "015_..." (Used for RHF or old UHF)
             # User Request: "Make them consistent"
             # New RHF uses 1-based index (target_idx), e.g. "016_HOMO.cube"
             patterns.append(f"{target_idx:03d}_*.cube") 
             
             # Legacy Fallback (optional, if user wants to load old files)
             # User Request: "comment out legacy" -> Strict uniformity.
             # patterns.append(f"{index:03d}_*.cube") # Legacy 0-based
             # patterns.append(f"{index}_*.cube")
        
        for p in patterns:
             full_p = os.path.join(self.result_dir, p)
             files = glob.glob(full_p)
             if files: break
        
        # Fallback to unpadded (legacy support) - handled in loop above
        
        if files:
            target = files[0] # Take first match
            # Call Parent method
            if hasattr(self.parent(), "load_file_by_path"):
                self.parent().load_file_by_path(target)
                # User Request: Bottom message instead of title
                self.status_label.setText(f"Loaded: {os.path.basename(target)}")
                # Clear title notify
                self.setWindowTitle("Orbital Energy Diagram")
        else:
             # File not found
             self.status_label.setText(f"File not found: {label}")
             # User Request: Use orbital number for label
             # Use safe 0-based syntax for worker
             mo_task_label = f"#{index}"
             
             # User Request: Confirm dialog "same with the rest"
             reply = QMessageBox.question(
                 self, 
                 "Confirm Analysis", 
                 f"Generate cube file for Orbital {label} (Index {index+1})?\nThis may take some time.",
                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
             )
             
             if reply == QMessageBox.StandardButton.Yes:
                 if hasattr(self.parent(), "generate_specific_orbital"):
                     self.status_label.setText(f"Generating {label}...")
                     
                     
                     # Force index-based request as per user requirement
                     # "make sure to use index num to generate or find"
                     
                     # We pass the explicit index to guaranteed unambiguous generation
                     self.parent().generate_specific_orbital(index, label, spin_suffix)

    def mouseMoveEvent(self, event):
        # Check if hovering over a clickable orbital level (when not dragging)
        if not (hasattr(self, 'dragging') and self.dragging):
            pos = event.position()
            point = pos.toPoint()
            
            hovering_over_orbital = False
            if hasattr(self, 'hit_zones'):
                hit_found = False
                for rect, index, label, spin_suffix in self.hit_zones:
                    if rect.contains(point):
                        hovering_over_orbital = True
                        hit_found = True
                        
                        # Show Tooltip with Index (User Request)
                        # Use 1-based index
                        idx_1b = index + 1
                        tip_text = f"Index: {idx_1b}"
                        if label: tip_text += f"\n{label}"
                        if spin_suffix: tip_text += f" ({spin_suffix.replace('_', '')})"
                        
                        QToolTip.showText(event.globalPosition().toPoint(), tip_text, self)
                        break
                
                if not hit_found:
                    QToolTip.hideText()
            
            # Update cursor based on hover state
            if hovering_over_orbital:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        
        # Handle drag-to-zoom functionality
        if hasattr(self, 'dragging') and self.dragging:
            current_y = event.position().y()
            delta_y = current_y - self.last_mouse_y
            
            # Zoom Logic (User Request: Drag to Zoom)
            # Drag DOWN (Positive Delta) -> Zoom OUT (Factor > 1)
            # Drag UP (Negative Delta) -> Zoom IN (Factor < 1)
            
            # Sensitivity
            factor = 1.0 + (delta_y * 0.01)
            
            # Bounds check
            if factor < 0.1: factor = 0.1
            if factor > 10.0: factor = 10.0
            
            span = self.current_max - self.current_min
            center = (self.current_min + self.current_max) / 2
            new_span = span * factor
            
            self.current_min = center - new_span / 2
            self.current_max = center + new_span / 2
            
            self.last_mouse_y = current_y
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        save_act = QAction("Save as PNG...", self)
        save_act.triggered.connect(self.save_image)
        menu.addAction(save_act)
        menu.exec(event.globalPos())

    def save_image(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Diagram", "orbital_diagram.png", "Images (*.png)")
        if fname:
            # User Request: Prevent exporting message (and UI controls)
            widgets_to_restore = []
            
            # Hide top buttons and bottom label
            if hasattr(self, 'unit_combo'): 
                self.unit_combo.setVisible(False)
                widgets_to_restore.append(self.unit_combo)

            if hasattr(self, 'lbl_unit'):
                self.lbl_unit.setVisible(False)
                widgets_to_restore.append(self.lbl_unit)
                # It's hard to reference. 
                # Alternative: Move controls to a container widget and hide that.
                # But layout is flat.
                # Just hiding save button and status label is likely enough as unit is small.
                # Actually, let's try to hide the layouts container if possible? No.
                
            if hasattr(self, 'btn_save'):
                self.btn_save.setVisible(False)
                widgets_to_restore.append(self.btn_save)
                
            if hasattr(self, 'status_label'):
                self.status_label.setVisible(False)
                widgets_to_restore.append(self.status_label)
            
            # Try to hide the "Unit:" label by iterating layout items?
            # It's the first item in btn_layout (item at index 0).
            # self.layout().itemAt(1) is btn_layout? 
            # btn_layout is itemAt(2) (after stretch and margin?) due to addLayout.
            
            # Simple approach: Capture only the paint area?
            # No, self.grab() behaves well.
            # Let's just hide what we can easily.
            
            QApplication.processEvents() # Ensure hidden
            
            pix = self.grab()
            pix.save(fname)
            
            # Restore
            for w in widgets_to_restore:
                w.setVisible(True)

    def update_unit(self, text):
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Reset Hit Zones
        self.hit_zones = [] # List of (QRect, index, label)
        from PyQt6.QtCore import QRect
        
        w = self.width()
        h = self.height()
        
        # Unit Conversion
        unit = self.unit_combo.currentText()
        factor = 1.0
        unit_label_str = "Ha"
        if unit == "eV":
            factor = nist.HARTREE2EV if nist else 27.211386245988
            unit_label_str = "eV"
        
        # Draw Background
        painter.fillRect(0, 0, w, h, QColor("white"))
        
        min_e = self.current_min
        max_e = self.current_max
        range_e = max_e - min_e
        if abs(range_e) < 1e-9: range_e = 1.0
        
        margin_top = 40
        margin_bottom = 80 # Increased for Save Button
        draw_h = h - margin_top - margin_bottom
        
        def val_to_y(val):
            rel = (val - min_e) / range_e
            return (h - margin_bottom) - (rel * draw_h)

        # --- Draw Axis ---
        pen_axis = QPen(QColor("black"), 2)
        painter.setPen(pen_axis)
        painter.drawLine(60, margin_top, 60, h - margin_bottom) # Shifted right for labels
        
        # Draw Unit Label
        font_axis = QFont("Arial", 10)
        painter.setFont(font_axis)
        painter.drawText(5, margin_top - 10, unit_label_str)

        # Draw Ticks (Standard tick logic omitted here for brevity as it follows later in original file, but we must ensure we don't break scope)
        # Wait, the original file has ticks logic between unit label and draw_levels...
        # My 'replace' messed up the flow.
        # I need to be careful. The ticks logic was seemingly preserved in the view after 2407?
        # In Step 771 view: Line 2407 is '# Dynamic Ticks...'.
        # My garbage lines ended at 2403 in step 780 view.
        # So I only corrupted the TOP part before ticks?
        # Let's check Step 780 view again.
        # Lines 2380-2403 in Step 780 contain the COMMENTS I pasted.
        # Lines 2404+ are `painter.fillRect...`.
        # This means I DUPLICATED the background code?
        # Yes, lines 2404 in step 780 match lines 2380 in step 771?
        # No.
        # I inserted the block at line 2373.
        # I need to DELETE the garbage block lines 2380-2403 and ensure the valid code follows.
        # BUT I also need to insert the layout variables (left_margin etc) which I wanted to add properly.
        # And I need to Fix `draw_levels` logic inside `draw_levels`.
        # `draw_levels` definition is FURTHER DOWN in the file (around 2560 in original, or 2482 in step 736 view).
        # My garbage block tried to patch `draw_levels` logic inside `paintEvent` scope erroneously.
        
        # Action: DELETE lines 2380-2403 (the garbage comments).
        # And THEN locate `draw_levels` definition properly and patch it there.
        # I will execute deletion first.
        


        # Dynamic Ticks (Round Numbers in Display Units)
        # 1. Determine Range in Display Units
        min_disp = min_e * factor
        max_disp = max_e * factor
        range_disp = max_disp - min_disp
        if abs(range_disp) < 1e-9: range_disp = 1.0
        
        # 2. Calculate Nice Step in Display Units
        import math
        target_ticks = 10
        raw_step = range_disp / target_ticks
        
        try:
             magnitude = 10 ** math.floor(math.log10(raw_step))
             residual = raw_step / magnitude
             if residual > 5: step = 10 * magnitude
             elif residual > 2: step = 5 * magnitude
             elif residual > 1: step = 2 * magnitude
             else: step = magnitude
        except:
             step = 1.0
             
        if step <= 0: step = 1.0
        
        # 3. Find First Tick in Display Units
        # e.g. min=-5.23, step=0.5 -> start=-5.0 (fail) or -5.5?
        # ceil(min/step)*step
        start_tick_disp = math.ceil(min_disp / step) * step
        
        # 4. Loop Ticks
        current_tick_disp = start_tick_disp
        # Use epsilon to handle float precision in loop
        while current_tick_disp <= max_disp + (step * 0.001):
             # Convert back to Internal Units for Y position
             val_internal = current_tick_disp / factor
             y = val_to_y(val_internal)
             
             if margin_top <= y <= (h - margin_bottom):
                 painter.drawLine(55, int(y), 60, int(y)) # Tick mark
                 
                 # Label (Use display value directly)
                 # Handle float precision "0.3000000004" -> "0.30"
                 label = f"{current_tick_disp:.2f}"
                 
                 # Right align text to x=50, Vertically Center
                 # Rect y-10 to y+10 for centering
                 painter.drawText(5, int(y) - 10, 45, 20, 
                                  Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, 
                                  label)
                 
             current_tick_disp += step

        # Draw Levels
        font = QFont("Arial", 12)
        painter.setFont(font)
        
        cols = 1 if not self.is_uhf else 2
        # Tighter layout for smaller window
        # Left margin for axis must fit Axis Ticks (x=50) AND Energy Labels (Orbital specific)
        # Energy labels are drawn at x1 - 85. x1 = left_margin + padding.
        # Need x1 >= 90+ for visibility? 
        # If x1=100 -> label at 15. 
        # left_margin=90 -> x1=100.
        # Let's use 120 to be safe and spacious.
        left_margin = 120 
        right_margin = 10 # Only for window edge, but column width absorbs text space
        avail_w = w - left_margin - right_margin
        col_width = avail_w / cols 
        
        # User Request: "Orbital lefter" and labels inside window.
        # Strategy: Left align the line within the column, leaving the rest for text.
        
        level_w = 50 # Fixed compact width (pixels)
        padding_left = 10 # Space from column start
        


        
        def find_somo_indices(energies_a, occ_a, energies_b, occ_b):
            """Find orbitals where Alpha is occupied but Beta is not (SOMO)"""
            somo_indices = set()
            
            # Use threshold 0.1 to avoid numerical precision errors
            occ_threshold = 0.1
            n_alpha = sum(1 for o in occ_a if o > occ_threshold)
            n_beta = sum(1 for o in occ_b if o > occ_threshold)
            
            start_somo = n_beta
            end_somo = n_alpha
            for i in range(start_somo, end_somo):
                somo_indices.add(i)
            return somo_indices

        if self.is_uhf:
            somo_set = find_somo_indices(self.energies_a, self.occ_a, self.energies_b, self.occ_b)
        else:
            somo_set = set()

        def draw_levels(energies, occs, col_idx, title):
            total_w = self.width()
            
            # Centering Logic
            # Centering Logic
            if not self.is_uhf:
                # User Request: Center RKS orbitals ("center but leftish")
                
                center_of_window = total_w / 2
                center_of_window = total_w / 2
                
                # User Request: "not centered the orbital. fix." -> Remove offset.
                line_center_x = center_of_window
                
                # Reverse calculate col_start
                raw_col_start = line_center_x - (level_w / 2) - padding_left
                
                # CLAMP against left_margin to prevent clipping of energy labels
                col_start = max(raw_col_start, left_margin)
                col_width = total_w 
            else:
                # UKS Logic: Use calculated column widths respecting margins
                # We use the outer scope 'col_width' and 'left_margin'
                # Note: 'col_width' in outer scope is 'avail_w / cols'
                
                # We should re-use col_width from outer scope if possible, 
                # but 'col_width' variable inside this function shadows outer scope if we assign to it?
                # Actually, we assign to 'col_width' in 'if' block, so it's local.
                # We need to recalculate it or access outer scope.
                # Outer scope: avail_w = w - left_margin - right_margin
                # cols = 2.
                
                # Re-calculate to be safe and explicit using 'avail_w' from outer scope
                u_col_width = avail_w / cols 
                col_start = left_margin + col_idx * u_col_width
                col_width = u_col_width # For title centering usage

            target_x1 = col_start + padding_left
            center_x = target_x1 + level_w/2
            
            painter.setPen(QColor("black"))
            
            # Title with Electron Count
            n_elec = sum(occs)
            title_text = f"{title}\n({n_elec:.0f}e)"
            
            fm = painter.fontMetrics()
            lines = title_text.split("\n")
            y_title_base = 20
            
            for line in lines:
                t_w = fm.horizontalAdvance(line)
                # User Request: "top label... at the position of appropriate... not one of the are at the center!!!"
                # "top label is not at the top of the orbital center"
                # Fix: Align title center with orbital line center (center_x)
                title_x = center_x - t_w / 2
                painter.drawText(int(title_x), y_title_base, line)
                y_title_base += 15
            
            homo_idx = -1
            for i, o in enumerate(occs):
                if o > 0: homo_idx = i
            
            # Lists
            occupied_items = []
            virtual_items = []
            
            for i, e, in enumerate(energies):
                if min_e <= e <= max_e:
                     item = (i, e, occs[i])
                     if occs[i] > 0:
                         occupied_items.append(item)
                     else:
                         virtual_items.append(item)
            
            occupied_items.sort(key=lambda x: x[1], reverse=True)
            virtual_items.sort(key=lambda x: x[1], reverse=False)
            
            # Colors and Labels
            is_alpha_col = (title == "Alpha")
            is_beta_col = (title == "Beta")
            
            # Color Definition
            # User Request: "rks should be black whole"
            # User Request: "do not change colors between filled or virtual in uks rks"
            if is_alpha_col:
                # Alpha: Uniform Red
                col_occ = QColor(180, 50, 50) 
                col_vir = QColor(180, 50, 50) 
            elif is_beta_col:
                # Beta: Uniform Blue
                col_occ = QColor(50, 50, 180)
                col_vir = QColor(50, 50, 180)
            else:
                # Restricted (RHF/RKS): Uniform Black
                col_occ = QColor("black")
                col_vir = QColor("black")

            def process_list(items, last_y_ref):
                new_last_y = last_y_ref
                for i_orig, e, occ_val in items:
                    y = val_to_y(e)
                    
                    is_occ = (occ_val > 0)
                    color = col_occ if is_occ else col_vir
                    pen = QPen(color, 2)
                    
                    is_somo = (is_alpha_col and i_orig in somo_set)
                    is_homo = (i_orig == homo_idx)
                    is_lumo = (i_orig == homo_idx + 1)
                    
                    # Highlight important levels
                    if is_somo or is_homo or is_lumo:
                        pen.setWidth(3)
                    
                    painter.setPen(pen)
                    x1 = center_x - level_w/2
                    x2 = center_x + level_w/2
                    painter.drawLine(int(x1), int(y), int(x2), int(y))
                    
                    # Electron Icons (Arrows)
                    if is_occ:
                        painter.setPen(QColor("black"))
                        # Center: center_x, y
                        # Font size increased (User Request)
                        f_icon = QFont("Arial", 16, QFont.Weight.Bold)
                        painter.setFont(f_icon)
                        
                        arrow_txt = ""
                        if not self.is_uhf:
                            # RKS/ROKS/RHF/ROHF
                            # Check occupancy for ROKS support (1.0 vs 2.0)
                            if abs(occ_val - 1.0) < 0.1:
                                arrow_txt = "↑" # Singly occupied (ROKS)
                            else:
                                arrow_txt = "↑↓" # Doubly occupied
                        else:
                            if is_alpha_col:
                                arrow_txt = "↑"
                            elif is_beta_col:
                                arrow_txt = "↓"
                        
                        # Adjust rect for centering
                        # Height needs to be enough for 16px font
                        rect_icon = QRect(int(x1), int(y)-14, int(level_w), 28)
                        painter.drawText(rect_icon, Qt.AlignmentFlag.AlignCenter, arrow_txt)
                        
                        # Restore font
                        painter.setFont(font)

                    # Store Hit Zone
                    rect_zone = QRect(int(x1), int(y)-4, int(x2-x1), 8)
                    
                    # Label Logic
                    # Scientific Labeling Logic (Revised)
                    label_txt = ""

                    # 1. ROKS SOMO Special Case
                    # ROKS: Restricted (not UHF) but with singly occupied orbital (occ ~ 1.0)
                    is_roks_somo = (not self.is_uhf) and (abs(occ_val - 1.0) < 0.1)

                    if is_roks_somo:
                        label_txt = "SOMO"
                    
                    # 2. HOMO / Occupied Labels
                    elif i_orig <= homo_idx:
                        diff = homo_idx - i_orig
                        if diff == 0:
                            # Standard HOMO (UHF/RHF)
                            # Or ROKS SOMO (but handled above, so this is only for UHF/RHF)
                            label_txt = "HOMO"
                        else:
                            # HOMO-n
                            # For ROKS: SOMO is at homo_idx. So homo_idx-1 is correctly "HOMO-1".
                            label_txt = f"HOMO-{diff}"
                    
                    # 3. LUMO / Virtual Labels
                    else:
                        diff = i_orig - (homo_idx + 1)
                        label_txt = "LUMO" if diff == 0 else f"LUMO+{diff}"
                    
                    # No overrides needed. Logic handled above.
                    
                    if label_txt:
                        painter.setPen(QColor("black"))
                        # Draw to the right of the line
                        painter.drawText(int(x2)+4, int(y)+4, label_txt)
                        
                    # User Request: Energy Values Missing / Unit Change
                    # Draw energy value to the left of the line
                    # Use factor and unit_label_str from outer scope (paintEvent)
                    vis_e_str = f"{e * factor:.2f} {unit_label_str}"
                    painter.setPen(QColor("black"))
                    # Calculate width to align right against the line
                    # x1 is left start of line. Draw to left of it.
                    # Ensure rect is wide enough and positioned correctly
                    rect_e = QRect(int(x1)-85, int(y)-7, 80, 14) 
                    painter.drawText(rect_e, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, vis_e_str)
                    
                    
                    # Store Hit Zone
                    # Format: (QRect, index, label_for_file_generation, spin_suffix)
                    # Expand width to include label (approx 80px) and height for easier clicking
                    r = QRect(int(x1), int(y)-7, int(level_w + 80), 14)
                    
                    # Determine file label (SOMO/HOMO/LUMO or just MO)
                    # "SELECTION ITEM NAME" -> Use the specific label if available
                    gen_label = label_txt if label_txt else f"MO_{i_orig+1}"
                    
                    spin_suffix = ""
                    if self.is_uhf:
                        spin_suffix = "_A" if is_alpha_col else "_B"
                        
                    self.hit_zones.append((r, i_orig, gen_label, spin_suffix))
                    
                return new_last_y

            process_list(occupied_items, -1000)
            process_list(virtual_items, 10000)

        
        if self.is_uhf:
            draw_levels(self.energies_a, self.occ_a, 0, "Alpha")
            draw_levels(self.energies_b, self.occ_b, 1, "Beta")
        else:
            draw_levels(self.energies_a, self.occ_a, 0, "Orbitals")




