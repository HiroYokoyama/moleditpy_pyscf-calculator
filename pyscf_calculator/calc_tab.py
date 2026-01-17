import os
import traceback
from rdkit import Chem

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QSpinBox, QCheckBox, QGroupBox, QFormLayout, 
    QMessageBox, QLineEdit, QFileDialog, QProgressBar, QTextEdit, QToolTip
)
from PyQt6.QtCore import Qt, QTimer

# Local Imports
try:
    from .worker import PySCFWorker
    from .utils import rdkit_to_xyz
    from .scan_dialog import ScanDialog
except ImportError:
    PySCFWorker = None
    ScanDialog = None
    rdkit_to_xyz = None

class CalcTab(QWidget):
    def __init__(self, parent_dialog, context, settings):
        super().__init__(parent_dialog)
        self.parent_dialog = parent_dialog
        self.context = context
        self.settings = settings
        self.worker = None
        self.scan_params = None
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)

        # --- Configuration Section ---
        config_group = QGroupBox("Calculation Settings")
        form_layout = QFormLayout()

        self.job_type_combo = QComboBox()
        self.job_type_combo.addItems([
            "Energy", 
            "Geometry Optimization", 
            "Frequency", 
            "Optimization + Frequency", 
            "Transition State Optimization", 
            "TS Optimization + Frequency",
            "TDDFT", 
            "Rigid Surface Scan", 
            "Relaxed Surface Scan"
        ])
        self.job_type_combo.currentTextChanged.connect(self.update_options)
        
        # Scan Config Button
        self.btn_scan_config = QPushButton("Configure Scan")
        self.btn_scan_config.clicked.connect(self.configure_scan)
        self.btn_scan_config.hide()
        
        h_job = QHBoxLayout()
        h_job.addWidget(self.job_type_combo)
        h_job.addWidget(self.btn_scan_config)
        
        form_layout.addRow("Job Type:", h_job)

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
        
        h_spin = QHBoxLayout()
        h_spin.addWidget(self.spin_input)
        
        form_layout.addRow("Spin Multiplicity (2S+1):", h_spin)
        
        # Auto Detect Button
        self.btn_auto_detect = QPushButton("Auto Detect")
        self.btn_auto_detect.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_auto_detect.clicked.connect(self.auto_detect_charge_spin)
        self.btn_auto_detect.setStyleSheet("padding: 2px 8px; margin-left: 5px;")
        
        form_layout.addRow("", self.btn_auto_detect)
        
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
        form_layout.addRow(self.check_symmetry) 
        
        self.check_break_sym = QCheckBox("Break Initial Guess Symmetry")
        self.check_break_sym.setChecked(False)
        self.check_break_sym.setToolTip("For UKS/UHF: Mix Alpha/Beta densities in initial guess to encourage spin polarization.")
        form_layout.addRow(self.check_break_sym)
        
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
        home_dir = os.path.expanduser("~")
        self.out_dir_edit.setText(os.path.join(home_dir, "PySCF_Results"))
        self.out_dir_edit.setToolTip("Path to save results. Relative paths (e.g. 'results') will be resolved relative to the current file location or default to the home directory.")
        
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_out_dir)
        
        h_box = QHBoxLayout()
        h_box.addWidget(self.out_dir_edit)
        h_box.addWidget(btn_browse)
        form_layout.addRow("Output Dir:", h_box)

        # Save as Default
        h_default = QHBoxLayout()
        h_default.addStretch()
        self.btn_save_default = QPushButton("Save as Default")
        self.btn_save_default.setStyleSheet("padding: 5px;") 
        self.btn_save_default.clicked.connect(self.parent_dialog.save_custom_defaults)
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

        self.job_type_combo.setCurrentText("Optimization + Frequency")

    def update_options(self, text=None):
        method = self.method_combo.currentText()
        is_dft = "KS" in method
        self.functional_combo.setEnabled(is_dft)
        
        job = self.job_type_combo.currentText()
        if "Scan" in job:
            if hasattr(self, 'btn_scan_config'): self.btn_scan_config.show()
        else:
            if hasattr(self, 'btn_scan_config'): self.btn_scan_config.hide()
        
        is_unrestricted = method in ["UKS", "UHF"]
        if hasattr(self, 'check_break_sym'):
             self.check_break_sym.setEnabled(is_unrestricted)
        
        job = self.job_type_combo.currentText()
        if hasattr(self, 'lbl_nstates') and hasattr(self, 'nstates_input'):
             is_tddft = (job == "TDDFT")
             self.lbl_nstates.setVisible(is_tddft)
             self.nstates_input.setVisible(is_tddft)

    def auto_detect_charge_spin(self):
        if not self.context or not self.context.current_molecule:
            QMessageBox.warning(self, "Warning", "No molecule loaded.")
            return

        try:
            mol = self.context.current_molecule
            total_electrons = 0
            has_transition_metal = False
            
            tm_nums = set(list(range(21, 31)) + list(range(39, 49)) + list(range(72, 81)))
            
            for atom in mol.GetAtoms():
                an = atom.GetAtomicNum()
                total_electrons += an
                if an in tm_nums:
                    has_transition_metal = True
            
            charge = Chem.GetFormalCharge(mol)
            net_electrons = total_electrons - charge
            
            if net_electrons % 2 == 0:
                suggested_spin = 1
            else:
                suggested_spin = 2
                
            idx_c = self.charge_input.findText(str(charge))
            if idx_c >= 0: self.charge_input.setCurrentIndex(idx_c)
            else: self.charge_input.setCurrentText(str(charge))
            
            target_str = str(suggested_spin)
            found = False
            for i in range(self.spin_input.count()):
                text = self.spin_input.itemText(i)
                if text.startswith(target_str + " "):
                    self.spin_input.setCurrentIndex(i)
                    found = True
                    break
            
            current_method = self.method_combo.currentText()
            new_method = current_method
            
            if suggested_spin == 1:
                if current_method == "UHF": new_method = "RHF"
                elif current_method == "UKS": new_method = "RKS"
            else:
                if current_method == "RHF": new_method = "UHF"
                elif current_method == "RKS": new_method = "UKS"
            
            if new_method != current_method:
                self.method_combo.setCurrentText(new_method)
                self.log(f"Auto-Detect: Switched method to {new_method} based on spin.")
            
            if has_transition_metal:
                self.log("Auto-Detect: Transition metal detected. High spin states may be possible.")
                QMessageBox.information(self, "Info", 
                    "Transition metal detected.\nCharge and Spin have been set to standard values (Low Spin),\nbut you may need to adjust Multiplicity manually for High Spin states.")
            else:
                self.log(f"Auto-Detect: Set Charge={charge}, Mult={suggested_spin} (Electrons={net_electrons})")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Auto-detection failed: {e}")

    def validate_spin_settings(self):
        try:
            if not self.context or not self.context.current_molecule: return
            
            mol = self.context.current_molecule
            total_protons = sum(atom.GetAtomicNum() for atom in mol.GetAtoms())
            
            try:
                charge = int(self.charge_input.currentText())
                spin_txt = self.spin_input.currentText()
                if " " in spin_txt:
                    mult = int(spin_txt.split(" ")[0])
                else:
                    mult = int(spin_txt)
            except:
                return
                
            electrons = total_protons - charge
            unpaired = mult - 1
            remaining = electrons - unpaired
            
            is_valid = (remaining >= 0) and (remaining % 2 == 0)
            
            if is_valid:
                self.spin_input.setStyleSheet("")
                self.spin_input.setToolTip("")
                self.charge_input.setStyleSheet("")
                self.charge_input.setToolTip("")
            else:
                style = "background-color: #ffcccc; color: black;"
                self.spin_input.setStyleSheet(style) 
                self.charge_input.setStyleSheet(style)
                
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

    def configure_scan(self):
        if not self.context or not self.context.current_molecule:
             QMessageBox.warning(self, "No Molecule", "Please load a molecule first.")
             return
             
        if ScanDialog is None:
             QMessageBox.critical(self, "Error", "ScanDialog not imported.")
             return

        if not hasattr(self, 'scan_params'): self.scan_params = None
        
        if hasattr(self, '_scan_config_dlg'):
            try: self._scan_config_dlg.close()
            except: pass
            
        self._scan_config_dlg = ScanDialog(self, self.context, initial_params=self.scan_params)
        self._scan_config_dlg.scan_configured.connect(self.on_scan_configured)
        self._scan_config_dlg.show()

    def on_scan_configured(self, params):
        self.scan_params = params
        QMessageBox.information(self, "Scan Configured", 
            f"Scan set: {self.scan_params['type']} ({self.scan_params['steps']} steps)")

    def get_spin_value(self):
        try:
            txt = self.spin_input.currentText()
            if " " in txt:
                return int(txt.split(" ")[0])
            return int(txt)
        except:
            return 1

    def run_calculation(self):
        if not self.context or not self.context.current_molecule:
            msg = "Error: No molecule loaded. Please load a molecule in the main window."
            self.log(msg)
            QMessageBox.warning(self, "No Molecule", "Please load a molecule first.")
            return

        raw_out_dir = self.out_dir_edit.text().strip()
        final_out_dir = raw_out_dir
        
        if not os.path.isabs(raw_out_dir):
            mw = self.context.get_main_window()
            current_path = getattr(mw, 'current_file_path', None)
            
            if not current_path:
                fallback_base = os.path.expanduser("~")
                fallback_path = os.path.abspath(os.path.join(fallback_base, raw_out_dir))
                
                msg = (f"You are using a relative path ('{raw_out_dir}') with an unsaved project.\n"
                       "Do you want to save the project first to establish a base directory?\n\n"
                       f"Clicking 'No' will save results to: {fallback_path}")
                
                reply = QMessageBox.question(self, "Unsaved Project", msg, 
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                
                if reply == QMessageBox.StandardButton.Yes:
                    if hasattr(mw, 'save_project'):
                        mw.save_project()
                    current_path = getattr(mw, 'current_file_path', None)
            
            if current_path:
                base_dir = os.path.dirname(current_path)
                final_out_dir = os.path.join(base_dir, raw_out_dir)
            else:
                final_out_dir = os.path.join(os.path.expanduser("~"), raw_out_dir)

        job_type = self.job_type_combo.currentText()
        if "Scan" in job_type:
            if not hasattr(self, 'scan_params') or not self.scan_params:
                reply = QMessageBox.question(self, "Scan Not Configured", 
                    "Scan parameters are missing. Configure now?", 
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    self.configure_scan()
                    if not self.scan_params: return
                else:
                    return

        config = {
            "job_type": job_type,
            "scan_params": getattr(self, 'scan_params', None),
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
            "out_dir": os.path.abspath(final_out_dir),
            "plugin_version": getattr(self.parent_dialog, 'version', '0.0.0')
        }
        
        try:
            os.makedirs(config["out_dir"], exist_ok=True)
        except Exception as e:
            self.log(f"Error creating output directory: {e}")
            return

        self.parent_dialog.btn_load_geom.setEnabled(False)
        self.parent_dialog.optimized_xyz = None 

        if PySCFWorker is None:
            self.log("Error: Could not import PySCFWorker. Check installation.")
            return

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.show()
        self.log_text.clear()
        self.log("Starting PySCF Calculation...\n---------------------------------")

        xyz_str = rdkit_to_xyz(self.context.current_molecule)
        
        self.worker = PySCFWorker(xyz_str, config)
        self.worker.log_signal.connect(self.log_append)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.result_signal.connect(self.parent_dialog.on_results) 
        
        self.worker.start()

    def stop_calculation(self):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.log("\nStopping calculation...")
            
            try:
                self.worker.log_signal.disconnect()
                self.worker.finished_signal.disconnect()
                self.worker.error_signal.disconnect()
                self.worker.result_signal.disconnect()
            except:
                pass

            if not self.worker.wait(500):
                self.worker.terminate()
                self.worker.wait()
            
            self.log("Calculation stopped.")
            self.cleanup_ui_state()

    def log(self, message):
        self.log_text.append(message)
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

    def log_append(self, text):
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_text.setTextCursor(cursor)

    def on_finished(self):
        self.parent_dialog.update_internal_state()
        
        if self.context:
             mw = self.context.get_main_window()
             if mw:
                 mw.has_unsaved_changes = True
                 mw.update_window_title()

        self.log("\n---------------------------------\nCalculation Finished.")
        self.cleanup_ui_state()

    def on_error(self, err_msg):
        self.log(f"\nERROR: {err_msg}")
        QMessageBox.critical(self, "Calculation Error", err_msg)
        self.cleanup_ui_state()

    def cleanup_ui_state(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.hide()
        self.worker = None

