import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QTextEdit, QProgressBar, QCheckBox, QGroupBox,
    QFormLayout, QMessageBox, QFileDialog, QTabWidget, QWidget, QLineEdit,
    QSpinBox, QListWidget, QListWidgetItem, QDoubleSpinBox
)
from PyQt6.QtCore import Qt
try:
    from .worker import PySCFWorker
except ImportError:
    # Fallback for when basic imports fail or circular deps (shouldn't happen here)
    PySCFWorker = None

import json
from rdkit import Chem

class PySCFDialog(QDialog):
    def __init__(self, parent=None, context=None, settings=None):
        super().__init__(parent)
        self.context = context
        self.settings = settings if settings is not None else {}
        self.setWindowTitle("PySCF Calculator")
        self.resize(600, 700)
        self.worker = None
        self.setup_ui()
        self.load_settings()

    def save_settings(self):
        # Update shared dictionary
        self.settings["job_type"] = self.job_type_combo.currentText()
        self.settings["method"] = self.method_combo.currentText()
        self.settings["functional"] = self.functional_combo.currentText()
        self.settings["basis"] = self.basis_combo.currentText()
        self.settings["charge"] = self.charge_input.currentText()
        self.settings["spin"] = self.spin_input.currentText()
        self.settings["out_dir"] = self.out_dir_edit.text()

    def load_settings(self):
        s = self.settings
        if "job_type" in s: self.job_type_combo.setCurrentText(s["job_type"])
        if "method" in s: self.method_combo.setCurrentText(s["method"])
        if "functional" in s: self.functional_combo.setCurrentText(s["functional"])
        if "basis" in s: self.basis_combo.setCurrentText(s["basis"])
        if "charge" in s: self.charge_input.setCurrentText(s["charge"])
        if "spin" in s: self.spin_input.setCurrentText(s["spin"])
        if "out_dir" in s: self.out_dir_edit.setText(s["out_dir"])

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
        self.job_type_combo.addItems(["Energy", "Geometry Optimization", "Frequency", "Optimization + Frequency", "ESP"])
        self.job_type_combo.currentTextChanged.connect(self.update_options)
        form_layout.addRow("Job Type:", self.job_type_combo)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["RKS", "RHF", "UKS", "UHF"])
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
        form_layout.addRow("Charge:", self.charge_input)

        self.spin_input = QComboBox()
        self.spin_input.addItems([str(i) for i in range(0, 6)])
        self.spin_input.setCurrentText("0")
        form_layout.addRow("Spin (2S):", self.spin_input)
        
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
        form_layout.addRow("", self.check_symmetry)
        
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
        
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_out_dir)
        
        h_box = QHBoxLayout()
        h_box.addWidget(self.out_dir_edit)
        h_box.addWidget(btn_browse)
        form_layout.addRow("Output Dir:", h_box)

        config_group.setLayout(form_layout)
        layout.addWidget(config_group)

        # --- Actions ---
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Calculation")
        self.run_btn.clicked.connect(self.run_calculation)
        self.run_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        
        # Disable if no molecule loaded
        if not self.context.current_molecule:
             self.run_btn.setEnabled(False)
             self.run_btn.setToolTip("Load a molecule to run calculations.")
             self.log("Plugin started in Viewer Mode (No molecule loaded).")
        
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
        
        # --- Analysis Selection ---
        analysis_group = QGroupBox("Post-Calculation Analysis")
        a_layout = QVBoxLayout(analysis_group)
        
        # Orbital/Property List
        self.orb_list = QListWidget()
        self.orb_list.setFixedHeight(120)
        a_layout.addWidget(self.orb_list)
        
        # Manual MO Input
        mo_input_layout = QHBoxLayout()
        self.mo_input = QLineEdit()
        self.mo_input.setPlaceholderText("MO Index (e.g. 15)")
        self.btn_add_mo = QPushButton("Add")
        self.btn_add_mo.setFixedWidth(60)
        self.btn_add_mo.clicked.connect(self.add_custom_mo)
        mo_input_layout.addWidget(self.mo_input)
        mo_input_layout.addWidget(self.btn_add_mo)
        a_layout.addLayout(mo_input_layout)
        
        # Populate with default range (can be done here or after calc)
        self.populate_analysis_options()
        
        # Button to run analysis
        self.btn_run_analysis = QPushButton("Generate & Visualize Selected")
        self.btn_run_analysis.clicked.connect(self.run_selected_analysis)
        self.btn_run_analysis.setEnabled(False)
        a_layout.addWidget(self.btn_run_analysis)
        
        # Energy Diagram Button
        self.btn_show_diagram = QPushButton("Show Orbital Energy Diagram")
        self.btn_show_diagram.clicked.connect(self.show_energy_diagram)
        self.btn_show_diagram.setEnabled(False)
        a_layout.addWidget(self.btn_show_diagram)
        
        # Thermo Button
        self.btn_show_thermo = QPushButton("Show Thermodynamic Properties")
        self.btn_show_thermo.clicked.connect(self.show_thermo_data)
        self.btn_show_thermo.setEnabled(False)
        a_layout.addWidget(self.btn_show_thermo)
        
        layout.addWidget(analysis_group)
        
        # File List (Results)
        layout.addWidget(QLabel("Visualization Files:"))
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
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
        self.iso_spin.setSingleStep(0.01)
        self.iso_spin.setValue(0.05)
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
        self.m_iso_spin.setValue(0.004) # Standard density iso
        self.m_iso_spin.valueChanged.connect(self.update_mapped_vis)
        m_iso_layout.addWidget(self.m_iso_spin)
        m_layout.addLayout(m_iso_layout)
        
        # Min/Max Controls
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Min/Max:"))
        
        self.m_min_spin = QDoubleSpinBox()
        self.m_min_spin.setRange(-10.0, 10.0)
        self.m_min_spin.setValue(-0.05)
        self.m_min_spin.setSingleStep(0.01)
        self.m_min_spin.valueChanged.connect(self.update_mapped_vis)
        range_layout.addWidget(self.m_min_spin)
        
        self.m_max_spin = QDoubleSpinBox()
        self.m_max_spin.setRange(-10.0, 10.0)
        self.m_max_spin.setValue(0.05)
        self.m_max_spin.setSingleStep(0.01)
        self.m_max_spin.valueChanged.connect(self.update_mapped_vis)
        range_layout.addWidget(self.m_max_spin)
        
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

    def on_file_selected(self, item):
        self.clear_3d_actors()
        path = item.toolTip() 
        if not os.path.exists(path): return

        # Check for ESP Pair (heuristic: if file is esp.cube, look for density.cube in same dir)
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        
        is_esp_pair = False
        if basename == "esp.cube" and os.path.exists(os.path.join(dirname, "density.cube")):
            is_esp_pair = True
            surf_file = os.path.join(dirname, "density.cube")
            prop_file = path
        
        if is_esp_pair:
            self.switch_to_mapped_mode(surf_file, prop_file)
        else:
            self.switch_to_standard_mode(path)

    def switch_to_standard_mode(self, path):
        self.mode = "standard"
        self.vis_controls.show()
        self.vis_controls.setEnabled(True)
        self.mapped_group.hide()
        
        # Clean mapped
        if self.mapped_visualizer: self.mapped_visualizer.clear_actors()
        
        self.loaded_file = path
        if self.visualizer.load_file(path):
            self.iso_spin.setRange(0.0001, self.visualizer.data_max)
            self.iso_spin.setValue(min(0.05, self.visualizer.data_max * 0.1))
            self.update_visualization()

    def switch_to_mapped_mode(self, surf_file, prop_file):
        self.mode = "mapped"
        self.vis_controls.hide()
        self.mapped_group.show()
        
        # Clean standard
        self.visualizer.clear_actors()
        
        from .vis import MappedVisualizer
        if not self.mapped_visualizer:
            self.mapped_visualizer = MappedVisualizer(self.context.get_main_window())
            
        if self.mapped_visualizer.load_files(surf_file, prop_file):
             # Auto settings
             self.m_iso_spin.setValue(0.002) # Standard for density
             
             # Use Mapped Range from Surface Sample (not full volume)
             p_min, p_max = self.mapped_visualizer.get_mapped_range(0.002)
             
             # Expand range slightly for UI
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
        if not self.mapped_visualizer: return
        iso = self.m_iso_spin.value()
        val_min = self.m_min_spin.value()
        val_max = self.m_max_spin.value()
        cmap = self.cmap_combo.currentText()
        # Use main opacity slider? Yes.
        opacity = self.op_slider.value() / 100.0
        
        self.mapped_visualizer.update_mesh(
            iso, opacity, cmap=cmap, clim=[val_min, val_max]
        )

    def update_visualization(self):
        if self.mode == "mapped":
            self.update_mapped_vis()
            return

        if not self.loaded_file: return
        
        val = self.iso_spin.value()
        opacity = self.op_slider.value() / 100.0
        
        self.visualizer.update_iso(val, self.color_p, self.color_n, opacity)

    def closeEvent(self, event):
        if self.visualizer: self.visualizer.clear_actors()
        if self.mapped_visualizer: self.mapped_visualizer.clear_actors()
        
        # Close Dock
        if hasattr(self, 'freq_dock') and self.freq_dock:
             mw = self.context.get_main_window()
             mw.removeDockWidget(self.freq_dock)
             self.freq_dock.close()
             self.freq_dock = None
             
        # Cleanup visualizer resources
        if hasattr(self, 'freq_vis') and self.freq_vis:
            try: self.freq_vis.cleanup()
            except: pass
            
        super().closeEvent(event)

    def update_options(self, text=None):
        method = self.method_combo.currentText()
        is_dft = "KS" in method
        self.functional_combo.setEnabled(is_dft)
        
        job = self.job_type_combo.currentText()
        # Maybe customize behavior based on job type (e.g. ESP doesn't need Opt params)

    def run_calculation(self):
        if not self.context or not self.context.current_molecule:
            self.log("Error: No molecule loaded.")
            return

        self.save_settings() # Save persistence

        # Prepare configuration
        config = {
            "job_type": self.job_type_combo.currentText(),
            "method": self.method_combo.currentText(),
            "functional": self.functional_combo.currentText(),
            "basis": self.basis_combo.currentText(),
            "charge": int(self.charge_input.currentText()),
            "spin": int(self.spin_input.currentText()),
            "threads": self.spin_threads.value(),
            "memory": self.spin_memory.value(),
            "symmetry": self.check_symmetry.isChecked(),
            "max_cycle": self.spin_cycles.value(),
            "conv_tol": self.edit_conv.text(),
            "out_dir": self.out_dir_edit.text()
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
        if self.worker and self.worker.isRunning():
            self.log("\nStopping calculation...")
            self.worker.terminate() # Forceful but necessary for long running C extensions sometimes
            # Ideally we check a flag, but PySCF C-code might guard
            # For now, simple terminate.
            self.worker.wait()
            self.cleanup_ui_state()

    def log(self, message):
        self.log_text.append(message)
        # Scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

    def log_append(self, text):
        # Insert without adding extra newlines if chunks are small
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_text.setTextCursor(cursor)

    def on_finished(self):
        self.log("\n---------------------------------\nCalculation Finished.")
        self.cleanup_ui_state()

    def on_error(self, err_msg):
        self.log(f"\nERROR: {err_msg}")
        QMessageBox.critical(self, "Calculation Error", err_msg)
        self.cleanup_ui_state()

    def clear_3d_actors(self):
        # Clear generic actors (Orbitals/Density)
        if hasattr(self, 'context') and self.context:
             mw = self.context.get_main_window()
             if hasattr(mw, 'plotter'):
                mw.plotter.remove_actor("pyscf_iso_p") 
                mw.plotter.remove_actor("pyscf_iso_n")
                mw.plotter.remove_actor("pyscf_mapped")
                mw.plotter.render()
        
        # Clear Freq Vis vectors
        if hasattr(self, 'freq_vis') and self.freq_vis:
            try:
                self.freq_vis.cleanup()
            except: pass

    def cleanup_ui_state(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.hide()
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
        
        if is_digit:
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
                     
                     if idx <= homo_i:
                         diff = homo_i - idx
                         lb = "HOMO" if diff == 0 else f"HOMO-{diff}"
                         display_label = f"{lb} (Index {idx})"
                     else:
                         # LUMO
                         lumo_i = homo_i + 1
                         diff = idx - lumo_i
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
        
        # Orbitals (HOMO-4 to LUMO+4)
        # We order them from LUMO+4 down to HOMO-4 for logical vertical stack?
        # Or just standard list. Let's do Standard list.
        
        range_mo = range(-4, 5) # -4 to +4
        
        # LUMOs (ascending energy usually displayed top? No, usually list)
        for i in reversed(range(0, 5)):
            if i == 0: label = "LUMO"
            else: label = f"LUMO+{i}"
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, label)
            self.orb_list.addItem(item)

        # HOMOs
        for i in range(0, 5):
            if i == 0: label = "HOMO"
            else: label = f"HOMO-{i}"
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, label)
            self.orb_list.addItem(item)
            
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

        self.log(f"Starting Analysis for: {', '.join(tasks)}...")
        
        # Create PropertyWorker
        # We need to import it. It might not be imported yet if it's new in worker.py
        from .worker import PropertyWorker
        
        # Determine Output Directory
        # Prioritize keeping files with the checkpoint
        import os
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
        files = result_data.get("files", [])
        files.sort() # Sort A-Z
        self.log(f"Generated {len(files)} new files.")
        
        last_added_item = None
        new_esp_item = None
        
        # Add to file list if not exists
        for fpath in files:
            name = os.path.basename(fpath)
            
            # Check/Add to list
            found_item = None
            for i in range(self.file_list.count()):
                it = self.file_list.item(i)
                if it.toolTip() == fpath:
                    found_item = it
                    break
            
            if not found_item:
                from PyQt6.QtWidgets import QListWidgetItem
                item = QListWidgetItem(name)
                item.setToolTip(fpath)
                self.file_list.addItem(item)
                found_item = item
            
            last_added_item = found_item
            
            # Check if this NEW file is an ESP file
            if name.startswith("esp") and name.endswith(".cube"):
                new_esp_item = found_item
        
        # Auto-select prioritization
        # Only select ESP if it was in the NEW results
        if new_esp_item:
             self.file_list.setCurrentItem(new_esp_item)
             self.on_file_selected(new_esp_item)
        elif last_added_item:
             self.file_list.setCurrentItem(last_added_item)
             self.on_file_selected(last_added_item)

    def on_results(self, result_data):
        # Handle post-processing, e.g., visualization
        self.log("Processing results...")
        
        if result_data.get("optimized_xyz"):
             self.optimized_xyz = result_data["optimized_xyz"]
             self.btn_load_geom.setEnabled(True)
             self.log("Optimization converged. Automatically updating geometry...")
             self.update_geometry(self.optimized_xyz)
        
        # Store Energy Data
        if result_data.get("mo_energy"):
             self.mo_data = {
                 "energy": result_data["mo_energy"],
                 "occ": result_data["mo_occ"],
                 "type": result_data.get("scf_type", "RHF")
             }
             self.btn_show_diagram.setEnabled(True)
             
             # Auto-Show Diagram as requested
             # self.show_energy_diagram() # Disabled per user request

        # Store chkfile & path
        if result_data.get("chkfile"):
            self.chkfile_path = result_data["chkfile"]
            self.last_out_dir = result_data.get("out_dir") # Store specific job folder
            self.btn_run_analysis.setEnabled(True)
            self.log(f"Checkpoint saved: {self.chkfile_path}")
            
            # --- Update Result Path Display ---
            if "out_dir" in result_data:
                self.result_path_display.setText(result_data["out_dir"])

        if result_data.get("cube_files"):
            # Inform user about generated cube files
            files = result_data["cube_files"]
            files.sort() # Sort A-Z
            self.log(f"Generated Cube Files: {len(files)}")
            
            # Add to list
            self.file_list.clear() 
            from PyQt6.QtWidgets import QListWidgetItem
            for fpath in files:
                name = os.path.basename(fpath)
                item = QListWidgetItem(name)
                item.setToolTip(fpath)
                self.file_list.addItem(item)

            self.tabs.setCurrentIndex(1)
            
        if result_data.get("freq_data"):
             self.log("Frequency Analysis available.")
             # Store data
             self.freq_data = result_data["freq_data"]
             
             # Create/Show Freq Visualizer
             try:
                 from .freq_vis import FreqVisualizer
                 mol = self.context.current_molecule
                 
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
                 
                 # Close existing dock if any
                 if hasattr(self, 'freq_dock') and self.freq_dock:
                     mw.removeDockWidget(self.freq_dock)
                     self.freq_dock.deleteLater()
                     self.freq_dock = None
                 
                 self.freq_dock = QDockWidget("PySCF Frequencies", mw)
                 self.freq_dock.setWidget(self.freq_vis)
                 self.freq_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
                 mw.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.freq_dock)
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
             
        # Auto-Switch to Visualization Tab
        self.tabs.setCurrentIndex(1)

    def load_result_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Result Directory")
        if not d: return
        
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
        
        # Run LoadWorker
        from .worker import LoadWorker
        if 'LoadWorker' not in globals() and 'LoadWorker' not in locals():
             # Safety import if not yet linked
             try: from .worker import LoadWorker
             except: pass
        
        self.load_worker = LoadWorker(chk_path)
        self.load_worker.finished_signal.connect(self.on_load_finished)
        self.load_worker.error_signal.connect(self.on_error)
        
        self.log(f"\nLoading result from: {d}...")
        self.progress_bar.show()
        self.load_worker.start()

    def on_load_finished(self, result_data):
         self.log("Result loaded successfully.")
         self.progress_bar.hide()
         
         # 1. Restore Checkpoint Path
         if result_data.get("chkfile"):
             self.chkfile_path = result_data["chkfile"]
             self.last_out_dir = os.path.dirname(self.chkfile_path)
             
         # 2. Restore Energy Data
         if result_data.get("mo_energy"):
             self.mo_data = {
                 "energy": result_data["mo_energy"],
                 "occ": result_data["mo_occ"],
                 "type": result_data.get("scf_type", "RHF")
             }
             self.btn_show_diagram.setEnabled(True)
             self.btn_run_analysis.setEnabled(True)
             
         # 3. Restore Geometry (Optional)
         if result_data.get("optimized_xyz"):
             self.optimized_xyz = result_data["optimized_xyz"]
             self.btn_load_geom.setEnabled(True)
             msg = QMessageBox.question(self, "Load Structure", "Do you want to update the 3D structure to the loaded geometry?", 
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
             if msg == QMessageBox.StandardButton.Yes:
                 self.update_geometry(self.optimized_xyz)
         
         # 4. Scan for Visualization Files
         self.file_list.clear()
         d = self.last_out_dir
         if d and os.path.exists(d):
             import glob
             # Cubes
             cubes = glob.glob(os.path.join(d, "*.cube"))
             cubes.sort() # Sort A-Z
             for c in cubes:
                 name = os.path.basename(c)
                 item = QListWidgetItem(name)
                 item.setToolTip(c)
                 self.file_list.addItem(item)
             self.log(f"Found {len(cubes)} existing visualization files.")

         # 5. Enable Checkboxes
         for i in range(self.orb_list.count()):
             item = self.orb_list.item(i)
             item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)
             # Optionally uncheck all initially
             item.setCheckState(Qt.CheckState.Unchecked)

    def update_geometry(self, xyz_content):
        # Use utils to update the specific molecule in context
        from .utils import update_molecule_from_xyz
        update_molecule_from_xyz(self.context, xyz_content)
        self.log("Geometry updated.")

    def show_energy_diagram(self):
        if not hasattr(self, 'mo_data'): return
        dlg = EnergyDiagramDialog(self.mo_data, self)
        dlg.exec()

    def show_thermo_data(self):
        if not hasattr(self, 'thermo_data'): 
            QMessageBox.information(self, "Info", "No thermodynamic data available.")
            return

        data = self.thermo_data
        # Format Text
        lines = []
        lines.append("Thermodynamic Properties (standard conditions)")
        lines.append("-" * 40)
        
        # Typical units: E, H, G are in Hartree. S is usually cal/mol*K or similar in PySCF output structure?
        # PySCF thermo.thermo returns E,H,G in Hartree/Particle. S in Hartree/K ?
        # Actually PySCF docs say outputs are Hartree.
        
        order = ["E_tot", "H_tot", "G_tot", "ZPE", "S_tot", "Cv_tot"]
        labels = {
            "E_tot": "Total Energy (E0 + ZPE + corrections)",
            "H_tot": "Enthalpy (H)",
            "G_tot": "Gibbs Free Energy (G)",
            "S_tot": "Entropy (S)",
            "ZPE": "Zero Point Energy",
            "Cv_tot": "Heat Capacity (Cv)"
        }
        
        for k in order:
            if k in data:
                v = data[k]
                label = labels.get(k, k)
                unit = "Ha"
                # Handle possible tuple or non-float
                if isinstance(v, (tuple, list)):
                     v = v[0] # Assume flow is (value, unit) but we force unit above
                
                try:
                    vf = float(v)
                    lines.append(f"{label:<30}: {vf:.6f} {unit}")
                except:
                     lines.append(f"{label:<30}: {v}")
        
        # Add others
        for k, v in data.items():
            if k not in order:
                 lines.append(f"{k}: {v}")
                
        QMessageBox.information(self, "Thermodynamics", "\n".join(lines))

from PyQt6.QtGui import QPainter, QPen, QColor, QFont
from PyQt6.QtCore import QPointF, QRectF

class EnergyDiagramDialog(QDialog):
    def __init__(self, mo_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Orbital Energy Diagram (Scroll to Zoom, Dbl-Click to Reset)")
        self.resize(600, 800)
        
        # Add Save Button overlay
        from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 20, 20) # margins
        layout.addStretch()
        
        btn_layout = QHBoxLayout()
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
        
        self.data = mo_data
        self.is_uhf = (self.data["type"] == "UHF")
        
        # Extract energies
        if self.is_uhf:
            self.energies_a = self.data["energy"][0]
            self.energies_b = self.data["energy"][1]
            self.occ_a = self.data["occ"][0]
            self.occ_b = self.data["occ"][1]
            all_e = self.energies_a + self.energies_b
        else:
            self.energies_a = self.data["energy"]
            self.occ_a = self.data["occ"]
            all_e = self.energies_a
            
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
        # Zoom Logic
        delta = event.angleDelta().y()
        # Zoom In (positive delta) -> factor < 1
        factor = 0.8 if delta > 0 else 1.25 
        
        span = self.current_max - self.current_min
        center = (self.current_min + self.current_max) / 2
        new_span = span * factor
        
        self.current_min = center - new_span / 2
        self.current_max = center + new_span / 2
        self.update()
        
    def mouseDoubleClickEvent(self, event):
        # Reset to full view
        self.current_min = self.full_min - 0.05 * (self.full_max - self.full_min)
        self.current_max = self.full_max + 0.05 * (self.full_max - self.full_min)
        self.update()



    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.last_mouse_y = event.position().y()

    def mouseMoveEvent(self, event):
        if hasattr(self, 'dragging') and self.dragging:
            current_y = event.position().y()
            delta_y = current_y - self.last_mouse_y
            
            # Update View
            # Pixels to Energy
            h = self.height()
            margin_top = 40
            margin_bottom = 40
            draw_h = h - margin_top - margin_bottom
            
            range_e = self.current_max - self.current_min
            if abs(range_e) < 1e-9: range_e = 1.0 # Avoid division by zero
            scale = range_e / draw_h # Energy per pixel
            
            # Dragging DOWN (Positive Delta_y) means we want to see higher energy levels
            # So we shift the visible energy window UP (increase min/max)
            change = delta_y * scale
            
            self.current_min += change
            self.current_max += change
            
            self.last_mouse_y = current_y
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False

    def contextMenuEvent(self, event):
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction
        menu = QMenu(self)
        save_act = QAction("Save as PNG...", self)
        save_act.triggered.connect(self.save_image)
        menu.addAction(save_act)
        menu.exec(event.globalPos())

    def save_image(self):
        from PyQt6.QtWidgets import QFileDialog
        fname, _ = QFileDialog.getSaveFileName(self, "Save Diagram", "orbital_diagram.png", "Images (*.png)")
        if fname:
            pix = self.grab()
            pix.save(fname)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
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
            # Clip logic if needed? QPainter handles out of bounds.
            rel = (val - min_e) / range_e
            return (h - margin_bottom) - (rel * draw_h)

        # Draw Axis Line
        pen_axis = QPen(QColor("black"), 2)
        painter.setPen(pen_axis)
        painter.drawLine(50, margin_top, 50, h - margin_bottom)
        
        # Axis Labels
        painter.drawText(5, margin_top + 10, f"{max_e:.2f}")
        painter.drawText(5, h - margin_bottom, f"{min_e:.2f}")
        painter.drawText(5, h//2, "E (Ha)")

        # Draw Levels
        font = QFont("Arial", 9)
        painter.setFont(font)
        
        cols = 1 if not self.is_uhf else 2
        col_width = (w - 100) / cols
        level_w = col_width * 0.6
        
        def draw_levels(energies, occs, col_idx, title):
            center_x = 100 + col_idx * col_width + col_width/2
            
            painter.setPen(QColor("black"))
            painter.drawText(int(center_x - 20), 20, title)
            
            homo_idx = -1
            for i, o in enumerate(occs):
                if o > 0: homo_idx = i
            
            # Filter visible levels and sort by energy for label collision logic
            visible_items = []
            # Add a buffer to min_e/max_e for levels just outside view
            buffer = range_e * 0.1 
            for i, e, in enumerate(energies):
                if min_e - buffer <= e <= max_e + buffer:
                     visible_items.append((i, e, occs[i]))
            
            # Sort by Energy (High to Low) for top-down label placement
            visible_items.sort(key=lambda x: x[1], reverse=True)
            
            last_label_y = -100 # Initialize far off-screen
            
            for i_orig, e, occ_val in visible_items:
                y = val_to_y(e)
                
                is_occ = (occ_val > 0)
                color = QColor("blue") if is_occ else QColor("red")
                pen = QPen(color, 2)
                
                # Highlight HOMO/LUMO
                is_homo = (i_orig == homo_idx)
                is_lumo = (i_orig == homo_idx + 1)
                
                if is_homo or is_lumo:
                    pen.setWidth(3)
                
                painter.setPen(pen)
                
                x1 = center_x - level_w/2
                x2 = center_x + level_w/2
                painter.drawLine(int(x1), int(y), int(x2), int(y))
                
                # Label Logic with Relative Labels
                if i_orig <= homo_idx:
                    diff = homo_idx - i_orig
                    rel_label = "HOMO" if diff == 0 else f"HOMO-{diff}"
                else:
                    diff = i_orig - (homo_idx + 1)
                    rel_label = "LUMO" if diff == 0 else f"LUMO+{diff}"
                
                label_text = f"{rel_label} ({e:.3f})"
                
                priority_label = (is_homo or is_lumo)
                
                # Check Overlap for non-priority labels
                # 12px is a rough estimate for font size 9 height + some padding
                pixel_gap = abs(y - last_label_y)
                
                if priority_label or pixel_gap > 12: 
                    painter.setPen(QColor("black"))
                    draw_x = int(x2) + 5
                    
                    if priority_label:
                        font_b = QFont("Arial", 9, QFont.Weight.Bold)
                        painter.setFont(font_b)
                        painter.drawText(draw_x, int(y)+4, label_text)
                        painter.setFont(font) # Reset font
                    else:
                        painter.drawText(draw_x, int(y)+4, label_text)
                    
                    last_label_y = y # Update collision reference for next label
        
        if self.is_uhf:
            draw_levels(self.energies_a, self.occ_a, 0, "Alpha")
            draw_levels(self.energies_b, self.occ_b, 1, "Beta")
        else:
            draw_levels(self.energies_a, self.occ_a, 0, "Orbitals")
