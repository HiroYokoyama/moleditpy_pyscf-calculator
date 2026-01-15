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
    QDockWidget, QApplication, QMenu
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
        
        title = "PySCF Calculator"
        self.version = version
        if version:
            title += f" v{version}"
        self.setWindowTitle(title)
        
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
        self.settings["version"] = self.version # Save version to project file
        
        # Persist History & Source
        if hasattr(self, 'calc_history'):
            self.settings["calc_history"] = self.calc_history
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

        # Local JSON Save (User Request)
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
        local_settings = {
             "root_path": self.out_dir_edit.text(),
             "threads": self.spin_threads.value(),
             "memory": self.spin_memory.value()
        }
        try:
             with open(json_path, 'w') as f:
                 json.dump(local_settings, f, indent=4)
        except Exception as e:
             print(f"Failed to save local settings: {e}")

    def load_settings(self):
        # Local JSON Load (Defaults)
        local_settings = {}
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
        if os.path.exists(json_path):
             try:
                 with open(json_path, 'r') as f:
                     local_settings = json.load(f)
                 # Root Path
                 rp = local_settings.get("root_path")
                 if rp:
                     self.out_dir_edit.setText(rp)
                 # Threads/Memory
                 if "threads" in local_settings and hasattr(self, 'spin_threads'):
                     self.spin_threads.setValue(local_settings["threads"])
                 if "memory" in local_settings and hasattr(self, 'spin_memory'):
                     self.spin_memory.setValue(local_settings["memory"])
             except Exception as e:
                 print(f"Failed to load local settings: {e}")

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
        self.calc_history = s.get("calc_history", [])
        self.struct_source = s.get("struct_source", None)
        # Update Label if UI ready (setup_ui called before load_settings)
        if hasattr(self, 'lbl_struct_source') and self.struct_source:
             self.lbl_struct_source.setText(f"Structure Source: {self.struct_source}")

        if self.calc_history:
             last_path = self.calc_history[-1]
             
             # Smart Reset Logic
             # 1. Get current file basename
             current_filename = None
             try:
                 if self.context:
                     mw = self.context.get_main_window()
                     if hasattr(mw, 'current_file_path') and mw.current_file_path:
                         current_filename = os.path.basename(mw.current_file_path)
             except: pass
             
             # 2. Get saved associated filename
             saved_filename = s.get("associated_filename", None)
             
             # 3. Determine if we should reset
             # Reset ONLY if:
             #   - We have a current file (not Untitled) AND Saved file matches -> Different files
             #   - OR We have a saved file (Named) AND Current is Untitled -> Reset (User Request)
             should_reset = False
             
             if saved_filename:
                 if current_filename and current_filename != saved_filename:
                     # Case 1: Different named files (A.xyz -> B.xyz)
                     should_reset = True
                     self.log(f"File changed ({saved_filename} -> {current_filename}). Resetting settings.")
                 elif not current_filename:
                     # Case 2: Named file -> Untitled (A.xyz -> Untitled / New)
                     should_reset = True
                     self.log(f"File closed ({saved_filename} -> Untitled). Resetting settings.")
             
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
                 self.job_type_combo.setCurrentText("Optimization + Frequency")
                 self.method_combo.setCurrentIndex(0) # RKS
                 self.functional_combo.setCurrentIndex(0) # b3lyp
                 self.basis_combo.setCurrentIndex(0) # sto-3g
                 self.charge_input.setCurrentText("0")
                 self.spin_input.setCurrentText("0")
                 
                 # Use Local Settings if available, else defaults
                 self.spin_threads.setValue(local_settings.get("threads", 0))
                 self.spin_memory.setValue(local_settings.get("memory", 4000))
                 if "root_path" in local_settings:
                     self.out_dir_edit.setText(local_settings["root_path"])
                     
                 self.check_symmetry.setChecked(False)
                 self.spin_cycles.setValue(100)
                 self.edit_conv.setText("1e-9")
                 
                 # Clear shared settings to prevent persistence from previous session
                 keys_to_clear = ["job_type", "method", "functional", "basis", 
                                  "charge", "spin", "threads", "memory", "struct_source", "associated_filename"]
                 for k in keys_to_clear:
                     if k in self.settings: del self.settings[k]
                     
             elif os.path.exists(last_path) and os.path.isdir(last_path):
                 # Auto-load if we decide NOT to reset (Same file OR Untitled)
                 self.log(f"Auto-loading latest result from history: {last_path}")
                 # Pass update_structure=False to prevent overwriting current molecule
                 QTimer.singleShot(200, lambda: self.load_result_folder(last_path, update_structure=False))

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
        self.job_type_combo.addItems(["Energy", "Geometry Optimization", "Frequency", "Optimization + Frequency"])
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
        
        # Structure Source Label
        self.lbl_struct_source = QLabel("")
        self.lbl_struct_source.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.lbl_struct_source)
        
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
            self.iso_spin.blockSignals(True)
            try:
                self.iso_spin.setRange(0.0001, max(10.0, self.visualizer.data_max))
                
                fname = os.path.basename(path).lower()
                if "density" in fname:
                     self.iso_spin.setValue(0.04)
                else:
                     self.iso_spin.setValue(0.04)
            finally:
                self.iso_spin.blockSignals(False)
            
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

        if not self.loaded_file: return
        
        val = self.iso_spin.value()
        opacity = self.op_slider.value() / 100.0
        
        self.visualizer.update_iso(val, self.color_p, self.color_n, opacity)

    def closeEvent(self, event):
        self.save_settings() # Save Settings on Close
        
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
            msg = "Error: No molecule loaded. Please load a molecule in the main window."
            self.log(msg)
            QMessageBox.warning(self, "No Molecule", "Please load a molecule first.")
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
            "out_dir": os.path.abspath(self.out_dir_edit.text())
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

    def generate_specific_orbital(self, label):
        """
        Public method to generate a specific orbital by label (e.g. 'HOMO', 'LUMO+1').
        Called by EnergyDiagramDialog.
        """
        self.run_specific_analysis([label])

    def run_specific_analysis(self, tasks, out_d=None):
        if not hasattr(self, 'chkfile_path') or not self.chkfile_path:
             QMessageBox.warning(self, "Error", "No checkpoint file found.")
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
                item = QListWidgetItem(name)
                item.setToolTip(fpath)
                self.file_list.addItem(item)
                found_item = item
            
            last_added_item = found_item
            
            # Check if this NEW file is an ESP file
            if name.startswith("esp") and name.endswith(".cube"):
                new_esp_item = found_item
        
        # Re-sort the file list alphabetically
        all_items = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            all_items.append((item.text(), item.toolTip()))
        
        # Store paths for reference (not item objects which will be deleted)
        last_path = last_added_item.toolTip() if last_added_item else None
        esp_path = new_esp_item.toolTip() if new_esp_item else None
        
        all_items.sort(key=lambda x: x[0])  # Sort by filename
        
        self.file_list.clear()
        last_added_item = None
        new_esp_item = None
        
        for name, path in all_items:
            item = QListWidgetItem(name)
            item.setToolTip(path)
            self.file_list.addItem(item)
            # Update references
            if path == last_path:
                last_added_item = item
            if path == esp_path:
                new_esp_item = item
        
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
        
        # Extract Output Directory first to ensure it's available for labels
        out_dir = result_data.get("out_dir")
        if out_dir:
            self.last_out_dir = out_dir
            
        if result_data.get("optimized_xyz"):
             self.optimized_xyz = result_data["optimized_xyz"]
             self.btn_load_geom.setEnabled(True)
             self.log("Optimization converged. Automatically updating geometry...")
             self.update_geometry(self.optimized_xyz)
             
             # Update Source Label
             src_name = "Result"
             src_path = ""
             if self.last_out_dir:
                 src_name = os.path.basename(self.last_out_dir)
                 src_path = self.last_out_dir
             
             self.struct_source = f"optimized from {src_name} ({src_path})"
             if hasattr(self, 'lbl_struct_source'):
                  self.lbl_struct_source.setText(f"Structure Source: {self.struct_source}")
        
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
                
                # Add to History
                d = result_data["out_dir"]
                if not hasattr(self, 'calc_history'): self.calc_history = []
                if d not in self.calc_history:
                    self.calc_history.append(d)
                
                # Redundant label update removed to respect correct format set earlier
            
            self.save_settings()

        if result_data.get("cube_files"):
            # Inform user about generated cube files
            files = result_data["cube_files"]
            files.sort() # Sort A-Z
            self.log(f"Generated Cube Files: {len(files)}")
            
            # Add to list
            self.file_list.clear() 
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
                 mol = self.context.current_molecule
                 
                 self.clear_3d_actors()
                 self.freq_vis = FreqVisualizer(
                     self.context.get_main_window(), 
                     mol, 
                     self.freq_data['freqs'], 
                     self.freq_data['modes'],
                     intensities=self.freq_data.get('intensities')
                 )
                 
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
             
        # Auto-Switch to Visualization Tab
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
        self.save_settings()
        
        # Run LoadWorker
        if LoadWorker is None:
             QMessageBox.critical(self, "Error", "PySCF worker (LoadWorker) is not available.\nThis likely means PySCF is not installed or failed to import.")
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
         # Using 'loaded_xyz' from LoadWorker which always returns the chkfile geometry
         if result_data.get("loaded_xyz"):
             self.optimized_xyz = result_data["loaded_xyz"]
             self.btn_load_geom.setEnabled(True)
         elif result_data.get("optimized_xyz"):
             # Fallback for old compatibility or if manually added
             self.optimized_xyz = result_data["optimized_xyz"]
             self.btn_load_geom.setEnabled(True)
         
         # Populate Options List so we can disable existing ones
         self.populate_analysis_options()

         # 4. Scan for Visualization Files
         self.file_list.clear()
         d = self.last_out_dir
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
                 if hasattr(self, 'chkfile_path') and self.chkfile_path:
                     chk_abs = os.path.abspath(self.chkfile_path)
                     full_path = os.path.dirname(chk_abs)
                     basename = os.path.basename(full_path)
                     # Differentiate label: Manual Load is just "Loaded from"
                     # The User specifically requested "optimized from" for optimizations,
                     # but for generic loads "Loaded from" is safer and requested implicitly.
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
                 except: pass

                 self.finalize_load(result_data, cubes)

             self.log("Optimized geometry loaded automatically.")
             QTimer.singleShot(100, update_and_finalize)
             
         else:
             # No geometry update needed, proceed immediately
             self.finalize_load(result_data, cubes)

    def finalize_load(self, result_data, cubes=None):
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
                     
                     # Properly close and remove old dock if it exists
                     if hasattr(self, 'freq_dock') and self.freq_dock:
                         try:
                             mw.removeDockWidget(self.freq_dock)
                             self.freq_dock.close()
                             self.freq_dock.setParent(None)
                             self.freq_dock = None
                         except:
                             pass
                     
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
        if hasattr(self, 'energy_dlg') and self.energy_dlg and self.energy_dlg.isVisible():
            self.energy_dlg.raise_()
            self.energy_dlg.activateWindow()
            return

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

class EnergyDiagramDialog(QDialog):
    def __init__(self, mo_data, parent=None, result_dir=None):
        super().__init__(parent)
        self.result_dir = result_dir
        self.setWindowTitle("Orbital Energy Diagram")
        self.resize(300, 600)
        
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
                  for rect, index, label in self.hit_zones:
                      # Check X-bounds first (strict)
                      if point.x() >= rect.left() and point.x() <= rect.right():
                          # Check Y-vicinity (e.g. +/- 10 pixels to allow slack)
                          center_y = rect.center().y()
                          dist = abs(y_click - center_y)
                          
                          # Allow clicking slightly outside exact rect if it's the closest one
                          # But usually we want strict containment? 
                          # The user said "especially close orbitals".
                          # Better to use strictly contained OR within very small margin.
                          # Let's stick to "If inside rect, compare distance".
                          
                          if rect.contains(point):
                              if dist < min_dist:
                                  min_dist = dist
                                  best_hit = (index, label)
                                  
             if best_hit:
                 self.try_load_cube(best_hit[0], best_hit[1])
                 return
             
             self.dragging = True
             self.last_mouse_y = event.position().y()

    def try_load_cube(self, index, label):
        if not self.result_dir:
            # QMessageBox.information(self, "Info", "No result directory linked.")
            return
            
        import glob
        # Pattern matching
        # Files like: "15_HOMO.cube" or "16_LUMO.cube" or "15_MO_15.cube"
        
        # We need to construct likely filenames based on index (which is reliable)
        # Search pattern: "15_*.cube"
        
        # Try padded first (normalized sorting)
        pattern = os.path.join(self.result_dir, f"{index:03d}_*.cube")
        files = glob.glob(pattern)
        
        # Fallback to unpadded (legacy support)
        if not files:
             pattern_legacy = os.path.join(self.result_dir, f"{index}_*.cube")
             files = glob.glob(pattern_legacy)
        
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
             
             reply = QMessageBox.question(
                 self, 
                 "Generate Cube File?", 
                 f"Cube file for Orbital {index+1} ({label}) not found.\nDo you want to generate it now?",
                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
             )
             
             if reply == QMessageBox.StandardButton.Yes:
                 if hasattr(self.parent(), "generate_specific_orbital"):
                     self.status_label.setText(f"Generating MO {index+1}...")
                     # Pass "MO N" format
                     self.parent().generate_specific_orbital(mo_task_label)

    def mouseMoveEvent(self, event):
        # Check if hovering over a clickable orbital level (when not dragging)
        if not (hasattr(self, 'dragging') and self.dragging):
            pos = event.position()
            point = pos.toPoint()
            
            hovering_over_orbital = False
            if hasattr(self, 'hit_zones'):
                for rect, index, label in self.hit_zones:
                    if rect.contains(point):
                        hovering_over_orbital = True
                        break
            
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
        menu.exec(event.globalPosition().toPoint())

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
        unit_label_str = "E (Ha)"
        if unit == "eV":
            factor = nist.HARTREE2EV if nist else 27.211386245988
            unit_label_str = "E (eV)"
        
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

        # --- Draw Axis ---
        pen_axis = QPen(QColor("black"), 2)
        painter.setPen(pen_axis)
        painter.drawLine(60, margin_top, 60, h - margin_bottom) # Shifted right for labels
        
        # Draw Unit Label
        font_axis = QFont("Arial", 10)
        painter.setFont(font_axis)
        painter.drawText(5, margin_top - 10, unit_label_str)

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
        # Left margin for axis is around 60.
        left_margin = 70 
        right_margin = 10 # Only for window edge, but column width absorbs text space
        avail_w = w - left_margin - right_margin
        col_width = avail_w / cols 
        
        # User Request: "Orbital lefter" and labels inside window.
        # Strategy: Left align the line within the column, leaving the rest for text.
        
        level_w = 50 # Fixed compact width (pixels)
        padding_left = 10 # Space from column start
        
        def draw_levels(energies, occs, col_idx, title):
            # Calculate center_x such that the line starts at padding_left
            # x1 = center_x - level_w/2
            # We want x1 = col_start + padding_left
            # col_start = left_margin + col_idx * col_width
            
            col_start = left_margin + col_idx * col_width
            target_x1 = col_start + padding_left
            center_x = target_x1 + level_w/2
            
            painter.setPen(QColor("black"))
            
            # User Request: Center "Orbital" title in the graph (column)
            # col_start is left edge of column, col_width is width.
            fm = painter.fontMetrics()
            t_w = fm.horizontalAdvance(title)
            title_x = col_start + (col_width - t_w) / 2
            
            painter.drawText(int(title_x), 20, title)
            
            homo_idx = -1
            for i, o in enumerate(occs):
                if o > 0: homo_idx = i
            
            # Filter visible levels and split into Occupied and Virtual
            # 1. Occupied: Sort High Energy -> Low Energy (Process HOMO first)
            occupied_items = []
            # 2. Virtual: Sort Low Energy -> High Energy (Process LUMO first)
            virtual_items = []
            
            for i, e, in enumerate(energies):
                if min_e <= e <= max_e:
                     # Item tuple: (original_index, energy, occupancy)
                     item = (i, e, occs[i])
                     if occs[i] > 0:
                         occupied_items.append(item)
                     else:
                         virtual_items.append(item)
            
            # Sort Strategy to prioritize HOMO/LUMO labels
            occupied_items.sort(key=lambda x: x[1], reverse=True)  # Descending (HOMO -> Core)
            virtual_items.sort(key=lambda x: x[1], reverse=False)  # Ascending (LUMO -> Vacuum)
            
            # Draw lists
            # We must maintain collision detection state for each stack separately? 
            # Or globally? If overlap, we skip. But virtuals move UP, occupied move DOWN.
            # Let's reset collision Y for each stack to ensure "closest to gap" draws first.
            
            # Draw Occupied
            last_label_y_occ = 100000 # Bottom-most
            last_label_y_virt = -100 # Top-most
            
            # Helper to draw
            def process_list(items, last_y_ref):
                new_last_y = last_y_ref
                for i_orig, e, occ_val in items:
                    y = val_to_y(e)
                    
                    is_occ = (occ_val > 0)
                    color = QColor("blue") if is_occ else QColor("red")
                    pen = QPen(color, 2)
                    
                    is_homo = (i_orig == homo_idx)
                    is_lumo = (i_orig == homo_idx + 1)
                    
                    if is_homo or is_lumo:
                        pen.setWidth(3)
                    
                    painter.setPen(pen)
                    x1 = center_x - level_w/2
                    x2 = center_x + level_w/2
                    painter.drawLine(int(x1), int(y), int(x2), int(y))
                    
                    # Store Hit Zone
                    rect = QRect(int(x1), int(y)-4, int(x2-x1), 8)
                    self.hit_zones.append( (rect, i_orig, "") )
                
                    # Labels
                    if i_orig <= homo_idx:
                        diff = homo_idx - i_orig
                        rel_label = "HOMO" if diff == 0 else f"HOMO-{diff}"
                    else:
                        diff = i_orig - (homo_idx + 1)
                        rel_label = "LUMO" if diff == 0 else f"LUMO+{diff}"
                    
                    # Unit Label is from outer scope (self.unit_combo)
                    u_str = "eV" if unit == "eV" else "Ha"
                    label_text = f"{rel_label} ({e * factor:.3f} {u_str})"
                    
                    # Update label in zone
                    self.hit_zones[-1] = (rect, i_orig, rel_label)
                    
                    priority_label = (is_homo or is_lumo)
                    pixel_gap = abs(y - new_last_y)
                    
                    if priority_label or pixel_gap > 12:
                         painter.setPen(QColor("black"))
                         draw_x = int(x2) + 5
                         if priority_label:
                             font_b = QFont("Arial", 12, QFont.Weight.Bold)
                             painter.setFont(font_b)
                             painter.drawText(draw_x, int(y)+4, label_text)
                             painter.setFont(font)
                         else:
                             painter.drawText(draw_x, int(y)+4, label_text)
                         new_last_y = y
                return new_last_y

            # Process independently
            # Occupied (moving down, but val_to_y increases? No, y decreases as Energy increases)
            # Y = H - (rel * H). High Energy = Small Y (Top). Low Energy = Large Y (Bottom).
            # Occupied list: High E (Small Y) -> Low E (Large Y).
            # So occupied start at Small Y. last_label_y should start at -100? No.
            # Wait. Scan order: Top -> Bottom.
            # Virtual List: Low E (Large Y) -> High E (Small Y).
            # Virtuals start NEAR GAP (Large Y) and go UP (Small Y).
            # Occupied List: High E (Small Y NEAR GAP) and fo DOWN (Large Y).
            
            # Since pixel_gap looks at abs diff, simply tracking the last drawn one works.
            # Virtuals: First is LUMO (Y=500). Draws. Last=500.
            # Next is LUMO+1 (Y=490). Gap=10. Skips.
            # Next is LUMO+2 (Y=480). Gap=20 from 500. Draws. Last=480.
            # Works perfectly.
            
            # Occupied: First is HOMO (Y=510). Draws. Last=510.
            # Next is HOMO-1 (Y=520). Gap=10. Skips.
            # Works perfectly.
            
            process_list(occupied_items, -1000) # Init far away
            process_list(virtual_items, 10000) # Init far away

        
        if self.is_uhf:
            draw_levels(self.energies_a, self.occ_a, 0, "Alpha")
            draw_levels(self.energies_b, self.occ_b, 1, "Beta")
        else:
            draw_levels(self.energies_a, self.occ_a, 0, "Orbitals")
