import os
import glob
import traceback
import csv
import json
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QListWidget, QListWidgetItem, QGroupBox, QLineEdit, 
    QMessageBox, QDoubleSpinBox, QSlider, QComboBox, 
    QTableWidget, QTableWidgetItem, QHeaderView, QDockWidget, QFileDialog, QDialog
)
from PyQt6.QtCore import Qt, QTimer


# Local Imports
try:
    from .worker import LoadWorker, PropertyWorker
    from .vis import CubeVisualizer, MappedVisualizer
    from .utils import update_molecule_from_xyz
    from .scan_results import ScanResultDialog
    from .energy_diag import EnergyDiagramDialog
except ImportError:
    LoadWorker = None
    PropertyWorker = None
    CubeVisualizer = None
    MappedVisualizer = None
    ScanResultDialog = None
    EnergyDiagramDialog = None
    update_molecule_from_xyz = None

try:
    from .freq_vis import FreqVisualizer
except ImportError:
    FreqVisualizer = None

class VisTab(QWidget):
    def __init__(self, parent_dialog, context):
        super().__init__(parent_dialog)
        self.parent_dialog = parent_dialog
        self.context = context
        
        # Data
        self.chkfile_path = None
        self.mo_data = None
        self.freq_data = None
        self.thermo_data = None
        self.last_out_dir = None
        self.optimized_xyz = None
        
        # Visualizers
        self.visualizer = CubeVisualizer(self.context.get_main_window())
        self.mapped_visualizer = None
        self.freq_vis = None
        self.freq_dock = None
        self.mode = "standard"
        self.loaded_file = None
        self.color_p = "blue"
        self.color_n = "red"
        
        # Workers
        self.prop_worker = None
        self.load_worker = None
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
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
        
        self.populate_analysis_options()
        
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
        self.iso_spin.setDecimals(4) 
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
        
        # Opacity
        op_layout = QHBoxLayout()
        op_layout.addWidget(QLabel("Opacity:"))
        self.op_slider = QSlider(Qt.Orientation.Horizontal)
        self.op_slider.setRange(0, 100)
        self.op_slider.setValue(40)
        self.op_slider.valueChanged.connect(self.update_visualization)
        op_layout.addWidget(self.op_slider)
        v_layout.addLayout(op_layout)
        
        # --- Mapped Controls ---
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
        self.m_iso_spin.setValue(0.004) 
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

    def log(self, msg):
        self.parent_dialog.log(msg)

    def load_result_folder(self, path=None, update_structure=True, is_opt_job=False):
        self.loading_update_struct = update_structure
        self._pending_is_opt = is_opt_job
        d = path
        if not d or isinstance(d, bool):
            d = QFileDialog.getExistingDirectory(self, "Select Result Directory")
        if not d: return
        
        d = os.path.abspath(d)
        
        # Check if this is a scan result folder (no checkpoint file needed)
        scan_csv = os.path.join(d, "scan_results.csv")
        scan_traj = os.path.join(d, "scan_trajectory.xyz")
        is_scan_result = os.path.exists(scan_csv) and os.path.exists(scan_traj)
        
        if is_scan_result:
            # Load scan results directly without checkpoint
            self.log(f"\nDetected scan results folder: {d}")
            # Do NOT update result_path_display for scans to keep primary result context
            # self.result_path_display.setText(d) 
            
            # Update History
            if not hasattr(self.parent_dialog, 'calc_history'): self.parent_dialog.calc_history = []
            if d not in self.parent_dialog.calc_history:
                self.parent_dialog.calc_history.append(d)
                self._history_changed = True
            self.parent_dialog.update_internal_state()
            
            # Load scan results
            try:
                self.load_scan_results(d)
            except Exception as e:
                self.log(f"Error loading scan results: {e}")
                QMessageBox.warning(self, "Error", f"Failed to load scan results: {e}")
            return
        
        # Regular checkpoint-based loading
        chk_path = os.path.join(d, "pyscf.chk")
        if not os.path.exists(chk_path):
             chk_path_alt = os.path.join(d, "checkpoint.chk")
             if os.path.exists(chk_path_alt):
                 chk_path = chk_path_alt
             else:
                 QMessageBox.warning(self, "Error", f"No checkpoint file (pyscf.chk) or scan results found in {d}")
                 return
        
        self.result_path_display.setText(d)

        # Update History
        if not hasattr(self.parent_dialog, 'calc_history'): self.parent_dialog.calc_history = []
        if d not in self.parent_dialog.calc_history:
            self.parent_dialog.calc_history.append(d)
            self._history_changed = True
            
        self.parent_dialog.update_internal_state()
            
        if LoadWorker is None:
             QMessageBox.critical(self, "Error", "LoadWorker not available.")
             return

        if self.load_worker and self.load_worker.isRunning():
            QMessageBox.warning(self, "Busy", "A result is already loading.")
            return

        self.load_worker = LoadWorker(chk_path)
        self.load_worker.finished_signal.connect(self.on_load_finished)
        self.load_worker.error_signal.connect(self.parent_dialog.on_error)
        
        self.log(f"\nLoading result from: {d}...")
        self.parent_dialog.progress_bar.show()
        self.load_worker.start()

    def load_scan_results(self, result_dir):
        """Load scan results from a folder without checkpoint file."""
        self.log("Loading scan results...")
        
        # Read CSV data
        csv_path = os.path.join(result_dir, "scan_results.csv")
        traj_path = os.path.join(result_dir, "scan_trajectory.xyz")
        
        scan_results = []
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    scan_results.append({
                        "step": int(row["Step"]),
                        "value": float(row["Value"]),
                        "energy": float(row["Energy"])
                    })
        except Exception as e:
            raise Exception(f"Failed to read scan CSV: {e}")
        
        # Parse trajectory
        trajectory = []
        try:
            with open(traj_path, 'r') as f:
                content = f.read()
                lines = content.splitlines()
                idx = 0
                while idx < len(lines):
                    if not lines[idx].strip():
                        idx += 1
                        continue
                    try:
                        natoms = int(lines[idx].strip())
                        block = "\n".join(lines[idx:idx+natoms+2])
                        trajectory.append(block)
                        idx += natoms + 2
                    except:
                        break
        except Exception as e:
            raise Exception(f"Failed to read trajectory: {e}")
        
        # Open scan results dialog
        try:
            if ScanResultDialog:
                dlg = ScanResultDialog(
                    scan_result_dir=result_dir,
                    parent=self.context.get_main_window(),
                    context=self.context,  # Pass context for 3D viewer updates
                    results=scan_results,
                    trajectory=trajectory
                )
                dlg.show()
                self.scan_dlg = dlg  # Keep reference
                self.log(f"Scan results loaded: {len(scan_results)} steps")
            else:
                raise Exception("ScanResultDialog not available")
        except Exception as e:
            raise Exception(f"Failed to open scan dialog: {e}")
        
        # Mark history as changed for saving
        if getattr(self, '_history_changed', False):
            if self.context:
                mw = self.context.get_main_window()
                if mw:
                    mw.has_unsaved_changes = True
                    mw.update_window_title()
            self._history_changed = False

    def on_load_finished(self, result_data):
         self.log("Result loaded successfully.")
         self.parent_dialog.progress_bar.hide()
         
         if getattr(self, '_history_changed', False):
             if self.context:
                 mw = self.context.get_main_window()
                 if mw:
                     mw.has_unsaved_changes = True
                     mw.update_window_title()
             self._history_changed = False
         
         try:
             # Cleanup
             if self.freq_vis:
                 try: self.freq_vis.cleanup()
                 except: pass
                 self.freq_vis = None
             
             if self.freq_dock:
                 try:
                     mw = self.context.get_main_window()
                     mw.removeDockWidget(self.freq_dock)
                     self.freq_dock.deleteLater()
                     self.freq_dock = None
                 except: pass
             
             self.clear_3d_actors()
             self.visualizer = None
             self.mapped_visualizer = None
             
         except Exception as cleanup_err:
             self.log(f"Warning during initial cleanup: {cleanup_err}")
         
         if result_data.get("chkfile"):
             self.chkfile_path = result_data["chkfile"]
             self.last_out_dir = os.path.dirname(self.chkfile_path)
             
         if result_data.get("mo_energy"):
             self.mo_data = {
                 "energies": result_data["mo_energy"],
                 "occupations": result_data["mo_occ"],
                 "type": result_data.get("scf_type", "RHF")
             }
             self.btn_show_diagram.setEnabled(True)
             self.btn_run_analysis.setEnabled(True)
             
         if result_data.get("loaded_xyz"):
             self.optimized_xyz = result_data["loaded_xyz"]
             self.parent_dialog.optimized_xyz = result_data["loaded_xyz"]  # Expose to parent
             self.btn_load_geom.setEnabled(True)
         elif result_data.get("optimized_xyz"):
             self.optimized_xyz = result_data["optimized_xyz"]
             self.parent_dialog.optimized_xyz = result_data["optimized_xyz"]  # Expose to parent
             self.btn_load_geom.setEnabled(True)
         
         try:
             self.populate_analysis_options()
         except Exception as e:
             self.log(f"ERROR populating analysis options: {e}")

         self.file_list.clear()
         d = self.last_out_dir
         cubes = []
         if d and os.path.exists(d):
             cubes = glob.glob(os.path.join(d, "*.cube"))
             cubes.sort()
             for c in cubes:
                 name = os.path.basename(c)
                 item = QListWidgetItem(name)
                 item.setToolTip(c)
                 self.file_list.addItem(item)
             self.log(f"Found {len(cubes)} existing visualization files.")
             self.disable_existing_analysis_items(cubes)

         if result_data.get("thermo_data"):
             self.thermo_data = result_data["thermo_data"]
             self.btn_show_thermo.setEnabled(True)
         
         self.parent_dialog.save_settings()

         should_update_geom = getattr(self, 'loading_update_struct', True)
         if should_update_geom and self.optimized_xyz:
             def update_and_finalize():
                # Strict Check: ONLY update source if it is an optimization result
                is_opt = result_data.get("optimized_xyz") or getattr(self, '_pending_is_opt', False)
                
                # Clear pending flag
                self._pending_is_opt = False
                
                if is_opt:
                    if self.chkfile_path:
                        chk_abs = os.path.abspath(self.chkfile_path)
                        full_path = os.path.dirname(chk_abs)
                        basename = os.path.basename(full_path)
                        src_str = f"optimized from {basename} ({full_path})"
                    else:
                        src_str = "optimized from Result (Unknown)"

                    self.parent_dialog.struct_source = src_str
                    self.lbl_struct_source.setText(f"Structure Source: {src_str}")
                    self.parent_dialog.save_settings()
                    
                    # Ensure project is marked as modified so settings are saved
                    try:
                        mw = self.context.get_main_window()
                        if mw:
                            mw.has_unsaved_changes = True
                            mw.update_window_title()
                    except: pass
                
                self.update_geometry(self.optimized_xyz)
                
                try:
                    mw = self.context.get_main_window()
                    if hasattr(mw, 'plotter'): mw.plotter.reset_camera()
                    is_manual_load = getattr(self, 'loading_update_struct', True)
                    if is_manual_load and hasattr(mw, 'minimize_2d_panel'):
                        mw.minimize_2d_panel()
                except: pass

                try:
                    self.finalize_load(result_data, cubes)
                except Exception as e:
                    self.log(f"Warning during finalize_load: {e}")
                    traceback.print_exc()

             self.log("Optimized geometry loaded automatically.")
             QTimer.singleShot(100, update_and_finalize)
             
         else:
             try:
                 self.finalize_load(result_data, cubes)
             except Exception as e:
                 self.log(f"Warning during finalize_load: {e}")

         QTimer.singleShot(150, lambda: self.parent_dialog.tabs.setCurrentIndex(1))

    def finalize_load(self, result_data, cubes=None):
         if self.freq_vis:
             try: 
                 self.freq_vis.cleanup()
                 self.freq_vis = None
             except: pass

         if self.freq_dock:
             try:
                 self.freq_dock.close()
                 self.freq_dock.deleteLater()
             except: pass
             self.freq_dock = None

         if hasattr(self, 'scan_dlg') and self.scan_dlg:
             try: self.scan_dlg.close()
             except: pass
         self.scan_dlg = None
         
         if hasattr(self, 'tddft_dlg') and self.tddft_dlg:
             try: self.tddft_dlg.close()
             except: pass
         self.tddft_dlg = None

         if result_data.get("freq_data"):
             self.log("Frequency data found in result.")
             self.freq_data = result_data["freq_data"]
             try:
                 mol = self.context.current_molecule
                 if mol:
                     self.log("Creating frequency visualizer...")
                     self.clear_3d_actors()
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
             except Exception as e:
                 self.log(f"Error opening Frequency Visualizer: {e}")
         elif result_data.get("freq_data"):
             if FreqVisualizer is None:
                 self.log("Frequency Visualizer module invalid or not imported.")
             else:
                 self.log("Frequency Visualizer skipped: No molecule loaded in context.")
         
         if result_data.get("tddft_data"):
             try:
                 from .tddft_table import TddftTable
                 self.tddft_dlg = TddftTable(self.context.get_main_window(), result_data["tddft_data"])
                 self.tddft_dlg.show()
             except Exception as e:
                 self.log(f"Error opening TDDFT Results: {e}")

         if result_data.get("scan_results"):
             self.log("Scan results found.")
             scan_res = result_data["scan_results"]
             traj_path = result_data.get("scan_trajectory_path")
             
             # Parse trajectory if path exists
             trajectory = []
             if traj_path and os.path.exists(traj_path):
                 try: 
                     with open(traj_path, 'r') as f:
                         # Simple parser or just read blocks
                         # For now, pass raw path or read it? 
                         # ScanResultDialog expects list of strings?
                         # Earlier I saw `trajectory` attribute. Let's read it into blocks if possible
                         # Basic XYZ parser:
                         content = f.read()
                         lines = content.splitlines()
                         idx = 0
                         while idx < len(lines):
                             if not lines[idx].strip(): 
                                 idx+=1
                                 continue
                             try:
                                 natoms = int(lines[idx].strip())
                                 block = "\n".join(lines[idx:idx+natoms+2])
                                 trajectory.append(block)
                                 idx += natoms + 2
                             except: break
                 except: pass

             try:
                 from .scan_results import ScanResultDialog
                 
                 # Use trajectories from result_data if available (likely fresher)
                 final_traj = trajectory
                 if result_data.get("scan_trajectory"):
                     final_traj = result_data["scan_trajectory"]
                 
                 dlg = ScanResultDialog(
                     scan_result_dir=result_data.get('out_dir'), 
                     parent=self.context.get_main_window(), 
                     results=scan_res, 
                     trajectory=final_traj,
                     context=self.context
                 )
                 dlg.show()
                 self.scan_dlg = dlg # Keep reference
                     
             except Exception as e:
                 self.log(f"Error opening Scan Results: {e}")
        
         if cubes:
             self.disable_existing_analysis_items(cubes)

    def populate_analysis_options(self):
        self.orb_list.clear()
        
        # Add Standard Options
        item_esp = QListWidgetItem("ESP (Electrostatic Potential + Density)")
        item_esp.setFlags(item_esp.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item_esp.setCheckState(Qt.CheckState.Unchecked)
        item_esp.setData(Qt.ItemDataRole.UserRole, "ESP")
        self.orb_list.addItem(item_esp)
        
        is_uhf = False
        is_roks = False
        occ_a = []
        occ_b = []
        
        scf_type = self.mo_data.get("type", "RHF") if self.mo_data else "RHF"
        
        if scf_type in ["UHF", "UKS"]:
            is_uhf = True
            item_sd = QListWidgetItem("Spin Density")
            item_sd.setFlags(item_sd.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item_sd.setCheckState(Qt.CheckState.Unchecked)
            item_sd.setData(Qt.ItemDataRole.UserRole, "SpinDensity")
            self.orb_list.addItem(item_sd)

            try:
                energies = self.mo_data.get("energies", [])
                occupations = self.mo_data.get("occupations", [])
                
                def safe_occ(occ_list):
                    try:
                        if occ_list is None: return []
                        if not isinstance(occ_list, (list, tuple)): return [occ_list]
                        if not occ_list: return []
                        first = occ_list[0]
                        if isinstance(first, (list, tuple)):
                             res = []
                             for x in occ_list:
                                 if isinstance(x, (list, tuple)) and len(x) > 0: res.append(x[0])
                                 else: res.append(0)
                             return res
                        return occ_list
                    except: return []

                if len(energies) >= 2 and isinstance(energies[0], list):
                    occ_a = safe_occ(occupations[0])
                    occ_b = safe_occ(occupations[1])
                elif len(energies) == 2 and isinstance(energies, list):
                     if len(occupations) >= 2:
                         occ_a = safe_occ(occupations[0])
                         occ_b = safe_occ(occupations[1])
                     else:
                         occ_a = safe_occ(occupations)
                else:
                    occ_a = safe_occ(occupations)
            except: pass
        
        elif scf_type in ["ROKS", "ROHF"]:
            is_roks = True
            try:
                occupations = self.mo_data.get("occupations", [])
                if isinstance(occupations, list) and len(occupations) >= 2:
                    if isinstance(occupations[0], list): occ_a = occupations[0]
                    elif hasattr(occupations[0], 'tolist'): occ_a = occupations[0].tolist()
                    else: occ_a = list(occupations[0]) if occupations[0] else []
                elif isinstance(occupations, list): occ_a = occupations
                elif hasattr(occupations, 'tolist'): occ_a = occupations.tolist()
            except: occ_a = []
        
        else:
            occs = self.mo_data.get("occupations", []) if self.mo_data else []
            if occs and isinstance(occs, list) and len(occs) > 0:
                 if isinstance(occs[0], (list, tuple)):
                      if len(occs) == 1: occ_a = list(occs[0])
                      else:
                           occ_a = []
                           for x in occs:
                               if isinstance(x, (list, tuple)) and len(x) > 0: occ_a.append(x[0])
                               else: occ_a.append(x)
                 else: occ_a = occs
            else: occ_a = []

        def add_orb_items(suffix="", label_suffix="", range_lumo=range(0, 5), range_homo=range(0, 5), check_somo=False):
            # LUMOs
            for i in reversed(range_lumo):
                label = "LUMO" if i == 0 else f"LUMO+{i}"
                full_label = f"{label}{label_suffix}"
                task_str = f"{label}{suffix}"
                
                item = QListWidgetItem(full_label)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setData(Qt.ItemDataRole.UserRole, task_str)
                self.orb_list.addItem(item)

            # HOMOs
            if "Beta" in label_suffix: current_occ = occ_b
            else: current_occ = occ_a
            
            occ_threshold = 0.1
            my_homo_idx = -1
            if current_occ:
                for idx, o in enumerate(current_occ):
                    if o > occ_threshold: my_homo_idx = idx
            
            for i in range_homo:
                target_idx = my_homo_idx - i
                if target_idx < 0: continue
                if current_occ and target_idx >= len(current_occ): continue

                label = "HOMO"
                is_roks_somo = False
                if not is_uhf and current_occ:
                    val = current_occ[target_idx]
                    if abs(val - 1.0) < 0.1: is_roks_somo = True
                
                if is_roks_somo: label = "SOMO"
                elif i == 0: label = "HOMO"
                else: label = f"HOMO-{i}"
                
                full_label = f"{label}{label_suffix}"
                worker_label = "HOMO" if i == 0 else f"HOMO-{i}"
                task_str = f"MO {target_idx+1}_{worker_label}{suffix}"
                
                item = QListWidgetItem(full_label)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setData(Qt.ItemDataRole.UserRole, task_str)
                self.orb_list.addItem(item)

        if is_uhf:
            add_orb_items(suffix="_A", label_suffix=" (Alpha)", check_somo=True)
            add_orb_items(suffix="_B", label_suffix=" (Beta)", check_somo=False)
        elif is_roks:
            add_orb_items(check_somo=True)
        else:
            add_orb_items()

    def disable_existing_analysis_items(self, cube_files):
        if not cube_files: return
        basenames = [os.path.basename(f).lower() for f in cube_files]
        for i in range(self.orb_list.count()):
            item = self.orb_list.item(i)
            task_data = item.data(Qt.ItemDataRole.UserRole)
            should_disable = False
            
            if task_data == "ESP":
                if "esp.cube" in basenames and "density.cube" in basenames: should_disable = True
            elif task_data and task_data != "ESP":
                search_label = task_data.upper().replace(" ", "")
                if search_label.startswith("MO"):
                    try:
                        mo_idx = search_label.replace("MO", "").strip()
                        padded_idx = f"{int(mo_idx):03d}"
                        for bn in basenames:
                            if (bn.startswith(f"{mo_idx}_") or bn.startswith(f"{padded_idx}_")) and bn.endswith(".cube"):
                                should_disable = True
                                break
                    except: pass
                else:
                    for bn in basenames:
                        if ".cube" in bn:
                            parts = bn.replace(".cube", "").split("_")
                            if len(parts) >= 2:
                                file_label = "_".join(parts[1:]).upper()
                                if file_label == search_label:
                                    should_disable = True
                                    break
            
            if should_disable:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setCheckState(Qt.CheckState.Unchecked)

    def run_selected_analysis(self):
        if not self.chkfile_path:
             QMessageBox.warning(self, "Error", "No checkpoint file found.")
             return
        tasks = []
        for i in range(self.orb_list.count()):
            item = self.orb_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                tasks.append(item.data(Qt.ItemDataRole.UserRole))
        if not tasks:
            QMessageBox.information(self, "Info", "No analysis selected.")
            return
        self.run_specific_analysis(tasks)
        


    def run_specific_analysis(self, tasks, out_d=None):
        if not self.chkfile_path: return
        if not os.path.exists(self.chkfile_path):
             QMessageBox.warning(self, "Error", f"Checkpoint file missing at: {self.chkfile_path}")
             return
        
        self.log(f"Starting Analysis for: {', '.join(tasks)}...")
        if not out_d:
            out_d = self.last_out_dir
            if not out_d and self.chkfile_path: out_d = os.path.dirname(self.chkfile_path)
            if not out_d: out_d = self.parent_dialog.out_dir_edit.text()

        self.prop_worker = PropertyWorker(self.chkfile_path, tasks, out_d)
        self.prop_worker.log_signal.connect(self.log) # Direct log
        self.prop_worker.finished_signal.connect(self.on_prop_finished)
        self.prop_worker.error_signal.connect(self.parent_dialog.on_error)
        self.prop_worker.result_signal.connect(self.on_prop_results)
        
        self.btn_run_analysis.setEnabled(False)
        self.parent_dialog.progress_bar.show()
        self.prop_worker.start()

    def load_file_by_path(self, path):
        path = os.path.normpath(path)
        # Try to find in file list
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            # Use tooltip for full path
            if os.path.normpath(item.toolTip()) == path:
                self.file_list.setCurrentItem(item)
                self.on_file_selected(item)
                return
        
        # Fallback: Load directly if exists
        if os.path.exists(path):
             self.visualizer.load_file(path)

    def generate_specific_orbital(self, index, label=None, spin_suffix=""):
        # index is 0-based index of the orbital
        # task string format: #<index><suffix>
        task = f"#{index}{spin_suffix}"
        self.run_specific_analysis([task])

    def on_prop_finished(self):
        self.log("\nAnalysis Finished.")
        self.btn_run_analysis.setEnabled(True)
        self.parent_dialog.progress_bar.hide()
        self.prop_worker = None
        for i in range(self.orb_list.count()):
            item = self.orb_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)

    def on_prop_results(self, result_data):
        new_files = result_data.get("files", [])
        if new_files:
            self.log(f"Generated {len(new_files)} new files.")
        
        self.file_list.clear()
        d = self.last_out_dir
        if not d and self.chkfile_path: d = os.path.dirname(self.chkfile_path)
        
        if d and os.path.exists(d):
            cubes = glob.glob(os.path.join(d, "*.cube"))
            cubes.sort()
            
            last_item = None
            target_item = None
            target_path = None
            if new_files:
                for nf in new_files:
                    if "esp" in os.path.basename(nf).lower():
                        target_path = nf
                        break
                if not target_path: target_path = new_files[-1]

            for c in cubes:
                name = os.path.basename(c)
                item = QListWidgetItem(name)
                item.setToolTip(c)
                self.file_list.addItem(item)
                
                if target_path and os.path.normpath(c) == os.path.normpath(target_path):
                    target_item = item
                last_item = item
            
            if target_item:
                 self.file_list.setCurrentItem(target_item)
                 self.file_list.scrollToItem(target_item)
                 self.on_file_selected(target_item)
            elif last_item and new_files:
                 self.file_list.setCurrentItem(last_item)
                 self.file_list.scrollToItem(last_item)
                 self.on_file_selected(last_item)

    def on_file_selected(self, item, previous=None):
        if not item: return
        self.clear_3d_actors()
        path = item.toolTip() 
        if not path or not os.path.exists(path): return

        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        
        is_esp_pair = False
        if basename.lower() == "esp.cube":
             density_path = os.path.join(dirname, "density.cube")
             if os.path.exists(density_path):
                is_esp_pair = True
                surf_file = density_path
                prop_file = path
        
        try:
            if is_esp_pair: self.switch_to_mapped_mode(surf_file, prop_file)
            else: self.switch_to_standard_mode(path)
        except Exception as e:
            self.log(f"Error loading visualization file: {e}")
            self.switch_to_standard_mode(path)

    def switch_to_standard_mode(self, path):
        self.mode = "standard"
        self.vis_controls.show()
        self.vis_controls.setEnabled(True)
        self.mapped_group.hide()
        
        if self.mapped_visualizer: 
             try: self.mapped_visualizer.clear_actors()
             except: pass
        
        self.loaded_file = path
        if not self.visualizer: self.visualizer = CubeVisualizer(self.context.get_main_window())

        if self.visualizer.load_file(path):
            self.iso_spin.blockSignals(True)
            try:
                new_max = max(10.0, self.visualizer.data_max)
                if new_max <= 0: new_max = 1.0
                self.iso_spin.setRange(0.0001, new_max)
                self.iso_spin.setValue(0.04)
            finally:
                self.iso_spin.blockSignals(False)
            self.update_visualization()

    def switch_to_mapped_mode(self, surf_file, prop_file):
        self.mode = "mapped"
        self.vis_controls.hide()
        self.mapped_group.show()
        
        if self.visualizer: 
            try: self.visualizer.clear_actors()
            except: pass
        
        if not self.mapped_visualizer:
            self.mapped_visualizer = MappedVisualizer(self.context.get_main_window())
            
        if self.mapped_visualizer.load_files(surf_file, prop_file):
             self.m_iso_spin.blockSignals(True)
             self.m_iso_spin.setValue(0.004)
             self.m_iso_spin.blockSignals(False)
             
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

    def update_visualization(self):
        if self.mode == "mapped":
            self.update_mapped_vis()
            return
        if not self.visualizer or not self.loaded_file: return
        val = self.iso_spin.value()
        opacity = self.op_slider.value() / 100.0
        self.visualizer.update_iso(val, self.color_p, self.color_n, opacity)

    def update_mapped_vis(self):
        if not self.mapped_visualizer: return
        iso = self.m_iso_spin.value()
        val_min = self.m_min_spin.value()
        val_max = self.m_max_spin.value()
        cmap = self.cmap_combo.currentText()
        opacity = self.m_op_slider.value() / 100.0
        self.mapped_visualizer.update_mesh(iso, opacity, cmap=cmap, clim=[val_min, val_max])

    def fit_mapped_range(self):
        if not self.mapped_visualizer: return
        iso = self.m_iso_spin.value()
        p_min, p_max = self.mapped_visualizer.get_mapped_range(iso)
        if p_max - p_min < 1e-9:
            p_max += 0.05
            p_min -= 0.05
        cur_min = self.m_min_spin.minimum()
        cur_max = self.m_max_spin.maximum()
        if p_min < cur_min: self.m_min_spin.setMinimum(p_min - 1.0)
        if p_max > cur_max: self.m_max_spin.setMaximum(p_max + 1.0)
        self.m_min_spin.setValue(p_min)
        self.m_max_spin.setValue(p_max)
        self.update_mapped_vis()

    def choose_color(self, mode):
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

    def clear_3d_actors(self):
        try:
             mw = self.context.get_main_window()
             if not mw or not hasattr(mw, 'plotter') or mw.plotter is None: return
             
             try: mw.plotter.remove_actor("pyscf_iso_p")
             except: pass
             try: mw.plotter.remove_actor("pyscf_iso_n")
             except: pass
             try: mw.plotter.remove_actor("pyscf_mapped")
             except: pass
             
             if self.visualizer:
                 try: self.visualizer.clear_actors()
                 except: pass
             if self.mapped_visualizer:
                 try: self.mapped_visualizer.clear_actors()
                 except: pass
             
             if not self.parent_dialog.closing:
                 try: mw.plotter.render()
                 except: pass
        except: pass
        
        if self.freq_vis:
            try: self.freq_vis.cleanup()
            except: pass

    def load_optimized_geometry(self):
        if self.optimized_xyz:
            self.update_geometry(self.optimized_xyz)
            QMessageBox.information(self, "Success", "Geometry updated with optimized structure.")

    def update_geometry(self, xyz):
        self.clear_3d_actors()
        if self.freq_vis:
            try: self.freq_vis.cleanup()
            except: pass
            self.freq_vis = None
        update_molecule_from_xyz(self.context, xyz)
        try:
            mw = self.context.get_main_window()
            if hasattr(mw, 'push_undo_state'): mw.push_undo_state()
        except: pass
        self.log("Geometry updated.")

    def add_custom_mo(self):
        text = self.mo_input.text().strip()
        if not text: return
        import re
        is_rel = re.match(r"^(HOMO|LUMO)([-+]\d+)?$", text, re.IGNORECASE)
        is_digit = text.isdigit()
        
        if not (is_digit or is_rel):
             QMessageBox.warning(self, "Invalid Input", "Please enter a valid integer index or relative label.")
             return
             
        display_label = text
        task_data = text
        
        match = re.match(r"^(\d+)([aAbB])$", text)
        if match:
             idx = int(match.group(1))
             suffix = match.group(2).lower()
             spin = "_A" if suffix == "a" else "_B"
             task_data = f"MO {idx}{spin}"
             display_label = f"MO {idx} ({'Alpha' if spin=='_A' else 'Beta'})"
        elif is_digit:
             idx = int(text)
             task_data = f"MO {idx}"
             display_label = f"MO {idx}"
             # Try to resolve relative label
             if self.mo_data:
                 try:
                     occ = self.mo_data["occupations"]
                     if isinstance(occ, list) and len(occ) > 0:
                         target = occ[0] if isinstance(occ[0], (list, tuple)) else occ
                         homo_i = -1
                         for i, o in enumerate(target):
                             if o > 0.5: homo_i = i
                             else: break
                         
                         comp_idx = idx - 1
                         if comp_idx <= homo_i:
                             diff = homo_i - comp_idx
                             lb = "HOMO" if diff == 0 else f"HOMO-{diff}"
                             display_label = f"{lb} (Index {idx})"
                         else:
                             lumo_i = homo_i + 1
                             diff = comp_idx - lumo_i
                             lb = "LUMO" if diff == 0 else f"LUMO+{diff}"
                             display_label = f"{lb} (Index {idx})"
                 except: pass
        else:
             task_data = text.upper().replace(" ", "")
             display_label = task_data
             
        for i in range(self.orb_list.count()):
             if self.orb_list.item(i).data(Qt.ItemDataRole.UserRole) == task_data: return

        item = QListWidgetItem(display_label)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        item.setData(Qt.ItemDataRole.UserRole, task_data)
        self.orb_list.addItem(item)
        self.mo_input.clear()

    def show_energy_diagram(self):
        if not self.mo_data: return
        if hasattr(self, 'energy_dlg') and self.energy_dlg:
             self.energy_dlg.close()
             self.energy_dlg = None
        self.energy_dlg = EnergyDiagramDialog(self.mo_data, parent=self, result_dir=self.last_out_dir)
        self.energy_dlg.show()
    
    def load_file_by_path(self, path):
         if not os.path.exists(path): return
         norm_path = os.path.normpath(path)
         found = False
         for i in range(self.file_list.count()):
             item = self.file_list.item(i)
             if os.path.normpath(item.toolTip()) == norm_path:
                 self.file_list.setCurrentItem(item)
                 self.file_list.scrollToItem(item)
                 self.on_file_selected(item)
                 found = True
                 break
         if not found: self.switch_to_standard_mode(path)

    def show_thermo_data(self):
        if not self.thermo_data: 
            QMessageBox.information(self, "Info", "No thermodynamic data available.")
            return

        data = self.thermo_data
        
        def flatten_value(v):
            if v is None: return (None, None)
            if isinstance(v, (int, float, bool, str)): return (v, None)
            if isinstance(v, (list, tuple)):
                if len(v) == 0: return (None, None)
                if len(v) == 1: return flatten_value(v[0])
                if len(v) == 2 and isinstance(v[1], str):
                     val, _ = flatten_value(v[0])
                     return (val, v[1])
                # recurse
                vals = []
                unit = None
                for i in v:
                    val, u = flatten_value(i)
                    if val is not None: vals.append(val)
                    if u: unit = u
                if len(vals) == 1: return (vals[0], unit)
                if vals: return (", ".join(map(str, vals)), unit)
                return (None, unit)
            return (str(v), None)

        dlg = QDialog(self)
        dlg.setWindowTitle("Thermodynamic Properties")
        dlg.resize(600, 400)  
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Thermodynamic Properties (standard conditions)"))
        
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Property", "Value", "Unit"])
        
        order = ["E_tot", "H_tot", "G_tot", "ZPE", "S_tot", "Cv_tot"]
        row = 0
        for k in order:
            if k in data:
                v = data[k]
                val, unit = flatten_value(v)
                if not unit: unit = "Ha"
                table.insertRow(row)
                table.setItem(row, 0, QTableWidgetItem(k))
                table.setItem(row, 1, QTableWidgetItem(f"{float(val):.6f}" if isinstance(val, (int,float)) else str(val)))
                table.setItem(row, 2, QTableWidgetItem(unit))
                row += 1
        
        for k, v in data.items():
            if k not in order:
                val, unit = flatten_value(v)
                table.insertRow(row)
                table.setItem(row, 0, QTableWidgetItem(k))
                table.setItem(row, 1, QTableWidgetItem(str(val)))
                table.setItem(row, 2, QTableWidgetItem(unit if unit else ""))
                row += 1
        
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table)
        
        btn_layout = QHBoxLayout()
        btn_export = QPushButton("Export CSV")
        btn_export.clicked.connect(lambda: self.export_thermo_csv(table))
        btn_layout.addWidget(btn_export)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)
        dlg.exec()

    def export_thermo_csv(self, table):
        fname, _ = QFileDialog.getSaveFileName(self, "Export CSV", "thermo.csv", "CSV (*.csv)")
        if not fname: return
        try:
            with open(fname, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
                writer.writerow(headers)
                for r in range(table.rowCount()):
                    row = [table.item(r, c).text() if table.item(r,c) else "" for c in range(table.columnCount())]
                    writer.writerow(row)
            QMessageBox.information(self, "Success", f"Exported to {fname}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def on_calculation_finished(self, result_data):
        """Called when a calculation finishes to auto-load results."""
        out_dir = result_data.get("out_dir")
        if out_dir:
            self.result_path_display.setText(out_dir)
            # update_structure=True to refresh geometry if optimized
            is_opt = bool(result_data.get("optimized_xyz"))
            self.load_result_folder(out_dir, update_structure=True, is_opt_job=is_opt)

    def close_freq_window(self):
        """Close frequency visualization dock/window if open."""
        if hasattr(self, 'freq_dock') and self.freq_dock:
            try: self.freq_dock.close()
            except: pass
        self.freq_dock = None
        self.freq_vis = None



