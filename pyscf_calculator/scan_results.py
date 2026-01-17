import os
import io
import traceback
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.collections

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QFileDialog, QMessageBox, QWidget, QComboBox,
    QSpinBox, QCheckBox, QFormLayout, QDialogButtonBox, QApplication
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdGeometry

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class ScanResultDialog(QDialog):
    def __init__(self, parent=None, results=None, trajectory=None, context=None, scan_type="Coordinate", scan_result_dir=None):
        super().__init__(parent)
        self.setWindowTitle(f"Scan Results: {scan_type}")
        self.resize(1200, 600)  # Enlarged width by 1.5x
        
        self.results = results  # List of {step, value, energy, ...}
        self.trajectory = trajectory # List of XYZ strings or RDKit Mols
        self.context = context
        self.scan_type = scan_type
        self.scan_result_dir = scan_result_dir
        self.frame_idx = 0
        self.is_playing = False
        self.base_mol = None
        
        if self.trajectory and len(self.trajectory) > 0:
            self.create_base_molecule()
        
        self.init_ui()
        self.plot_data()
        
        # Set focus to Play button so user can start manually with Enter/Space
        if hasattr(self, 'btn_play'):
            self.btn_play.setFocus()
            self.btn_play.setDefault(True)

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Graph
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.canvas)
        
        # Connect pick and hover events
        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)

        # Create annotation for tooltip (hidden by default)
        self.annot = self.canvas.axes.annotate("", xy=(0,0), xytext=(20,20),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w", alpha=0.9),
                            arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

        # 2. Controls
        ctrl_layout = QHBoxLayout()
        
        self.btn_prev = QPushButton("<<")
        self.btn_prev.clicked.connect(self.prev_frame)
        ctrl_layout.addWidget(self.btn_prev)

        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        ctrl_layout.addWidget(self.btn_play)

        self.btn_next = QPushButton(">>")
        self.btn_next.clicked.connect(self.next_frame)
        ctrl_layout.addWidget(self.btn_next)
        
        # Unselect Button
        self.btn_unselect = QPushButton("Clear Selection")
        self.btn_unselect.clicked.connect(self.clear_selection)
        ctrl_layout.addWidget(self.btn_unselect)
        
        # Save Graph Button
        self.btn_save = QPushButton("Save Graph")
        self.btn_save.clicked.connect(self.save_plot)
        ctrl_layout.addWidget(self.btn_save)
        
        # Save CSV Button
        self.btn_save_csv = QPushButton("Save CSV")
        self.btn_save_csv.clicked.connect(self.save_csv)
        ctrl_layout.addWidget(self.btn_save_csv)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(self.trajectory) - 1 if self.trajectory else 0)
        self.slider.setMinimumWidth(300)  # Ensure slider is visible
        self.slider.valueChanged.connect(self.on_slider_change)
        ctrl_layout.addWidget(self.slider)

        self.lbl_frame = QLabel("Frame: 0")
        ctrl_layout.addWidget(self.lbl_frame)
        
        # Energy unit selector
        ctrl_layout.addWidget(QLabel("Units:"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Hartree", "kJ/mol", "kcal/mol"])
        self.unit_combo.setCurrentIndex(1)  # Default to kJ/mol
        self.unit_combo.currentTextChanged.connect(self.on_unit_changed)
        ctrl_layout.addWidget(self.unit_combo)

        # Relative Energy Checkbox
        self.chk_relative = QCheckBox("Relative")
        self.chk_relative.setChecked(True) # Relative by default as requested
        self.chk_relative.toggled.connect(self.plot_data)
        ctrl_layout.addWidget(self.chk_relative)

        layout.addLayout(ctrl_layout)
        
        # 3. Export
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        
        self.btn_gif = QPushButton("Save GIF")
        self.btn_gif.clicked.connect(self.save_gif)
        self.btn_gif.setEnabled(HAS_PIL and len(self.trajectory) > 0 if self.trajectory else False)
        export_layout.addWidget(self.btn_gif)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        export_layout.addWidget(self.btn_close)
        
        layout.addLayout(export_layout)

        # Timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame_auto)

    def plot_data(self):
        if not self.results: return
        
        # Conversion factors
        HARTREE_TO_KJMOL = 2625.5
        HARTREE_TO_KCALMOL = 627.509
        
        unit = self.unit_combo.currentText() if hasattr(self, 'unit_combo') else "Hartree"
        is_rel = self.chk_relative.isChecked() if hasattr(self, 'chk_relative') else False
        
        x = [r['value'] for r in self.results]
        y_hartree_abs = [r['energy'] for r in self.results]
        
        # Calculate Energy based on Relative setting
        if is_rel:
            min_e = min(y_hartree_abs)
            y_hartree = [e - min_e for e in y_hartree_abs]
            ylabel_prefix = "Relative Energy"
        else:
            y_hartree = y_hartree_abs
            ylabel_prefix = "Energy"
        
        # Convert energies based on selected unit
        if unit == "kJ/mol":
            y = [e * HARTREE_TO_KJMOL for e in y_hartree]
            ylabel = f"{ylabel_prefix} (kJ/mol)"
        elif unit == "kcal/mol":
            y = [e * HARTREE_TO_KCALMOL for e in y_hartree]
            ylabel = f"{ylabel_prefix} (kcal/mol)"
        else:  # Hartree
            y = y_hartree
            ylabel = f"{ylabel_prefix} (Hartree)"
        
        self.canvas.axes.clear()
        line, = self.canvas.axes.plot(x, y, 'b-', label='Energy', picker=5) # Enable picker
        self.scatter = self.canvas.axes.scatter(x, y, c='red', s=25, picker=5, zorder=5) # Use scatter for easier hover detection
        
        # Labeling
        xlabel = "Coordinate"
        if "Bond" in self.scan_type or "Dist" in self.scan_type:
            xlabel = "Bond Length (Å)"
        elif "Angle" in self.scan_type or "Dihedral" in self.scan_type:
            xlabel = "Angle (Degrees)"
            
        self.canvas.axes.set_xlabel(xlabel)
        self.canvas.axes.set_ylabel(ylabel)
        self.canvas.axes.set_title(f"Scan Profile: {self.scan_type}")
        self.canvas.axes.grid(True)
        
        # Highlight current frame
        self.highlight_point(0)

    def on_unit_changed(self, unit):
        """Replot when energy unit changes"""
        self.plot_data()
        # Restore highlight if exists
        if hasattr(self, 'frame_idx') and self.frame_idx >= 0:
            self.highlight_point(self.frame_idx)
        
        self.canvas.draw()

    def highlight_point(self, idx):
        # Remove old highlight if exists
        if hasattr(self, '_highlight_marker'):
            try: self._highlight_marker.remove()
            except: pass
        if hasattr(self, '_highlight_line'):
            try: self._highlight_line.remove()
            except: pass
            
        # Get coordinate value
        x = self.results[idx]['value']
        
        # Get absolute energy and calculate energy according to relative checkbox
        y_abs = self.results[idx]['energy']
        is_rel = self.chk_relative.isChecked() if hasattr(self, 'chk_relative') else False
        
        if is_rel:
            all_energies = [r['energy'] for r in self.results]
            min_e = min(all_energies)
            y_hartree = y_abs - min_e
        else:
            y_hartree = y_abs
        
        # Apply same conversion as plot_data
        HARTREE_TO_KJMOL = 2625.5
        HARTREE_TO_KCALMOL = 627.509
        unit = self.unit_combo.currentText() if hasattr(self, 'unit_combo') else "Hartree"
        
        if unit == "kJ/mol":
            y = y_hartree * HARTREE_TO_KJMOL
        elif unit == "kcal/mol":
            y = y_hartree * HARTREE_TO_KCALMOL
        else:  # Hartree
            y = y_hartree
        
        # 1. Large distinct marker (Red circle)
        self._highlight_marker, = self.canvas.axes.plot(x, y, 'ro', markersize=10, markeredgecolor='darkred', markeredgewidth=2, zorder=10)
        
        # 2. Vertical Line
        self._highlight_line = self.canvas.axes.axvline(x=x, color='gray', linestyle='--', alpha=0.7, zorder=0)
        
        self.canvas.draw()

    def create_base_molecule(self):
        """Create a base molecule with topology from the first frame."""
        if not self.trajectory:
            return
        
        try:
            # Match AnimatedXYZPlayer logic more closely for reliable bond detection
            xyz_block = self.trajectory[0]
            lines = xyz_block.strip().split('\n')
            if len(lines) < 3: return
            
            mol = Chem.RWMol()
            coord_start = 0
            if lines[0].strip().isdigit():
                coord_start = 2
            
            coords = []
            for line in lines[coord_start:]:
                parts = line.split()
                if len(parts) >= 4:
                    sym = parts[0]
                    try:
                        x, y, z = map(float, parts[1:4])
                        coords.append((x, y, z))
                        try:
                            atom = Chem.Atom(sym)
                        except:
                            atom = Chem.Atom('C') # Fallback
                        mol.AddAtom(atom)
                    except ValueError:
                        continue
            
            # Add conformer
            conf = Chem.Conformer(mol.GetNumAtoms())
            for idx, (x, y, z) in enumerate(coords):
                conf.SetAtomPosition(idx, rdGeometry.Point3D(x, y, z))
            mol.AddConformer(conf)

            # Establish topology
            mw = self.context.get_main_window()
            
            # Use same logic as xyz_giffer: prefer main window's estimate_bonds_from_distances
            if hasattr(mw, 'estimate_bonds_from_distances'):
                try:
                    mw.estimate_bonds_from_distances(mol)
                except Exception as e:
                    print(f"estimate_bonds_from_distances failed: {e}")
            
            # Also try rdDetermineBonds as a secondary supplement if 0 bonds were found
            if mol.GetNumBonds() == 0:
                try:
                    from rdkit.Chem import rdDetermineBonds
                    rdDetermineBonds.DetermineConnectivity(mol)
                    rdDetermineBonds.DetermineBondOrders(mol)
                except Exception as e:
                    print(f"rdDetermineBonds fallback failed: {e}")
            
            self.base_mol = mol.GetMol()
            
            # Set as current molecule in context
            if self.context:
                self.context.current_molecule = self.base_mol
            
            # Initial draw
            if hasattr(mw, 'draw_molecule_3d'):
                mw.draw_molecule_3d(self.base_mol)
                if hasattr(mw, 'plotter'):
                    mw.plotter.reset_camera()
                    mw.plotter.update()
                    mw.plotter.render()
        except Exception as e:
            print(f"Error creating base molecule: {e}")
            traceback.print_exc()

    def on_pick(self, event):
        if event.artist and hasattr(event, 'ind'):
            idx = event.ind[0]
            self.set_frame(idx)

    def on_hover(self, event):
        """Update and show tooltip on hover"""
        vis = self.annot.get_visible()
        if event.inaxes == self.canvas.axes:
            if hasattr(self, 'scatter') and self.scatter:
                cont, ind = self.scatter.contains(event)
                if cont:
                    idx = ind['ind'][0]
                    # offsets for scatter are at idx
                    pos = self.scatter.get_offsets()[idx]
                    
                    self.annot.xy = pos
                    
                    unit = self.unit_combo.currentText()
                    val = self.results[idx]['value']
                    energy = self.results[idx]['energy']
                    
                    is_rel = self.chk_relative.isChecked()
                    disp_energy = energy
                    if is_rel:
                        min_e = min([r['energy'] for r in self.results])
                        disp_energy -= min_e
                    
                    if unit == "kJ/mol": disp_energy *= 2625.5
                    elif unit == "kcal/mol": disp_energy *= 627.509
                    
                    # High precision for exact values
                    text = f"X: {val:.6f}\nY: {disp_energy:.8f} {unit}"
                    self.annot.set_text(text)
                    self.annot.set_visible(True)
                    self.canvas.draw_idle()
                    return
        
        if vis:
            self.annot.set_visible(False)
            self.canvas.draw_idle()

    def set_frame(self, idx):
        if not self.trajectory or idx < 0 or idx >= len(self.trajectory):
            return
            
        self.frame_idx = idx
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.lbl_frame.setText(f"Frame: {idx}")
        
        self.highlight_point(idx)
        self.update_viewer(idx)

    def update_viewer(self, idx):
        if not self.context or not self.trajectory: return
        
        xyz = self.trajectory[idx]
        
        if self.base_mol:
            # Efficient update: just change coordinates
            try:
                # Parse coordinates from XYZ string
                lines = xyz.strip().split('\n')
                coord_start = 0
                if lines[0].strip().isdigit():
                    coord_start = 2
                
                conf = self.base_mol.GetConformer()
                for i, line in enumerate(lines[coord_start:]):
                    parts = line.split()
                    if len(parts) >= 4:
                        x, y, z = map(float, parts[1:4])
                        conf.SetAtomPosition(i, rdGeometry.Point3D(x, y, z))
                
                # Trigger redraw in main window
                # The context.current_molecule setter might not trigger redraw if it's the same object
                # So we might need to call draw_molecule_3d directly if available
                mw = self.context.get_main_window()
                if hasattr(mw, 'draw_molecule_3d'):
                    mw.draw_molecule_3d(self.base_mol)
                else:
                    self.context.current_molecule = self.base_mol
            except Exception as e:
                # Fallback to full reload if efficient update fails
                from .utils import update_molecule_from_xyz
                update_molecule_from_xyz(self.context, xyz, mark_modified=False)
        else:
            # Fallback
            from .utils import update_molecule_from_xyz
            update_molecule_from_xyz(self.context, xyz, mark_modified=False)

    def on_slider_change(self, val):
        self.set_frame(val)

    def toggle_play(self):
        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("Play")
            self.is_playing = False
        else:
            self.timer.start(500) # 500ms
            self.btn_play.setText("Pause")
            self.is_playing = True
            
    def save_plot(self):
        try:
            default_dir = self.scan_result_dir if self.scan_result_dir else ""
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Graph", default_dir,
                "Images (*.png *.jpg *.svg *.pdf);;All Files (*)"
            )
            if path:
                self.canvas.fig.savefig(path, dpi=300)
                QMessageBox.information(self, "Saved", f"Graph saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save graph: {e}")

    def save_csv(self):
        if not self.results:
            return
        try:
            default_dir = self.scan_result_dir if self.scan_result_dir else ""
            default_path = os.path.join(default_dir, "scan_export.csv") if default_dir else "scan_export.csv"
            
            path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV", default_path,
                "CSV Files (*.csv);;All Files (*)"
            )
            if path:
                import csv
                keys = self.results[0].keys()
                with open(path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(self.results)
                QMessageBox.information(self, "Saved", f"Results saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save CSV: {e}")

    def prev_frame(self):
        self.set_frame(self.frame_idx - 1)

    def clear_selection(self):
        """Remove highlight marker and line from the graph"""
        try:
            if hasattr(self, '_highlight_marker') and self._highlight_marker:
                self._highlight_marker.remove()
                self._highlight_marker = None
        except: pass
        
        try:
            if hasattr(self, '_highlight_line') and self._highlight_line:
                self._highlight_line.remove()
                self._highlight_line = None
        except: pass
        
        self.canvas.draw()

    def next_frame(self):
        self.set_frame(self.frame_idx + 1)
        
    def next_frame_auto(self):
        next_idx = self.frame_idx + 1
        if next_idx >= len(self.trajectory):
            next_idx = 0 # Loop
        self.set_frame(next_idx)

    def closeEvent(self, event):
        """Stop animation before closing dialog."""
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
        event.accept()

    def save_gif(self):
        """Export animation as GIF with advanced options (from animated_xyz_giffer)"""
        if not self.trajectory:
            return
        
        if not HAS_PIL:
            QMessageBox.warning(self, "Error", "PIL (Pillow) module is required for GIF export.\nPlease install it via: pip install Pillow")
            return

        # Pause if playing
        was_playing = self.is_playing
        if self.is_playing:
            self.toggle_play()

        # Dialog for settings
        dialog = QDialog(self)
        dialog.setWindowTitle("Export GIF Settings")
        form = QFormLayout(dialog)
        
        spin_fps = QSpinBox()
        spin_fps.setRange(1, 60)
        spin_fps.setValue(10)  # Default 10 FPS
        
        chk_transparent = QCheckBox()
        chk_transparent.setChecked(True)
        
        form.addRow("FPS:", spin_fps)
        form.addRow("Transparent Background:", chk_transparent)

        chk_loop = QCheckBox()
        chk_loop.setChecked(True)
        form.addRow("Loop Animation:", chk_loop)
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        form.addRow(btns)
        
        if dialog.exec() != QDialog.DialogCode.Accepted:
            if was_playing:
                self.toggle_play()
            return

        target_fps = spin_fps.value()
        use_transparent = chk_transparent.isChecked()
        use_loop = chk_loop.isChecked()

        # File Dialog
        default_name = os.path.join(self.scan_result_dir, "scan_animation.gif") if self.scan_result_dir else "scan_animation.gif"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save GIF", default_name, "GIF Files (*.gif)"
        )
        if not file_path:
            if was_playing:
                self.toggle_play()
            return
            
        if not file_path.lower().endswith('.gif'):
             file_path += '.gif'

        # Progress Dialog
        self.setCursor(Qt.CursorShape.WaitCursor)
        from PyQt6.QtWidgets import QProgressDialog, QApplication
        progress = QProgressDialog("Generating GIF...", "Cancel", 0, len(self.trajectory), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        
        try:
            original_frame_idx = self.frame_idx
            images = []
            
            mw = self.context.get_main_window() if self.context else None
            if not mw or not hasattr(mw, 'plotter'):
                raise Exception("3D plotter not available")
            
            for i in range(len(self.trajectory)):
                if progress.wasCanceled(): 
                    break
                
                self.set_frame(i)
                QApplication.processEvents() # Process events to ensure viewer updates
                
                # Force update/render as in xyz_giffer
                mw.plotter.update()
                mw.plotter.render()
                
                # Capture screenshot
                img_array = mw.plotter.screenshot(transparent_background=use_transparent, return_img=True)
                
                if img_array is not None:
                    img = Image.fromarray(img_array)
                    images.append(img)
                
                progress.setValue(i + 1)
            
            # Save GIF with advanced settings
            if images:
                gif_frames = []
                duration_ms = int(1000 / target_fps)
                
                for img in images:
                    if use_transparent:
                         # Advanced transparency handling for GIF
                         img = img.convert("RGBA")
                         alpha = img.split()[3]
                         
                         # Create binary mask
                         mask = Image.eval(alpha, lambda a: 255 if a <= 128 else 0)
                         
                         # Quantize to 255 colors (leaving 1 for transparency)
                         img_p = img.convert("RGB").quantize(colors=255)
                         
                         # Paste transparent color index into transparent regions
                         img_p.paste(255, mask)
                         
                         gif_frames.append(img_p)
                    else:
                         gif_frames.append(img)

                # Save with parameters
                save_params = {
                    "save_all": True,
                    "append_images": gif_frames[1:],
                    "duration": duration_ms,
                    "disposal": 2,
                }
                
                if use_transparent:
                    save_params["transparency"] = 255

                if use_loop:
                    save_params["loop"] = 0  # Infinite loop
                
                gif_frames[0].save(file_path, **save_params)
                QMessageBox.information(self, "Success", f"Saved GIF to:\n{file_path}")
            else:
                QMessageBox.warning(self, "Error", "Failed to capture frames.")
                
            # Restore original frame
            self.set_frame(original_frame_idx)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save GIF:\n{e}")
        finally:
             self.setCursor(Qt.CursorShape.ArrowCursor)
             progress.close()
             if was_playing:
                 self.toggle_play()
