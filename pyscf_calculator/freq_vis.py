from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTreeWidget, QTreeWidgetItem, QHeaderView, QDoubleSpinBox, 
    QSlider, QCheckBox, QGroupBox, QSpinBox, QDialog, 
    QFileDialog, QMessageBox, QApplication, QFormLayout, QDialogButtonBox # Added QFormLayout, QDialogButtonBox
)
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QPaintEvent, QPalette
from PyQt6.QtCore import Qt, QTimer, QPointF
import numpy as np
import traceback

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

class FreqVisualizer(QWidget):
    def __init__(self, main_window, mol, freqs, modes, intensities=None):
        super().__init__()
        self.mw = main_window
        self.mol = mol  # RDKit Mol
        self.freqs = freqs # List of frequencies
        self.modes = modes # List of mode vectors (each is N_atoms x 3)
        self.intensities = intensities # List of intensities (optional)
        
        # Store original coordinates
        self.base_coords = []
        conf = self.mol.GetConformer()
        for i in range(self.mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            self.base_coords.append((p.x, p.y, p.z))
            
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate_frame)
        self.animation_step = 0
        self.is_playing = False
        self.vector_actor = None
        
        self.init_ui()
        self.populate_list()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # List
        layout.addWidget(QLabel("Vibrational Modes:"))
        self.list_freq = QTreeWidget()
        self.list_freq.setColumnCount(2) # Changed to 2
        self.list_freq.setHeaderLabels(["Mode", "Freq (cm⁻¹)"]) # Removed Intensity
        self.list_freq.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.list_freq.currentItemChanged.connect(self.on_freq_selected)
        layout.addWidget(self.list_freq)
        
        self.btn_select_none = QPushButton("Select None")
        self.btn_select_none.clicked.connect(self.select_none)
        layout.addWidget(self.btn_select_none)
        
        # Controls
        ctrl_group = QGroupBox("Animation Controls")
        c_layout = QVBoxLayout(ctrl_group)
        
        # Vectors
        vec_layout = QHBoxLayout()
        self.chk_vectors = QCheckBox("Show Vectors")
        self.chk_vectors.setChecked(True)
        self.chk_vectors.stateChanged.connect(self.update_vectors)
        vec_layout.addWidget(self.chk_vectors)
        
        vec_layout.addWidget(QLabel("Scale:"))
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.1, 50.0)
        self.spin_scale.setValue(2.0) # Default to 2.0 as requested
        self.spin_scale.valueChanged.connect(self.on_scale_changed)
        vec_layout.addWidget(self.spin_scale)
        c_layout.addLayout(vec_layout)
        
        # Frequency Scaling
        freq_scale_layout = QHBoxLayout()
        freq_scale_layout.addWidget(QLabel("Freq Scale:"))
        self.spin_freq_scale = QDoubleSpinBox()
        self.spin_freq_scale.setRange(0.0, 2.0)
        self.spin_freq_scale.setValue(1.0)
        self.spin_freq_scale.setSingleStep(0.001)
        self.spin_freq_scale.setDecimals(4)
        self.spin_freq_scale.valueChanged.connect(self.update_list_and_spectrum)
        freq_scale_layout.addWidget(self.spin_freq_scale)
        freq_scale_layout.addStretch()
        c_layout.addLayout(freq_scale_layout)
        
        # Animation
        anim_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        anim_layout.addWidget(self.btn_play)
        
        # self.btn_spectrum = QPushButton("Show Spectrum")
        # self.btn_spectrum.clicked.connect(self.show_spectrum)
        # anim_layout.addWidget(self.btn_spectrum)
        
        anim_layout.addWidget(QLabel("Amplitude:"))
        self.spin_amp = QDoubleSpinBox()
        self.spin_amp.setRange(0.01, 10.0)
        self.spin_amp.setSingleStep(0.01)
        self.spin_amp.setDecimals(2)
        self.spin_amp.setValue(1.0)
        anim_layout.addWidget(self.spin_amp)
        
        c_layout.addLayout(anim_layout)
        
        # Export GIF
        self.btn_gif = QPushButton("Export GIF")
        self.btn_gif.clicked.connect(self.save_as_gif)
        self.btn_gif.setEnabled(HAS_PIL)
        c_layout.addWidget(self.btn_gif)
        
        layout.addWidget(ctrl_group)
        self.setLayout(layout)
        
    def populate_list(self):
        self.update_list_and_spectrum()
    
    def update_list_and_spectrum(self):
        self.list_freq.clear()
        sf = self.spin_freq_scale.value()
        
        for i, freq in enumerate(self.freqs):
            item = QTreeWidgetItem()
            item.setText(0, str(i + 1))
            
            scaled_freq = freq * sf
            item.setText(1, f"{scaled_freq:.2f}")
            
            # if self.intensities and i < len(self.intensities):
            #     val = self.intensities[i]
            #     if val is not None:
            #          item.setText(2, f"{val:.2f}")
            #     else:
            #          item.setText(2, "-")
            # else:
            #     item.setText(2, "-")
            
            item.setTextAlignment(0, Qt.AlignmentFlag.AlignCenter)
            item.setTextAlignment(1, Qt.AlignmentFlag.AlignCenter)
            # item.setTextAlignment(2, Qt.AlignmentFlag.AlignCenter)
            self.list_freq.addTopLevelItem(item)

    def on_scale_changed(self):
        self.update_vectors()
            
    def on_freq_selected(self, current, previous):
        if not current: 
            return # select_none handles reset
            
        if self.is_playing:
             pass 
        else:
            self.reset_geometry()
            self.update_vectors()
            
    def select_none(self):
        self.list_freq.clearSelection()
        self.list_freq.setCurrentItem(None)
        
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            self.btn_play.setText("Play")
        
        self.reset_geometry()
        self.update_vectors()
        QApplication.processEvents()

    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            self.btn_play.setText("Play")
            self.reset_geometry() # Reset to rest
            QApplication.processEvents()
        else:
            if not self.list_freq.currentItem(): return
            self.timer.start(50)
            self.is_playing = True
            self.btn_play.setText("Stop")
            
    def show_spectrum(self):
        if not self.freqs: return
        
        if self.intensities is None or len(self.intensities) == 0:
            return

        intensities = self.intensities
        
        # Handle None explicitly in list too
        safe_ints = []
        for i in intensities:
            if i is None: safe_ints.append(0.0)
            else: safe_ints.append(float(i))
            
        # Apply scaling to freqs for display
        sf = self.spin_freq_scale.value()
        scaled_freqs = [f * sf for f in self.freqs]
            
        dlg = SpectrumDialog(scaled_freqs, safe_ints, parent=self)
        dlg.exec()

    def reset_geometry(self):
        conf = self.mol.GetConformer()
        from rdkit.Geometry import Point3D
        for i, (x, y, z) in enumerate(self.base_coords):
            conf.SetAtomPosition(i, Point3D(x, y, z))

        # Redraw once
        if hasattr(self.mw, 'draw_molecule_3d'):
             self.mw.draw_molecule_3d(self.mol)
        
        if not self.is_playing: 
             self.update_vectors()

    def update_vectors(self):
        # CLEANUP: Remove vectors and OTHER actors (Orbital/Mapped)
        if hasattr(self.mw, 'plotter'):
            if self.vector_actor:
                self.mw.plotter.remove_actor(self.vector_actor)
                self.vector_actor = None
            
            # Remove potential orbital actors to avoid interference
            self.mw.plotter.remove_actor("pyscf_iso_p")
            self.mw.plotter.remove_actor("pyscf_iso_n")
            self.mw.plotter.remove_actor("pyscf_mapped")
            
        if not self.chk_vectors.isChecked(): 
            if hasattr(self.mw, 'plotter'): self.mw.plotter.render()
            return
            
        item = self.list_freq.currentItem()
        if not item: # Select None case
            if hasattr(self.mw, 'plotter'): self.mw.plotter.render()
            return

        idx = self.list_freq.indexOfTopLevelItem(item)
        if idx >= len(self.modes): return
        
        mode = self.modes[idx] # (N, 3) array
        scale = self.spin_scale.value() # This is vector scale
        
        # Prepare data for pyvista
        coords = np.array(self.base_coords)
        vectors = np.array(mode)
        
        # Add Arrows
        try:
           self.vector_actor = self.mw.plotter.add_arrows(coords, vectors, mag=scale, color='lightgreen', show_scalar_bar=False)
           self.mw.plotter.render()
        except: pass

    def animate_frame(self):
        if not self.is_playing: return
        
        item = self.list_freq.currentItem()
        if not item: 
            self.toggle_play()
            return
        idx = self.list_freq.indexOfTopLevelItem(item)
        mode = self.modes[idx]
        
        self.animation_step += 1
        # Match ORCA logic: 20 frames per cycle
        cycle_pos = (self.animation_step % 20) / 20.0
        phase = cycle_pos * 2 * np.pi
        
        # Amplitude handling
        amp = self.spin_amp.value()
        
        factor = np.sin(phase) * amp
        
        # Update atoms
        conf = self.mol.GetConformer()
        from rdkit.Geometry import Point3D
        
        for i, (bx, by, bz) in enumerate(self.base_coords):
            dx, dy, dz = mode[i]
            nx = bx + dx * factor
            ny = by + dy * factor
            nz = bz + dz * factor
            conf.SetAtomPosition(i, Point3D(nx, ny, nz))
            
        if hasattr(self.mw, 'draw_molecule_3d'):
             self.mw.draw_molecule_3d(self.mol)
        
        # Remove vectors during animation to avoid clutter/mismatch
        if self.vector_actor and hasattr(self.mw, 'plotter'):
             self.mw.plotter.remove_actor(self.vector_actor)
             self.vector_actor = None
        self.mw.plotter.render()



    def save_as_gif(self):
        if not self.mol: return
        
        was_playing = self.is_playing
        if self.is_playing:
            self.toggle_play()

        curr = self.list_freq.currentItem()
        if not curr:
            QMessageBox.warning(self, "Select Frequency", "Please select a frequency to export.")
            return
        idx = self.list_freq.indexOfTopLevelItem(curr)
        if idx < 0: return

        dialog = QDialog(self)
        dialog.setWindowTitle("Export GIF Settings")
        form = QFormLayout(dialog)
        
        spin_fps = QSpinBox()
        spin_fps.setRange(1, 60)
        spin_fps.setValue(20)
        
        # Check standard transparency preference
        chk_transparent = QCheckBox()
        chk_transparent.setChecked(True)
        
        chk_hq = QCheckBox()
        chk_hq.setChecked(True)
        
        form.addRow("FPS:", spin_fps)
        form.addRow("Transparent Background:", chk_transparent)
        form.addRow("High Quality (Adaptive):", chk_hq)
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        form.addRow(btns)
        
        if dialog.exec() != QDialog.DialogCode.Accepted:
            if was_playing: self.toggle_play()
            return

        target_fps = spin_fps.value()
        use_transparent = chk_transparent.isChecked()
        use_hq = chk_hq.isChecked()
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save GIF", "", "GIF Files (*.gif)")
        if not file_path:
            if was_playing: self.toggle_play()
            return
            
        if not file_path.lower().endswith('.gif'):
             file_path += '.gif'
             
        # Generate Frames
        images = []
        mode = self.modes[idx]
        
        self.reset_geometry()
        
        try:
            # 20 frames cycle
            for i in range(20):
                cycle_pos = i / 20.0
                phase = cycle_pos * 2 * np.pi
                scale = self.spin_amp.value() 
                factor = np.sin(phase) * scale
                
                # Apply displacement manually
                conf = self.mol.GetConformer()
                from rdkit.Geometry import Point3D
                for j, (bx, by, bz) in enumerate(self.base_coords):
                    dx, dy, dz = mode[j]
                    nx = bx + dx * factor
                    ny = by + dy * factor
                    nz = bz + dz * factor
                    conf.SetAtomPosition(j, Point3D(nx, ny, nz))
                
                if hasattr(self.mw, 'draw_molecule_3d'):
                     self.mw.draw_molecule_3d(self.mol)
                
                if self.chk_vectors.isChecked():
                     pass 

                if hasattr(self.mw, 'plotter'):
                    self.mw.plotter.render()
                    img_array = self.mw.plotter.screenshot(transparent_background=use_transparent, return_img=True)
                    if img_array is not None:
                         img = Image.fromarray(img_array)
                         images.append(img)
            
            if images:
                duration_ms = int(1000 / target_fps)
                processed_images = []
                for img in images:
                    if use_hq:
                        if use_transparent:
                            alpha = img.split()[3]
                            img_rgb = img.convert("RGB")
                            img_p = img_rgb.convert('P', palette=Image.Palette.ADAPTIVE, colors=255)
                            mask = Image.eval(alpha, lambda a: 255 if a <= 128 else 0)
                            img_p.paste(255, mask)
                            img_p.info['transparency'] = 255
                            processed_images.append(img_p)
                        else:
                            processed_images.append(img.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))
                    else:
                         processed_images.append(img) 

                processed_images[0].save(
                    file_path, 
                    save_all=True, 
                    append_images=processed_images[1:], 
                    optimize=False, 
                    duration=duration_ms, 
                    loop=0, 
                    disposal=2
                )
                QMessageBox.information(self, "Success", f"Saved GIF to {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save GIF: {e}\n{traceback.format_exc()}")
        
        finally:
            self.reset_geometry()
            if was_playing: self.toggle_play()

    def cleanup(self):
        if hasattr(self.mw, 'plotter') and self.vector_actor:
             self.mw.plotter.remove_actor(self.vector_actor)
             self.vector_actor = None
             self.mw.plotter.render()

class SpectrumDialog(QDialog):
    def __init__(self, freqs, intensities, title="IR Spectrum", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 600)
        
        self.freqs = np.array(freqs)
        self.intensities = np.array(intensities)
        
        layout = QVBoxLayout(self)
        
        # Plot
        self.plot_widget = SpectrumWidget(self.freqs, self.intensities)
        layout.addWidget(self.plot_widget)
        
        # Controls
        ctrl_layout = QHBoxLayout()
        
        # Broadening
        ctrl_layout.addWidget(QLabel("Broadening (cm⁻¹):"))
        self.spin_width = QDoubleSpinBox()
        self.spin_width.setRange(1.0, 500.0)
        self.spin_width.setValue(20.0) 
        self.spin_width.valueChanged.connect(self.update_plot)
        ctrl_layout.addWidget(self.spin_width)
        
        # Range
        ctrl_layout.addWidget(QLabel("Max Wavenumber:"))
        self.spin_max_wn = QDoubleSpinBox()
        self.spin_max_wn.setRange(100.0, 10000.0)
        self.spin_max_wn.setValue(4000.0)
        self.spin_max_wn.setSingleStep(100.0)
        self.spin_max_wn.valueChanged.connect(self.update_plot)
        ctrl_layout.addWidget(self.spin_max_wn)
        
        # Invert Intensity
        self.chk_invert_y = QCheckBox("Invert Intensity")
        self.chk_invert_y.stateChanged.connect(self.update_plot)
        ctrl_layout.addWidget(self.chk_invert_y)
        
        # Gaussian vs Lorentzian
        # Removed per user request (Default to Gaussian)
        # self.chk_gaussian = QCheckBox("Gaussian Profile")
        # self.chk_gaussian.setChecked(True) 
        # self.chk_gaussian.stateChanged.connect(self.update_plot)
        # ctrl_layout.addWidget(self.chk_gaussian)

        ctrl_layout.addStretch()
        
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        ctrl_layout.addWidget(btn_close)
        
        layout.addLayout(ctrl_layout)
        
        self.update_plot()
        
    def update_plot(self):
        self.plot_widget.set_params(
            width=self.spin_width.value(),
            max_wn=self.spin_max_wn.value(),
            invert_y=self.chk_invert_y.isChecked(),
            use_gaussian=True # Hardcoded to True
        )

class SpectrumWidget(QWidget):
    def __init__(self, freqs, intensities):
        super().__init__()
        self.freqs = freqs
        self.intensities = intensities
        self.setBackgroundRole(QPalette.ColorRole.Base)
        self.setAutoFillBackground(True)
        
        # Defaults
        self.width_val = 20.0
        self.max_wn = 4000.0
        self.invert_y = False
        self.use_gaussian = True
        
        self.curve_x = []
        self.curve_y = []
        self.recalc_curve()
        
    def set_params(self, width, max_wn, invert_y, use_gaussian):
        self.width_val = width
        self.max_wn = max_wn
        self.invert_y = invert_y
        self.use_gaussian = use_gaussian
        self.recalc_curve()
        self.update()
        
    def recalc_curve(self):
        if len(self.freqs) == 0: return
        
        # Create X grid
        self.curve_x = np.linspace(0, self.max_wn, 1000)
        self.curve_y = np.zeros_like(self.curve_x)
        
        sigma = self.width_val # FWHM-like parameter
        
        import math
        
        if self.use_gaussian:
             # Gaussian: exp( - (x-mu)^2 / (2*sigma^2) ) ... actually FWHM to sigma conversion?
             # FWHM = 2.355 * stdev.  sigma input is treated as FWHM here usually.
             s = sigma / 2.355
             for f, i in zip(self.freqs, self.intensities):
                 # Normalized Gaussian? Or just height? Usually user wants peak height ~ intensity
                 # Exp factor
                 self.curve_y += i * np.exp(- (self.curve_x - f)**2 / (2 * s**2))
        else:
             # Lorentzian: gamma / ((x-x0)^2 + gamma^2)
             # FWHM = 2*gamma. gamma = sigma/2
             gamma = sigma / 2.0
             for f, i in zip(self.freqs, self.intensities):
                 self.curve_y += i * (gamma**2 / ((self.curve_x - f)**2 + gamma**2))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Margins
        ml, mr, mt, mb = 60, 20, 20, 40
        plot_w = w - ml - mr
        plot_h = h - mt - mb
        
        # Ranges
        min_f = 0.0
        max_f = self.max_wn
        f_range = max_f - min_f
        
        # Max Y
        max_y = np.max(self.curve_y) if len(self.curve_y) > 0 else 1.0
        if max_y == 0: max_y = 1.0
        
        # Draw Background
        painter.fillRect(0, 0, w, h, QColor("white"))
        
        # Axes
        painter.setPen(QPen(QColor("black"), 1))
        painter.drawLine(ml, h - mb, w - mr, h - mb) # X
        painter.drawLine(ml, mt, ml, h - mb) # Y
        
        # X Ticks (Inverted: High on Left)
        n_ticks = 10
        painter.setPen(QColor("#ddd"))
        for i in range(n_ticks + 1):
             val = min_f + (f_range * i / n_ticks)
             # Map: Inverted
             # Val 0 (min) -> Right (w-mr)
             # Val Max -> Left (ml)
             x_px = ml + (max_f - val) / f_range * plot_w
             
             painter.drawLine(int(x_px), mt, int(x_px), h - mb)
             
             painter.setPen(QColor("black"))
             painter.drawText(int(x_px) - 20, h - mb + 5, 40, 15, Qt.AlignmentFlag.AlignCenter, f"{int(val)}")
             painter.setPen(QColor("#ddd"))
             
        # Plot Curve
        if len(self.curve_x) > 0:
            painter.setPen(QPen(QColor("blue"), 2))
            path_pts = []
            
            for i, val in enumerate(self.curve_x):
                # Clamp
                if val < 0 or val > max_f: continue
                
                # X Map
                x_px = ml + (max_f - val) / f_range * plot_w
                
                # Y Map
                # Invert Y if requested (Transmittance style: 0 at top, 1 at bottom? Or just flip?)
                # Standard Spectrum: 0 at Bottom (h-mb). 
                # If "Invert Intensity": 0 at Top (mt)? Or plot goes DOWN?
                # Usually Inverted Intensity means peaks point Down (like Transmittance).
                # 0 intensity = Top line (100% T). Max intensity = Bottom.
                
                y_norm = self.curve_y[i] / max_y
                
                if self.invert_y:
                    y_px = mt + y_norm * plot_h
                else:
                    y_px = (h - mb) - y_norm * plot_h
                    
                path_pts.append(QPointF(x_px, y_px))
            
            if path_pts:
                painter.drawPolyline(path_pts)
            
        # Draw Sticks (Peaks)
        painter.setPen(QPen(QColor("red"), 1))
        for f, i in zip(self.freqs, self.intensities):
            if f < 0 or f > max_f: continue
            
            x_px = ml + (max_f - f) / f_range * plot_w
            
            y_norm = i / max_y if max_y > 0 else 0
            
            if self.invert_y:
                y_base = mt
                y_tip = mt + y_norm * plot_h
            else:
                y_base = h - mb
                y_tip = (h - mb) - y_norm * plot_h
                
            painter.drawLine(QPointF(x_px, y_base), QPointF(x_px, y_tip))
            
        # Labels
        painter.setPen(QColor("black"))
        painter.drawText(w//2, h-5, "Wavenumber (cm⁻¹)")
        
        painter.save()
        painter.translate(15, h//2)
        painter.rotate(-90)
        label = "Intensity (Inverted)" if self.invert_y else "Intensity"
        painter.drawText(0, 0, label)
        painter.restore()


