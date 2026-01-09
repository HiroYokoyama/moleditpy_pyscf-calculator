from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTreeWidget, QTreeWidgetItem, QHeaderView, QDoubleSpinBox, 
    QSlider, QCheckBox, QGroupBox, QSpinBox, QDialog, 
    QFileDialog, QMessageBox
)
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QPaintEvent
from PyQt6.QtCore import Qt, QTimer
import numpy as np
import traceback


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
        self.list_freq.setColumnCount(3)
        self.list_freq.setHeaderLabels(["Mode", "Freq (cm⁻¹)", "Intensity"])
        self.list_freq.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.list_freq.currentItemChanged.connect(self.on_freq_selected)
        layout.addWidget(self.list_freq)
        
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
        self.spin_scale.setValue(5.0) # PySCF modes might be normalized differently?
        self.spin_scale.valueChanged.connect(self.update_vectors)
        vec_layout.addWidget(self.spin_scale)
        c_layout.addLayout(vec_layout)
        
        # Animation
        anim_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        anim_layout.addWidget(self.btn_play)
        
        self.btn_spectrum = QPushButton("Show Spectrum")
        self.btn_spectrum.clicked.connect(self.show_spectrum)
        anim_layout.addWidget(self.btn_spectrum)
        
        anim_layout.addWidget(QLabel("Amplitude:"))
        self.slider_amp = QSlider(Qt.Orientation.Horizontal)
        self.slider_amp.setRange(1, 100)
        self.slider_amp.setValue(20)
        anim_layout.addWidget(self.slider_amp)
        
        c_layout.addLayout(anim_layout)
        
        layout.addWidget(ctrl_group)
        self.setLayout(layout)
        
    def populate_list(self):
        self.list_freq.clear()
        for i, freq in enumerate(self.freqs):
            # PySCF might return imaginary freq as negative? or complex?
            # Assuming float.
            item = QTreeWidgetItem()
            item.setText(0, str(i + 1))
            item.setText(1, f"{freq:.2f}")
            if self.intensities and i < len(self.intensities):
                item.setText(2, f"{self.intensities[i]:.2f}")
            else:
                item.setText(2, "-")
            self.list_freq.addTopLevelItem(item)
            
    def on_freq_selected(self, current, previous):
        self.reset_geometry()
        self.update_vectors()
        
    def toggle_play(self):
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.btn_play.setText("Play")
            self.reset_geometry()
        else:
            if not self.list_freq.currentItem(): return
            self.timer.start(50)
            self.is_playing = True
            self.btn_play.setText("Stop")
            
    def show_spectrum(self):
        if not self.freqs: return
        # Mock intensities if not present?
        intensities = self.intensities if self.intensities else [1.0] * len(self.freqs)
        dlg = SpectrumDialog(self.freqs, intensities, self)
        dlg.exec()

    def reset_geometry(self):
        conf = self.mol.GetConformer()
        from rdkit.Geometry import Point3D
        for i, (x, y, z) in enumerate(self.base_coords):
            conf.SetAtomPosition(i, Point3D(x, y, z))
        
        # Redraw
        if hasattr(self.mw, 'draw_molecule_3d'):
             self.mw.draw_molecule_3d(self.mol)
             # Restore vectors if needed
             if not self.is_playing: self.update_vectors()

    def update_vectors(self):
        # Clear existing
        if self.vector_actor and hasattr(self.mw, 'plotter'):
            self.mw.plotter.remove_actor(self.vector_actor)
            self.vector_actor = None
            
        if not self.chk_vectors.isChecked(): 
            self.mw.plotter.render()
            return
            
        item = self.list_freq.currentItem()
        if not item: return
        idx = self.list_freq.indexOfTopLevelItem(item)
        
        if idx >= len(self.modes): return
        
        mode = self.modes[idx] # (N, 3) array
        scale = self.spin_scale.value()
        
        # Prepare data for pyvista
        coords = np.array(self.base_coords)
        vectors = np.array(mode)
        
        # Add Arrows
        try:
           self.vector_actor = self.mw.plotter.add_arrows(coords, vectors, mag=scale, color='lightgreen', show_scalar_bar=False)
           self.mw.plotter.render()
        except: pass

    def animate_frame(self):
        item = self.list_freq.currentItem()
        if not item: 
            self.toggle_play()
            return
        idx = self.list_freq.indexOfTopLevelItem(item)
        mode = self.modes[idx]
        
        self.animation_step += 1
        phase = (self.animation_step % 20) / 20.0 * 2 * np.pi
        amp = self.slider_amp.value() / 20.0 # Scaling factor
        
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
        
        if self.vector_actor and hasattr(self.mw, 'plotter'):
             self.mw.plotter.remove_actor(self.vector_actor)
             self.vector_actor = None
        self.mw.plotter.render()



class SpectrumDialog(QDialog):
    def __init__(self, freqs, intensities, parent=None):
        super().__init__(parent)
        self.setWindowTitle('IR Spectrum')
        self.resize(800, 600)
        
        self.freqs = np.array(freqs)
        self.intensities = np.array(intensities)
        
        layout = QVBoxLayout(self)
        
        self.plot_widget = SpectrumPlotWidget(self.freqs, self.intensities)
        layout.addWidget(self.plot_widget)
        
        controls = QHBoxLayout()
        controls.addWidget(QLabel('Broadening (cm⁻¹):'))
        self.spin_fwhm = QSpinBox()
        self.spin_fwhm.setRange(1, 500)
        self.spin_fwhm.setValue(50)
        self.spin_fwhm.valueChanged.connect(lambda v: self.plot_widget.set_fwhm(v))
        controls.addWidget(self.spin_fwhm)
        
        controls.addWidget(QLabel('Max WN:'))
        self.spin_max = QSpinBox()
        self.spin_max.setRange(0, 5000)
        self.spin_max.setValue(4000)
        self.spin_max.setSingleStep(100)
        self.spin_max.valueChanged.connect(self.on_range_changed)
        controls.addWidget(self.spin_max)
        
        btn_close = QPushButton('Close')
        btn_close.clicked.connect(self.accept)
        controls.addWidget(btn_close)
        
        layout.addLayout(controls)
        self.setLayout(layout)
        self.on_range_changed()

    def on_range_changed(self):
        self.plot_widget.set_range(0, self.spin_max.value())

class SpectrumPlotWidget(QWidget):
    def __init__(self, freqs, intensities, parent=None):
        super().__init__(parent)
        self.freqs = freqs
        self.intensities = intensities
        self.fwhm = 50.0
        self.min_x = 0
        self.max_x = 4000
        self.curve_x = []
        self.curve_y = []
        self.setAutoFillBackground(True)
        self.setStyleSheet('background-color: white;')
        self.recalc_curve()

    def set_fwhm(self, val):
        self.fwhm = val
        self.recalc_curve()
        self.update()

    def set_range(self, mn, mx):
        self.min_x = mn
        self.max_x = mx
        self.recalc_curve()
        self.update()

    def recalc_curve(self):
        self.curve_x = np.linspace(self.min_x, self.max_x, 1000)
        self.curve_y = np.zeros_like(self.curve_x)
        
        sigma = self.fwhm / 2.355
        
        for f, i in zip(self.freqs, self.intensities):
            # Lorentzian
            gamma = self.fwhm / 2.0
            # L = (1/pi) * (gamma / ((x-x0)^2 + gamma^2))
            # Scaled by intensity
            peak = i * (gamma / ((self.curve_x - f)**2 + gamma**2))
            self.curve_y += peak
            
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Margins
        mx, my = 50, 40
        plot_w = w - 2*mx
        plot_h = h - 2*my
        
        # Draw Axes
        painter.drawLine(mx, h-my, w-mx, h-my) # X
        painter.drawLine(mx, h-my, mx, my)     # Y
        
        if len(self.curve_y) == 0: return
        
        max_y = np.max(self.curve_y)
        if max_y == 0: max_y = 1.0
        
        # Plot Curve
        path_pts = []
        for i, x_val in enumerate(self.curve_x):
            if x_val < self.min_x or x_val > self.max_x: continue
            
            px = mx + (x_val - self.min_x) / (self.max_x - self.min_x) * plot_w
            py = (h - my) - (self.curve_y[i] / max_y) * plot_h
            path_pts.append((px, py))
            
        if len(path_pts) > 1:
            from PyQt6.QtGui import QPainterPath
            path = QPainterPath()
            path.moveTo(path_pts[0][0], path_pts[0][1])
            for px, py in path_pts[1:]:
                path.lineTo(px, py)
                
            pen = QPen(QColor('blue'), 2)
            painter.setPen(pen)
            painter.drawPath(path)
            
        # Draw Peaks (Sticks)
        pen_stick = QPen(QColor('red'), 2)
        painter.setPen(pen_stick)
        for f, i in zip(self.freqs, self.intensities):
            if f < self.min_x or f > self.max_x: continue
            px = mx + (f - self.min_x) / (self.max_x - self.min_x) * plot_w
            py_base = h - my
            py_top = py_base - 10 
            painter.drawLine(int(px), int(py_base), int(px), int(py_top))


