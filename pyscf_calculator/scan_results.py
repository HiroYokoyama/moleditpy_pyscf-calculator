import csv
import logging
import os

import matplotlib
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)
from rdkit import Chem
from rdkit.Chem import rdGeometry

matplotlib.use("QtAgg")  # must precede matplotlib backend imports
import matplotlib.collections
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

_HARTREE_TO_KJMOL = 2625.5
_HARTREE_TO_KCALMOL = 627.509
_UNIT_FACTORS = {
    "Hartree": 1.0,
    "kJ/mol": _HARTREE_TO_KJMOL,
    "kcal/mol": _HARTREE_TO_KCALMOL,
}
# matplotlib raises these when an artist is already removed
_ARTIST_GONE = (ValueError, NotImplementedError, AttributeError)

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class ScanResultDialog(QDialog):
    def __init__(
        self,
        parent=None,
        results=None,
        trajectory=None,
        context=None,
        scan_type="Coordinate",
        scan_result_dir=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Scan Results: {scan_type}")
        self.resize(1200, 600)  # Enlarged width by 1.5x

        self.results = results  # List of {step, value, energy, ...}
        self.trajectory = trajectory  # List of XYZ strings or RDKit Mols
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
        if getattr(self, "btn_play", None) is not None:
            self.btn_play.setFocus()
            self.btn_play.setDefault(True)

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Graph
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.canvas)

        # Connect pick and hover events
        self.canvas.mpl_connect("pick_event", self.on_pick)
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

        # Create annotation for tooltip (hidden by default)
        self.annot = self.canvas.axes.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox={"boxstyle": "round", "fc": "w", "alpha": 0.9},
            arrowprops={"arrowstyle": "->"},
        )
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
        self.chk_relative.setChecked(True)  # Relative by default as requested
        self.chk_relative.toggled.connect(self.plot_data)
        ctrl_layout.addWidget(self.chk_relative)

        # Dynamic Bonds Checkbox
        self.chk_dynamic_bonds = QCheckBox("Dynamic Bonds")
        self.chk_dynamic_bonds.setToolTip(
            "Recalculate bonds at every frame (slower, but shows bond breaking/forming)"
        )
        self.chk_dynamic_bonds.setChecked(False)
        self.chk_dynamic_bonds.toggled.connect(
            lambda: self.update_viewer(self.frame_idx)
        )
        ctrl_layout.addWidget(self.chk_dynamic_bonds)

        layout.addLayout(ctrl_layout)

        # 3. Export
        export_layout = QHBoxLayout()
        export_layout.addStretch()

        self.btn_gif = QPushButton("Save GIF")
        self.btn_gif.clicked.connect(self.save_gif)
        self.btn_gif.setEnabled(
            HAS_PIL and len(self.trajectory) > 0 if self.trajectory else False
        )
        export_layout.addWidget(self.btn_gif)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        export_layout.addWidget(self.btn_close)

        layout.addLayout(export_layout)

        # Timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame_auto)

    def _energy_settings(self):
        """(unit, is_relative) as chosen in the controls."""
        unit = (
            self.unit_combo.currentText()
            if getattr(self, "unit_combo", None) is not None
            else "Hartree"
        )
        is_rel = (
            self.chk_relative.isChecked()
            if getattr(self, "chk_relative", None) is not None
            else False
        )
        return unit, is_rel

    def _to_display(self, energy_ha):
        """An energy (Hartree) in the chosen unit, relative to the scan's
        minimum when "Relative" is ticked. Used by the plot, the highlight
        marker and the hover text alike."""
        unit, is_rel = self._energy_settings()
        ref = min(r["energy"] for r in self.results) if is_rel else 0.0
        return (energy_ha - ref) * _UNIT_FACTORS.get(unit, 1.0)

    def plot_data(self):
        if not self.results:
            return

        unit, is_rel = self._energy_settings()
        x = [r["value"] for r in self.results]
        y = [self._to_display(r["energy"]) for r in self.results]
        prefix = "Relative Energy" if is_rel else "Energy"
        ylabel = f"{prefix} ({unit if unit in _UNIT_FACTORS else 'Hartree'})"

        self.canvas.axes.clear()
        self.canvas.axes.plot(x, y, "b-", label="Energy", picker=5)
        # Unconverged SCF points get a hollow marker: their energy can sit
        # far off the surface and must not read as a real feature.
        conv = [bool(r.get("converged", True)) for r in self.results]
        self.scatter = self.canvas.axes.scatter(
            x,
            y,
            c=["red" if ok else "none" for ok in conv],
            edgecolors="red",
            s=25,
            picker=5,
            zorder=5,
        )  # Use scatter for easier hover detection
        if not all(conv):
            self.canvas.axes.scatter(
                [], [], c="none", edgecolors="red", s=25, label="SCF not converged"
            )
            self.canvas.axes.legend(loc="best")

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
        if getattr(self, "frame_idx", None) is not None and self.frame_idx >= 0:
            self.highlight_point(self.frame_idx)

        self.canvas.draw()

    def highlight_point(self, idx):
        # Remove old highlight if exists
        self._remove_highlight()
        x = self.results[idx]["value"]
        y = self._to_display(self.results[idx]["energy"])

        # 1. Large distinct marker (Red circle)
        (self._highlight_marker,) = self.canvas.axes.plot(
            x,
            y,
            "ro",
            markersize=10,
            markeredgecolor="darkred",
            markeredgewidth=2,
            zorder=10,
        )

        # 2. Vertical Line
        self._highlight_line = self.canvas.axes.axvline(
            x=x, color="gray", linestyle="--", alpha=0.7, zorder=0
        )

        self.canvas.draw()

    def _remove_highlight(self):
        for attr in ("_highlight_marker", "_highlight_line"):
            artist = getattr(self, attr, None)
            if artist is not None:
                try:
                    artist.remove()
                except _ARTIST_GONE as exc:
                    logger.debug("%s already removed: %s", attr, exc)
                setattr(self, attr, None)

    def create_base_molecule(self):
        """Create a base molecule with topology from the first frame."""
        if not self.trajectory:
            return

        try:
            # Match AnimatedXYZPlayer logic more closely for reliable bond detection
            xyz_block = self.trajectory[0]
            lines = xyz_block.strip().split("\n")
            if len(lines) < 3:
                return

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
                        except (RuntimeError, ValueError):
                            atom = Chem.Atom("C")  # unknown symbol
                        mol.AddAtom(atom)
                    except ValueError:
                        continue

            # Add conformer
            conf = Chem.Conformer(mol.GetNumAtoms())
            for idx, (x, y, z) in enumerate(coords):
                conf.SetAtomPosition(idx, rdGeometry.Point3D(x, y, z))
            mol.AddConformer(conf)

            # Establish topology
            mw = self.context.get_main_window() if self.context else None

            # Use same logic as xyz_giffer: prefer main window's estimate_bonds_from_distances
            iom = getattr(mw, "io_manager", None)
            if iom and hasattr(iom, "estimate_bonds_from_distances"):
                try:
                    iom.estimate_bonds_from_distances(mol)
                except (RuntimeError, ValueError) as _e:
                    logger.warning("estimate_bonds_from_distances failed: %s", _e)

            # Also try rdDetermineBonds as a secondary supplement if 0 bonds were found
            if mol.GetNumBonds() == 0:
                try:
                    from rdkit.Chem import rdDetermineBonds

                    rdDetermineBonds.DetermineConnectivity(mol)
                    rdDetermineBonds.DetermineBondOrders(mol)
                except (RuntimeError, ValueError) as e:  # RDKit: ValueError subclasses
                    logger.warning("rdDetermineBonds fallback failed: %s", e)

            self.base_mol = mol.GetMol()

            # Set as current molecule in context
            if self.context:
                self.context.current_molecule = self.base_mol
                # context.current_molecule's setter only pushes the mol to
                # the 3D view; it does not touch the unsaved-changes flag
                # (see PluginContext.current_mol in the main app), so we
                # must mark the project modified ourselves — otherwise the
                # document's molecule is silently swapped for this
                # topology-reconstructed scan frame with no dirty indicator.
                self.context.mark_project_modified()

                # Initial draw
                self.context.draw_molecule_3d(self.base_mol)
                self.context.reset_3d_camera()
            if hasattr(mw, "view_3d_manager") and hasattr(
                mw.view_3d_manager, "plotter"
            ):
                mw.view_3d_manager.plotter.update()
                mw.view_3d_manager.plotter.render()
        # runs while the dialog is built: log, never abort the host (PyQt6)
        except Exception:
            logger.exception("Error creating base molecule")

    def on_pick(self, event):
        if event.artist and hasattr(event, "ind"):
            idx = event.ind[0]
            self.set_frame(idx)

    def on_hover(self, event):
        """Update and show tooltip on hover"""
        vis = self.annot.get_visible()
        if event.inaxes == self.canvas.axes and getattr(self, "scatter", None):
            cont, ind = self.scatter.contains(event)
            if cont:
                idx = ind["ind"][0]
                self.annot.xy = self.scatter.get_offsets()[idx]
                unit, _ = self._energy_settings()
                val = self.results[idx]["value"]
                disp_energy = self._to_display(self.results[idx]["energy"])
                # full precision: the tooltip is where exact values are read
                text = f"X: {val:.6f}\nY: {disp_energy:.8f} {unit}"
                if not self.results[idx].get("converged", True):
                    text += "\nSCF NOT CONVERGED"
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
        if not self.context or not self.trajectory:
            return

        xyz = self.trajectory[idx]

        # Check for Dynamic Bonds setting
        use_dynamic_bonds = (
            self.chk_dynamic_bonds.isChecked()
            if getattr(self, "chk_dynamic_bonds", None) is not None
            else False
        )

        if self.base_mol and not use_dynamic_bonds:
            # Efficient update: just change coordinates
            try:
                # Parse coordinates from XYZ string
                lines = xyz.strip().split("\n")
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
                self.context.draw_molecule_3d(self.base_mol)
            except (ValueError, IndexError, RuntimeError):
                # frame does not fit the base molecule: reload it fully
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
            self.timer.start(500)  # 500ms
            self.btn_play.setText("Pause")
            self.is_playing = True

    def save_plot(self):
        try:
            default_dir = self.scan_result_dir if self.scan_result_dir else ""
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Graph",
                default_dir,
                "Images (*.png *.jpg *.svg *.pdf);;All Files (*)",
            )
            if path:
                self.canvas.fig.savefig(path, dpi=300)
                QMessageBox.information(self, "Saved", f"Graph saved to:\n{path}")
        except (OSError, ValueError) as e:  # unwritable path / unknown format
            QMessageBox.critical(self, "Error", f"Failed to save graph: {e}")

    def save_csv(self):
        if not self.results:
            return
        try:
            default_dir = self.scan_result_dir if self.scan_result_dir else ""
            default_path = (
                os.path.join(default_dir, "scan_export.csv")
                if default_dir
                else "scan_export.csv"
            )

            path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV", default_path, "CSV Files (*.csv);;All Files (*)"
            )
            if path:
                keys = self.results[0].keys()
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(self.results)
                QMessageBox.information(self, "Saved", f"Results saved to:\n{path}")
        except (OSError, ValueError, csv.Error) as e:
            QMessageBox.critical(self, "Error", f"Failed to save CSV: {e}")

    def prev_frame(self):
        self.set_frame(self.frame_idx - 1)

    def clear_selection(self):
        """Remove highlight marker and line from the graph"""
        self._remove_highlight()

        self.canvas.draw()

    def next_frame(self):
        self.set_frame(self.frame_idx + 1)

    def next_frame_auto(self):
        next_idx = self.frame_idx + 1
        if next_idx >= len(self.trajectory):
            next_idx = 0  # Loop
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
            QMessageBox.warning(
                self,
                "Error",
                "PIL (Pillow) module is required for GIF export.\nPlease install it via: pip install Pillow",
            )
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

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
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
        default_name = (
            os.path.join(self.scan_result_dir, "scan_animation.gif")
            if self.scan_result_dir
            else "scan_animation.gif"
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save GIF", default_name, "GIF Files (*.gif)"
        )
        if not file_path:
            if was_playing:
                self.toggle_play()
            return

        if not file_path.lower().endswith(".gif"):
            file_path += ".gif"

        # Progress Dialog
        self.setCursor(Qt.CursorShape.WaitCursor)
        progress = QProgressDialog(
            "Generating GIF...", "Cancel", 0, len(self.trajectory), self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)

        try:
            original_frame_idx = self.frame_idx
            images = []

            mw = self.context.get_main_window() if self.context else None
            if not mw or not (
                hasattr(mw, "view_3d_manager")
                and hasattr(mw.view_3d_manager, "plotter")
            ):
                raise RuntimeError("3D plotter not available")

            for i in range(len(self.trajectory)):
                if progress.wasCanceled():
                    break

                self.set_frame(i)
                QApplication.processEvents()  # Process events to ensure viewer updates

                # Force update/render as in xyz_giffer
                mw.view_3d_manager.plotter.update()
                mw.view_3d_manager.plotter.render()

                # Capture screenshot
                img_array = mw.view_3d_manager.plotter.screenshot(
                    transparent_background=use_transparent, return_img=True
                )

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

        # button slot: PIL / VTK screenshot errors of any kind are reported
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Error", f"Failed to save GIF:\n{e}")
        finally:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            progress.close()
            if was_playing:
                self.toggle_play()
