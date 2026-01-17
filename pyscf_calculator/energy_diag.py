import os
import glob
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QFileDialog, QMessageBox, QMenu, QApplication,
    QToolTip
)
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QAction

try:
    import nist
except ImportError:
    nist = None

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
                        if label: tip_text += f"\\n{label}"
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
                
            if hasattr(self, 'btn_save'):
                self.btn_save.setVisible(False)
                widgets_to_restore.append(self.btn_save)
                
            if hasattr(self, 'status_label'):
                self.status_label.setVisible(False)
                widgets_to_restore.append(self.status_label)
            
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
                 label = f"{current_tick_disp:.2f}"
                 
                 # Right align text to x=50, Vertically Center
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
        left_margin = 120 
        right_margin = 10 # Only for window edge, but column width absorbs text space
        avail_w = w - left_margin - right_margin
        
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
            if not self.is_uhf:
                center_of_window = total_w / 2
                line_center_x = center_of_window
                
                # Reverse calculate col_start
                raw_col_start = line_center_x - (level_w / 2) - padding_left
                
                # CLAMP against left_margin to prevent clipping of energy labels
                col_start = max(raw_col_start, left_margin)
            else:
                # UKS Logic: Use calculated column widths respecting margins
                u_col_width = avail_w / cols 
                col_start = left_margin + col_idx * u_col_width

            target_x1 = col_start + padding_left
            center_x = target_x1 + level_w/2
            
            painter.setPen(QColor("black"))
            
            # Title with Electron Count
            n_elec = sum(occs)
            title_text = f"{title}\\n({n_elec:.0f}e)"
            
            fm = painter.fontMetrics()
            lines = title_text.split("\\n")
            y_title_base = 20
            
            for line in lines:
                t_w = fm.horizontalAdvance(line)
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
            if is_alpha_col:
                col_occ = QColor(180, 50, 50) 
                col_vir = QColor(180, 50, 50) 
            elif is_beta_col:
                col_occ = QColor(50, 50, 180)
                col_vir = QColor(50, 50, 180)
            else:
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
                        f_icon = QFont("Arial", 16, QFont.Weight.Bold)
                        painter.setFont(f_icon)
                        
                        arrow_txt = ""
                        if not self.is_uhf:
                            if abs(occ_val - 1.0) < 0.1:
                                arrow_txt = "↑" # Singly occupied (ROKS)
                            else:
                                arrow_txt = "↑↓" # Doubly occupied
                        else:
                            if is_alpha_col:
                                arrow_txt = "↑"
                            elif is_beta_col:
                                arrow_txt = "↓"
                        
                        rect_icon = QRect(int(x1), int(y)-14, int(level_w), 28)
                        painter.drawText(rect_icon, Qt.AlignmentFlag.AlignCenter, arrow_txt)
                        
                        painter.setFont(font)

                    # Label Logic
                    label_txt = ""

                    # 1. ROKS SOMO Special Case
                    is_roks_somo = (not self.is_uhf) and (abs(occ_val - 1.0) < 0.1)

                    if is_roks_somo:
                        label_txt = "SOMO"
                    elif i_orig <= homo_idx:
                        diff = homo_idx - i_orig
                        if diff == 0:
                            label_txt = "HOMO"
                        else:
                            label_txt = f"HOMO-{diff}"
                    else:
                        diff = i_orig - (homo_idx + 1)
                        label_txt = "LUMO" if diff == 0 else f"LUMO+{diff}"
                    
                    if label_txt:
                        painter.setPen(QColor("black"))
                        painter.drawText(int(x2)+4, int(y)+4, label_txt)
                        
                    # Energy Values
                    vis_e_str = f"{e * factor:.2f} {unit_label_str}"
                    painter.setPen(QColor("black"))
                    rect_e = QRect(int(x1)-85, int(y)-7, 80, 14) 
                    painter.drawText(rect_e, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, vis_e_str)
                    
                    # Store Hit Zone
                    r = QRect(int(x1), int(y)-7, int(level_w + 80), 14)
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
