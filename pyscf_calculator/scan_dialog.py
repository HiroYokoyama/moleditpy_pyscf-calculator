import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QMessageBox, QGroupBox, QFormLayout, QTableWidget, 
    QTableWidgetItem, QAbstractItemView, QHeaderView
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

class ScanDialog(QDialog):
    scan_configured = pyqtSignal(dict)

    def __init__(self, parent=None, context=None, initial_params=None):
        super().__init__(parent)
        self.context = context
        self.mw = context.get_main_window() if context else None
        
        # Scan Parameters
        self.selected_atoms = []
        self.scan_params = {} # {type, atoms, start, end, steps}
        
        self.setWindowTitle("Surface Scan Setup")
        self.resize(650, 350)
        self.init_ui()
        
        # Restore saved params if available
        if initial_params:
            if 'atoms' in initial_params:
                self.selected_atoms = initial_params['atoms']
                # Try to restore visual selection in viewer
                try:
                    if self.mw:
                        # Restore Unordered Set (for visual highlighting)
                        if hasattr(self.mw, 'selected_atoms_3d'):
                            self.mw.selected_atoms_3d = set(self.selected_atoms)
                        
                        # Restore ORDERED List (crucial for Angle/Dihedral definition)
                        if hasattr(self.mw, 'selected_atoms_for_measurement'):
                            self.mw.selected_atoms_for_measurement = list(self.selected_atoms)

                        if hasattr(self.mw, 'gl_widget'):
                            self.mw.gl_widget.update()
                except: pass
            
            # Update UI state first (calculates current value)
            self.update_ui_state()
            
            # Then overwrite with saved values if present
            if 'start' in initial_params: self.edit_start.setText(str(initial_params['start']))
            if 'end' in initial_params: self.edit_end.setText(str(initial_params['end']))
            if 'steps' in initial_params: self.edit_steps.setText(str(initial_params['steps']))

        # Auto-update timer for selection
        self.sel_timer = QTimer(self)
        self.sel_timer.timeout.connect(self._auto_update_selection)
        self.sel_timer.start(200)
        
        # Auto-activate Selection Mode (Measurement Mode)
        try:
            if self.mw and hasattr(self.mw, 'toggle_measurement_mode'):
                # Check current state
                self.was_measurement_active = getattr(self.mw, 'measurement_mode', False)
                if not self.was_measurement_active:
                    self.mw.toggle_measurement_mode(True)
                    if hasattr(self.mw, 'measurement_action'):
                        self.mw.measurement_action.setChecked(True)
        except Exception as e:
            print(f"Failed to activate selection mode: {e}")

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Instruction
        lbl_info = QLabel("Select 2, 3, or 4 atoms in the viewer to define the scan coordinate.")
        lbl_info.setWordWrap(True)
        layout.addWidget(lbl_info)

        # 2. Selection Display
        self.lbl_selection = QLabel("Selected: None")
        self.lbl_selection.setStyleSheet("font-weight: bold; color: #333;")
        layout.addWidget(self.lbl_selection)

        # 3. Parameter Group
        self.grp_params = QGroupBox("Scan Parameters")
        form = QFormLayout()

        self.lbl_type = QLabel("Type: -")
        form.addRow("Scan Type:", self.lbl_type)

        self.lbl_current = QLabel("Current Value: -")
        form.addRow("Current Value:", self.lbl_current)

        self.edit_start = QLineEdit()
        self.edit_start.setPlaceholderText("Start value")
        form.addRow("Start:", self.edit_start)

        self.edit_end = QLineEdit()
        self.edit_end.setPlaceholderText("End value")
        form.addRow("End:", self.edit_end)

        self.edit_steps = QLineEdit("10")
        form.addRow("Steps:", self.edit_steps)

        self.grp_params.setLayout(form)
        layout.addWidget(self.grp_params)
        
        # Disable initially
        self.grp_params.setEnabled(False)

        # 4. Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.btn_ok = QPushButton("OK")
        self.btn_ok.clicked.connect(self.accept_scan)
        self.btn_ok.setEnabled(False)
        btn_layout.addWidget(self.btn_ok)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        layout.addLayout(btn_layout)

    def _auto_update_selection(self):
        """Check main window selection and update state."""
        if not self.mw: return

        # Get Selection (Logic adapted from atom_colorizer)
        new_selection = []
        
        # 1. Check measurement selection (First Priority - Preserves Order)
        if hasattr(self.mw, 'selected_atoms_for_measurement') and self.mw.selected_atoms_for_measurement:
             # Typically list of ints in click order
             new_selection = [x for x in self.mw.selected_atoms_for_measurement if isinstance(x, int)]

        # 2. Check direct 3D selection (Fallback - Unordered Set)
        elif hasattr(self.mw, 'selected_atoms_3d') and self.mw.selected_atoms_3d:
            # Note: Set is unordered. We can't guarantee A-B-C order. 
            # But sorting breaks geometry too. Better to rely on Measurement Mode.
            new_selection = list(self.mw.selected_atoms_3d)

        # Limit to 4 atoms
        if len(new_selection) > 4:
            new_selection = new_selection[:4]
            
        # Update if changed
        if new_selection != self.selected_atoms:
            self.selected_atoms = new_selection
            self.update_ui_state()

    def update_ui_state(self):
        n = len(self.selected_atoms)
        self.lbl_selection.setText(f"Selected: {self.selected_atoms}")
        
        if n in [2, 3, 4]:
            self.grp_params.setEnabled(True)
            self.btn_ok.setEnabled(True)
            self.calculate_current_value()
        else:
            self.grp_params.setEnabled(False)
            self.btn_ok.setEnabled(False)
            self.lbl_type.setText("Type: -")
            self.lbl_current.setText("Current Value: -")

    def calculate_current_value(self):
        if not self.context or not self.context.current_molecule:
            return

        mol = self.context.current_molecule
        conf = mol.GetConformer()
        
        picked = self.selected_atoms
        val = 0.0
        
        try:
            if len(picked) == 2:
                self.scan_type = "Dist"
                p1 = conf.GetAtomPosition(picked[0])
                p2 = conf.GetAtomPosition(picked[1])
                val = (p1 - p2).Length() # RDKit Point3D supports subtraction -> Point3D, check Length()
                # Correct RDKit Point3D usage if needed:
                # val = p1.Distance(p2) # Usually available
                
            elif len(picked) == 3:
                self.scan_type = "Angle"
                val = rdMolTransforms.GetAngleDeg(conf, picked[0], picked[1], picked[2])
                
            elif len(picked) == 4:
                self.scan_type = "Dihedral"
                val = rdMolTransforms.GetDihedralDeg(conf, picked[0], picked[1], picked[2], picked[3])
            
            # Format
            self.lbl_type.setText(f"Type: {self.scan_type}")
            self.lbl_current.setText(f"Current Value: {val:.3f}")
            
            # Always update Start value when selection changes (current geometry)
            self.edit_start.setText(f"{val:.3f}")
            # Auto-fill End if empty? Maybe not, leave it to user
            
        except Exception as e:
            self.lbl_current.setText("Error calc value")
            print(f"ScanDialog Calc Error: {e}")

    def accept_scan(self):
        try:
            start = float(self.edit_start.text())
            end = float(self.edit_end.text())
            steps = int(self.edit_steps.text())
            
            if steps < 2:
                QMessageBox.warning(self, "Invalid Input", "Steps must be >= 2.")
                return

            self.scan_params = {
                "type": self.scan_type,
                "atoms": self.selected_atoms,
                "start": start,
                "end": end,
                "end": end,
                "steps": steps
            }
            self.scan_configured.emit(self.scan_params)
            
            # Deactivate Selection Mode
            try:
                if self.mw and hasattr(self.mw, 'toggle_measurement_mode'):
                    # User requested exit from select mode
                    self.mw.toggle_measurement_mode(False)
                    if hasattr(self.mw, 'measurement_action'):
                        self.mw.measurement_action.setChecked(False)
            except:
                pass
                
            self.accept()
            
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values.")
