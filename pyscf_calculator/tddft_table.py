from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, 
    QHeaderView, QPushButton, QHBoxLayout, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt
import csv

class TddftTable(QDialog):
    def __init__(self, parent=None, results=None):
        super().__init__(parent)
        self.setWindowTitle("TDDFT Results")
        self.resize(600, 400)
        self.results = results # List of dicts
        
        # Modeless dialog needs to delete on close? 
        # Or kept alive by parent? parent keeps reference.
        # Ensure it doesn't block.
        self.setModal(False)
        
        self.init_ui()
        self.populate()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "State", "Energy (eV)", "Wavelength (nm)", "Osc. Strength", "Total E (Ha)"
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("Save CSV")
        self.btn_save.clicked.connect(self.save_csv)
        btn_layout.addWidget(self.btn_save)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        btn_layout.addWidget(self.btn_close)
        
        layout.addLayout(btn_layout)

    def populate(self):
        if not self.results: return
        
        self.table.setRowCount(len(self.results))
        for i, row in enumerate(self.results):
            # Items
            # state, excitation_energy_ev, wavelength_nm, oscillator_strength, energy_total
            state_item = QTableWidgetItem(str(row.get('state', i+1)))
            state_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            ev_val = row.get('excitation_energy_ev', 0.0)
            ev_item = QTableWidgetItem(f"{ev_val:.4f}")
            ev_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
            nm_val = row.get('wavelength_nm', 0.0)
            nm_txt = f"{nm_val:.2f}" if nm_val != float('inf') else "inf"
            nm_item = QTableWidgetItem(nm_txt)
            nm_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
            osc_val = row.get('oscillator_strength', 0.0)
            osc_item = QTableWidgetItem(f"{osc_val:.4f}")
            osc_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            tot_val = row.get('energy_total', 0.0)
            tot_item = QTableWidgetItem(f"{tot_val:.6f}")
            tot_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            self.table.setItem(i, 0, state_item)
            self.table.setItem(i, 1, ev_item)
            self.table.setItem(i, 2, nm_item)
            self.table.setItem(i, 3, osc_item)
            self.table.setItem(i, 4, tot_item)

    def save_csv(self):
        if not self.results: return
        try:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save TDDFT CSV", "tddft_results.csv",
                "CSV Files (*.csv);;All Files (*)"
            )
            if path:
                with open(path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "state", "excitation_energy_ev", "wavelength_nm", 
                        "oscillator_strength", "energy_total"
                    ])
                    writer.writeheader()
                    writer.writerows(self.results)
                QMessageBox.information(self, "Saved", f"Saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")
