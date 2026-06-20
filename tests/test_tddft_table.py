"""
tests/test_tddft_table.py
Unit tests for data formatting in the TDDFT result widget via headless stubs.
"""

import os
import sys
import types
import unittest
import importlib.util
from unittest.mock import MagicMock, patch, mock_open


def _load_module_direct(relpath, module_name):
    src = os.path.join(os.path.dirname(__file__), "..", relpath)
    src = os.path.normpath(src)
    spec = importlib.util.spec_from_file_location(module_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_stubs():
    qt_core = types.ModuleType("PyQt6.QtCore")

    class _Qt:
        class AlignmentFlag:
            AlignCenter = 0
            AlignRight = 1
            AlignVCenter = 2

    qt_core.Qt = _Qt
    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core
    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtCore"] = qt_core

    class _QDialog:
        def __init__(self, parent=None):
            pass

        def setWindowTitle(self, *a):
            pass

        def resize(self, *a):
            pass

        def setModal(self, *a):
            pass

        def close(self):
            pass

    class _QWidget:
        pass

    class _QHeaderView:
        class ResizeMode:
            Stretch = 1

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QDialog = _QDialog
    qt_widgets.QWidget = _QWidget
    qt_widgets.QHeaderView = _QHeaderView

    for name in [
        "QVBoxLayout",
        "QHBoxLayout",
        "QTableWidget",
        "QPushButton",
        "QMessageBox",
        "QFileDialog",
    ]:
        setattr(qt_widgets, name, MagicMock)

    qt_widgets.QTableWidgetItem = lambda *a, **k: MagicMock()

    pyqt6.QtWidgets = qt_widgets
    sys.modules["PyQt6.QtWidgets"] = qt_widgets


_install_stubs()

tddft_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "tddft_table.py"),
    "pyscf_calculator_tddft_table_under_test",
)
TddftTable = tddft_mod.TddftTable


class TestTddftTable(unittest.TestCase):
    def setUp(self):
        self.mock_results = [
            {
                "state": 1,
                "excitation_energy_ev": 4.5678,
                "wavelength_nm": 271.3,
                "oscillator_strength": 0.015,
                "energy_total": -76.4321,
            },
            {
                "state": 2,
                "excitation_energy_ev": 5.0,
                "wavelength_nm": float("inf"),
                "oscillator_strength": 0.0,
                "energy_total": -76.0,
            },
        ]

    def test_populate_scales_and_formats(self):
        # We check that table items were created with correct numeric formats
        with patch.object(tddft_mod, "QTableWidgetItem") as MockItem:
            table = TddftTable.__new__(TddftTable)
            table.results = self.mock_results
            table.table = MagicMock()

            table.populate()

            # Look at calls to QTableWidgetItem
            calls = MockItem.call_args_list
            args = [c[0][0] for c in calls if c[0]]

            self.assertIn("4.5678", args)
            self.assertIn("271.30", args)
            self.assertIn("0.0150", args)
            self.assertIn("-76.432100", args)

            # Wavelength parsing of infinity
            self.assertIn("inf", args)

    @patch("builtins.open", new_callable=mock_open)
    def test_save_csv(self, mock_file):
        table = TddftTable.__new__(TddftTable)
        table.results = self.mock_results

        orig_file = getattr(tddft_mod.QFileDialog, "getSaveFileName", None)
        orig_msg = getattr(tddft_mod.QMessageBox, "information", None)

        tddft_mod.QFileDialog.getSaveFileName = MagicMock(
            return_value=("/fake/path.csv", "")
        )
        tddft_mod.QMessageBox.information = MagicMock()

        try:
            table.save_csv()
        finally:
            if orig_file is not None:
                tddft_mod.QFileDialog.getSaveFileName = orig_file
            if orig_msg is not None:
                tddft_mod.QMessageBox.information = orig_msg

        mock_file.assert_called_with("/fake/path.csv", "w", newline="")
        # CSV content includes headers and data
        handle = mock_file()
        handle.write.assert_any_call(
            "state,excitation_energy_ev,wavelength_nm,oscillator_strength,energy_total\r\n"
        )
        # Check that it tried to write row 1
        handle.write.assert_any_call("1,4.5678,271.3,0.015,-76.4321\r\n")

    def test_populate_empty(self):
        table = TddftTable.__new__(TddftTable)
        table.results = []
        table.table = MagicMock()

        table.populate()
        table.table.setRowCount.assert_not_called()

    def test_save_csv_empty(self):
        table = TddftTable.__new__(TddftTable)
        table.results = []
        # should return implicitly without doing anything
        table.save_csv()

    @patch("builtins.open")
    def test_save_csv_exception(self, mock_open_func):
        table = TddftTable.__new__(TddftTable)
        table.results = self.mock_results

        tddft_mod.QFileDialog.getSaveFileName = MagicMock(
            return_value=("/fake/path.csv", "")
        )
        tddft_mod.QMessageBox.critical = MagicMock()

        # Make opening file raise Exception
        mock_open_func.side_effect = PermissionError("Cannot open file")

        table.save_csv()
        tddft_mod.QMessageBox.critical.assert_called_once()

    @patch.object(tddft_mod, "QTableWidget")
    @patch.object(tddft_mod, "QVBoxLayout")
    @patch.object(tddft_mod, "QHBoxLayout")
    @patch.object(tddft_mod, "QPushButton")
    def test_init_ui(self, mock_btn, mock_hbox, mock_vbox, mock_table):
        table = TddftTable.__new__(TddftTable)
        tddft_mod.TddftTable.__init__(table, results=self.mock_results)

        self.assertEqual(table.results, self.mock_results)
        mock_table.return_value.setColumnCount.assert_called_with(5)


if __name__ == "__main__":
    unittest.main()
