"""
tests/test_calc_tab.py
Unit tests for the main calculation tab configuration building without requiring UI interactions.
"""

import os
import sys
import types
import unittest
import importlib.util
from unittest.mock import MagicMock


def _load_module_direct(relpath, module_name):
    """Load a .py file as a module without going through the package __init__."""
    src = os.path.join(os.path.dirname(__file__), "..", relpath)
    src = os.path.normpath(src)
    spec = importlib.util.spec_from_file_location(module_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_stubs():
    # PyQt6.QtCore stubs
    qt_core = types.ModuleType("PyQt6.QtCore")

    class _QThread:
        def __init__(self):
            pass

        def start(self):
            pass

        def isRunning(self):
            return False

        def wait(self, ms=0):
            return True

        def terminate(self):
            pass

        @staticmethod
        def msleep(ms):
            pass

    qt_core.QThread = _QThread
    qt_core.pyqtSignal = lambda *a, **kw: MagicMock()

    class _Qt:
        class CursorShape:
            PointingHandCursor = None

        class AlignmentFlag:
            AlignRight = None

        class Orientation:
            Horizontal = None

    qt_core.Qt = _Qt

    class _QTimer:
        @staticmethod
        def singleShot(ms, fn):
            pass

    qt_core.QTimer = _QTimer

    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core
    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtCore"] = qt_core

    class _QDialog:
        def __init__(self, *args, **kwargs):
            pass

        def setWindowTitle(self, *a):
            pass

        def resize(self, *a):
            pass

    class _QWidget:
        def __init__(self, *args, **kwargs):
            pass

    # PyQt6.QtWidgets stubs
    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QWidget = _QWidget
    qt_widgets.QDialog = _QDialog

    for name in [
        "QVBoxLayout",
        "QHBoxLayout",
        "QLabel",
        "QComboBox",
        "QPushButton",
        "QSpinBox",
        "QCheckBox",
        "QGroupBox",
        "QFormLayout",
        "QMessageBox",
        "QLineEdit",
        "QFileDialog",
        "QProgressBar",
        "QTextEdit",
        "QSizePolicy",
        "QScrollArea",
        "QFrame",
        "QTabWidget",
        "QToolTip",
    ]:
        setattr(qt_widgets, name, MagicMock)
    pyqt6.QtWidgets = qt_widgets
    sys.modules["PyQt6.QtWidgets"] = qt_widgets

    # rdkit stubs
    rdkit = types.ModuleType("rdkit")
    rdkit_chem = types.ModuleType("rdkit.Chem")
    rdkit_chem.Chem = MagicMock()
    rdkit_chem.GetFormalCharge = MagicMock(return_value=0)
    rdkit.Chem = rdkit_chem
    sys.modules["rdkit"] = rdkit
    sys.modules["rdkit.Chem"] = rdkit_chem
    sys.modules["rdkit.Chem.rdMolTransforms"] = MagicMock()
    sys.modules["pyscf"] = None  # mock missing PySCF cleanly


_install_stubs()

# Now load calc_tab directly
_calc_tab_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "calc_tab.py"),
    "pyscf_calculator_calc_tab_under_test",
)
CalcTab = _calc_tab_mod.CalcTab


class TestCalcTabConfig(unittest.TestCase):
    def setUp(self):
        self.tab = CalcTab.__new__(CalcTab)
        self.tab.scan_params = None

        # Setup standard mocked UI components for configuration
        self.tab.job_type_combo = MagicMock()
        self.tab.method_combo = MagicMock()
        self.tab.functional_combo = MagicMock()
        self.tab.basis_combo = MagicMock()
        self.tab.charge_input = MagicMock()
        self.tab.spin_input = MagicMock()
        self.tab.nstates_input = MagicMock()
        self.tab.out_dir_edit = MagicMock()

        self.tab.spin_memory = MagicMock()
        self.tab.spin_threads = MagicMock()
        self.tab.check_symmetry = MagicMock()
        self.tab.check_break_sym = MagicMock()
        self.tab.spin_grid_level = MagicMock()
        self.tab.edit_conv = MagicMock()
        self.tab.spin_cycles = MagicMock()
        self.tab.solvent_combo = MagicMock()

        self.tab.parent_dialog = MagicMock()
        self.tab.parent_dialog.btn_load_geom = MagicMock()

        self.tab.progress_bar = MagicMock()
        self.tab.log_text = MagicMock()
        self.tab.run_btn = MagicMock()
        self.tab.stop_btn = MagicMock()

        self.tab.context = MagicMock()
        self.tab.context.current_molecule = "mock123"

    def _run_calc_and_get_config(self, **kwargs):
        self.tab.job_type_combo.currentText.return_value = kwargs.get(
            "job", "Optimization"
        )
        self.tab.method_combo.currentText.return_value = kwargs.get("method", "RKS")
        self.tab.functional_combo.currentText.return_value = kwargs.get("func", "b3lyp")
        self.tab.basis_combo.currentText.return_value = kwargs.get("basis", "sto-3g")
        self.tab.charge_input.currentText.return_value = kwargs.get("charge", "0")
        self.tab.spin_input.currentText.return_value = kwargs.get("spin", "0")
        self.tab.out_dir_edit.text.return_value = "/mock/dir"

        self.tab.nstates_input.value.return_value = kwargs.get("nstates", 3)
        self.tab.spin_threads.value.return_value = kwargs.get("threads", 4)
        self.tab.spin_memory.value.return_value = kwargs.get("memory", 4000)
        self.tab.check_symmetry.isChecked.return_value = kwargs.get("symm", False)
        self.tab.check_break_sym.isChecked.return_value = kwargs.get(
            "break_symm", False
        )
        self.tab.spin_grid_level.value.return_value = kwargs.get("grid_level", 3)
        self.tab.edit_conv.text.return_value = kwargs.get("conv", "1e-9")
        self.tab.spin_cycles.value.return_value = kwargs.get("cycles", 100)
        self.tab.solvent_combo.currentText.return_value = kwargs.get(
            "solvent", "None (Vacuum)"
        )

        # Inject PySCFWorker explicitly so calc_tab doesn't think it's None due to the missing pyscf stub
        MockWorker = MagicMock()
        _calc_tab_mod.PySCFWorker = MockWorker
        _calc_tab_mod.rdkit_to_xyz = lambda m: "H 0 0 0"
        _calc_tab_mod.os = MagicMock()
        _calc_tab_mod.os.path = os.path  # Keep path functions working
        _calc_tab_mod.os.path.isabs = lambda p: True

        self.tab.run_calculation()

        if MockWorker.called:
            args, _ = MockWorker.call_args
            xyz_str, config = args
            return config
        return None

    def test_build_config_standard_dft(self):
        config = self._run_calc_and_get_config()

        self.assertEqual(config["method"], "RKS")
        self.assertEqual(config["functional"], "b3lyp")
        self.assertEqual(config["basis"], "sto-3g")
        self.assertEqual(config["charge"], 0)
        self.assertEqual(config["spin"], 0)
        self.assertEqual(config["nstates"], 3)
        self.assertEqual(config["threads"], 4)
        self.assertEqual(config["memory"], 4000)
        self.assertEqual(config["symmetry"], False)
        self.assertEqual(config["grid_level"], 3)
        self.assertEqual(config["conv_tol"], "1e-9")
        self.assertEqual(config["max_cycle"], 100)
        self.assertEqual(config["solvent"], "None (Vacuum)")
        self.assertEqual(config["job_type"], "Optimization")

    def test_build_config_solvent(self):
        config = self._run_calc_and_get_config(solvent="water")
        self.assertEqual(config["solvent"], "water")

    def test_build_config_scan_job_includes_params(self):
        self.tab.scan_params = {"steps": 10, "start": 1.0, "end": 2.0}
        config = self._run_calc_and_get_config(job="Rigid Scan")

        self.assertEqual(config["job_type"], "Rigid Scan")
        self.assertIn("scan_params", config)
        self.assertEqual(config["scan_params"]["steps"], 10)


if __name__ == "__main__":
    unittest.main()
