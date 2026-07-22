"""
tests/test_calc_tab_coverage.py

Additional coverage for pyscf_calculator/calc_tab.py: setup_ui(), update_options(),
auto_detect_charge_spin(), validate_spin_settings(), browse_out_dir(),
configure_scan()/on_scan_configured(), get_spin_value(), run_calculation() edge
branches, stop_calculation()/_on_worker_stopped(), log()/log_append(),
on_finished()/on_error()/cleanup_ui_state().

Not covered here: the "happy path" run_calculation() config-building already
exercised by tests/test_calc_tab.py (not duplicated).
"""

import os
import sys
import types
import unittest
import importlib.util
from unittest.mock import MagicMock, patch


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

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QWidget = _QWidget
    qt_widgets.QDialog = _QDialog

    # Plain factory (not the MagicMock class itself): calling e.g.
    # QPushButton("Configure Scan") on the bare MagicMock class would bind the
    # string positional arg to MagicMock's own `spec` kwarg, producing a
    # str-spec'd mock that lacks attributes like `.clicked`.
    def _widget_factory(*args, **kwargs):
        return MagicMock()

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
        setattr(qt_widgets, name, _widget_factory)
    pyqt6.QtWidgets = qt_widgets
    sys.modules["PyQt6.QtWidgets"] = qt_widgets

    rdkit = types.ModuleType("rdkit")
    rdkit_chem = types.ModuleType("rdkit.Chem")
    rdkit_chem.Chem = MagicMock()
    rdkit_chem.GetFormalCharge = MagicMock(return_value=0)
    rdkit.Chem = rdkit_chem
    sys.modules["rdkit"] = rdkit
    sys.modules["rdkit.Chem"] = rdkit_chem
    sys.modules["rdkit.Chem.rdMolTransforms"] = MagicMock()
    sys.modules["pyscf"] = None


_install_stubs()

_calc_tab_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "calc_tab.py"),
    "pyscf_calculator_calc_tab_coverage_under_test",
)
CalcTab = _calc_tab_mod.CalcTab


def _mock_message_box():
    """QMessageBox is a bare `MagicMock` *class* in the stub; class-attribute
    access like `QMessageBox.warning(...)` raises AttributeError on the class
    object itself. Replace it with an instance so those calls resolve."""
    mb = MagicMock()
    mb.StandardButton.Yes = 1
    mb.StandardButton.No = 2
    return mb


class TestSetupUI(unittest.TestCase):
    def test_setup_ui_builds_widgets_no_molecule(self):
        parent = MagicMock()
        context = MagicMock()
        context.current_molecule = None
        tab = CalcTab(parent, context, {})

        self.assertIsNotNone(tab.job_type_combo)
        self.assertIsNotNone(tab.method_combo)
        self.assertIsNotNone(tab.functional_combo)
        self.assertIsNotNone(tab.basis_combo)
        self.assertIsNotNone(tab.solvent_combo)
        self.assertIsNotNone(tab.charge_input)
        self.assertIsNotNone(tab.spin_input)
        self.assertIsNotNone(tab.out_dir_edit)
        self.assertIsNotNone(tab.run_btn)
        self.assertIsNotNone(tab.stop_btn)

    def test_setup_ui_schedules_auto_detect_when_molecule_present(self):
        parent = MagicMock()
        context = MagicMock()
        context.current_molecule = "some_mol"
        with patch.object(_calc_tab_mod.QTimer, "singleShot") as mock_single_shot:
            tab = CalcTab(parent, context, {})
        mock_single_shot.assert_called_once()
        self.assertEqual(mock_single_shot.call_args[0][1], tab.auto_detect_charge_spin)


class _BaseTabTest(unittest.TestCase):
    def setUp(self):
        self.tab = CalcTab.__new__(CalcTab)
        self.tab.scan_params = None

        self.tab.job_type_combo = MagicMock()
        self.tab.method_combo = MagicMock()
        self.tab.functional_combo = MagicMock()
        self.tab.basis_combo = MagicMock()
        self.tab.charge_input = MagicMock()
        self.tab.spin_input = MagicMock()
        self.tab.nstates_input = MagicMock()
        self.tab.lbl_nstates = MagicMock()
        self.tab.out_dir_edit = MagicMock()
        self.tab.btn_scan_config = MagicMock()
        self.tab.check_break_sym = MagicMock()

        self.tab.spin_memory = MagicMock()
        self.tab.spin_threads = MagicMock()
        self.tab.check_symmetry = MagicMock()
        self.tab.spin_grid_level = MagicMock()
        self.tab.edit_conv = MagicMock()
        self.tab.spin_cycles = MagicMock()
        self.tab.solvent_combo = MagicMock()

        self.tab.parent_dialog = MagicMock()
        self.tab.parent_dialog.btn_load_geom = MagicMock()
        self.tab.parent_dialog.version = "1.2.3"

        self.tab.progress_bar = MagicMock()
        self.tab.log_text = MagicMock()
        self.tab.run_btn = MagicMock()
        self.tab.stop_btn = MagicMock()

        self.tab.context = MagicMock()
        self.tab.context.current_molecule = "mock123"

        self._mb = _mock_message_box()
        _calc_tab_mod.QMessageBox = self._mb


class TestUpdateOptions(_BaseTabTest):
    def test_dft_method_enables_functional(self):
        self.tab.method_combo.currentText.return_value = "UKS"
        self.tab.job_type_combo.currentText.return_value = "Energy"
        self.tab.update_options()
        self.tab.functional_combo.setEnabled.assert_called_with(True)
        self.tab.check_break_sym.setEnabled.assert_called_with(True)

    def test_hf_method_disables_functional_and_break_sym(self):
        self.tab.method_combo.currentText.return_value = "RHF"
        self.tab.job_type_combo.currentText.return_value = "Energy"
        self.tab.update_options()
        self.tab.functional_combo.setEnabled.assert_called_with(False)
        self.tab.check_break_sym.setEnabled.assert_called_with(False)

    def test_scan_job_shows_scan_button(self):
        self.tab.method_combo.currentText.return_value = "RKS"
        self.tab.job_type_combo.currentText.return_value = "Rigid Surface Scan"
        self.tab.update_options()
        self.tab.btn_scan_config.show.assert_called_once()

    def test_non_scan_job_hides_scan_button(self):
        self.tab.method_combo.currentText.return_value = "RKS"
        self.tab.job_type_combo.currentText.return_value = "Energy"
        self.tab.update_options()
        self.tab.btn_scan_config.hide.assert_called_once()

    def test_tddft_job_shows_nstates(self):
        self.tab.method_combo.currentText.return_value = "RKS"
        self.tab.job_type_combo.currentText.return_value = "TDDFT"
        self.tab.update_options()
        self.tab.lbl_nstates.setVisible.assert_called_with(True)
        self.tab.nstates_input.setVisible.assert_called_with(True)

    def test_non_tddft_job_hides_nstates(self):
        self.tab.method_combo.currentText.return_value = "RKS"
        self.tab.job_type_combo.currentText.return_value = "Energy"
        self.tab.update_options()
        self.tab.lbl_nstates.setVisible.assert_called_with(False)
        self.tab.nstates_input.setVisible.assert_called_with(False)


class TestAutoDetectChargeSpin(_BaseTabTest):
    def _make_atom(self, atomic_num):
        atom = MagicMock()
        atom.GetAtomicNum.return_value = atomic_num
        return atom

    def test_no_molecule_warns(self):
        self.tab.context.current_molecule = None
        self.tab.auto_detect_charge_spin()
        self._mb.warning.assert_called_once()

    def test_no_context_warns(self):
        self.tab.context = None
        self.tab.auto_detect_charge_spin()
        self._mb.warning.assert_called_once()

    def test_even_electrons_singlet_switches_u_to_r(self):
        mol = MagicMock()
        mol.GetAtoms.return_value = [self._make_atom(1), self._make_atom(1)]
        self.tab.context.current_molecule = mol
        _calc_tab_mod.Chem.GetFormalCharge = MagicMock(return_value=0)
        self.tab.charge_input.findText.return_value = -1
        self.tab.spin_input.count.return_value = 6
        self.tab.spin_input.itemText.side_effect = [
            "1 (Singlet)",
            "2 (Doublet)",
            "3 (Triplet)",
            "4 (Quartet)",
            "5 (Quintet)",
            "6 (Sextet)",
        ]
        self.tab.method_combo.currentText.return_value = "UHF"

        self.tab.auto_detect_charge_spin()

        self.tab.charge_input.setCurrentText.assert_called_with("0")
        self.tab.spin_input.setCurrentIndex.assert_called_with(0)
        self.tab.method_combo.setCurrentText.assert_called_with("RHF")

    def test_even_electrons_singlet_switches_uks_to_rks(self):
        mol = MagicMock()
        mol.GetAtoms.return_value = [self._make_atom(1), self._make_atom(1)]
        self.tab.context.current_molecule = mol
        _calc_tab_mod.Chem.GetFormalCharge = MagicMock(return_value=0)
        self.tab.charge_input.findText.return_value = -1
        self.tab.spin_input.count.return_value = 6
        self.tab.spin_input.itemText.side_effect = [
            "1 (Singlet)",
            "2 (Doublet)",
            "3 (Triplet)",
            "4 (Quartet)",
            "5 (Quintet)",
            "6 (Sextet)",
        ]
        self.tab.method_combo.currentText.return_value = "UKS"

        self.tab.auto_detect_charge_spin()

        self.tab.method_combo.setCurrentText.assert_called_with("RKS")

    def test_odd_electrons_doublet_switches_rhf_to_uhf(self):
        mol = MagicMock()
        mol.GetAtoms.return_value = [self._make_atom(1)]
        self.tab.context.current_molecule = mol
        _calc_tab_mod.Chem.GetFormalCharge = MagicMock(return_value=0)
        self.tab.charge_input.findText.return_value = -1
        self.tab.spin_input.count.return_value = 6
        self.tab.spin_input.itemText.side_effect = [
            "1 (Singlet)",
            "2 (Doublet)",
            "3 (Triplet)",
            "4 (Quartet)",
            "5 (Quintet)",
            "6 (Sextet)",
        ]
        self.tab.method_combo.currentText.return_value = "RHF"

        self.tab.auto_detect_charge_spin()

        self.tab.method_combo.setCurrentText.assert_called_with("UHF")

    def test_odd_electrons_doublet_switches_r_to_u_and_tm_message(self):
        # Atomic number 26 (Fe) is a transition metal
        mol = MagicMock()
        mol.GetAtoms.return_value = [self._make_atom(26), self._make_atom(1)]
        self.tab.context.current_molecule = mol
        _calc_tab_mod.Chem.GetFormalCharge = MagicMock(return_value=0)
        self.tab.charge_input.findText.return_value = 3
        self.tab.spin_input.count.return_value = 6
        self.tab.spin_input.itemText.side_effect = [
            "1 (Singlet)",
            "2 (Doublet)",
            "3 (Triplet)",
            "4 (Quartet)",
            "5 (Quintet)",
            "6 (Sextet)",
        ]
        self.tab.method_combo.currentText.return_value = "RKS"

        self.tab.auto_detect_charge_spin()

        self.tab.charge_input.setCurrentIndex.assert_called_with(3)
        self.tab.method_combo.setCurrentText.assert_called_with("UKS")
        self._mb.information.assert_called_once()

    def test_exception_is_swallowed_and_warns(self):
        mol = MagicMock()
        mol.GetAtoms.side_effect = RuntimeError("boom")
        self.tab.context.current_molecule = mol
        self.tab.auto_detect_charge_spin()
        self._mb.warning.assert_called_once()


class TestValidateSpinSettings(_BaseTabTest):
    def _make_mol(self, n_atoms_protons=2):
        mol = MagicMock()
        atoms = []
        for _ in range(n_atoms_protons):
            a = MagicMock()
            a.GetAtomicNum.return_value = 1
            atoms.append(a)
        mol.GetAtoms.return_value = atoms
        return mol

    def test_no_context_returns_immediately(self):
        self.tab.context = None
        self.tab.validate_spin_settings()  # must not raise

    def test_no_molecule_returns_immediately(self):
        self.tab.context.current_molecule = None
        self.tab.validate_spin_settings()

    def test_valid_settings_clear_styles(self):
        self.tab.context.current_molecule = self._make_mol(2)
        self.tab.charge_input.currentText.return_value = "0"
        self.tab.spin_input.currentText.return_value = "1 (Singlet)"
        self.tab.validate_spin_settings()
        self.tab.spin_input.setStyleSheet.assert_called_with("")
        self.tab.charge_input.setStyleSheet.assert_called_with("")

    def test_invalid_negative_remaining_sets_error_style(self):
        self.tab.context.current_molecule = self._make_mol(2)
        self.tab.charge_input.currentText.return_value = "0"
        # 2 electrons, mult=6 -> unpaired=5 > electrons
        self.tab.spin_input.currentText.return_value = "6 (Sextet)"
        self.tab.validate_spin_settings()
        style = "background-color: #ffcccc; color: black;"
        self.tab.spin_input.setStyleSheet.assert_called_with(style)
        self.assertIn(
            "More unpaired electrons", self.tab.spin_input.setToolTip.call_args[0][0]
        )

    def test_invalid_parity_mismatch_sets_error_style(self):
        self.tab.context.current_molecule = self._make_mol(2)
        self.tab.charge_input.currentText.return_value = "0"
        # 2 electrons, mult=2 -> unpaired=1, remaining=1 (odd) -> invalid
        self.tab.spin_input.currentText.return_value = "2 (Doublet)"
        self.tab.validate_spin_settings()
        self.assertIn(
            "cannot have multiplicity", self.tab.spin_input.setToolTip.call_args[0][0]
        )

    def test_bad_charge_text_returns_without_crash(self):
        self.tab.context.current_molecule = self._make_mol(2)
        self.tab.charge_input.currentText.return_value = "not-a-number"
        self.tab.validate_spin_settings()  # inner except swallows and returns

    def test_outer_exception_is_silenced(self):
        mol = MagicMock()
        mol.GetAtoms.side_effect = RuntimeError("boom")
        self.tab.context.current_molecule = mol
        self.tab.validate_spin_settings()  # must not raise


class TestBrowseOutDir(_BaseTabTest):
    def test_sets_text_when_dir_selected(self):
        _calc_tab_mod.QFileDialog.getExistingDirectory = MagicMock(
            return_value="/some/dir"
        )
        self.tab.browse_out_dir()
        self.tab.out_dir_edit.setText.assert_called_with("/some/dir")

    def test_no_op_when_cancelled(self):
        _calc_tab_mod.QFileDialog.getExistingDirectory = MagicMock(return_value="")
        self.tab.browse_out_dir()
        self.tab.out_dir_edit.setText.assert_not_called()


class TestConfigureScan(_BaseTabTest):
    def test_no_molecule_warns(self):
        self.tab.context.current_molecule = None
        self.tab.configure_scan()
        self._mb.warning.assert_called_once()

    def test_scan_dialog_none_shows_critical(self):
        with patch.object(_calc_tab_mod, "ScanDialog", None):
            self.tab.configure_scan()
        self._mb.critical.assert_called_once()

    def test_opens_new_scan_dialog(self):
        mock_dialog_cls = MagicMock()
        mock_instance = MagicMock()
        mock_dialog_cls.return_value = mock_instance
        with patch.object(_calc_tab_mod, "ScanDialog", mock_dialog_cls):
            self.tab.configure_scan()
        mock_instance.scan_configured.connect.assert_called_once_with(
            self.tab.on_scan_configured
        )
        mock_instance.show.assert_called_once()

    def test_closes_existing_dialog_first(self):
        old_dlg = MagicMock()
        self.tab._scan_config_dlg = old_dlg
        mock_dialog_cls = MagicMock()
        with patch.object(_calc_tab_mod, "ScanDialog", mock_dialog_cls):
            self.tab.configure_scan()
        old_dlg.close.assert_called_once()

    def test_existing_dialog_close_exception_silenced(self):
        old_dlg = MagicMock()
        old_dlg.close.side_effect = RuntimeError("boom")
        self.tab._scan_config_dlg = old_dlg
        mock_dialog_cls = MagicMock()
        with patch.object(_calc_tab_mod, "ScanDialog", mock_dialog_cls):
            self.tab.configure_scan()  # must not raise

    def test_on_scan_configured_sets_params_and_informs(self):
        params = {"type": "Dist", "steps": 5}
        self.tab.on_scan_configured(params)
        self.assertEqual(self.tab.scan_params, params)
        self._mb.information.assert_called_once()


class TestGetSpinValue(_BaseTabTest):
    def test_with_space(self):
        self.tab.spin_input.currentText.return_value = "3 (Triplet)"
        self.assertEqual(self.tab.get_spin_value(), 3)

    def test_without_space(self):
        self.tab.spin_input.currentText.return_value = "4"
        self.assertEqual(self.tab.get_spin_value(), 4)

    def test_exception_returns_one(self):
        self.tab.spin_input.currentText.side_effect = RuntimeError("boom")
        self.assertEqual(self.tab.get_spin_value(), 1)


class TestRunCalculationEdgeCases(_BaseTabTest):
    def _wire_common(self, job="Energy", out_dir="/abs/results"):
        self.tab.job_type_combo.currentText.return_value = job
        self.tab.method_combo.currentText.return_value = "RKS"
        self.tab.functional_combo.currentText.return_value = "b3lyp"
        self.tab.basis_combo.currentText.return_value = "sto-3g"
        self.tab.charge_input.currentText.return_value = "0"
        self.tab.spin_input.currentText.return_value = "1 (Singlet)"
        self.tab.out_dir_edit.text.return_value = out_dir
        self.tab.nstates_input.value.return_value = 10
        self.tab.spin_threads.value.return_value = 0
        self.tab.spin_memory.value.return_value = 4000
        self.tab.check_symmetry.isChecked.return_value = False
        self.tab.check_break_sym.isChecked.return_value = False
        self.tab.spin_grid_level.value.return_value = 3
        self.tab.edit_conv.text.return_value = "1e-9"
        self.tab.spin_cycles.value.return_value = 100
        self.tab.solvent_combo.currentText.return_value = "None (Vacuum)"

    def test_no_molecule_logs_and_warns(self):
        self.tab.context.current_molecule = None
        self.tab.log = MagicMock()
        self.tab.run_calculation()
        self.tab.log.assert_called_once()
        self._mb.warning.assert_called_once()

    def test_relative_path_unsaved_project_reply_no_falls_back_home(self):
        # Note: test_calc_tab.py's helper permanently monkeypatches the real
        # os.path.isabs() to always return True (it assigns
        # `_calc_tab_mod.os.path = os.path` then rebinds `.isabs` on that
        # shared object). Force the correct relative-path behavior here
        # regardless of prior test-order pollution.
        self._wire_common(out_dir="results")
        self._mb.question.return_value = self._mb.StandardButton.No
        mw = MagicMock()
        mw.init_manager = None
        self.tab.context.get_main_window.return_value = mw
        _calc_tab_mod.PySCFWorker = None  # short-circuit before start()

        with patch.object(_calc_tab_mod.os.path, "isabs", return_value=False):
            self.tab.run_calculation()
        self._mb.question.assert_called_once()

    def test_relative_path_unsaved_project_reply_yes_saves_project(self):
        self._wire_common(out_dir="results")
        self._mb.question.return_value = self._mb.StandardButton.Yes
        mw = MagicMock()
        mw.init_manager = None
        mw.io_manager.save_project = MagicMock()
        self.tab.context.get_main_window.return_value = mw
        _calc_tab_mod.PySCFWorker = None

        with patch.object(_calc_tab_mod.os.path, "isabs", return_value=False):
            self.tab.run_calculation()
        mw.io_manager.save_project.assert_called_once()

    def test_relative_path_with_saved_project_uses_base_dir(self):
        self._wire_common(out_dir="results")
        mw = MagicMock()
        mw.init_manager.current_file_path = os.path.join("proj", "file.pmeprj")
        self.tab.context.get_main_window.return_value = mw
        _calc_tab_mod.PySCFWorker = None

        with patch.object(_calc_tab_mod.os.path, "isabs", return_value=False):
            self.tab.run_calculation()
        self._mb.question.assert_not_called()

    def test_scan_job_not_configured_reply_no_returns(self):
        self._wire_common(job="Rigid Surface Scan", out_dir=os.path.abspath("out"))
        self.tab.scan_params = None
        self._mb.question.return_value = self._mb.StandardButton.No
        _calc_tab_mod.PySCFWorker = MagicMock()

        self.tab.run_calculation()
        _calc_tab_mod.PySCFWorker.assert_not_called()

    def test_scan_job_not_configured_reply_yes_but_still_empty_returns(self):
        self._wire_common(job="Rigid Surface Scan", out_dir=os.path.abspath("out"))
        self.tab.scan_params = None
        self._mb.question.return_value = self._mb.StandardButton.Yes
        self.tab.configure_scan = MagicMock()  # doesn't set scan_params
        _calc_tab_mod.PySCFWorker = MagicMock()

        self.tab.run_calculation()
        self.tab.configure_scan.assert_called_once()
        _calc_tab_mod.PySCFWorker.assert_not_called()

    def test_scan_job_configured_via_prompt_proceeds(self):
        self._wire_common(job="Rigid Surface Scan", out_dir=os.path.abspath("out"))
        self.tab.scan_params = None
        self._mb.question.return_value = self._mb.StandardButton.Yes

        def _configure():
            self.tab.scan_params = {"type": "Dist", "steps": 5}

        self.tab.configure_scan = MagicMock(side_effect=_configure)
        mock_worker_cls = MagicMock()
        _calc_tab_mod.PySCFWorker = mock_worker_cls
        _calc_tab_mod.rdkit_to_xyz = lambda m: "H 0 0 0"

        self.tab.run_calculation()
        mock_worker_cls.assert_called_once()

    def test_makedirs_exception_logs_and_returns(self):
        self._wire_common(out_dir=os.path.abspath("out"))
        self.tab.log = MagicMock()
        with patch.object(
            _calc_tab_mod.os, "makedirs", side_effect=OSError("disk full")
        ):
            self.tab.run_calculation()
        self.tab.log.assert_called_with("Error creating output directory: disk full")

    def test_worker_none_logs_error(self):
        self._wire_common(out_dir=os.path.abspath("out"))
        self.tab.log = MagicMock()
        _calc_tab_mod.PySCFWorker = None
        with patch.object(_calc_tab_mod.os, "makedirs"):
            self.tab.run_calculation()
        self.tab.log.assert_called_with(
            "Error: Could not import PySCFWorker. Check installation."
        )

    def test_successful_run_starts_worker(self):
        self._wire_common(out_dir=os.path.abspath("out"))
        mock_worker_instance = MagicMock()
        mock_worker_cls = MagicMock(return_value=mock_worker_instance)
        _calc_tab_mod.PySCFWorker = mock_worker_cls
        _calc_tab_mod.rdkit_to_xyz = lambda m: "H 0 0 0"

        with patch.object(_calc_tab_mod.os, "makedirs"):
            self.tab.run_calculation()

        mock_worker_instance.start.assert_called_once()
        self.tab.run_btn.setEnabled.assert_called_with(False)
        self.tab.stop_btn.setEnabled.assert_called_with(True)
        self.tab.progress_bar.show.assert_called_once()


class TestStopCalculation(_BaseTabTest):
    def test_no_worker_returns_immediately(self):
        self.tab.worker = None
        self.tab.stop_calculation()  # must not raise

    def test_worker_not_running_returns_immediately(self):
        worker = MagicMock()
        worker.isRunning.return_value = False
        self.tab.worker = worker
        self.tab.log = MagicMock()
        self.tab.stop_calculation()
        self.tab.log.assert_not_called()

    def test_stream_close_exception_silenced(self):
        worker = MagicMock()
        worker.isRunning.return_value = True
        worker._stream = MagicMock()
        worker._stream.close.side_effect = RuntimeError("boom")
        worker.wait.return_value = True
        self.tab.worker = worker
        self.tab.log = MagicMock()
        self.tab.stop_calculation()  # must not raise
        worker.finished.connect.assert_called_with(self.tab._on_worker_stopped)

    def test_disconnect_exception_silenced(self):
        worker = MagicMock()
        worker.isRunning.return_value = True
        worker._stream = None
        worker.log_signal.disconnect.side_effect = RuntimeError("boom")
        worker.wait.return_value = True
        self.tab.worker = worker
        self.tab.log = MagicMock()
        self.tab.stop_calculation()  # must not raise

    def test_force_terminate_when_wait_times_out(self):
        worker = MagicMock()
        worker.isRunning.return_value = True
        worker._stream = None
        worker.wait.return_value = False
        self.tab.worker = worker
        self.tab.log = MagicMock()
        self.tab.stop_calculation()
        worker.terminate.assert_called_once()

    def test_no_terminate_when_wait_succeeds(self):
        worker = MagicMock()
        worker.isRunning.return_value = True
        worker._stream = None
        worker.wait.return_value = True
        self.tab.worker = worker
        self.tab.log = MagicMock()
        self.tab.stop_calculation()
        worker.terminate.assert_not_called()


class TestOnWorkerStopped(_BaseTabTest):
    def test_none_worker_returns(self):
        self.tab.worker = None
        self.tab.cleanup_ui_state = MagicMock()
        self.tab._on_worker_stopped()
        self.tab.cleanup_ui_state.assert_not_called()

    def test_clears_worker_and_cleans_up(self):
        self.tab.worker = MagicMock()
        self.tab.cleanup_ui_state = MagicMock()
        self.tab._on_worker_stopped()
        self.assertIsNone(self.tab.worker)
        self.tab.cleanup_ui_state.assert_called_once()


class TestLogging(_BaseTabTest):
    def test_log_appends_and_moves_cursor(self):
        cursor = MagicMock()
        self.tab.log_text.textCursor.return_value = cursor
        self.tab.log("hello")
        self.tab.log_text.append.assert_called_with("hello")
        cursor.movePosition.assert_called_once()
        self.tab.log_text.setTextCursor.assert_called_with(cursor)

    def test_log_append_inserts_text(self):
        cursor = MagicMock()
        self.tab.log_text.textCursor.return_value = cursor
        self.tab.log_append("chunk")
        cursor.insertText.assert_called_with("chunk")
        self.tab.log_text.setTextCursor.assert_called_with(cursor)


class TestOnFinishedAndOnError(_BaseTabTest):
    def test_on_finished_no_context(self):
        self.tab.context = None
        self.tab.log = MagicMock()
        self.tab.cleanup_ui_state = MagicMock()
        self.tab.on_finished()
        self.tab.parent_dialog.update_internal_state.assert_called_once()
        self.tab.cleanup_ui_state.assert_called_once()

    def test_on_finished_with_context_and_splitter(self):
        mw = MagicMock()
        splitter = MagicMock()
        mw.init_manager.splitter = splitter
        self.tab.context.get_main_window.return_value = mw
        self.tab.cleanup_ui_state = MagicMock()
        self.tab.on_finished()
        self.tab.context.mark_project_modified.assert_called_once()
        splitter.setCollapsible.assert_called_with(0, True)

    def test_on_finished_mw_falsy_skips_splitter(self):
        self.tab.context.get_main_window.return_value = None
        self.tab.cleanup_ui_state = MagicMock()
        self.tab.on_finished()  # must not raise

    def test_on_error_logs_and_shows_critical(self):
        self.tab.log = MagicMock()
        self.tab.cleanup_ui_state = MagicMock()
        self.tab.on_error("bad stuff")
        self.tab.log.assert_called_with("\nERROR: bad stuff")
        self._mb.critical.assert_called_once()
        self.tab.cleanup_ui_state.assert_called_once()


class TestCleanupUiState(_BaseTabTest):
    def test_resets_ui_flags(self):
        self.tab.worker = MagicMock()
        self.tab.cleanup_ui_state()
        self.tab.run_btn.setEnabled.assert_called_with(True)
        self.tab.stop_btn.setEnabled.assert_called_with(False)
        self.tab.progress_bar.hide.assert_called_once()
        self.assertIsNone(self.tab.worker)


if __name__ == "__main__":
    unittest.main()
