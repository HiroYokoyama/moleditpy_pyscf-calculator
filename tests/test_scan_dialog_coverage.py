"""
tests/test_scan_dialog_coverage.py

Additional coverage for pyscf_calculator/scan_dialog.py: __init__() restore/
selection-mode branches, init_ui(), _auto_update_selection(),
calculate_current_value() (Dist/Angle/Dihedral/exception/no-molecule),
accept_scan() steps<2 branch, closeEvent().

Not covered here: update_ui_state()/accept_scan() happy paths already
exercised by tests/test_scan_dialog.py (not duplicated).
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
    qt_core.pyqtSignal = lambda *a, **kw: MagicMock()

    class _QTimer:
        def __init__(self, *a):
            self.timeout = MagicMock()
            self._active = False

        def start(self, ms):
            self._active = True

        def stop(self):
            self._active = False

        def isActive(self):
            return self._active

    qt_core.QTimer = _QTimer

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

        def accept(self):
            pass

        def reject(self):
            pass

        def closeEvent(self, event):
            pass

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QDialog = _QDialog

    def _widget_factory(*args, **kwargs):
        return MagicMock()

    for name in [
        "QVBoxLayout",
        "QHBoxLayout",
        "QLabel",
        "QLineEdit",
        "QPushButton",
        "QMessageBox",
        "QGroupBox",
        "QFormLayout",
    ]:
        setattr(qt_widgets, name, _widget_factory)

    pyqt6.QtWidgets = qt_widgets
    sys.modules["PyQt6.QtWidgets"] = qt_widgets
    sys.modules["rdkit"] = MagicMock()
    sys.modules["rdkit.Chem"] = MagicMock()


_install_stubs()

scan_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "scan_dialog.py"),
    "pyscf_calculator_scan_dialog_coverage_under_test",
)
ScanDialog = scan_mod.ScanDialog

# QMessageBox is a bare factory function in the stub; replace with a real
# mock instance so `QMessageBox.warning(...)` resolves.
scan_mod.QMessageBox = MagicMock()


class TestInitBasic(unittest.TestCase):
    def test_init_no_context_sets_defaults(self):
        dlg = ScanDialog(parent=None, context=None)
        self.assertIsNone(dlg.mw)
        self.assertEqual(dlg.selected_atoms, [])
        self.assertEqual(dlg.scan_params, {})
        self.assertFalse(dlg.was_measurement_active)

    def test_init_with_context_calls_get_main_window(self):
        context = MagicMock()
        mw = MagicMock()
        mw.edit_3d_manager = MagicMock()
        mw.edit_3d_manager.measurement_mode = False
        context.get_main_window.return_value = mw
        dlg = ScanDialog(parent=None, context=context)
        self.assertIs(dlg.mw, mw)
        mw.edit_3d_manager.toggle_measurement_mode.assert_called_once_with(True)

    def test_init_activation_exception_silenced(self):
        context = MagicMock()
        mw = MagicMock()
        context.get_main_window.return_value = mw
        # Make hasattr(...) True but toggle raise
        mw.edit_3d_manager.measurement_mode = False
        mw.edit_3d_manager.toggle_measurement_mode.side_effect = RuntimeError("boom")
        dlg = ScanDialog(parent=None, context=context)  # must not raise
        self.assertIsNotNone(dlg)

    def test_init_already_measuring_does_not_toggle(self):
        context = MagicMock()
        mw = MagicMock()
        mw.edit_3d_manager.measurement_mode = True
        context.get_main_window.return_value = mw
        dlg = ScanDialog(parent=None, context=context)
        self.assertTrue(dlg.was_measurement_active)
        mw.edit_3d_manager.toggle_measurement_mode.assert_not_called()

    def test_init_sets_measurement_action_checked(self):
        context = MagicMock()
        mw = MagicMock()
        mw.edit_3d_manager.measurement_mode = False
        context.get_main_window.return_value = mw
        ScanDialog(parent=None, context=context)
        mw.init_manager.measurement_action.setChecked.assert_called_with(True)


class TestInitRestoreParams(unittest.TestCase):
    def test_restores_atoms_and_visual_selection(self):
        context = MagicMock()
        mw = MagicMock()
        e3d = MagicMock()
        e3d.measurement_mode = False
        e3d.selected_atoms_3d = None
        e3d.selected_atoms_for_measurement = None
        mw.edit_3d_manager = e3d
        context.get_main_window.return_value = mw

        dlg = ScanDialog(parent=None, context=context, initial_params={"atoms": [1, 2]})
        self.assertEqual(dlg.selected_atoms, [1, 2])
        self.assertEqual(e3d.selected_atoms_3d, {1, 2})
        self.assertEqual(e3d.selected_atoms_for_measurement, [1, 2])
        e3d.update_3d_selection_display.assert_called_once()

    def test_restore_visual_selection_exception_silenced(self):
        context = MagicMock()
        mw = MagicMock()
        mw.edit_3d_manager.measurement_mode = False
        context.get_main_window.return_value = mw
        # getattr(self.mw, "edit_3d_manager", None) raising inside try
        with patch.object(ScanDialog, "update_ui_state", side_effect=lambda: None):
            with patch("builtins.getattr", side_effect=getattr):
                pass  # not needed; simulate failure via broken mw attribute access instead

        # Force failure a simpler way: make hasattr() raise by giving a
        # pathological e3d whose attribute access blows up.
        class _BadE3D:
            @property
            def selected_atoms_3d(self):
                raise RuntimeError("boom")

        mw.edit_3d_manager = _BadE3D()
        dlg = ScanDialog(
            parent=None, context=context, initial_params={"atoms": [1, 2]}
        )  # must not raise
        self.assertEqual(dlg.selected_atoms, [1, 2])

    def test_restores_start_end_steps_after_update_ui_state(self):
        context = MagicMock()
        context.get_main_window.return_value = None
        dlg = ScanDialog(
            parent=None,
            context=context,
            initial_params={"start": "1.0", "end": "2.0", "steps": "8"},
        )
        dlg.edit_start.setText.assert_any_call("1.0")
        dlg.edit_end.setText.assert_any_call("2.0")
        dlg.edit_steps.setText.assert_any_call("8")

    def test_no_atoms_key_skips_restore_block(self):
        context = MagicMock()
        context.get_main_window.return_value = None
        dlg = ScanDialog(parent=None, context=context, initial_params={"start": "1.0"})
        self.assertEqual(dlg.selected_atoms, [])


class TestAutoUpdateSelection(unittest.TestCase):
    def _make_dialog(self):
        dlg = ScanDialog.__new__(ScanDialog)
        dlg.context = MagicMock()
        dlg.selected_atoms = []
        dlg.lbl_selection = MagicMock()
        dlg.grp_params = MagicMock()
        dlg.btn_ok = MagicMock()
        dlg.lbl_type = MagicMock()
        dlg.lbl_current = MagicMock()
        return dlg

    def test_no_mw_returns_immediately(self):
        dlg = self._make_dialog()
        dlg.mw = None
        dlg._auto_update_selection()  # must not raise

    def test_no_edit_3d_manager_returns(self):
        dlg = self._make_dialog()
        dlg.mw = MagicMock(spec=[])
        dlg._auto_update_selection()  # must not raise

    def test_measurement_selection_takes_priority(self):
        dlg = self._make_dialog()
        mw = MagicMock()
        e3d = MagicMock()
        e3d.selected_atoms_for_measurement = [3, 1, "bad", 2]
        e3d.selected_atoms_3d = {9, 9, 9}
        mw.edit_3d_manager = e3d
        dlg.mw = mw
        with patch.object(dlg, "update_ui_state") as mock_update:
            dlg._auto_update_selection()
        self.assertEqual(dlg.selected_atoms, [3, 1, 2])
        mock_update.assert_called_once()

    def test_falls_back_to_selected_atoms_3d(self):
        dlg = self._make_dialog()
        mw = MagicMock()
        e3d = MagicMock()
        e3d.selected_atoms_for_measurement = []
        e3d.selected_atoms_3d = {5, 6}
        mw.edit_3d_manager = e3d
        dlg.mw = mw
        with patch.object(dlg, "update_ui_state"):
            dlg._auto_update_selection()
        self.assertEqual(sorted(dlg.selected_atoms), [5, 6])

    def test_limits_to_four_atoms(self):
        dlg = self._make_dialog()
        mw = MagicMock()
        e3d = MagicMock()
        e3d.selected_atoms_for_measurement = [1, 2, 3, 4, 5]
        mw.edit_3d_manager = e3d
        dlg.mw = mw
        with patch.object(dlg, "update_ui_state"):
            dlg._auto_update_selection()
        self.assertEqual(dlg.selected_atoms, [1, 2, 3, 4])

    def test_no_change_skips_update_ui_state(self):
        dlg = self._make_dialog()
        dlg.selected_atoms = [1, 2]
        mw = MagicMock()
        e3d = MagicMock()
        e3d.selected_atoms_for_measurement = [1, 2]
        mw.edit_3d_manager = e3d
        dlg.mw = mw
        with patch.object(dlg, "update_ui_state") as mock_update:
            dlg._auto_update_selection()
        mock_update.assert_not_called()


class TestCalculateCurrentValue(unittest.TestCase):
    def _make_dialog(self):
        dlg = ScanDialog.__new__(ScanDialog)
        dlg.lbl_type = MagicMock()
        dlg.lbl_current = MagicMock()
        dlg.edit_start = MagicMock()
        return dlg

    def test_no_context_returns(self):
        dlg = self._make_dialog()
        dlg.context = None
        dlg.selected_atoms = [0, 1]
        dlg.calculate_current_value()  # must not raise
        dlg.lbl_type.setText.assert_not_called()

    def test_no_molecule_returns(self):
        dlg = self._make_dialog()
        dlg.context = MagicMock()
        dlg.context.current_molecule = None
        dlg.selected_atoms = [0, 1]
        dlg.calculate_current_value()
        dlg.lbl_type.setText.assert_not_called()

    def test_distance_two_atoms(self):
        dlg = self._make_dialog()
        mol = MagicMock()
        conf = MagicMock()
        p1 = MagicMock()
        p2 = MagicMock()
        diff = MagicMock()
        diff.Length.return_value = 1.5
        p1.__sub__ = MagicMock(return_value=diff)
        conf.GetAtomPosition.side_effect = [p1, p2]
        mol.GetConformer.return_value = conf
        dlg.context = MagicMock()
        dlg.context.current_molecule = mol
        dlg.selected_atoms = [0, 1]

        dlg.calculate_current_value()

        self.assertEqual(dlg.scan_type, "Dist")
        dlg.lbl_type.setText.assert_called_with("Type: Dist")
        dlg.lbl_current.setText.assert_called_with("Current Value: 1.500")
        dlg.edit_start.setText.assert_called_with("1.500")

    def test_angle_three_atoms(self):
        dlg = self._make_dialog()
        mol = MagicMock()
        conf = MagicMock()
        mol.GetConformer.return_value = conf
        dlg.context = MagicMock()
        dlg.context.current_molecule = mol
        dlg.selected_atoms = [0, 1, 2]

        with patch.object(scan_mod.rdMolTransforms, "GetAngleDeg", return_value=109.5):
            dlg.calculate_current_value()

        self.assertEqual(dlg.scan_type, "Angle")
        dlg.lbl_current.setText.assert_called_with("Current Value: 109.500")

    def test_dihedral_four_atoms(self):
        dlg = self._make_dialog()
        mol = MagicMock()
        conf = MagicMock()
        mol.GetConformer.return_value = conf
        dlg.context = MagicMock()
        dlg.context.current_molecule = mol
        dlg.selected_atoms = [0, 1, 2, 3]

        with patch.object(
            scan_mod.rdMolTransforms, "GetDihedralDeg", return_value=-60.0
        ):
            dlg.calculate_current_value()

        self.assertEqual(dlg.scan_type, "Dihedral")
        dlg.lbl_current.setText.assert_called_with("Current Value: -60.000")

    def test_exception_sets_error_label(self):
        dlg = self._make_dialog()
        mol = MagicMock()
        conf = MagicMock()
        conf.GetAtomPosition.side_effect = RuntimeError("boom")
        mol.GetConformer.return_value = conf
        dlg.context = MagicMock()
        dlg.context.current_molecule = mol
        dlg.selected_atoms = [0, 1]

        dlg.calculate_current_value()

        dlg.lbl_current.setText.assert_called_with("Error calc value")


class TestAcceptScanStepsValidation(unittest.TestCase):
    def _make_dialog(self):
        dlg = ScanDialog.__new__(ScanDialog)
        dlg.edit_start = MagicMock()
        dlg.edit_end = MagicMock()
        dlg.edit_steps = MagicMock()
        dlg.scan_configured = MagicMock()
        dlg.scan_type = "Dist"
        dlg.selected_atoms = [0, 1]
        dlg.mw = None
        return dlg

    def test_steps_below_two_warns_and_returns(self):
        dlg = self._make_dialog()
        dlg.edit_start.text.return_value = "1.0"
        dlg.edit_end.text.return_value = "2.0"
        dlg.edit_steps.text.return_value = "1"

        with patch.object(scan_mod.QMessageBox, "warning") as mock_warn:
            dlg.accept_scan()

        mock_warn.assert_called_once()
        dlg.scan_configured.emit.assert_not_called()

    def test_deactivate_selection_mode_exception_silenced(self):
        dlg = self._make_dialog()
        dlg.edit_start.text.return_value = "1.0"
        dlg.edit_end.text.return_value = "2.0"
        dlg.edit_steps.text.return_value = "10"
        mw = MagicMock()
        mw.edit_3d_manager.toggle_measurement_mode.side_effect = RuntimeError("boom")
        dlg.mw = mw

        with patch.object(dlg, "accept"):
            dlg.accept_scan()  # must not raise

        dlg.scan_configured.emit.assert_called_once()

    def test_deactivates_selection_mode_on_success(self):
        dlg = self._make_dialog()
        dlg.edit_start.text.return_value = "1.0"
        dlg.edit_end.text.return_value = "2.0"
        dlg.edit_steps.text.return_value = "10"
        mw = MagicMock()
        dlg.mw = mw

        with patch.object(dlg, "accept") as mock_accept:
            dlg.accept_scan()

        mw.edit_3d_manager.toggle_measurement_mode.assert_called_once_with(False)
        mw.init_manager.measurement_action.setChecked.assert_called_with(False)
        mock_accept.assert_called_once()


class TestCloseEvent(unittest.TestCase):
    def _make_dialog(self):
        dlg = ScanDialog.__new__(ScanDialog)
        dlg.mw = None
        dlg.was_measurement_active = False
        return dlg

    def test_stops_active_timer(self):
        dlg = self._make_dialog()
        timer = MagicMock()
        timer.isActive.return_value = True
        dlg.sel_timer = timer
        dlg.closeEvent(MagicMock())
        timer.stop.assert_called_once()

    def test_no_timer_attribute_no_crash(self):
        dlg = self._make_dialog()
        dlg.closeEvent(MagicMock())  # must not raise

    def test_inactive_timer_not_stopped(self):
        dlg = self._make_dialog()
        timer = MagicMock()
        timer.isActive.return_value = False
        dlg.sel_timer = timer
        dlg.closeEvent(MagicMock())
        timer.stop.assert_not_called()

    def test_restores_measurement_mode_and_action(self):
        dlg = self._make_dialog()
        dlg.was_measurement_active = True
        mw = MagicMock()
        dlg.mw = mw
        dlg.closeEvent(MagicMock())
        mw.edit_3d_manager.toggle_measurement_mode.assert_called_once_with(True)
        mw.init_manager.measurement_action.setChecked.assert_called_once_with(True)

    def test_exception_in_restore_silenced(self):
        dlg = self._make_dialog()
        mw = MagicMock()
        mw.edit_3d_manager.toggle_measurement_mode.side_effect = RuntimeError("boom")
        dlg.mw = mw
        dlg.closeEvent(MagicMock())  # must not raise


if __name__ == "__main__":
    unittest.main()
