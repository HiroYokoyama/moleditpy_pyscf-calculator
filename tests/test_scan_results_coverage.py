"""
tests/test_scan_results_coverage.py

Additional coverage for pyscf_calculator/scan_results.py: __init__()/init_ui(),
plot_data() branches, on_unit_changed(), highlight_point(),
create_base_molecule() edge branches, on_pick()/on_hover(), set_frame(),
update_viewer() branches, on_slider_change(), toggle_play(), save_plot()/
save_csv() exception paths, prev/next_frame(), next_frame_auto(),
clear_selection(), closeEvent(), save_gif().

Not covered here: plot_data() relative/absolute happy paths and save_csv()
happy path already exercised by tests/test_scan_results.py, and
create_base_molecule()'s mark_project_modified regression tests (not
duplicated).
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
    qt_core.pyqtSignal = lambda *a, **kw: MagicMock()

    class _Qt:
        class Orientation:
            Horizontal = 1

        class CursorShape:
            WaitCursor = 2
            ArrowCursor = 3

        class WindowModality:
            WindowModal = 4

    qt_core.Qt = _Qt

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

        def setCursor(self, *a):
            pass

        def close(self):
            pass

        def exec(self):
            return 1

        class DialogCode:
            Accepted = 1
            Rejected = 0

    class _QSlider:
        def __init__(self, *a):
            self.valueChanged = MagicMock()

        def setRange(self, *a):
            pass

        def setMinimumWidth(self, *a):
            pass

        def setValue(self, *a):
            pass

        def blockSignals(self, *a):
            pass

    class _QDialogButtonBox:
        def __init__(self, *a, **k):
            self.accepted = MagicMock()
            self.rejected = MagicMock()

        class StandardButton:
            Ok = 1
            Cancel = 2

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QDialog = _QDialog
    qt_widgets.QSlider = _QSlider
    qt_widgets.QDialogButtonBox = _QDialogButtonBox

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
        "QComboBox",
        "QSpinBox",
        "QCheckBox",
        "QFileDialog",
        "QProgressDialog",
        "QApplication",
    ]:
        setattr(qt_widgets, name, _widget_factory)

    pyqt6.QtWidgets = qt_widgets
    sys.modules["PyQt6.QtWidgets"] = qt_widgets

    sys.modules["rdkit"] = MagicMock()
    sys.modules["rdkit.Chem"] = MagicMock()
    sys.modules["rdkit.Chem.rdGeometry"] = MagicMock()

    sys.modules["matplotlib"] = MagicMock()
    sys.modules["matplotlib.backends.backend_qtagg"] = MagicMock()
    sys.modules["matplotlib.figure"] = MagicMock()
    sys.modules["matplotlib.collections"] = MagicMock()
    sys.modules["PIL"] = MagicMock()

    backend = types.ModuleType("matplotlib.backends.backend_qtagg")

    class _FigureCanvasQTAgg:
        def __init__(self, *a, **k):
            pass

    class _FigureCanvasQTAgg2(_FigureCanvasQTAgg):
        def mpl_connect(self, *a, **k):
            pass

        def draw(self):
            pass

        def draw_idle(self):
            pass

    backend.FigureCanvasQTAgg = _FigureCanvasQTAgg2
    sys.modules["matplotlib.backends.backend_qtagg"] = backend


_install_stubs()

# scan_results.py's update_viewer() fallback branch does a *bare* (unwrapped)
# `from .utils import update_molecule_from_xyz` relative import. Loading the
# module standalone (module_name not a real submodule) makes that raise
# ImportError at call time. Fake a minimal parent package so the relative
# import resolves to the real utils.py (itself already covered/tested
# elsewhere and safe to import with rdkit stubbed).
_FAKE_PKG_NAME = "pyscf_calculator_scan_results_coverage_pkg"
_fake_pkg = types.ModuleType(_FAKE_PKG_NAME)
_fake_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "pyscf_calculator")]
sys.modules[_FAKE_PKG_NAME] = _fake_pkg
_utils_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "utils.py"), f"{_FAKE_PKG_NAME}.utils"
)
sys.modules[f"{_FAKE_PKG_NAME}.utils"] = _utils_mod
_fake_pkg.utils = _utils_mod


def _load_scan_results_as_package_submodule():
    src = os.path.join(
        os.path.dirname(__file__), "..", "pyscf_calculator", "scan_results.py"
    )
    src = os.path.normpath(src)
    module_name = f"{_FAKE_PKG_NAME}.scan_results"
    spec = importlib.util.spec_from_file_location(
        module_name, src, submodule_search_locations=[]
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _FAKE_PKG_NAME
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


sr_mod = _load_scan_results_as_package_submodule()
ScanResultDialog = sr_mod.ScanResultDialog

# QMessageBox/QFileDialog are bare factory functions in the stub; replace
# with real mock instances so class-level calls resolve.
sr_mod.QMessageBox = MagicMock()
sr_mod.QFileDialog = MagicMock()
sr_mod.QApplication = MagicMock()
sr_mod.QProgressDialog = MagicMock()


def _make_bare_dialog(results=None, trajectory=None):
    dlg = ScanResultDialog.__new__(ScanResultDialog)
    dlg.results = results
    dlg.trajectory = trajectory
    dlg.context = MagicMock()
    dlg.scan_type = "Dist"
    dlg.scan_result_dir = "/tmp"
    dlg.frame_idx = 0
    dlg.is_playing = False
    dlg.base_mol = None

    dlg.unit_combo = MagicMock()
    dlg.unit_combo.currentText.return_value = "Hartree"
    dlg.chk_relative = MagicMock()
    dlg.chk_relative.isChecked.return_value = False
    dlg.chk_dynamic_bonds = MagicMock()
    dlg.chk_dynamic_bonds.isChecked.return_value = False
    dlg.canvas = MagicMock()
    dlg.canvas.axes = MagicMock()
    dlg.canvas.axes.plot.return_value = [MagicMock()]
    dlg.canvas.axes.axvline.return_value = MagicMock()
    dlg.slider = MagicMock()
    dlg.lbl_frame = MagicMock()
    dlg.timer = MagicMock()
    dlg.btn_play = MagicMock()
    dlg.annot = MagicMock()
    return dlg


class TestInit(unittest.TestCase):
    def test_init_without_trajectory_skips_base_molecule(self):
        with (
            patch.object(ScanResultDialog, "create_base_molecule") as mock_create,
            patch.object(ScanResultDialog, "plot_data"),
        ):
            dlg = ScanResultDialog(
                parent=None, results=[{"value": 1, "energy": -1}], trajectory=None
            )
        mock_create.assert_not_called()
        self.assertIsNone(dlg.base_mol)

    def test_init_with_trajectory_creates_base_molecule(self):
        with (
            patch.object(ScanResultDialog, "create_base_molecule") as mock_create,
            patch.object(ScanResultDialog, "plot_data"),
        ):
            ScanResultDialog(
                parent=None,
                results=[{"value": 1, "energy": -1}],
                trajectory=["xyz1"],
            )
        mock_create.assert_called_once()

    def test_init_focuses_play_button(self):
        with (
            patch.object(ScanResultDialog, "create_base_molecule"),
            patch.object(ScanResultDialog, "plot_data"),
        ):
            dlg = ScanResultDialog(
                parent=None, results=[{"value": 1, "energy": -1}], trajectory=None
            )
        dlg.btn_play.setFocus.assert_called_once()
        dlg.btn_play.setDefault.assert_called_once_with(True)

    def test_init_ui_no_trajectory_gif_disabled(self):
        with (
            patch.object(ScanResultDialog, "create_base_molecule"),
            patch.object(ScanResultDialog, "plot_data"),
        ):
            dlg = ScanResultDialog(
                parent=None, results=[{"value": 1, "energy": -1}], trajectory=None
            )
        dlg.btn_gif.setEnabled.assert_called_with(False)


class TestPlotDataEdgeCases(unittest.TestCase):
    def test_no_results_returns_immediately(self):
        dlg = _make_bare_dialog(results=None)
        dlg.plot_data()  # must not raise
        dlg.canvas.axes.clear.assert_not_called()

    def test_kcalmol_unit_label(self):
        dlg = _make_bare_dialog(
            results=[{"value": 1.0, "energy": -1.0}, {"value": 2.0, "energy": -2.0}]
        )
        dlg.unit_combo.currentText.return_value = "kcal/mol"
        dlg.chk_relative.isChecked.return_value = False
        dlg.plot_data()
        dlg.canvas.axes.set_ylabel.assert_called_with("Energy (kcal/mol)")

    def test_bond_scan_type_xlabel(self):
        dlg = _make_bare_dialog(
            results=[{"value": 1.0, "energy": -1.0}, {"value": 2.0, "energy": -2.0}]
        )
        dlg.scan_type = "Bond Scan"
        dlg.plot_data()
        dlg.canvas.axes.set_xlabel.assert_called_with("Bond Length (Å)")

    def test_dihedral_scan_type_xlabel(self):
        dlg = _make_bare_dialog(
            results=[{"value": 1.0, "energy": -1.0}, {"value": 2.0, "energy": -2.0}]
        )
        dlg.scan_type = "Dihedral"
        dlg.plot_data()
        dlg.canvas.axes.set_xlabel.assert_called_with("Angle (Degrees)")

    def test_generic_scan_type_default_xlabel(self):
        dlg = _make_bare_dialog(
            results=[{"value": 1.0, "energy": -1.0}, {"value": 2.0, "energy": -2.0}]
        )
        dlg.scan_type = "Custom"
        dlg.plot_data()
        dlg.canvas.axes.set_xlabel.assert_called_with("Coordinate")

    def test_no_unit_combo_defaults_hartree(self):
        dlg = _make_bare_dialog(
            results=[{"value": 1.0, "energy": -1.0}, {"value": 2.0, "energy": -2.0}]
        )
        del dlg.unit_combo
        dlg.plot_data()
        dlg.canvas.axes.set_ylabel.assert_called_with("Energy (Hartree)")

    def test_no_chk_relative_defaults_absolute(self):
        dlg = _make_bare_dialog(
            results=[{"value": 1.0, "energy": -1.0}, {"value": 2.0, "energy": -2.0}]
        )
        del dlg.chk_relative
        dlg.plot_data()
        dlg.canvas.axes.set_ylabel.assert_called_with("Energy (Hartree)")


class TestOnUnitChanged(unittest.TestCase):
    def test_replots_and_redraws(self):
        dlg = _make_bare_dialog(
            results=[{"value": 1.0, "energy": -1.0}, {"value": 2.0, "energy": -2.0}]
        )
        dlg.frame_idx = 0
        with patch.object(dlg, "plot_data") as mock_plot, patch.object(
            dlg, "highlight_point"
        ) as mock_hl:
            dlg.on_unit_changed("kJ/mol")
        mock_plot.assert_called_once()
        mock_hl.assert_called_once_with(0)
        dlg.canvas.draw.assert_called_once()

    def test_no_frame_idx_skips_highlight(self):
        dlg = _make_bare_dialog(
            results=[{"value": 1.0, "energy": -1.0}, {"value": 2.0, "energy": -2.0}]
        )
        dlg.frame_idx = None
        with patch.object(dlg, "plot_data"), patch.object(
            dlg, "highlight_point"
        ) as mock_hl:
            dlg.on_unit_changed("kJ/mol")
        mock_hl.assert_not_called()


class TestHighlightPoint(unittest.TestCase):
    def _make_dlg(self):
        return _make_bare_dialog(
            results=[
                {"value": 1.0, "energy": -1.0},
                {"value": 2.0, "energy": -2.0},
                {"value": 3.0, "energy": -1.5},
            ]
        )

    def test_removes_old_marker_and_line(self):
        dlg = self._make_dlg()
        dlg._highlight_marker = MagicMock()
        dlg._highlight_line = MagicMock()
        old_marker = dlg._highlight_marker
        old_line = dlg._highlight_line
        dlg.highlight_point(0)
        old_marker.remove.assert_called_once()
        old_line.remove.assert_called_once()

    def test_marker_remove_exception_silenced(self):
        dlg = self._make_dlg()
        dlg._highlight_marker = MagicMock()
        dlg._highlight_marker.remove.side_effect = RuntimeError("boom")
        dlg.highlight_point(0)  # must not raise

    def test_line_remove_exception_silenced(self):
        dlg = self._make_dlg()
        dlg._highlight_line = MagicMock()
        dlg._highlight_line.remove.side_effect = RuntimeError("boom")
        dlg.highlight_point(0)  # must not raise

    def test_relative_kjmol_conversion(self):
        dlg = self._make_dlg()
        dlg.chk_relative.isChecked.return_value = True
        dlg.unit_combo.currentText.return_value = "kJ/mol"
        dlg.highlight_point(1)  # min energy index -> rel 0
        args, kwargs = dlg.canvas.axes.plot.call_args
        self.assertAlmostEqual(args[1], 0.0)

    def test_kcalmol_conversion_absolute(self):
        dlg = self._make_dlg()
        dlg.chk_relative.isChecked.return_value = False
        dlg.unit_combo.currentText.return_value = "kcal/mol"
        dlg.highlight_point(0)
        args, kwargs = dlg.canvas.axes.plot.call_args
        self.assertAlmostEqual(args[1], -1.0 * sr_mod._HARTREE_TO_KCALMOL)

    def test_no_unit_combo_defaults_hartree(self):
        dlg = self._make_dlg()
        del dlg.unit_combo
        dlg.highlight_point(0)
        args, kwargs = dlg.canvas.axes.plot.call_args
        self.assertAlmostEqual(args[1], -1.0)


class TestCreateBaseMoleculeBranches(unittest.TestCase):
    def test_no_trajectory_returns(self):
        dlg = ScanResultDialog.__new__(ScanResultDialog)
        dlg.trajectory = None
        dlg.create_base_molecule()  # must not raise

    def test_too_few_lines_returns(self):
        dlg = ScanResultDialog.__new__(ScanResultDialog)
        dlg.trajectory = ["only one line"]
        dlg.context = None
        dlg.create_base_molecule()  # must not raise
        self.assertIsNone(getattr(dlg, "base_mol", None))

    def test_estimate_bonds_exception_silenced(self):
        dlg = ScanResultDialog.__new__(ScanResultDialog)
        dlg.trajectory = ["2\ncomment\nH 0.0 0.0 0.0\nH 0.0 0.0 1.0"]
        context = MagicMock()
        mw = MagicMock()
        mw.io_manager.estimate_bonds_from_distances.side_effect = RuntimeError("boom")
        context.get_main_window.return_value = mw
        dlg.context = context
        dlg.create_base_molecule()  # must not raise
        self.assertIsNotNone(dlg.base_mol)

    def test_no_io_manager_skips_bond_estimation(self):
        dlg = ScanResultDialog.__new__(ScanResultDialog)
        dlg.trajectory = ["2\ncomment\nH 0.0 0.0 0.0\nH 0.0 0.0 1.0"]
        context = MagicMock()
        mw = MagicMock(spec=[])
        context.get_main_window.return_value = mw
        dlg.context = context
        dlg.create_base_molecule()  # must not raise

    def test_zero_bonds_triggers_rddeterminebonds_fallback(self):
        dlg = ScanResultDialog.__new__(ScanResultDialog)
        dlg.trajectory = ["2\ncomment\nH 0.0 0.0 0.0\nH 0.0 0.0 1.0"]
        context = MagicMock()
        mw = MagicMock(spec=[])
        context.get_main_window.return_value = mw
        dlg.context = context

        rddb = MagicMock()
        with patch.dict(sys.modules, {"rdkit.Chem.rdDetermineBonds": rddb}):
            dlg.create_base_molecule()

        self.assertIsNotNone(dlg.base_mol)

    def test_rddeterminebonds_exception_silenced(self):
        dlg = ScanResultDialog.__new__(ScanResultDialog)
        dlg.trajectory = ["2\ncomment\nH 0.0 0.0 0.0\nH 0.0 0.0 1.0"]
        context = MagicMock()
        mw = MagicMock(spec=[])
        context.get_main_window.return_value = mw
        dlg.context = context

        rddb = MagicMock()
        rddb.DetermineConnectivity.side_effect = RuntimeError("boom")
        with patch.dict(sys.modules, {"rdkit.Chem.rdDetermineBonds": rddb}):
            dlg.create_base_molecule()  # must not raise

    def test_plotter_update_render_called_when_available(self):
        dlg = ScanResultDialog.__new__(ScanResultDialog)
        dlg.trajectory = ["2\ncomment\nH 0.0 0.0 0.0\nH 0.0 0.0 1.0"]
        context = MagicMock()
        mw = MagicMock()
        context.get_main_window.return_value = mw
        dlg.context = context
        dlg.create_base_molecule()
        mw.view_3d_manager.plotter.update.assert_called_once()
        mw.view_3d_manager.plotter.render.assert_called_once()

    def test_invalid_coordinate_line_is_skipped(self):
        dlg = ScanResultDialog.__new__(ScanResultDialog)
        dlg.trajectory = [
            "3\ncomment\nH 0.0 0.0 0.0\nH not_a_number 0.0 1.0\nH 0.0 0.0 2.0"
        ]
        dlg.context = None
        dlg.create_base_molecule()  # must not raise
        self.assertIsNotNone(dlg.base_mol)


class TestOnPick(unittest.TestCase):
    def test_pick_event_sets_frame(self):
        dlg = _make_bare_dialog(results=[{"value": 1, "energy": -1}])
        event = MagicMock()
        event.artist = MagicMock()
        event.ind = [2]
        with patch.object(dlg, "set_frame") as mock_set:
            dlg.on_pick(event)
        mock_set.assert_called_once_with(2)

    def test_no_artist_no_op(self):
        dlg = _make_bare_dialog(results=[{"value": 1, "energy": -1}])
        event = MagicMock()
        event.artist = None
        with patch.object(dlg, "set_frame") as mock_set:
            dlg.on_pick(event)
        mock_set.assert_not_called()


class TestOnHover(unittest.TestCase):
    def _make_dlg(self):
        dlg = _make_bare_dialog(
            results=[{"value": 1.0, "energy": -1.0}, {"value": 2.0, "energy": -2.0}]
        )
        dlg.scatter = MagicMock()
        return dlg

    def test_hover_over_point_shows_tooltip(self):
        dlg = self._make_dlg()
        dlg.annot.get_visible.return_value = False
        event = MagicMock()
        event.inaxes = dlg.canvas.axes
        dlg.scatter.contains.return_value = (True, {"ind": [0]})
        dlg.scatter.get_offsets.return_value = [(1.0, -1.0), (2.0, -2.0)]
        dlg.unit_combo.currentText.return_value = "Hartree"
        dlg.chk_relative.isChecked.return_value = False

        dlg.on_hover(event)

        dlg.annot.set_text.assert_called_once()
        dlg.annot.set_visible.assert_called_with(True)
        dlg.canvas.draw_idle.assert_called_once()

    def test_hover_relative_kjmol(self):
        dlg = self._make_dlg()
        dlg.annot.get_visible.return_value = False
        event = MagicMock()
        event.inaxes = dlg.canvas.axes
        dlg.scatter.contains.return_value = (True, {"ind": [1]})
        dlg.scatter.get_offsets.return_value = [(1.0, -1.0), (2.0, -2.0)]
        dlg.unit_combo.currentText.return_value = "kJ/mol"
        dlg.chk_relative.isChecked.return_value = True

        dlg.on_hover(event)
        text = dlg.annot.set_text.call_args[0][0]
        self.assertIn("Y: 0.00000000", text)

    def test_hover_kcalmol(self):
        dlg = self._make_dlg()
        dlg.annot.get_visible.return_value = False
        event = MagicMock()
        event.inaxes = dlg.canvas.axes
        dlg.scatter.contains.return_value = (True, {"ind": [0]})
        dlg.scatter.get_offsets.return_value = [(1.0, -1.0), (2.0, -2.0)]
        dlg.unit_combo.currentText.return_value = "kcal/mol"
        dlg.chk_relative.isChecked.return_value = False

        dlg.on_hover(event)  # must not raise

    def test_hover_outside_axes_hides_visible_tooltip(self):
        dlg = self._make_dlg()
        dlg.annot.get_visible.return_value = True
        event = MagicMock()
        event.inaxes = None

        dlg.on_hover(event)

        dlg.annot.set_visible.assert_called_with(False)
        dlg.canvas.draw_idle.assert_called_once()

    def test_hover_no_scatter_no_crash(self):
        dlg = self._make_dlg()
        dlg.scatter = None
        dlg.annot.get_visible.return_value = False
        event = MagicMock()
        event.inaxes = dlg.canvas.axes

        dlg.on_hover(event)  # must not raise

    def test_hover_not_contained_and_not_visible_no_op(self):
        dlg = self._make_dlg()
        dlg.annot.get_visible.return_value = False
        event = MagicMock()
        event.inaxes = dlg.canvas.axes
        dlg.scatter.contains.return_value = (False, {})

        dlg.on_hover(event)
        dlg.annot.set_visible.assert_not_called()


class TestSetFrame(unittest.TestCase):
    def test_out_of_range_negative_is_noop(self):
        dlg = _make_bare_dialog(trajectory=["a", "b"])
        dlg.set_frame(-1)
        dlg.slider.setValue.assert_not_called()

    def test_out_of_range_too_high_is_noop(self):
        dlg = _make_bare_dialog(trajectory=["a", "b"])
        dlg.set_frame(5)
        dlg.slider.setValue.assert_not_called()

    def test_no_trajectory_is_noop(self):
        dlg = _make_bare_dialog(trajectory=None)
        dlg.set_frame(0)
        dlg.slider.setValue.assert_not_called()

    def test_valid_frame_updates_ui(self):
        dlg = _make_bare_dialog(
            results=[{"value": 1, "energy": -1}, {"value": 2, "energy": -2}],
            trajectory=["a", "b"],
        )
        with patch.object(dlg, "highlight_point") as mock_hl, patch.object(
            dlg, "update_viewer"
        ) as mock_uv:
            dlg.set_frame(1)
        self.assertEqual(dlg.frame_idx, 1)
        dlg.lbl_frame.setText.assert_called_with("Frame: 1")
        mock_hl.assert_called_once_with(1)
        mock_uv.assert_called_once_with(1)


class TestUpdateViewer(unittest.TestCase):
    def test_no_context_returns(self):
        dlg = _make_bare_dialog(trajectory=["a"])
        dlg.context = None
        dlg.update_viewer(0)  # must not raise

    def test_no_trajectory_returns(self):
        dlg = _make_bare_dialog(trajectory=None)
        dlg.update_viewer(0)  # must not raise

    def test_base_mol_efficient_update_path(self):
        dlg = _make_bare_dialog(trajectory=["2\ncomment\nH 0 0 0\nH 0 0 1"])
        dlg.base_mol = MagicMock()
        conf = MagicMock()
        dlg.base_mol.GetConformer.return_value = conf
        dlg.chk_dynamic_bonds.isChecked.return_value = False

        dlg.update_viewer(0)

        dlg.context.draw_molecule_3d.assert_called_once_with(dlg.base_mol)

    def test_efficient_update_exception_falls_back(self):
        dlg = _make_bare_dialog(trajectory=["2\ncomment\nH 0 0 0\nH 0 0 1"])
        dlg.base_mol = MagicMock()
        dlg.base_mol.GetConformer.side_effect = RuntimeError("boom")
        dlg.chk_dynamic_bonds.isChecked.return_value = False

        dlg.update_viewer(0)  # must not raise (falls back to full reload)

    def test_dynamic_bonds_uses_full_reload(self):
        dlg = _make_bare_dialog(trajectory=["2\ncomment\nH 0 0 0\nH 0 0 1"])
        dlg.base_mol = MagicMock()
        dlg.chk_dynamic_bonds.isChecked.return_value = True

        dlg.update_viewer(0)  # must not raise; base_mol path skipped
        dlg.base_mol.GetConformer.assert_not_called()

    def test_no_base_mol_uses_full_reload(self):
        dlg = _make_bare_dialog(trajectory=["2\ncomment\nH 0 0 0\nH 0 0 1"])
        dlg.base_mol = None
        dlg.update_viewer(0)  # must not raise

    def test_no_chk_dynamic_bonds_defaults_false(self):
        dlg = _make_bare_dialog(trajectory=["2\ncomment\nH 0 0 0\nH 0 0 1"])
        dlg.base_mol = MagicMock()
        del dlg.chk_dynamic_bonds
        dlg.update_viewer(0)
        dlg.context.draw_molecule_3d.assert_called_once_with(dlg.base_mol)


class TestOnSliderChange(unittest.TestCase):
    def test_delegates_to_set_frame(self):
        dlg = _make_bare_dialog()
        with patch.object(dlg, "set_frame") as mock_set:
            dlg.on_slider_change(3)
        mock_set.assert_called_once_with(3)


class TestTogglePlay(unittest.TestCase):
    def test_starts_playing(self):
        dlg = _make_bare_dialog()
        dlg.is_playing = False
        dlg.toggle_play()
        dlg.timer.start.assert_called_once_with(500)
        dlg.btn_play.setText.assert_called_with("Pause")
        self.assertTrue(dlg.is_playing)

    def test_stops_playing(self):
        dlg = _make_bare_dialog()
        dlg.is_playing = True
        dlg.toggle_play()
        dlg.timer.stop.assert_called_once()
        dlg.btn_play.setText.assert_called_with("Play")
        self.assertFalse(dlg.is_playing)


class TestSavePlot(unittest.TestCase):
    def test_success_shows_information(self):
        dlg = _make_bare_dialog()
        with patch.object(
            sr_mod.QFileDialog, "getSaveFileName", return_value=("/out.png", "")
        ), patch.object(sr_mod.QMessageBox, "information") as mock_info:
            dlg.save_plot()
        dlg.canvas.fig.savefig.assert_called_once_with("/out.png", dpi=300)
        mock_info.assert_called_once()

    def test_cancelled_dialog_no_op(self):
        dlg = _make_bare_dialog()
        with patch.object(
            sr_mod.QFileDialog, "getSaveFileName", return_value=("", "")
        ):
            dlg.save_plot()
        dlg.canvas.fig.savefig.assert_not_called()

    def test_exception_shows_critical(self):
        dlg = _make_bare_dialog()
        with patch.object(
            sr_mod.QFileDialog, "getSaveFileName", return_value=("/out.png", "")
        ), patch.object(sr_mod.QMessageBox, "critical") as mock_crit:
            dlg.canvas.fig.savefig.side_effect = RuntimeError("boom")
            dlg.save_plot()
        mock_crit.assert_called_once()


class TestSaveCsv(unittest.TestCase):
    def test_no_results_returns(self):
        dlg = _make_bare_dialog(results=None)
        dlg.save_csv()  # must not raise

    def test_exception_shows_critical(self):
        dlg = _make_bare_dialog(results=[{"value": 1, "energy": -1}])
        with patch.object(
            sr_mod.QFileDialog, "getSaveFileName", return_value=("/out.csv", "")
        ), patch("builtins.open", side_effect=OSError("disk full")), patch.object(
            sr_mod.QMessageBox, "critical"
        ) as mock_crit:
            dlg.save_csv()
        mock_crit.assert_called_once()

    def test_no_scan_result_dir_uses_bare_filename(self):
        dlg = _make_bare_dialog(results=[{"value": 1, "energy": -1}])
        dlg.scan_result_dir = None
        with patch.object(
            sr_mod.QFileDialog, "getSaveFileName", return_value=("", "")
        ) as mock_dialog:
            dlg.save_csv()
        args = mock_dialog.call_args[0]
        self.assertEqual(args[2], "scan_export.csv")


class TestFrameNavigation(unittest.TestCase):
    def test_prev_frame(self):
        dlg = _make_bare_dialog()
        dlg.frame_idx = 3
        with patch.object(dlg, "set_frame") as mock_set:
            dlg.prev_frame()
        mock_set.assert_called_once_with(2)

    def test_next_frame(self):
        dlg = _make_bare_dialog()
        dlg.frame_idx = 3
        with patch.object(dlg, "set_frame") as mock_set:
            dlg.next_frame()
        mock_set.assert_called_once_with(4)

    def test_next_frame_auto_wraps(self):
        dlg = _make_bare_dialog(trajectory=["a", "b"])
        dlg.frame_idx = 1
        with patch.object(dlg, "set_frame") as mock_set:
            dlg.next_frame_auto()
        mock_set.assert_called_once_with(0)

    def test_next_frame_auto_no_wrap(self):
        dlg = _make_bare_dialog(trajectory=["a", "b", "c"])
        dlg.frame_idx = 0
        with patch.object(dlg, "set_frame") as mock_set:
            dlg.next_frame_auto()
        mock_set.assert_called_once_with(1)


class TestClearSelection(unittest.TestCase):
    def test_removes_marker_and_line(self):
        dlg = _make_bare_dialog()
        dlg._highlight_marker = MagicMock()
        dlg._highlight_line = MagicMock()
        marker, line = dlg._highlight_marker, dlg._highlight_line
        dlg.clear_selection()
        marker.remove.assert_called_once()
        line.remove.assert_called_once()
        self.assertIsNone(dlg._highlight_marker)
        self.assertIsNone(dlg._highlight_line)
        dlg.canvas.draw.assert_called_once()

    def test_marker_remove_exception_silenced(self):
        dlg = _make_bare_dialog()
        dlg._highlight_marker = MagicMock()
        dlg._highlight_marker.remove.side_effect = RuntimeError("boom")
        dlg.clear_selection()  # must not raise

    def test_line_remove_exception_silenced(self):
        dlg = _make_bare_dialog()
        dlg._highlight_line = MagicMock()
        dlg._highlight_line.remove.side_effect = RuntimeError("boom")
        dlg.clear_selection()  # must not raise

    def test_no_markers_no_crash(self):
        dlg = _make_bare_dialog()
        dlg.clear_selection()  # must not raise
        dlg.canvas.draw.assert_called_once()


class TestCloseEvent(unittest.TestCase):
    def test_stops_timer_when_playing(self):
        dlg = _make_bare_dialog()
        dlg.is_playing = True
        event = MagicMock()
        dlg.closeEvent(event)
        dlg.timer.stop.assert_called_once()
        self.assertFalse(dlg.is_playing)
        event.accept.assert_called_once()

    def test_no_op_when_not_playing(self):
        dlg = _make_bare_dialog()
        dlg.is_playing = False
        event = MagicMock()
        dlg.closeEvent(event)
        dlg.timer.stop.assert_not_called()
        event.accept.assert_called_once()


class TestSaveGif(unittest.TestCase):
    def _make_dlg(self, trajectory=None):
        dlg = _make_bare_dialog(trajectory=trajectory)
        dlg.setCursor = MagicMock()
        return dlg

    def test_no_trajectory_returns(self):
        dlg = self._make_dlg(trajectory=None)
        dlg.save_gif()  # must not raise

    def test_no_pil_shows_warning(self):
        dlg = self._make_dlg(trajectory=["a"])
        with patch.object(sr_mod, "HAS_PIL", False), patch.object(
            sr_mod.QMessageBox, "warning"
        ) as mock_warn:
            dlg.save_gif()
        mock_warn.assert_called_once()

    def test_pauses_if_playing_before_settings_dialog(self):
        dlg = self._make_dlg(trajectory=["a"])
        dlg.is_playing = True
        with patch.object(sr_mod, "HAS_PIL", True), patch.object(
            dlg, "toggle_play"
        ) as mock_toggle, patch.object(sr_mod, "QDialog") as mock_dialog_cls:
            mock_dialog_instance = MagicMock()
            mock_dialog_instance.exec.return_value = 0  # Rejected
            mock_dialog_cls.return_value = mock_dialog_instance
            mock_dialog_cls.DialogCode.Accepted = 1
            dlg.save_gif()
        # Paused before opening the settings dialog, then resumed once the
        # dialog is rejected (pause + resume = 2 calls).
        self.assertEqual(mock_toggle.call_count, 2)

    def test_dialog_rejected_returns(self):
        dlg = self._make_dlg(trajectory=["a"])
        with patch.object(sr_mod, "HAS_PIL", True), patch.object(
            sr_mod, "QDialog"
        ) as mock_dialog_cls:
            mock_dialog_instance = MagicMock()
            mock_dialog_instance.exec.return_value = 0
            mock_dialog_cls.return_value = mock_dialog_instance
            mock_dialog_cls.DialogCode.Accepted = 1
            with patch.object(sr_mod, "QFileDialog") as mock_fd:
                # NOTE: some sibling test files permanently monkeypatch the
                # real `unittest.mock.MagicMock` *class* itself (e.g.
                # `sr_mod.QFileDialog.getSaveFileName = MagicMock(...)` where
                # QFileDialog is bound to the bare MagicMock class), so a
                # fresh mock's un-set attributes can inherit stale call
                # history from that shared class attribute. Force a clean
                # instance-level attribute before asserting.
                mock_fd.getSaveFileName = MagicMock()
                dlg.save_gif()
                mock_fd.getSaveFileName.assert_not_called()

    def test_no_file_path_returns(self):
        dlg = self._make_dlg(trajectory=["a"])
        with patch.object(sr_mod, "HAS_PIL", True), patch.object(
            sr_mod, "QDialog"
        ) as mock_dialog_cls, patch.object(
            sr_mod, "QFileDialog"
        ) as mock_fd, patch.object(sr_mod, "QProgressDialog") as mock_progress_cls:
            mock_dialog_instance = MagicMock()
            mock_dialog_instance.exec.return_value = 1
            mock_dialog_cls.return_value = mock_dialog_instance
            mock_dialog_cls.DialogCode.Accepted = 1
            mock_fd.getSaveFileName.return_value = ("", "")
            dlg.save_gif()
            mock_progress_cls.assert_not_called()

    def test_no_plotter_raises_and_shows_critical(self):
        dlg = self._make_dlg(trajectory=["a", "b"])
        dlg.context = MagicMock()
        mw = MagicMock(spec=[])
        dlg.context.get_main_window.return_value = mw

        with patch.object(sr_mod, "HAS_PIL", True), patch.object(
            sr_mod, "QDialog"
        ) as mock_dialog_cls, patch.object(
            sr_mod, "QFileDialog"
        ) as mock_fd, patch.object(
            sr_mod, "QProgressDialog"
        ) as mock_progress_cls, patch.object(
            sr_mod.QMessageBox, "critical"
        ) as mock_crit:
            mock_dialog_instance = MagicMock()
            mock_dialog_instance.exec.return_value = 1
            mock_dialog_cls.return_value = mock_dialog_instance
            mock_dialog_cls.DialogCode.Accepted = 1
            mock_fd.getSaveFileName.return_value = ("/out", "")
            mock_progress_instance = MagicMock()
            mock_progress_instance.wasCanceled.return_value = False
            mock_progress_cls.return_value = mock_progress_instance

            dlg.save_gif()

        mock_crit.assert_called_once()
        self.assertEqual(dlg.setCursor.call_count, 2)  # WaitCursor then ArrowCursor

    def test_successful_gif_export(self):
        dlg = self._make_dlg(trajectory=["a", "b"])
        dlg.context = MagicMock()
        mw = MagicMock()
        img_array = MagicMock()
        mw.view_3d_manager.plotter.screenshot.return_value = img_array
        dlg.context.get_main_window.return_value = mw

        fake_image = MagicMock()
        fake_img_instance = MagicMock()
        fake_image.fromarray.return_value = fake_img_instance
        fake_img_instance.convert.return_value = fake_img_instance
        fake_img_instance.split.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        fake_img_instance.quantize.return_value = fake_img_instance
        fake_image.eval.return_value = MagicMock()

        with patch.object(sr_mod, "HAS_PIL", True), patch.object(
            sr_mod, "QDialog"
        ) as mock_dialog_cls, patch.object(
            sr_mod, "QFileDialog"
        ) as mock_fd, patch.object(
            sr_mod, "QProgressDialog"
        ) as mock_progress_cls, patch.object(
            sr_mod.QMessageBox, "information"
        ) as mock_info, patch.object(
            sr_mod, "Image", fake_image
        ), patch.object(dlg, "set_frame"):
            mock_dialog_instance = MagicMock()
            mock_dialog_instance.exec.return_value = 1
            mock_dialog_cls.return_value = mock_dialog_instance
            mock_dialog_cls.DialogCode.Accepted = 1
            mock_fd.getSaveFileName.return_value = ("/out", "")
            mock_progress_instance = MagicMock()
            mock_progress_instance.wasCanceled.return_value = False
            mock_progress_cls.return_value = mock_progress_instance

            dlg.save_gif()

        mock_info.assert_called_once()
        fake_img_instance.save.assert_called_once()

    def test_no_images_captured_shows_warning(self):
        dlg = self._make_dlg(trajectory=["a", "b"])
        dlg.context = MagicMock()
        mw = MagicMock()
        mw.view_3d_manager.plotter.screenshot.return_value = None
        dlg.context.get_main_window.return_value = mw

        with patch.object(sr_mod, "HAS_PIL", True), patch.object(
            sr_mod, "QDialog"
        ) as mock_dialog_cls, patch.object(
            sr_mod, "QFileDialog"
        ) as mock_fd, patch.object(
            sr_mod, "QProgressDialog"
        ) as mock_progress_cls, patch.object(
            sr_mod.QMessageBox, "warning"
        ) as mock_warn, patch.object(dlg, "set_frame"):
            mock_dialog_instance = MagicMock()
            mock_dialog_instance.exec.return_value = 1
            mock_dialog_cls.return_value = mock_dialog_instance
            mock_dialog_cls.DialogCode.Accepted = 1
            mock_fd.getSaveFileName.return_value = ("/out.gif", "")
            mock_progress_instance = MagicMock()
            mock_progress_instance.wasCanceled.return_value = False
            mock_progress_cls.return_value = mock_progress_instance

            dlg.save_gif()

        mock_warn.assert_called_once()

    def test_progress_cancel_stops_loop_early(self):
        dlg = self._make_dlg(trajectory=["a", "b", "c"])
        dlg.context = MagicMock()
        mw = MagicMock()
        dlg.context.get_main_window.return_value = mw

        with patch.object(sr_mod, "HAS_PIL", True), patch.object(
            sr_mod, "QDialog"
        ) as mock_dialog_cls, patch.object(
            sr_mod, "QFileDialog"
        ) as mock_fd, patch.object(
            sr_mod, "QProgressDialog"
        ) as mock_progress_cls, patch.object(
            sr_mod.QMessageBox, "warning"
        ), patch.object(dlg, "set_frame") as mock_set_frame:
            mock_dialog_instance = MagicMock()
            mock_dialog_instance.exec.return_value = 1
            mock_dialog_cls.return_value = mock_dialog_instance
            mock_dialog_cls.DialogCode.Accepted = 1
            mock_fd.getSaveFileName.return_value = ("/out.gif", "")
            mock_progress_instance = MagicMock()
            mock_progress_instance.wasCanceled.return_value = True  # cancel immediately
            mock_progress_cls.return_value = mock_progress_instance

            dlg.save_gif()

        # The capture loop breaks immediately (no per-frame set_frame calls),
        # but the original frame is always restored once via set_frame after
        # the loop, regardless of cancellation.
        mock_set_frame.assert_called_once_with(dlg.frame_idx)


if __name__ == "__main__":
    unittest.main()
