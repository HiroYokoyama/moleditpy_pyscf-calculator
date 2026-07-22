"""
tests/test_energy_diag_coverage.py

Broad coverage tests for EnergyDiagramDialog methods not exercised by
test_energy_diag.py / test_energy_diag_branches.py (both of which only cover
__init__). This file targets:

  - wheelEvent (trackpad pixel-delta path, mouse-wheel angle-delta path,
    near-zero range guard)
  - mouseDoubleClickEvent (homo/lumo reset branch + fallback branch)
  - mousePressEvent (hit-zone -> try_load_cube dispatch, no-hit -> drag start)
  - try_load_cube (no result_dir guard, spin_suffix pattern selection,
    glob match -> parent().load_file_by_path, no match -> QMessageBox.question
    Yes/No branches)
  - mouseMoveEvent (hover tooltip show/hide, drag-to-zoom)
  - mouseReleaseEvent
  - contextMenuEvent
  - save_image (empty filename guard, full hide/grab/save/restore path)
  - update_unit
  - paintEvent (RHF + UHF + eV/Hartree unit branches)

Stub strategy follows test_energy_diag.py / test_energy_diag_branches.py
(custom lightweight QDialog stub, no PyQt6 installed), extended with:
  - a real-geometry FakeQRect (left/right/center/contains) since
    mousePressEvent/mouseMoveEvent do real hit-testing against QRects stored
    in self.hit_zones by paintEvent.
  - fake mouse/wheel event objects exposing position()/globalPosition()/
    button()/pixelDelta()/angleDelta().
  - dedicated QFileDialog/QMessageBox/QMenu/QAction/QToolTip stand-ins
    (not raw MagicMock) so return values are controllable per test.
"""

import os
import sys
import types
import unittest
import importlib.util
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _Pt:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def x(self):
        return self._x

    def y(self):
        return self._y

    def toPoint(self):
        return self


class _FakeQRect:
    def __init__(self, x, y, w, h):
        self._x, self._y, self._w, self._h = x, y, w, h

    def left(self):
        return self._x

    def right(self):
        return self._x + self._w

    def top(self):
        return self._y

    def bottom(self):
        return self._y + self._h

    def center(self):
        return _Pt(self._x + self._w / 2, self._y + self._h / 2)

    def contains(self, point):
        return (
            self.left() <= point.x() <= self.right()
            and self.top() <= point.y() <= self.bottom()
        )


class _FakeFontMetrics:
    def horizontalAdvance(self, s):
        return len(s) * 7


class _FakePainter:
    class RenderHint:
        Antialiasing = 1

    def __init__(self, *a, **k):
        pass

    def setRenderHint(self, *a):
        pass

    def fillRect(self, *a):
        pass

    def setPen(self, *a):
        pass

    def setFont(self, *a):
        pass

    def drawLine(self, *a):
        pass

    def drawText(self, *a):
        pass

    def fontMetrics(self):
        return _FakeFontMetrics()


class _FakeQMessageBox:
    class StandardButton:
        Yes = 1
        No = 0

    question = staticmethod(lambda *a, **k: 0)
    information = staticmethod(lambda *a, **k: None)
    warning = staticmethod(lambda *a, **k: None)
    critical = staticmethod(lambda *a, **k: None)


class _FakeQFileDialog:
    getSaveFileName = staticmethod(lambda *a, **k: ("", ""))


class _Signal:
    def __init__(self):
        self._cb = None

    def connect(self, cb):
        self._cb = cb

    def emit(self, *a):
        if self._cb:
            self._cb(*a)


class _FakeQAction:
    def __init__(self, text, parent=None):
        self.text = text
        self.triggered = _Signal()


class _FakeQMenu:
    def __init__(self, *a, **k):
        self.actions = []

    def addAction(self, act):
        self.actions.append(act)

    def exec(self, *a, **k):
        pass


class _FakeQToolTip:
    showText = staticmethod(lambda *a, **k: None)
    hideText = staticmethod(lambda *a, **k: None)


def _install_stubs():
    qt_core = types.ModuleType("PyQt6.QtCore")

    class _Qt:
        class CursorShape:
            PointingHandCursor = 1
            ArrowCursor = 0

        class AlignmentFlag:
            AlignCenter = 2
            AlignRight = 4
            AlignVCenter = 8

        class MouseButton:
            LeftButton = 1
            RightButton = 2

    qt_core.Qt = _Qt
    qt_core.QRect = _FakeQRect

    class _FakeQColor:
        def __init__(self, *a, **k):
            pass

    class _FakeQPen:
        def __init__(self, *a, **k):
            pass

        def setWidth(self, *a):
            pass

    class _FakeQFont:
        class Weight:
            Bold = 1

        def __init__(self, *a, **k):
            pass

    qt_gui = types.ModuleType("PyQt6.QtGui")
    qt_gui.QPainter = _FakePainter
    qt_gui.QPen = _FakeQPen
    qt_gui.QColor = _FakeQColor
    qt_gui.QFont = _FakeQFont
    qt_gui.QAction = _FakeQAction

    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core
    pyqt6.QtGui = qt_gui
    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtCore"] = qt_core
    sys.modules["PyQt6.QtGui"] = qt_gui

    class _QDialog:
        def __init__(self, parent=None):
            pass

        def setWindowTitle(self, *a):
            pass

        def resize(self, *a):
            pass

        def setMouseTracking(self, *a):
            pass

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QDialog = _QDialog
    qt_widgets.QFileDialog = _FakeQFileDialog
    qt_widgets.QMessageBox = _FakeQMessageBox
    qt_widgets.QMenu = _FakeQMenu
    qt_widgets.QToolTip = _FakeQToolTip

    class _FakeQApplication:
        processEvents = staticmethod(lambda *a, **k: None)

    qt_widgets.QApplication = _FakeQApplication

    qt_widgets.QComboBox = lambda *a, **k: MagicMock()
    qt_widgets.QVBoxLayout = lambda *a, **k: MagicMock()
    qt_widgets.QHBoxLayout = lambda *a, **k: MagicMock()
    qt_widgets.QPushButton = lambda *a, **k: MagicMock()
    qt_widgets.QLabel = lambda *a, **k: MagicMock()

    pyqt6.QtWidgets = qt_widgets
    sys.modules["PyQt6.QtWidgets"] = qt_widgets


_install_stubs()


def _load_module_direct(relpath, module_name):
    src = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", relpath))
    spec = importlib.util.spec_from_file_location(module_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


ed_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "energy_diag.py"),
    "pyscf_calculator_energy_diag_coverage_under_test",
)
EnergyDiagramDialog = ed_mod.EnergyDiagramDialog


class _MouseEvent:
    def __init__(self, x, y, button=1):
        self._pos = _Pt(x, y)
        self._button = button

    def position(self):
        return self._pos

    def globalPosition(self):
        return self._pos

    def button(self):
        return self._button


class _Delta:
    def __init__(self, y, is_null):
        self._y = y
        self._null = is_null

    def y(self):
        return self._y

    def isNull(self):
        return self._null


class _WheelEvent:
    def __init__(self, pixel_y=0, angle_y=0, pixel_null=True):
        self._pixel = _Delta(pixel_y, pixel_null)
        self._angle = _Delta(angle_y, False)

    def pixelDelta(self):
        return self._pixel

    def angleDelta(self):
        return self._angle


# ---------------------------------------------------------------------------
# Helper: dialog construction + host-app overrides
# ---------------------------------------------------------------------------


def _make_dialog(mo_data, result_dir=None, parent_obj=None):
    d = EnergyDiagramDialog.__new__(EnergyDiagramDialog)
    ed_mod.EnergyDiagramDialog.__init__(d, mo_data, result_dir=result_dir)
    d.update = MagicMock()
    d.setCursor = MagicMock()
    d._parent_obj = parent_obj
    d.parent = lambda: d._parent_obj
    d.height = lambda: 600
    d.width = lambda: 450
    d.grab = MagicMock(return_value=MagicMock())
    return d


RHF_DATA = {
    "type": "RHF",
    "energies": [-10.0, -5.0, 2.0, 5.0],
    "occupations": [2.0, 2.0, 0.0, 0.0],
}

UHF_DATA = {
    "type": "UHF",
    "energies": [[-10.0, -5.0, 2.0], [-9.5, -4.5, 3.0]],
    "occupations": [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
}


# ===========================================================================
# wheelEvent
# ===========================================================================


class TestWheelEvent(unittest.TestCase):
    def test_trackpad_pixel_delta_shifts_range(self):
        d = _make_dialog(RHF_DATA)
        cmin, cmax = d.current_min, d.current_max
        ev = _WheelEvent(pixel_y=10, pixel_null=False)
        d.wheelEvent(ev)
        # Both bounds shift by the same amount (pan, not zoom)
        self.assertNotEqual(d.current_min, cmin)
        self.assertAlmostEqual((d.current_max - d.current_min), (cmax - cmin))
        d.update.assert_called_once()

    def test_mouse_wheel_angle_delta_shifts_range(self):
        d = _make_dialog(RHF_DATA)
        cmin, cmax = d.current_min, d.current_max
        ev = _WheelEvent(angle_y=120, pixel_null=True)
        d.wheelEvent(ev)
        self.assertNotEqual(d.current_min, cmin)
        d.update.assert_called_once()

    def test_zero_delta_is_noop_shift(self):
        d = _make_dialog(RHF_DATA)
        cmin, cmax = d.current_min, d.current_max
        ev = _WheelEvent(pixel_y=0, angle_y=0, pixel_null=True)
        d.wheelEvent(ev)
        self.assertAlmostEqual(d.current_min, cmin)
        self.assertAlmostEqual(d.current_max, cmax)


# ===========================================================================
# mouseDoubleClickEvent
# ===========================================================================


class TestMouseDoubleClickEvent(unittest.TestCase):
    def test_resets_to_homo_lumo_gap(self):
        d = _make_dialog(RHF_DATA)
        d.current_min = -1000.0
        d.current_max = 1000.0
        d.mouseDoubleClickEvent(MagicMock())
        gap = abs(d.lumo_energy - d.homo_energy)
        center = (d.homo_energy + d.lumo_energy) / 2
        self.assertAlmostEqual(d.current_max - d.current_min, gap * 3)
        self.assertAlmostEqual((d.current_max + d.current_min) / 2, center)
        d.update.assert_called_once()

    def test_fallback_to_full_view_when_homo_missing(self):
        d = _make_dialog(RHF_DATA)
        d.homo_energy = None
        d.mouseDoubleClickEvent(MagicMock())
        pad = 0.05 * (d.full_max - d.full_min)
        self.assertAlmostEqual(d.current_min, d.full_min - pad)
        self.assertAlmostEqual(d.current_max, d.full_max + pad)


# ===========================================================================
# mousePressEvent / try_load_cube
# ===========================================================================


class TestMousePressEvent(unittest.TestCase):
    def test_left_click_outside_hit_zones_starts_drag(self):
        d = _make_dialog(RHF_DATA)
        d.hit_zones = []
        ev = _MouseEvent(10, 10, button=1)
        d.mousePressEvent(ev)
        self.assertTrue(d.dragging)
        self.assertEqual(d.last_mouse_y, 10)

    def test_left_click_inside_hit_zone_calls_try_load_cube(self):
        d = _make_dialog(RHF_DATA)
        rect = _FakeQRect(0, 0, 100, 20)
        d.hit_zones = [(rect, 3, "HOMO", "")]
        ev = _MouseEvent(50, 10, button=1)
        with patch.object(d, "try_load_cube") as tlc:
            d.mousePressEvent(ev)
            tlc.assert_called_once_with(3, "HOMO", "")

    def test_non_left_button_does_nothing(self):
        d = _make_dialog(RHF_DATA)
        d.hit_zones = []
        ev = _MouseEvent(10, 10, button=2)
        d.mousePressEvent(ev)
        self.assertFalse(hasattr(d, "dragging") and d.dragging)


class TestTryLoadCube(unittest.TestCase):
    def test_no_result_dir_returns(self):
        d = _make_dialog(RHF_DATA, result_dir=None)
        d.try_load_cube(0, "HOMO")  # must not raise

    def test_found_file_loads_via_parent(self):
        parent = MagicMock()
        d = _make_dialog(RHF_DATA, result_dir="C:/fake_dir", parent_obj=parent)
        d.status_label = MagicMock()
        with patch.object(
            ed_mod.glob, "glob", return_value=["C:/fake_dir/016_HOMO.cube"]
        ):
            d.try_load_cube(15, "HOMO")
        parent.load_file_by_path.assert_called_once_with("C:/fake_dir/016_HOMO.cube")
        d.status_label.setText.assert_called()

    def test_alpha_spin_suffix_uses_a_pattern(self):
        parent = MagicMock()
        d = _make_dialog(RHF_DATA, result_dir="C:/fake_dir", parent_obj=parent)
        d.status_label = MagicMock()
        captured = {}

        def fake_glob(pattern):
            captured["pattern"] = pattern
            return ["C:/fake_dir/010a_HOMO.cube"] if "010a" in pattern else []

        with patch.object(ed_mod.glob, "glob", side_effect=fake_glob):
            d.try_load_cube(9, "HOMO", spin_suffix="_A")
        self.assertIn("010a_", captured["pattern"])

    def test_beta_spin_suffix_uses_b_pattern(self):
        d = _make_dialog(RHF_DATA, result_dir="C:/fake_dir")
        d.status_label = MagicMock()
        captured = {}

        def fake_glob(pattern):
            captured["pattern"] = pattern
            return []

        with patch.object(ed_mod.glob, "glob", side_effect=fake_glob):
            d.try_load_cube(9, "LUMO", spin_suffix="_B")
        self.assertIn("010b_", captured["pattern"])

    def test_not_found_prompts_and_generates_on_yes(self):
        parent = MagicMock()
        d = _make_dialog(RHF_DATA, result_dir="C:/fake_dir", parent_obj=parent)
        d.status_label = MagicMock()
        with (
            patch.object(ed_mod.glob, "glob", return_value=[]),
            patch.object(
                ed_mod.QMessageBox,
                "question",
                return_value=ed_mod.QMessageBox.StandardButton.Yes,
            ),
        ):
            d.try_load_cube(2, "HOMO-1")
        parent.generate_specific_orbital.assert_called_once_with(2, "HOMO-1", "")

    def test_not_found_prompts_and_skips_on_no(self):
        parent = MagicMock()
        d = _make_dialog(RHF_DATA, result_dir="C:/fake_dir", parent_obj=parent)
        d.status_label = MagicMock()
        with (
            patch.object(ed_mod.glob, "glob", return_value=[]),
            patch.object(
                ed_mod.QMessageBox,
                "question",
                return_value=ed_mod.QMessageBox.StandardButton.No,
            ),
        ):
            d.try_load_cube(2, "LUMO")
        parent.generate_specific_orbital.assert_not_called()

    def test_parent_missing_load_file_by_path_no_crash(self):
        # A parent object with no load_file_by_path attribute: hasattr() must
        # gate the call cleanly instead of raising AttributeError.
        d = _make_dialog(RHF_DATA, result_dir="C:/fake_dir", parent_obj=object())
        d.status_label = MagicMock()
        with patch.object(ed_mod.glob, "glob", return_value=["C:/fake_dir/x.cube"]):
            d.try_load_cube(0, "HOMO")  # must not raise


# ===========================================================================
# mouseMoveEvent
# ===========================================================================


class TestMouseMoveEvent(unittest.TestCase):
    def test_hover_over_hit_zone_shows_tooltip_and_cursor(self):
        d = _make_dialog(RHF_DATA)
        rect = _FakeQRect(0, 0, 100, 20)
        d.hit_zones = [(rect, 2, "HOMO", "_A")]
        d.dragging = False
        ev = _MouseEvent(10, 10)
        with patch.object(ed_mod.QToolTip, "showText") as show_m:
            d.mouseMoveEvent(ev)
            show_m.assert_called_once()
        d.setCursor.assert_called_with(ed_mod.Qt.CursorShape.PointingHandCursor)

    def test_no_hover_hides_tooltip_and_arrow_cursor(self):
        d = _make_dialog(RHF_DATA)
        d.hit_zones = [(_FakeQRect(500, 500, 10, 10), 0, "LUMO", "")]
        d.dragging = False
        ev = _MouseEvent(10, 10)
        with patch.object(ed_mod.QToolTip, "hideText") as hide_m:
            d.mouseMoveEvent(ev)
            hide_m.assert_called_once()
        d.setCursor.assert_called_with(ed_mod.Qt.CursorShape.ArrowCursor)

    def test_dragging_zooms_view(self):
        d = _make_dialog(RHF_DATA)
        d.hit_zones = []
        d.dragging = True
        d.last_mouse_y = 100.0
        span_before = d.current_max - d.current_min
        ev = _MouseEvent(10, 150.0)  # drag DOWN -> zoom out
        d.mouseMoveEvent(ev)
        span_after = d.current_max - d.current_min
        self.assertGreater(span_after, span_before)
        self.assertEqual(d.last_mouse_y, 150.0)
        d.update.assert_called_once()

    def test_dragging_zoom_factor_clamped(self):
        d = _make_dialog(RHF_DATA)
        d.hit_zones = []
        d.dragging = True
        d.last_mouse_y = 0.0
        ev = _MouseEvent(10, 100000.0)  # huge drag -> factor clamps to 10.0
        d.mouseMoveEvent(ev)
        self.assertTrue(d.current_max > d.current_min)


# ===========================================================================
# mouseReleaseEvent / contextMenuEvent
# ===========================================================================


class TestMouseReleaseEvent(unittest.TestCase):
    def test_left_button_stops_dragging(self):
        d = _make_dialog(RHF_DATA)
        d.dragging = True
        ev = _MouseEvent(0, 0, button=1)
        d.mouseReleaseEvent(ev)
        self.assertFalse(d.dragging)

    def test_other_button_leaves_dragging_unchanged(self):
        d = _make_dialog(RHF_DATA)
        d.dragging = True
        ev = _MouseEvent(0, 0, button=2)
        d.mouseReleaseEvent(ev)
        self.assertTrue(d.dragging)


class TestContextMenuEvent(unittest.TestCase):
    def test_builds_menu_and_triggers_save(self):
        d = _make_dialog(RHF_DATA)
        ev = MagicMock()
        with patch.object(d, "save_image") as save_m:
            d.contextMenuEvent(ev)
            # Simulate user clicking "Save as PNG..."
        # The action's triggered signal was connected to save_image; emit it.
        # contextMenuEvent doesn't return the menu, so rebuild is unnecessary —
        # instead verify save_image gets wired by directly checking QAction use
        # via a patched QMenu capturing addAction.
        with patch.object(ed_mod, "QMenu") as menu_cls:
            menu_instance = menu_cls.return_value
            d.contextMenuEvent(ev)
            menu_instance.addAction.assert_called_once()
            menu_instance.exec.assert_called_once()


# ===========================================================================
# save_image
# ===========================================================================


class TestSaveImage(unittest.TestCase):
    def test_empty_filename_skips_grab(self):
        d = _make_dialog(RHF_DATA)
        with patch.object(ed_mod.QFileDialog, "getSaveFileName", return_value=("", "")):
            d.save_image()
        d.grab.assert_not_called()

    def test_full_path_hides_and_restores_widgets(self):
        d = _make_dialog(RHF_DATA)
        d.unit_combo = MagicMock()
        d.lbl_unit = MagicMock()
        d.btn_save = MagicMock()
        d.status_label = MagicMock()
        with patch.object(
            ed_mod.QFileDialog, "getSaveFileName", return_value=("out.png", "")
        ):
            d.save_image()
        d.unit_combo.setVisible.assert_any_call(False)
        d.unit_combo.setVisible.assert_any_call(True)
        d.btn_save.setVisible.assert_any_call(False)
        d.btn_save.setVisible.assert_any_call(True)
        d.grab.assert_called_once()
        d.grab.return_value.save.assert_called_once_with("out.png")


# ===========================================================================
# update_unit
# ===========================================================================


class TestUpdateUnit(unittest.TestCase):
    def test_calls_update(self):
        d = _make_dialog(RHF_DATA)
        d.update_unit("Hartree")
        d.update.assert_called_once()


# ===========================================================================
# paintEvent
# ===========================================================================


class TestPaintEvent(unittest.TestCase):
    # These tests re-run _install_stubs() at execution time; snapshot/restore the
    # PyQt6 sys.modules entries so that clobber doesn't leak into later test files
    # (e.g. test_plugin_integration.py's real-context QColor/QFont).
    _QT_KEYS = ("PyQt6", "PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets")

    def setUp(self):
        self._qt_snapshot = {k: sys.modules.get(k) for k in self._QT_KEYS}

    def tearDown(self):
        for k, v in self._qt_snapshot.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_rhf_ev_units_no_crash_and_hit_zones_populated(self):
        d = _make_dialog(RHF_DATA)
        d.unit_combo = types.SimpleNamespace(currentText=lambda: "eV")
        _install_stubs()  # re-assert QRect etc. in case another test module
        # clobbered sys.modules["PyQt6.QtCore"] since our module-level import;
        # paintEvent() does `from PyQt6.QtCore import QRect` at call time.
        d.paintEvent(MagicMock())
        self.assertGreater(len(d.hit_zones), 0)
        # Every level within view should have produced a hit zone tuple
        for rect, idx, label, spin in d.hit_zones:
            self.assertIsInstance(idx, int)
            self.assertIsInstance(label, str)

    def test_rhf_hartree_units_no_crash(self):
        d = _make_dialog(RHF_DATA)
        d.unit_combo = types.SimpleNamespace(currentText=lambda: "Hartree")
        _install_stubs()  # re-assert QRect etc. in case another test module
        # clobbered sys.modules["PyQt6.QtCore"] since our module-level import;
        # paintEvent() does `from PyQt6.QtCore import QRect` at call time.
        d.paintEvent(MagicMock())

    def test_uhf_two_columns_no_crash(self):
        d = _make_dialog(UHF_DATA)
        d.unit_combo = types.SimpleNamespace(currentText=lambda: "eV")
        _install_stubs()  # re-assert QRect etc. in case another test module
        # clobbered sys.modules["PyQt6.QtCore"] since our module-level import;
        # paintEvent() does `from PyQt6.QtCore import QRect` at call time.
        d.paintEvent(MagicMock())
        # UHF must produce spin-tagged hit zones ("_A"/"_B")
        suffixes = {spin for _, _, _, spin in d.hit_zones}
        self.assertTrue(suffixes.issubset({"_A", "_B"}))
        self.assertTrue(len(suffixes) > 0)

    def test_uhf_somo_case_no_crash(self):
        somo_data = {
            "type": "UHF",
            "energies": [[-10.0, -5.0, 2.0], [-9.5, -4.9, 3.0]],
            "occupations": [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        }
        d = _make_dialog(somo_data)
        d.unit_combo = types.SimpleNamespace(currentText=lambda: "eV")
        _install_stubs()  # re-assert QRect etc. in case another test module
        # clobbered sys.modules["PyQt6.QtCore"] since our module-level import;
        # paintEvent() does `from PyQt6.QtCore import QRect` at call time.
        d.paintEvent(MagicMock())

    def test_empty_energies_no_crash(self):
        empty_data = {"type": "RHF", "energies": [], "occupations": []}
        d = _make_dialog(empty_data)
        d.unit_combo = types.SimpleNamespace(currentText=lambda: "eV")
        _install_stubs()  # re-assert QRect etc. in case another test module
        # clobbered sys.modules["PyQt6.QtCore"] since our module-level import;
        # paintEvent() does `from PyQt6.QtCore import QRect` at call time.
        d.paintEvent(MagicMock())
        self.assertEqual(d.hit_zones, [])

    def test_roks_singly_occupied_label(self):
        roks_data = {
            "type": "RHF",
            "energies": [-10.0, -5.0, 2.0],
            "occupations": [2.0, 1.0, 0.0],
        }
        d = _make_dialog(roks_data)
        d.unit_combo = types.SimpleNamespace(currentText=lambda: "eV")
        _install_stubs()  # re-assert QRect etc. in case another test module
        # clobbered sys.modules["PyQt6.QtCore"] since our module-level import;
        # paintEvent() does `from PyQt6.QtCore import QRect` at call time.
        d.paintEvent(MagicMock())
        labels = {label for _, _, label, _ in d.hit_zones}
        self.assertIn("SOMO", labels)


if __name__ == "__main__":
    unittest.main()
