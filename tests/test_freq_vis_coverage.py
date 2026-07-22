"""
tests/test_freq_vis_coverage.py

Broad coverage tests for freq_vis.py methods not exercised by
test_freq_vis_normalizer.py / test_freq_vis_spectrum.py:

  - FreqVisualizer.init_ui / populate_list (real widget construction)
  - update_list_and_spectrum, on_scale_changed, on_freq_selected, select_none
  - _update_timer_interval, toggle_play
  - show_spectrum (guards + dialog creation)
  - reset_geometry, update_vectors (all branches)
  - animate_frame (playing / not-playing / cached-idx / toggle_play fallback)
  - save_as_gif (early-return guards + full success + exception path)
  - cleanup (normal + exception-silenced)
  - SpectrumDialog.__init__ / update_plot
  - SpectrumWidget.paintEvent (both invert_y states, both broadening types)

Stub strategy: superset of the stubs used in test_freq_vis_normalizer.py and
test_freq_vis_spectrum.py, plus:
  - an _AutoMock mixin so QWidget/QDialog subclasses (FreqVisualizer,
    SpectrumDialog) can be freely instantiated while any undefined method
    call auto-resolves to a MagicMock (avoids the "subclassing MagicMock
    directly" recursion trap).
  - real-ish QSpinBox/QDoubleSpinBox/QCheckBox fakes so save_as_gif's locally
    constructed dialog controls behave like real widgets (value()/
    isChecked() round-trip).
  - a controllable QFileDialog/QMessageBox so save_as_gif branches can be
    driven deterministically.
"""

import os
import sys
import types
import unittest
import importlib.util
import numpy as np
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _AutoMock:
    """__getattr__ fallback: any undefined attribute becomes a cached MagicMock."""

    def __getattr__(self, name):
        m = MagicMock()
        object.__setattr__(self, name, m)
        return m


class _QWidget:
    # NOTE: deliberately NOT an _AutoMock subclass. FreqVisualizer/SpectrumWidget
    # rely on real getattr(self, "_foo", default)/hasattr(self, "_foo") semantics
    # for their own private state (e.g. _animating, _cached_anim_idx); an
    # auto-vivifying __getattr__ would make hasattr() always True and defeat
    # those guards.
    def __init__(self, *a, **kw):
        pass

    def setBackgroundRole(self, *a):
        pass

    def setAutoFillBackground(self, *a):
        pass

    def setLayout(self, *a):
        pass

    def update(self):
        pass

    def width(self):
        return getattr(self, "_w", 400)

    def height(self):
        return getattr(self, "_h", 300)


class _QDialog:
    class DialogCode:
        Accepted = 1
        Rejected = 0

    def __init__(self, parent=None):
        self._exec_result = _QDialog.DialogCode.Accepted

    def resize(self, *a):
        pass

    def setWindowTitle(self, *a):
        pass

    def exec(self):
        return self._exec_result

    def accept(self):
        pass

    def reject(self):
        pass


class _Signal:
    def connect(self, *a, **k):
        pass


class _FakeSpinBox:
    def __init__(self, *a, **k):
        self._v = 0

    def setRange(self, *a):
        pass

    def setSingleStep(self, *a):
        pass

    def setDecimals(self, *a):
        pass

    def setFixedWidth(self, *a):
        pass

    def setValue(self, v):
        self._v = v

    def value(self):
        return self._v

    @property
    def valueChanged(self):
        return _Signal()


class _FakeCheckBox:
    def __init__(self, *a, **k):
        self._c = False

    def setChecked(self, v):
        self._c = bool(v)

    def isChecked(self):
        return self._c

    @property
    def stateChanged(self):
        return _Signal()


class _FakeFileDialog:
    getSaveFileName = staticmethod(lambda *a, **k: ("", ""))


class _FakeMessageBox:
    information = staticmethod(lambda *a, **k: None)
    warning = staticmethod(lambda *a, **k: None)
    critical = staticmethod(lambda *a, **k: None)


class _FakePainter(_AutoMock):
    class RenderHint:
        Antialiasing = 1

    def __init__(self, *a, **k):
        pass


class _FakeGeneric(_AutoMock):
    """Generic stand-in for leaf Qt widgets constructed with positional args
    (e.g. QLabel("text")). Plain MagicMock can't be used as the class here:
    calling MagicMock(some_positional_arg) treats the first positional arg as
    Mock's own `spec` kwarg, which breaks when the arg is itself an instance
    (e.g. QVBoxLayout(self) inside init_ui)."""

    def __init__(self, *a, **k):
        pass


class _FakeQApplication:
    processEvents = staticmethod(lambda *a, **k: None)


class _FakeQDialogButtonBox(_AutoMock):
    class StandardButton:
        Ok = 1
        Cancel = 2

    def __init__(self, *a, **k):
        pass


class _FakeQHeaderView(_AutoMock):
    class ResizeMode:
        ResizeToContents = 1

    def __init__(self, *a, **k):
        pass


def _install_stubs():
    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QWidget = _QWidget
    qt_widgets.QDialog = _QDialog
    qt_widgets.QSpinBox = _FakeSpinBox
    qt_widgets.QDoubleSpinBox = _FakeSpinBox
    qt_widgets.QCheckBox = _FakeCheckBox
    qt_widgets.QFileDialog = _FakeFileDialog
    qt_widgets.QMessageBox = _FakeMessageBox
    qt_widgets.QApplication = _FakeQApplication
    qt_widgets.QDialogButtonBox = _FakeQDialogButtonBox
    qt_widgets.QHeaderView = _FakeQHeaderView
    for name in [
        "QVBoxLayout",
        "QHBoxLayout",
        "QPushButton",
        "QLabel",
        "QTreeWidget",
        "QTreeWidgetItem",
        "QGroupBox",
        "QFormLayout",
    ]:
        setattr(qt_widgets, name, _FakeGeneric)

    class _FakeQColor:
        def __init__(self, *a, **k):
            pass

    class _FakeQPen:
        def __init__(self, *a, **k):
            pass

        def setWidth(self, *a):
            pass

    qt_gui = types.ModuleType("PyQt6.QtGui")
    qt_gui.QPainter = _FakePainter
    qt_gui.QPen = _FakeQPen
    qt_gui.QColor = _FakeQColor

    class _QPalette:
        class ColorRole:
            Base = 0

    qt_gui.QPalette = _QPalette

    class _FakeQPointF:
        def __init__(self, *a, **k):
            pass

    qt_core = types.ModuleType("PyQt6.QtCore")

    class _Qt:
        class AlignmentFlag:
            AlignCenter = 1

    qt_core.Qt = _Qt
    qt_core.QTimer = MagicMock
    qt_core.QPointF = _FakeQPointF

    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtWidgets = qt_widgets
    pyqt6.QtGui = qt_gui
    pyqt6.QtCore = qt_core

    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtWidgets"] = qt_widgets
    sys.modules["PyQt6.QtGui"] = qt_gui
    sys.modules["PyQt6.QtCore"] = qt_core
    sys.modules.setdefault("rdkit", MagicMock())
    sys.modules.setdefault("rdkit.Geometry", MagicMock())
    sys.modules.setdefault("PIL", MagicMock())
    sys.modules.setdefault("PIL.Image", MagicMock())


_install_stubs()


def _load_freq_vis_mod():
    src = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "pyscf_calculator", "freq_vis.py")
    )
    mod_name = "pyscf_calculator_freq_vis_coverage_test"
    spec = importlib.util.spec_from_file_location(mod_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_freq_vis_mod()
FreqVisualizer = _mod.FreqVisualizer
SpectrumDialog = _mod.SpectrumDialog
SpectrumWidget = _mod.SpectrumWidget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mol(n_atoms=2):
    mock_mol = MagicMock()
    mock_mol.GetConformer.return_value.GetPositions.return_value = np.zeros(
        (n_atoms, 3)
    )
    return mock_mol


def _make_fv_full_init(freqs=None, modes=None, intensities=None, mw=None, context=None):
    """Construct via the REAL init_ui + populate_list path (for coverage)."""
    if freqs is None:
        freqs = [500.0, 1000.0]
    if modes is None:
        modes = [np.zeros((2, 3)) for _ in freqs] if freqs else [np.zeros((2, 3))]
    mock_mol = _make_mol()
    if mw is None:
        mw = types.SimpleNamespace()
    fv = FreqVisualizer(mw, mock_mol, freqs, modes, intensities, context=context)
    return fv


def _make_fv(freqs=None, modes=None, intensities=None, mw=None, context=None):
    """Construct with init_ui/populate_list patched out; widgets mocked manually."""
    if freqs is None:
        freqs = [500.0, 1000.0]
    if modes is None:
        modes = [np.zeros((2, 3)) for _ in freqs] if freqs else [np.zeros((2, 3))]
    mock_mol = _make_mol()
    if mw is None:
        mw = types.SimpleNamespace()

    with (
        patch.object(FreqVisualizer, "init_ui", lambda self: None),
        patch.object(FreqVisualizer, "populate_list", lambda self: None),
    ):
        fv = FreqVisualizer(mw, mock_mol, freqs, modes, intensities, context=context)

    fv.list_freq = MagicMock()
    fv.timer = MagicMock()
    fv.btn_play = MagicMock()
    fv.spin_fps = MagicMock()
    fv.spin_fps.value.return_value = 20
    fv.chk_vectors = MagicMock()
    fv.chk_vectors.isChecked.return_value = True
    fv.spin_scale = MagicMock()
    fv.spin_scale.value.return_value = 2.0
    fv.spin_amp = MagicMock()
    fv.spin_amp.value.return_value = 1.0
    fv.spin_freq_scale = MagicMock()
    fv.spin_freq_scale.value.return_value = 1.0
    fv.btn_gif = MagicMock()
    return fv


# ===========================================================================
# init_ui / populate_list — real widget construction path
# ===========================================================================


class TestInitUiRealPath(unittest.TestCase):
    def test_full_construction_sets_widgets(self):
        fv = _make_fv_full_init([500.0, 1000.0])
        self.assertIsNotNone(fv.list_freq)
        self.assertIsNotNone(fv.spin_scale)
        self.assertIsNotNone(fv.spin_freq_scale)
        self.assertIsNotNone(fv.spin_fps)
        self.assertIsNotNone(fv.spin_amp)
        self.assertIsNotNone(fv.btn_play)
        self.assertIsNotNone(fv.btn_gif)

    def test_populate_list_runs_update_list_and_spectrum(self):
        # Real populate_list() delegates to update_list_and_spectrum(); the
        # tree widget must receive one item per frequency.
        fv = _make_fv_full_init([500.0, 1000.0, 1500.0])
        # list_freq is a real QTreeWidget()-equivalent MagicMock instance;
        # addTopLevelItem should have been called 3 times.
        self.assertEqual(fv.list_freq.addTopLevelItem.call_count, 3)

    def test_empty_freqs_populate_list_no_items(self):
        fv = _make_fv_full_init([])
        fv.list_freq.addTopLevelItem.assert_not_called()

    def test_btn_gif_enabled_reflects_has_pil(self):
        _make_fv_full_init([500.0])
        # HAS_PIL True in this stub env (PIL mocked as importable)
        self.assertTrue(_mod.HAS_PIL)


# ===========================================================================
# update_list_and_spectrum
# ===========================================================================


class TestUpdateListAndSpectrum(unittest.TestCase):
    def test_scales_frequencies(self):
        fv = _make_fv([500.0, 1000.0])
        fv.spin_freq_scale.value.return_value = 2.0
        items = []
        fv.list_freq.addTopLevelItem.side_effect = lambda it: items.append(it)
        fv.update_list_and_spectrum()
        fv.list_freq.clear.assert_called_once()
        self.assertEqual(len(items), 2)
        items[0].setText.assert_any_call(1, "1000.00")
        items[1].setText.assert_any_call(1, "2000.00")


# ===========================================================================
# on_scale_changed / on_freq_selected / select_none
# ===========================================================================


class TestOnScaleChanged(unittest.TestCase):
    def test_delegates_to_update_vectors(self):
        fv = _make_fv()
        with patch.object(fv, "update_vectors") as m:
            fv.on_scale_changed()
            m.assert_called_once()


class TestOnFreqSelected(unittest.TestCase):
    def test_none_current_returns_early(self):
        fv = _make_fv()
        with (
            patch.object(fv, "reset_geometry") as reset_m,
            patch.object(fv, "update_vectors") as upd_m,
        ):
            fv.on_freq_selected(None, MagicMock())
            reset_m.assert_not_called()
            upd_m.assert_not_called()

    def test_selection_caches_index_and_resets(self):
        fv = _make_fv()
        fv.is_playing = False
        fv.list_freq.indexOfTopLevelItem.return_value = 1
        current = MagicMock()
        with (
            patch.object(fv, "reset_geometry") as reset_m,
            patch.object(fv, "update_vectors") as upd_m,
        ):
            fv.on_freq_selected(current, None)
            self.assertEqual(fv._cached_anim_idx, 1)
            reset_m.assert_called_once()
            upd_m.assert_called_once()

    def test_selection_while_playing_skips_reset(self):
        fv = _make_fv()
        fv.is_playing = True
        fv.list_freq.indexOfTopLevelItem.return_value = 0
        with (
            patch.object(fv, "reset_geometry") as reset_m,
            patch.object(fv, "update_vectors") as upd_m,
        ):
            fv.on_freq_selected(MagicMock(), None)
            reset_m.assert_not_called()
            upd_m.assert_not_called()


class TestSelectNone(unittest.TestCase):
    def test_clears_selection_and_resets(self):
        fv = _make_fv()
        fv.is_playing = False
        with (
            patch.object(fv, "reset_geometry") as reset_m,
            patch.object(fv, "update_vectors") as upd_m,
        ):
            fv.select_none()
        fv.list_freq.clearSelection.assert_called_once()
        fv.list_freq.setCurrentItem.assert_called_once_with(None)
        reset_m.assert_called_once()
        upd_m.assert_called_once()

    def test_stops_animation_if_playing(self):
        fv = _make_fv()
        fv.is_playing = True
        with (
            patch.object(fv, "reset_geometry"),
            patch.object(fv, "update_vectors"),
        ):
            fv.select_none()
        self.assertFalse(fv.is_playing)
        fv.timer.stop.assert_called_once()
        fv.btn_play.setText.assert_called_with("Play")


# ===========================================================================
# _update_timer_interval / toggle_play
# ===========================================================================


class TestUpdateTimerInterval(unittest.TestCase):
    def test_restarts_when_active(self):
        fv = _make_fv()
        fv.timer.isActive.return_value = True
        fv.spin_fps.value.return_value = 25
        fv._update_timer_interval()
        fv.timer.stop.assert_called_once()
        fv.timer.start.assert_called_once_with(40)  # 1000/25

    def test_noop_when_inactive(self):
        fv = _make_fv()
        fv.timer.isActive.return_value = False
        fv._update_timer_interval()
        fv.timer.start.assert_not_called()


class TestTogglePlay(unittest.TestCase):
    def test_start_playing(self):
        fv = _make_fv()
        fv.is_playing = False
        fv.list_freq.currentItem.return_value = MagicMock()
        fv.spin_fps.value.return_value = 20
        fv.toggle_play()
        self.assertTrue(fv.is_playing)
        fv.timer.start.assert_called_once_with(50)
        fv.btn_play.setText.assert_called_with("Stop")

    def test_start_without_selection_noop(self):
        fv = _make_fv()
        fv.is_playing = False
        fv.list_freq.currentItem.return_value = None
        fv.toggle_play()
        self.assertFalse(fv.is_playing)
        fv.timer.start.assert_not_called()

    def test_stop_playing_resets_geometry(self):
        fv = _make_fv()
        fv.is_playing = True
        with patch.object(fv, "reset_geometry") as reset_m:
            fv.toggle_play()
        self.assertFalse(fv.is_playing)
        fv.timer.stop.assert_called_once()
        fv.btn_play.setText.assert_called_with("Play")
        reset_m.assert_called_once()


# ===========================================================================
# show_spectrum
# ===========================================================================


class TestShowSpectrum(unittest.TestCase):
    def test_no_freqs_returns(self):
        fv = _make_fv(freqs=[])
        with patch.object(_mod, "SpectrumDialog") as dlg_cls:
            fv.show_spectrum()
            dlg_cls.assert_not_called()

    def test_no_intensities_returns(self):
        fv = _make_fv(freqs=[500.0], intensities=None)
        with patch.object(_mod, "SpectrumDialog") as dlg_cls:
            fv.show_spectrum()
            dlg_cls.assert_not_called()

    def test_empty_intensities_returns(self):
        fv = _make_fv(freqs=[500.0], intensities=[])
        with patch.object(_mod, "SpectrumDialog") as dlg_cls:
            fv.show_spectrum()
            dlg_cls.assert_not_called()

    def test_creates_dialog_with_scaled_freqs_and_safe_intensities(self):
        fv = _make_fv(freqs=[500.0, 1000.0], intensities=[1.0, None])
        fv.spin_freq_scale.value.return_value = 2.0
        with patch.object(_mod, "SpectrumDialog") as dlg_cls:
            instance = dlg_cls.return_value
            fv.show_spectrum()
            args, kwargs = dlg_cls.call_args
            self.assertEqual(args[0], [1000.0, 2000.0])
            self.assertEqual(args[1], [1.0, 0.0])
            instance.exec.assert_called_once()


# ===========================================================================
# reset_geometry
# ===========================================================================


class TestResetGeometry(unittest.TestCase):
    def test_uses_context_when_present(self):
        ctx = MagicMock()
        fv = _make_fv(context=ctx)
        fv.is_playing = True  # avoid recursive update_vectors call
        fv.reset_geometry()
        ctx.draw_molecule_3d.assert_called_once_with(fv.mol)

    def test_uses_view_3d_manager_when_no_context(self):
        vm = types.SimpleNamespace(draw_molecule_3d=MagicMock())
        mw = types.SimpleNamespace(view_3d_manager=vm)
        fv = _make_fv(mw=mw, context=None)
        fv.is_playing = True
        fv.reset_geometry()
        vm.draw_molecule_3d.assert_called_once_with(fv.mol)

    def test_calls_update_vectors_when_not_playing(self):
        fv = _make_fv()
        fv.is_playing = False
        with patch.object(fv, "update_vectors") as upd_m:
            fv.reset_geometry()
            upd_m.assert_called_once()

    def test_skips_update_vectors_when_playing(self):
        fv = _make_fv()
        fv.is_playing = True
        with patch.object(fv, "update_vectors") as upd_m:
            fv.reset_geometry()
            upd_m.assert_not_called()


# ===========================================================================
# update_vectors
# ===========================================================================


class TestUpdateVectors(unittest.TestCase):
    def _mw_with_plotter(self):
        plotter = MagicMock()
        vm = types.SimpleNamespace(plotter=plotter)
        return types.SimpleNamespace(view_3d_manager=vm), plotter

    def test_removes_stale_vector_actor(self):
        mw, plotter = self._mw_with_plotter()
        fv = _make_fv(mw=mw)
        fv.vector_actor = "old_actor"
        fv.chk_vectors.isChecked.return_value = False
        fv.update_vectors()
        plotter.remove_actor.assert_any_call("old_actor")
        self.assertIsNone(fv.vector_actor)

    def test_unchecked_renders_and_returns(self):
        mw, plotter = self._mw_with_plotter()
        fv = _make_fv(mw=mw)
        fv.chk_vectors.isChecked.return_value = False
        fv.update_vectors()
        plotter.render.assert_called()

    def test_no_current_item_renders_and_returns(self):
        mw, plotter = self._mw_with_plotter()
        fv = _make_fv(mw=mw)
        fv.chk_vectors.isChecked.return_value = True
        fv.list_freq.currentItem.return_value = None
        fv.update_vectors()
        plotter.render.assert_called()

    def test_idx_out_of_range_returns(self):
        mw, plotter = self._mw_with_plotter()
        fv = _make_fv(mw=mw, modes=[np.zeros((2, 3))])
        fv.chk_vectors.isChecked.return_value = True
        fv.list_freq.currentItem.return_value = MagicMock()
        fv.list_freq.indexOfTopLevelItem.return_value = 99
        fv.update_vectors()
        plotter.add_arrows.assert_not_called()

    def test_adds_arrows_for_valid_mode(self):
        mw, plotter = self._mw_with_plotter()
        modes = [np.ones((2, 3)), np.zeros((2, 3))]
        fv = _make_fv(mw=mw, modes=modes)
        fv.chk_vectors.isChecked.return_value = True
        fv.list_freq.currentItem.return_value = MagicMock()
        fv.list_freq.indexOfTopLevelItem.return_value = 0
        fv.spin_scale.value.return_value = 3.5
        fv.update_vectors()
        plotter.add_arrows.assert_called_once()
        _, kwargs = plotter.add_arrows.call_args
        self.assertEqual(kwargs["mag"], 3.5)
        self.assertEqual(kwargs["color"], "lightgreen")
        plotter.render.assert_called()

    def test_no_view_3d_manager_does_not_crash(self):
        fv = _make_fv(mw=types.SimpleNamespace())
        fv.chk_vectors.isChecked.return_value = True
        fv.list_freq.currentItem.return_value = MagicMock()
        fv.list_freq.indexOfTopLevelItem.return_value = 0
        fv.update_vectors()  # must not raise

    def test_exception_during_cleanup_is_silenced(self):
        mw, plotter = self._mw_with_plotter()
        plotter.remove_actor.side_effect = RuntimeError("boom")
        fv = _make_fv(mw=mw)
        fv.vector_actor = "actor1"
        fv.chk_vectors.isChecked.return_value = False
        fv.update_vectors()  # must not raise


# ===========================================================================
# animate_frame
# ===========================================================================


class TestAnimateFrame(unittest.TestCase):
    def test_not_playing_returns_immediately(self):
        fv = _make_fv()
        fv.is_playing = False
        fv.animate_frame()
        fv.list_freq.currentItem.assert_not_called()

    def test_animating_guard_skips_frame(self):
        fv = _make_fv()
        fv.is_playing = True
        fv._animating = True
        fv.animate_frame()
        fv.list_freq.currentItem.assert_not_called()

    def test_no_current_item_calls_toggle_play(self):
        fv = _make_fv()
        fv.is_playing = True
        fv.list_freq.currentItem.return_value = None
        with patch.object(fv, "toggle_play") as tp_m:
            fv.animate_frame()
            tp_m.assert_called_once()

    def test_advances_animation_step_and_updates_geometry(self):
        ctx = MagicMock()
        modes = [np.ones((2, 3))]
        fv = _make_fv(modes=modes, context=ctx)
        fv.is_playing = True
        fv.list_freq.currentItem.return_value = MagicMock()
        fv._cached_anim_idx = 0
        fv.animation_step = 0
        fv.spin_amp.value.return_value = 1.0
        fv.animate_frame()
        self.assertEqual(fv.animation_step, 1)
        ctx.draw_molecule_3d.assert_called_once_with(fv.mol)
        self.assertFalse(fv._animating)

    def test_caches_index_when_missing(self):
        modes = [np.zeros((2, 3)), np.ones((2, 3))]
        fv = _make_fv(modes=modes)
        fv.is_playing = True
        item = MagicMock()
        fv.list_freq.currentItem.return_value = item
        fv.list_freq.indexOfTopLevelItem.return_value = 1
        self.assertFalse(hasattr(fv, "_cached_anim_idx"))
        fv.animate_frame()
        self.assertEqual(fv._cached_anim_idx, 1)


# ===========================================================================
# save_as_gif
# ===========================================================================


class TestSaveAsGifGuards(unittest.TestCase):
    def test_no_mol_returns(self):
        fv = _make_fv()
        fv.mol = None
        fv.save_as_gif()  # must not raise

    def test_no_selection_warns_and_returns(self):
        fv = _make_fv()
        fv.list_freq.currentItem.return_value = None
        with patch.object(_mod.QMessageBox, "warning") as warn_m:
            fv.save_as_gif()
            warn_m.assert_called_once()

    def test_negative_index_returns(self):
        fv = _make_fv()
        fv.list_freq.currentItem.return_value = MagicMock()
        fv.list_freq.indexOfTopLevelItem.return_value = -1
        fv.save_as_gif()  # must not raise

    def test_dialog_rejected_returns(self):
        fv = _make_fv()
        fv.list_freq.currentItem.return_value = MagicMock()
        fv.list_freq.indexOfTopLevelItem.return_value = 0
        with patch.object(_mod, "QDialog") as dlg_cls:
            dlg_cls.return_value.exec.return_value = (
                _mod.QDialog.DialogCode.Rejected + 5
            )
            dlg_cls.DialogCode = _mod.QDialog.DialogCode
            fv.save_as_gif()  # must not raise / not proceed to file dialog

    def test_no_file_path_chosen_returns(self):
        fv = _make_fv()
        fv.list_freq.currentItem.return_value = MagicMock()
        fv.list_freq.indexOfTopLevelItem.return_value = 0
        with patch.object(_mod.QFileDialog, "getSaveFileName", return_value=("", "")):
            fv.save_as_gif()  # must not raise

    def test_was_playing_resumes_after_cancel(self):
        fv = _make_fv()
        fv.is_playing = True
        fv.list_freq.currentItem.return_value = None
        with patch.object(fv, "toggle_play") as tp_m:
            fv.save_as_gif()
            # toggle_play called once to stop playback before the warning path
            tp_m.assert_called_once()


class TestSaveAsGifFullPath(unittest.TestCase):
    def _fv_ready(self):
        plotter = MagicMock()
        plotter.screenshot.return_value = np.zeros((4, 4, 4), dtype=np.uint8)
        vm = types.SimpleNamespace(plotter=plotter, draw_molecule_3d=MagicMock())
        mw = types.SimpleNamespace(view_3d_manager=vm)
        modes = [np.ones((2, 3))]
        fv = _make_fv(mw=mw, modes=modes)
        fv.list_freq.currentItem.return_value = MagicMock()
        fv.list_freq.indexOfTopLevelItem.return_value = 0
        return fv, plotter

    def test_success_path_saves_gif(self):
        fv, plotter = self._fv_ready()
        gif_path = os.path.join(
            os.environ.get("TEMP", "."), "freq_vis_coverage_test.gif"
        )
        with (
            patch.object(
                _mod.QFileDialog, "getSaveFileName", return_value=(gif_path, "")
            ),
            patch.object(_mod.time, "sleep"),
            patch.object(_mod.QMessageBox, "information") as info_m,
        ):
            fv.save_as_gif()
        info_m.assert_called_once()

    def test_success_path_appends_gif_extension(self):
        fv, plotter = self._fv_ready()
        gif_path = os.path.join(
            os.environ.get("TEMP", "."), "freq_vis_coverage_test_noext"
        )
        with (
            patch.object(
                _mod.QFileDialog, "getSaveFileName", return_value=(gif_path, "")
            ),
            patch.object(_mod.time, "sleep"),
            patch.object(_mod.QMessageBox, "information") as info_m,
        ):
            fv.save_as_gif()
        # Success message reports the actual saved filename, which must have
        # had ".gif" appended (input path had no extension).
        msg = info_m.call_args[0][2]
        self.assertTrue(msg.endswith(".gif"))

    def test_exception_during_frame_loop_shows_critical(self):
        fv, plotter = self._fv_ready()
        plotter.render.side_effect = RuntimeError("render failed")
        gif_path = os.path.join(os.environ.get("TEMP", "."), "freq_vis_err.gif")
        with (
            patch.object(
                _mod.QFileDialog, "getSaveFileName", return_value=(gif_path, "")
            ),
            patch.object(_mod.time, "sleep"),
            patch.object(_mod.QMessageBox, "critical") as crit_m,
        ):
            fv.save_as_gif()
        crit_m.assert_called_once()

    def test_was_playing_toggled_back_on_after_export(self):
        fv, plotter = self._fv_ready()
        fv.is_playing = True
        gif_path = os.path.join(os.environ.get("TEMP", "."), "freq_vis_resume.gif")
        with (
            patch.object(
                _mod.QFileDialog, "getSaveFileName", return_value=(gif_path, "")
            ),
            patch.object(_mod.time, "sleep"),
            patch.object(_mod.QMessageBox, "information"),
        ):
            # toggle_play is called twice for real: once to stop (is_playing True->False)
            # before export, once to resume (False->True) after.
            fv.save_as_gif()
        self.assertTrue(fv.is_playing)


# ===========================================================================
# cleanup
# ===========================================================================


class TestCleanup(unittest.TestCase):
    def test_stops_active_timer_and_removes_actor(self):
        plotter = MagicMock()
        vm = types.SimpleNamespace(plotter=plotter)
        mw = types.SimpleNamespace(view_3d_manager=vm)
        fv = _make_fv(mw=mw)
        fv.timer.isActive.return_value = True
        fv.vector_actor = "actor_x"
        fv.cleanup()
        fv.timer.stop.assert_called_once()
        self.assertFalse(fv.is_playing)
        plotter.remove_actor.assert_called_once_with("actor_x")
        self.assertIsNone(fv.vector_actor)

    def test_no_timer_or_actor_is_noop(self):
        fv = _make_fv()
        fv.timer.isActive.return_value = False
        fv.vector_actor = None
        fv.cleanup()  # must not raise

    def test_exception_is_silenced(self):
        plotter = MagicMock()
        plotter.remove_actor.side_effect = RuntimeError("fail")
        vm = types.SimpleNamespace(plotter=plotter)
        mw = types.SimpleNamespace(view_3d_manager=vm)
        fv = _make_fv(mw=mw)
        fv.vector_actor = "x"
        fv.cleanup()  # must not raise


# ===========================================================================
# SpectrumDialog
# ===========================================================================


class TestSpectrumDialog(unittest.TestCase):
    def test_construction_creates_plot_widget(self):
        dlg = SpectrumDialog([500.0, 1000.0], [1.0, 0.5])
        self.assertIsInstance(dlg.plot_widget, SpectrumWidget)
        self.assertEqual(list(dlg.freqs), [500.0, 1000.0])

    def test_update_plot_forwards_current_control_values(self):
        dlg = SpectrumDialog([1000.0], [1.0])
        dlg.spin_width.setValue(55.0)
        dlg.spin_max_wn.setValue(2500.0)
        dlg.chk_invert_y.setChecked(True)
        dlg.update_plot()
        self.assertEqual(dlg.plot_widget.width_val, 55.0)
        self.assertEqual(dlg.plot_widget.max_wn, 2500.0)
        self.assertTrue(dlg.plot_widget.invert_y)
        self.assertTrue(dlg.plot_widget.use_gaussian)


# ===========================================================================
# SpectrumWidget.paintEvent
# ===========================================================================


class TestSpectrumWidgetPaintEvent(unittest.TestCase):
    def _widget(self, invert_y=False, use_gaussian=True):
        w = SpectrumWidget([1000.0, 2000.0], [1.0, 0.5])
        w.invert_y = invert_y
        w.use_gaussian = use_gaussian
        w.recalc_curve()
        return w

    def test_paint_event_normal(self):
        w = self._widget()
        w.paintEvent(MagicMock())  # must not raise

    def test_paint_event_inverted(self):
        w = self._widget(invert_y=True)
        w.paintEvent(MagicMock())  # must not raise

    def test_paint_event_lorentzian(self):
        w = self._widget(use_gaussian=False)
        w.paintEvent(MagicMock())  # must not raise

    def test_paint_event_empty_curve(self):
        w = SpectrumWidget([], [])
        w.paintEvent(MagicMock())  # must not raise; max_y falls back to 1.0

    def test_paint_event_all_zero_intensity(self):
        w = SpectrumWidget([1000.0], [0.0])
        w.paintEvent(MagicMock())  # max_y == 0 -> falls back to 1.0


if __name__ == "__main__":
    unittest.main()
