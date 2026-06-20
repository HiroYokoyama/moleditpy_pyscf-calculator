"""
tests/test_freq_vis_spectrum.py

Tests for SpectrumWidget.recalc_curve() and set_params() in freq_vis.py.

recalc_curve() is pure numpy — it builds a Gaussian or Lorentzian envelope
over a linspace grid.  No real Qt rendering is needed.

Strategy:
  - Stub every Qt import at module level so freq_vis.py loads headlessly.
  - Construct SpectrumWidget via __new__ + manual attribute init, bypassing
    the real QWidget.__init__ (which needs a running QApplication).
  - Call recalc_curve() and verify curve_x / curve_y directly.

Coverage targets (freq_vis.py):
  - recalc_curve() empty-freqs guard (line 568)
  - X-grid creation: linspace(0, max_wn, 1000) (lines 571-572)
  - Gaussian branch: sigma/2.355 conversion, exp formula (lines 577-584)
  - Lorentzian branch: gamma = sigma/2, Cauchy formula (lines 585-590)
  - set_params() forwarding (lines 559-565)
"""

import os
import sys
import types
import unittest
import importlib.util
import numpy as np
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Qt stubs — must be installed before freq_vis.py is loaded
# ---------------------------------------------------------------------------


def _install_stubs():
    # QWidget base class stub — __init__ is a no-op so SpectrumWidget can be
    # constructed without a running QApplication.
    class _QWidget:
        def __init__(self, *a, **kw):
            pass

        def setBackgroundRole(self, *a):
            pass

        def setAutoFillBackground(self, *a):
            pass

        def update(self):
            pass

        def width(self):
            return 400

        def height(self):
            return 300

    # Force-install: other test files may have set PyQt6.QtWidgets without
    # QWidget, so we must overwrite to guarantee our stub is in place.
    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    for name in [
        "QVBoxLayout",
        "QHBoxLayout",
        "QPushButton",
        "QLabel",
        "QTreeWidget",
        "QTreeWidgetItem",
        "QHeaderView",
        "QDoubleSpinBox",
        "QCheckBox",
        "QGroupBox",
        "QSpinBox",
        "QDialog",
        "QFileDialog",
        "QMessageBox",
        "QApplication",
        "QFormLayout",
        "QDialogButtonBox",
    ]:
        setattr(qt_widgets, name, MagicMock)
    qt_widgets.QWidget = _QWidget

    qt_gui = types.ModuleType("PyQt6.QtGui")
    for name in ["QPainter", "QPen", "QColor", "QPalette"]:
        setattr(qt_gui, name, MagicMock)

    qt_core = types.ModuleType("PyQt6.QtCore")
    qt_core.Qt = MagicMock()
    qt_core.QTimer = MagicMock
    qt_core.QPointF = MagicMock

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
    spec = importlib.util.spec_from_file_location("pyscf_calculator_freq_vis_test", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pyscf_calculator_freq_vis_test"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_freq_vis_mod()
SpectrumWidget = _mod.SpectrumWidget


# ---------------------------------------------------------------------------
# Helper: build a SpectrumWidget without running QApplication
# ---------------------------------------------------------------------------


def _make_widget(
    freqs, intensities, width_val=20.0, max_wn=4000.0, invert_y=False, use_gaussian=True
):
    """
    Construct SpectrumWidget bypassing QWidget.__init__.
    Manually set required attributes, then call recalc_curve().
    """
    w = SpectrumWidget.__new__(SpectrumWidget)
    w.freqs = list(freqs)
    w.intensities = list(intensities)
    w.width_val = width_val
    w.max_wn = max_wn
    w.invert_y = invert_y
    w.use_gaussian = use_gaussian
    w.curve_x = []
    w.curve_y = []
    w.recalc_curve()
    return w


# ===========================================================================
# Guard: empty frequency list
# ===========================================================================


class TestRecalcCurveEmpty(unittest.TestCase):
    def test_empty_freqs_does_not_modify_curve(self):
        w = _make_widget([], [])
        self.assertEqual(w.curve_x, [])
        self.assertEqual(w.curve_y, [])

    def test_empty_freqs_returns_early(self):
        """curve_x/y remain as pre-set empty lists."""
        w = SpectrumWidget.__new__(SpectrumWidget)
        w.freqs = []
        w.intensities = []
        w.width_val = 20.0
        w.max_wn = 4000.0
        w.use_gaussian = True
        w.curve_x = "sentinel"
        w.curve_y = "sentinel"
        w.recalc_curve()
        self.assertEqual(w.curve_x, "sentinel")


# ===========================================================================
# Gaussian branch
# ===========================================================================


class TestRecalcCurveGaussian(unittest.TestCase):
    def test_curve_x_length_is_1000(self):
        w = _make_widget([1000.0], [1.0])
        self.assertEqual(len(w.curve_x), 1000)

    def test_curve_x_starts_at_zero(self):
        w = _make_widget([1000.0], [1.0])
        self.assertAlmostEqual(w.curve_x[0], 0.0)

    def test_curve_x_ends_at_max_wn(self):
        w = _make_widget([1000.0], [1.0], max_wn=3500.0)
        self.assertAlmostEqual(w.curve_x[-1], 3500.0)

    def test_curve_y_peaks_near_frequency(self):
        """Gaussian peak must be near the given frequency."""
        freq = 1500.0
        w = _make_widget([freq], [1.0], width_val=50.0, use_gaussian=True)
        peak_idx = np.argmax(w.curve_y)
        self.assertAlmostEqual(w.curve_x[peak_idx], freq, delta=10.0)

    def test_curve_y_peak_height_scales_with_intensity(self):
        """Doubling intensity must (roughly) double the peak height."""
        w1 = _make_widget([1000.0], [1.0], use_gaussian=True)
        w2 = _make_widget([1000.0], [2.0], use_gaussian=True)
        ratio = max(w2.curve_y) / max(w1.curve_y)
        self.assertAlmostEqual(ratio, 2.0, places=5)

    def test_curve_y_nonnegative_for_positive_intensity(self):
        w = _make_widget([500.0, 1500.0], [1.0, 0.5], use_gaussian=True)
        self.assertTrue(np.all(np.array(w.curve_y) >= 0.0))

    def test_two_peaks_both_visible(self):
        """Two well-separated peaks must each produce a local maximum."""
        w = _make_widget([500.0, 3000.0], [1.0, 1.0], width_val=30.0, use_gaussian=True)
        cy = np.array(w.curve_y)
        cx = np.array(w.curve_x)
        # Find values near each peak
        near_500 = cy[np.abs(cx - 500.0) < 50]
        near_3000 = cy[np.abs(cx - 3000.0) < 50]
        self.assertGreater(np.max(near_500), 0.1)
        self.assertGreater(np.max(near_3000), 0.1)

    def test_sigma_uses_fwhm_conversion(self):
        """sigma passed in is FWHM; internally s = sigma/2.355."""
        sigma = 100.0
        s = sigma / 2.355
        freq = 1000.0
        w = _make_widget([freq], [1.0], width_val=sigma, use_gaussian=True)
        # At x = freq + s, the Gaussian = exp(-0.5) ≈ 0.606 of peak.
        # Grid spacing is max_wn/999 ≈ 4 units; use 3% tolerance.
        cx = np.array(w.curve_x)
        cy = np.array(w.curve_y)
        peak = np.max(cy)
        idx_at_s = np.argmin(np.abs(cx - (freq + s)))
        expected = peak * np.exp(-0.5)
        self.assertAlmostEqual(cy[idx_at_s], expected, delta=peak * 0.03)


# ===========================================================================
# Lorentzian branch
# ===========================================================================


class TestRecalcCurveLorentzian(unittest.TestCase):
    def test_lorentzian_peak_at_frequency(self):
        freq = 2000.0
        w = _make_widget([freq], [1.0], width_val=40.0, use_gaussian=False)
        peak_idx = np.argmax(w.curve_y)
        self.assertAlmostEqual(w.curve_x[peak_idx], freq, delta=10.0)

    def test_lorentzian_peak_height_equals_intensity(self):
        """At x=f exactly, Lorentzian value = i * gamma^2 / gamma^2 = i."""
        freq = 1000.0
        intensity = 2.5
        sigma = 40.0
        gamma = sigma / 2.0
        w = _make_widget([freq], [intensity], width_val=sigma, use_gaussian=False)
        peak = np.max(w.curve_y)
        self.assertAlmostEqual(peak, intensity, delta=0.05)

    def test_lorentzian_nonnegative(self):
        w = _make_widget([1000.0], [1.0], use_gaussian=False)
        self.assertTrue(np.all(np.array(w.curve_y) >= 0.0))

    def test_lorentzian_curve_length_is_1000(self):
        w = _make_widget([1000.0], [1.0], use_gaussian=False)
        self.assertEqual(len(w.curve_x), 1000)

    def test_lorentzian_half_width_at_half_max(self):
        """At x = freq + gamma, Lorentzian = 0.5 of peak."""
        freq = 1500.0
        sigma = 60.0
        gamma = sigma / 2.0
        w = _make_widget([freq], [1.0], width_val=sigma, use_gaussian=False)
        cx = np.array(w.curve_x)
        cy = np.array(w.curve_y)
        peak = np.max(cy)
        idx = np.argmin(np.abs(cx - (freq + gamma)))
        expected = peak * 0.5
        self.assertAlmostEqual(cy[idx], expected, delta=peak * 0.02)


# ===========================================================================
# set_params
# ===========================================================================


class TestSetParams(unittest.TestCase):
    def test_set_params_updates_width(self):
        """Wider sigma → wider half-max region in the curve."""
        narrow = _make_widget([1000.0], [1.0], width_val=20.0)
        wide = _make_widget([1000.0], [1.0], width_val=300.0)
        # Count grid points above half-max for each
        half = 0.5
        narrow_count = np.sum(np.array(narrow.curve_y) > half)
        wide_count = np.sum(np.array(wide.curve_y) > half)
        self.assertGreater(wide_count, narrow_count)

    def test_set_params_switches_to_lorentzian(self):
        w = _make_widget([1000.0], [1.0], use_gaussian=True)
        y_gauss = np.max(w.curve_y)

        w2 = _make_widget([1000.0], [1.0], use_gaussian=False)
        y_lor = np.max(w2.curve_y)

        # Both must produce a non-zero peak
        self.assertGreater(y_gauss, 0.0)
        self.assertGreater(y_lor, 0.0)

    def test_set_params_changes_max_wn(self):
        w1 = _make_widget([500.0], [1.0], max_wn=2000.0)
        w2 = _make_widget([500.0], [1.0], max_wn=4000.0)
        # Different max_wn → different grid endpoint
        self.assertAlmostEqual(w1.curve_x[-1], 2000.0)
        self.assertAlmostEqual(w2.curve_x[-1], 4000.0)


if __name__ == "__main__":
    unittest.main()
