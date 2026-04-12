"""
tests/test_freq_vis_normalizer.py

Tests for FreqVisualizer frequency normalization (__init__ lines 29-46)
and SpectrumWidget real construction + set_params() (lines 543-565).

Coverage targets (freq_vis.py):
  - FreqVisualizer.__init__ frequency list/tuple/ndarray unwrapping (32-34)
  - Complex frequency handling: imaginary → negative real (37-44)
  - Zero-length inner list → 0.0 (34)
  - SpectrumWidget.__init__ full path via real constructor (543-557)
  - SpectrumWidget.set_params() (559-565)
"""
import os
import sys
import types
import unittest
import importlib.util
import numpy as np
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Qt stubs — force-install to avoid interference from other test files
# ---------------------------------------------------------------------------

def _install_stubs():
    class _QWidget:
        def __init__(self, *a, **kw): pass
        def setBackgroundRole(self, *a): pass
        def setAutoFillBackground(self, *a): pass
        def update(self): pass
        def width(self): return 400
        def height(self): return 300

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    for name in ["QVBoxLayout", "QHBoxLayout", "QPushButton", "QLabel",
                 "QTreeWidget", "QTreeWidgetItem", "QHeaderView",
                 "QDoubleSpinBox", "QCheckBox", "QGroupBox", "QSpinBox",
                 "QDialog", "QFileDialog", "QMessageBox", "QApplication",
                 "QFormLayout", "QDialogButtonBox"]:
        setattr(qt_widgets, name, MagicMock)
    qt_widgets.QWidget = _QWidget

    qt_gui = types.ModuleType("PyQt6.QtGui")
    for name in ["QPainter", "QPen", "QColor"]:
        setattr(qt_gui, name, MagicMock)
    # QPalette needs ColorRole.Base attribute access
    class _QPalette:
        class ColorRole:
            Base = 0
    qt_gui.QPalette = _QPalette

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
    mod_name = "pyscf_calculator_freq_vis_norm_test"
    spec = importlib.util.spec_from_file_location(mod_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_freq_vis_mod()
FreqVisualizer = _mod.FreqVisualizer
SpectrumWidget = _mod.SpectrumWidget


# ---------------------------------------------------------------------------
# Helper: build FreqVisualizer bypassing UI (init_ui / populate_list mocked)
# ---------------------------------------------------------------------------

def _make_freq_viz(freqs, modes=None, intensities=None):
    """
    Construct FreqVisualizer with mocked Qt/RDKit dependencies.
    init_ui and populate_list are patched out.
    Returns the FreqVisualizer instance.
    """
    if modes is None:
        n = max(1, len(freqs))
        modes = [np.zeros((2, 3)) for _ in range(n)]

    mock_mol = MagicMock()
    mock_mol.GetConformer.return_value.GetPositions.return_value = np.zeros((2, 3))
    mock_mw = MagicMock()

    with patch.object(FreqVisualizer, "init_ui", lambda self: None), \
         patch.object(FreqVisualizer, "populate_list", lambda self: None):
        fv = FreqVisualizer(mock_mw, mock_mol, freqs, modes, intensities)

    return fv


# ===========================================================================
# 1. FreqVisualizer frequency normalization
# ===========================================================================

class TestFreqVisualizerNormalization(unittest.TestCase):
    """FreqVisualizer.__init__ normalizes freqs to a list of plain floats."""

    def test_plain_float_list(self):
        fv = _make_freq_viz([500.0, 1000.0, 2000.0])
        self.assertEqual(fv.freqs, [500.0, 1000.0, 2000.0])

    def test_list_wrapped_scalar(self):
        """Each freq given as a 1-element list → unwrapped to scalar."""
        fv = _make_freq_viz([[500.0], [1000.0]])
        self.assertEqual(fv.freqs, [500.0, 1000.0])

    def test_tuple_wrapped_scalar(self):
        fv = _make_freq_viz([(500.0,), (1000.0,)])
        self.assertEqual(fv.freqs, [500.0, 1000.0])

    def test_ndarray_wrapped_scalar(self):
        fv = _make_freq_viz([np.array([500.0]), np.array([1000.0])])
        self.assertEqual(fv.freqs, [500.0, 1000.0])

    def test_empty_inner_list_becomes_zero(self):
        """Empty list/array inside → 0.0."""
        fv = _make_freq_viz([[]])
        self.assertEqual(fv.freqs, [0.0])

    def test_complex_freq_pure_real(self):
        """Complex with zero imag part → use real part."""
        fv = _make_freq_viz([complex(1500.0, 0.0)])
        self.assertAlmostEqual(fv.freqs[0], 1500.0)

    def test_complex_freq_imaginary_becomes_negative(self):
        """Complex with nonzero imag → -(abs(imag)) convention."""
        fv = _make_freq_viz([complex(0.0, 300.0)])
        self.assertAlmostEqual(fv.freqs[0], -300.0)

    def test_complex_freq_mixed(self):
        """complex(real, imag) with imag != 0 → -(abs imag)."""
        fv = _make_freq_viz([complex(100.0, 200.0)])
        self.assertAlmostEqual(fv.freqs[0], -200.0)

    def test_all_freqs_are_floats(self):
        fv = _make_freq_viz([500, 1000, 1500])   # plain ints
        for f in fv.freqs:
            self.assertIsInstance(f, float)

    def test_empty_freq_list(self):
        fv = _make_freq_viz([])
        self.assertEqual(fv.freqs, [])

    def test_modes_stored_as_ndarray(self):
        modes = [np.zeros((2, 3)), np.ones((2, 3))]
        fv = _make_freq_viz([500.0, 1000.0], modes=modes)
        self.assertIsInstance(fv.modes, np.ndarray)
        self.assertEqual(fv.modes.shape, (2, 2, 3))

    def test_intensities_stored(self):
        fv = _make_freq_viz([500.0, 1000.0], intensities=[1.0, 0.5])
        self.assertEqual(fv.intensities, [1.0, 0.5])

    def test_intensities_none_when_omitted(self):
        fv = _make_freq_viz([500.0])
        self.assertIsNone(fv.intensities)


# ===========================================================================
# 2. SpectrumWidget real constructor
# ===========================================================================

class TestSpectrumWidgetInit(unittest.TestCase):
    """SpectrumWidget.__init__ sets defaults and calls recalc_curve."""

    def test_constructor_sets_freqs(self):
        w = SpectrumWidget([500.0, 1000.0], [1.0, 0.5])
        self.assertEqual(w.freqs, [500.0, 1000.0])

    def test_constructor_sets_intensities(self):
        w = SpectrumWidget([500.0], [2.0])
        self.assertEqual(w.intensities, [2.0])

    def test_constructor_default_width_val(self):
        w = SpectrumWidget([500.0], [1.0])
        self.assertEqual(w.width_val, 20.0)

    def test_constructor_default_max_wn(self):
        w = SpectrumWidget([500.0], [1.0])
        self.assertEqual(w.max_wn, 4000.0)

    def test_constructor_default_use_gaussian(self):
        w = SpectrumWidget([500.0], [1.0])
        self.assertTrue(w.use_gaussian)

    def test_constructor_runs_recalc(self):
        """After __init__, curve_x/y must be populated (recalc was called)."""
        w = SpectrumWidget([1000.0], [1.0])
        self.assertEqual(len(w.curve_x), 1000)
        self.assertEqual(len(w.curve_y), 1000)


# ===========================================================================
# 3. SpectrumWidget.set_params — real method call
# ===========================================================================

class TestSpectrumWidgetSetParams(unittest.TestCase):

    def test_set_params_updates_width_val(self):
        w = SpectrumWidget([1000.0], [1.0])
        w.set_params(50.0, 3000.0, False, True)
        self.assertEqual(w.width_val, 50.0)

    def test_set_params_updates_max_wn(self):
        w = SpectrumWidget([1000.0], [1.0])
        w.set_params(20.0, 3000.0, False, True)
        self.assertEqual(w.max_wn, 3000.0)

    def test_set_params_updates_invert_y(self):
        w = SpectrumWidget([1000.0], [1.0])
        w.set_params(20.0, 4000.0, True, True)
        self.assertTrue(w.invert_y)

    def test_set_params_switches_to_lorentzian(self):
        w = SpectrumWidget([1000.0], [1.0])
        w.set_params(20.0, 4000.0, False, False)
        self.assertFalse(w.use_gaussian)

    def test_set_params_triggers_recalc(self):
        """After set_params, curve_x must reflect new max_wn."""
        w = SpectrumWidget([1000.0], [1.0])
        w.set_params(20.0, 2000.0, False, True)
        self.assertAlmostEqual(w.curve_x[-1], 2000.0)

    def test_set_params_switches_broadening_type(self):
        w = SpectrumWidget([1000.0], [1.0])
        w.set_params(20.0, 4000.0, False, True)   # Gaussian
        y_gauss = max(w.curve_y)
        w.set_params(20.0, 4000.0, False, False)  # Lorentzian
        y_lor = max(w.curve_y)
        # Both must produce non-zero peaks
        self.assertGreater(y_gauss, 0.0)
        self.assertGreater(y_lor, 0.0)


if __name__ == "__main__":
    unittest.main()
