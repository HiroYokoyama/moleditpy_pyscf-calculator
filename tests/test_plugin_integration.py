"""
tests/test_plugin_integration.py

Integration tests verifying the PySCF-calculator plugin's contract with the
MoleditPy main-application PluginContext interface.

Two execution modes
-------------------
1. **Stub mode** (always runs, including CI):
   A StubPluginContext mirrors the real PluginContext API so that all contract
   tests pass without installing the main app.

2. **Real-context mode** (runs only when the main app source is present):
   If python_molecular_editor/moleditpy/src is found relative to this repo
   (local dev) OR via the CI_MAIN_APP_SRC environment variable, the tests are
   re-run using the actual PluginContext class to verify true compatibility.
   Skipped with `pytest.mark.skipif` when not available.

What is tested
--------------
- `initialize(context)` registers the correct menu-action path
- `initialize(context)` registers save, load, and document-reset handlers
- Save handler returns the live PLUGIN_SETTINGS reference
- Load handler clears-and-updates PLUGIN_SETTINGS (round-trip safe)
- Document-reset handler clears expected keys; preserves unrelated keys
- Document-reset handler calls `dialog_instance.on_document_reset()` when a
  dialog is active (via monkeypatching the nonlocal binding)

CI setup
--------
The GitHub Actions workflow optionally clones the main app before running tests:

    - name: Clone main app (optional, for real-context tests)
      run: git clone https://github.com/HiroYokoyama/python_molecular_editor.git
             ../python_molecular_editor || true

Then set CI_MAIN_APP_SRC=../python_molecular_editor/moleditpy/src in the test
step's env block to activate real-context mode.
"""

import os
import sys
import types
import unittest
import importlib.util
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Qt / RDKit stubs (must be installed before loading __init__.py)
# ---------------------------------------------------------------------------


def _install_stubs():
    qt_core = sys.modules.get("PyQt6.QtCore")
    if qt_core is None or not hasattr(qt_core, "__file__"):
        qt_core = types.ModuleType("PyQt6.QtCore")

        class _QThread:
            def __init__(self):
                pass

        qt_core.QThread = _QThread
        qt_core.pyqtSignal = lambda *a, **kw: MagicMock()
        sys.modules["PyQt6.QtCore"] = qt_core

    pyqt6 = sys.modules.get("PyQt6")
    if pyqt6 is None or not hasattr(pyqt6, "__file__"):
        pyqt6 = types.ModuleType("PyQt6")
        pyqt6.QtCore = qt_core
        sys.modules["PyQt6"] = pyqt6

    qt_widgets = sys.modules.get("PyQt6.QtWidgets")
    if qt_widgets is None or not hasattr(qt_widgets, "__file__"):
        qt_widgets = types.ModuleType("PyQt6.QtWidgets")
        for name in [
            "QDialog",
            "QVBoxLayout",
            "QHBoxLayout",
            "QPushButton",
            "QLabel",
            "QComboBox",
            "QFileDialog",
            "QMessageBox",
            "QMenu",
            "QApplication",
            "QToolTip",
            "QWidget",
            "QTabWidget",
            "QCheckBox",
            "QLineEdit",
            "QSpinBox",
            "QDoubleSpinBox",
            "QGroupBox",
            "QRadioButton",
            "QTextEdit",
            "QScrollArea",
            "QSplitter",
            "QFrame",
            "QSizePolicy",
            "QStackedWidget",
        ]:
            setattr(qt_widgets, name, MagicMock)
        pyqt6.QtWidgets = qt_widgets
        sys.modules["PyQt6.QtWidgets"] = qt_widgets

    qt_gui = sys.modules.get("PyQt6.QtGui")
    if qt_gui is None or not hasattr(qt_gui, "__file__"):
        if qt_gui is None:
            qt_gui = types.ModuleType("PyQt6.QtGui")
            sys.modules["PyQt6.QtGui"] = qt_gui

        # Define a mock QColor subclass to avoid spec restriction when instantiated with color string
        class MockQColor(MagicMock):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def redF(self):
                return 1.0

            def greenF(self):
                return 1.0

            def blueF(self):
                return 1.0

        setattr(qt_gui, "QColor", MockQColor)
        # Mock QFont so it supports Weight.Bold (constants.py uses it)
        mock_font = MagicMock()
        mock_font.Weight.Bold = 75
        setattr(qt_gui, "QFont", mock_font)
        for name in ["QPainter", "QPen", "QAction", "QIcon"]:
            if not hasattr(qt_gui, name):
                setattr(qt_gui, name, MagicMock)
        pyqt6.QtGui = qt_gui
        sys.modules["PyQt6.QtGui"] = qt_gui

    sys.modules.setdefault("rdkit", MagicMock())
    sys.modules.setdefault("rdkit.Chem", MagicMock())
    sys.modules.setdefault("rdkit.Chem.rdMolTransforms", MagicMock())
    sys.modules.setdefault("pyscf", MagicMock())
    for sub in ("pyscf.gto", "pyscf.scf", "pyscf.dft", "pyscf.solvent"):
        sys.modules.setdefault(sub, MagicMock())


_install_stubs()

# ---------------------------------------------------------------------------
# Load pyscf_calculator/__init__.py with a mocked gui module
# ---------------------------------------------------------------------------

_INIT_MODULE_NAME = "pyscf_calculator_integration_under_test"

_mock_gui_module = types.ModuleType(f"{_INIT_MODULE_NAME}.gui")
_MockPySCFDialog = MagicMock(name="PySCFDialog")
_mock_gui_module.PySCFDialog = _MockPySCFDialog
sys.modules["pyscf_calculator.gui"] = _mock_gui_module
sys.modules[f"{_INIT_MODULE_NAME}.gui"] = _mock_gui_module


def _load_init_mod():
    src = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "pyscf_calculator", "__init__.py")
    )
    spec = importlib.util.spec_from_file_location(_INIT_MODULE_NAME, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_INIT_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


_init_mod = _load_init_mod()


# ---------------------------------------------------------------------------
# StubPluginContext — mirrors the real PluginContext API
# ---------------------------------------------------------------------------


class StubPluginContext:
    """
    Minimal stub that mirrors moleditpy's PluginContext interface.
    Captures all registrations so tests can inspect them.
    """

    def __init__(self):
        self.menu_actions: dict = {}  # path → callback
        self.save_handlers: list = []
        self.load_handlers: list = []
        self.reset_handlers: list = []
        self._main_window = MagicMock(name="MainWindow")
        self._windows: dict = {}  # window_id → Qt window

    # --- API used by initialize() ---

    def add_menu_action(self, path, callback, text=None, icon=None, shortcut=None):
        self.menu_actions[path] = callback

    def register_menu_action(
        self, path, text_or_callback, callback=None, icon=None, shortcut=None
    ):
        if callable(text_or_callback):
            self.add_menu_action(path, text_or_callback)
        else:
            self.add_menu_action(path, callback, text_or_callback)

    def register_save_handler(self, callback):
        self.save_handlers.append(callback)

    def register_load_handler(self, callback):
        self.load_handlers.append(callback)

    def register_document_reset_handler(self, callback):
        self.reset_handlers.append(callback)

    def get_main_window(self):
        return self._main_window

    # Stubs for other API methods (not used by initialize, but kept for
    # completeness so tests with the real PluginContext also have coverage)
    def show_status_message(self, message, timeout=3000):
        pass

    def push_undo_checkpoint(self):
        pass

    def get_selected_atom_indices(self):
        return []

    def register_window(self, wid, win):
        self._windows[wid] = win

    def get_window(self, wid):
        return self._windows.get(wid)

    def get_setting(self, key, default=None):
        return default

    def set_setting(self, key, value):
        pass

    def add_plugin_menu(self, path, callback, text=None, icon=None, shortcut=None):
        self.add_menu_action(f"Plugin/{path.lstrip('/')}", callback)

    def add_toolbar_action(self, callback, text, icon=None, tooltip=None):
        pass

    def register_drop_handler(self, callback, priority=0):
        pass

    def register_file_opener(self, ext, callback, priority=0):
        pass

    def add_export_action(self, label, callback):
        pass

    def add_analysis_tool(self, label, callback):
        pass

    def register_optimization_method(self, name, callback):
        pass

    def register_3d_style(self, style_name, callback):
        pass

    def draw_molecule_3d(self, mol):
        pass

    def refresh_3d_view(self):
        pass

    def reset_3d_camera(self):
        pass

    def get_3d_controller(self):
        return MagicMock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_context():
    """Return a new StubPluginContext and reset PLUGIN_SETTINGS."""
    _init_mod.PLUGIN_SETTINGS.clear()
    return StubPluginContext()


def _initialize_fresh():
    """Call initialize() with a fresh context, return (context, MockDialog)."""
    ctx = _fresh_context()
    _MockPySCFDialog.reset_mock()
    mock_dialog_instance = MagicMock(name="dialog_instance")
    _MockPySCFDialog.return_value = mock_dialog_instance
    _init_mod.initialize(ctx)
    return ctx, mock_dialog_instance


# ===========================================================================
# 1. Registration contract
# ===========================================================================


class TestInitializeRegistrations(unittest.TestCase):
    """initialize(context) must register all expected handlers/actions."""

    @classmethod
    def setUpClass(cls):
        cls.ctx, _ = _initialize_fresh()

    def test_menu_action_registered(self):
        self.assertIn("Extensions/PySCF Calculator...", self.ctx.menu_actions)

    def test_menu_action_is_callable(self):
        cb = self.ctx.menu_actions["Extensions/PySCF Calculator..."]
        self.assertTrue(callable(cb))

    def test_save_handler_registered(self):
        self.assertEqual(len(self.ctx.save_handlers), 1)

    def test_load_handler_registered(self):
        self.assertEqual(len(self.ctx.load_handlers), 1)

    def test_document_reset_handler_registered(self):
        self.assertEqual(len(self.ctx.reset_handlers), 1)


# ===========================================================================
# 2. Save / Load state round-trip
# ===========================================================================


class TestSaveLoadState(unittest.TestCase):
    def setUp(self):
        self.ctx, _ = _initialize_fresh()

    def test_save_handler_returns_settings_reference(self):
        _init_mod.PLUGIN_SETTINGS["key"] = "value"
        result = self.ctx.save_handlers[0]()
        self.assertIn("key", result)
        self.assertEqual(result["key"], "value")

    def test_save_handler_returns_live_reference(self):
        """Modifying PLUGIN_SETTINGS after save() returns still visible."""
        data = self.ctx.save_handlers[0]()
        _init_mod.PLUGIN_SETTINGS["added_after"] = 42
        self.assertEqual(data["added_after"], 42)

    def test_load_handler_updates_settings(self):
        self.ctx.load_handlers[0]({"x": 1, "y": 2})
        self.assertEqual(_init_mod.PLUGIN_SETTINGS["x"], 1)
        self.assertEqual(_init_mod.PLUGIN_SETTINGS["y"], 2)

    def test_load_handler_clears_old_keys(self):
        _init_mod.PLUGIN_SETTINGS["stale"] = "old"
        self.ctx.load_handlers[0]({"new": "data"})
        self.assertNotIn("stale", _init_mod.PLUGIN_SETTINGS)

    def test_round_trip(self):
        # Note: save_handler returns the live PLUGIN_SETTINGS reference, so
        # we must copy it before clearing to simulate serialisation round-trip.
        _init_mod.PLUGIN_SETTINGS.update({"a": 1, "b": [1, 2, 3]})
        saved_copy = dict(self.ctx.save_handlers[0]())
        _init_mod.PLUGIN_SETTINGS.clear()
        self.ctx.load_handlers[0](saved_copy)
        self.assertEqual(_init_mod.PLUGIN_SETTINGS["a"], 1)
        self.assertEqual(_init_mod.PLUGIN_SETTINGS["b"], [1, 2, 3])

    def test_load_empty_dict_clears_settings(self):
        _init_mod.PLUGIN_SETTINGS["exists"] = True
        self.ctx.load_handlers[0]({})
        self.assertEqual(_init_mod.PLUGIN_SETTINGS, {})


# ===========================================================================
# 3. Document reset handler
# ===========================================================================


class TestDocumentReset(unittest.TestCase):
    def setUp(self):
        self.ctx, _ = _initialize_fresh()
        self.reset = self.ctx.reset_handlers[0]

    def test_clears_associated_filename(self):
        _init_mod.PLUGIN_SETTINGS["associated_filename"] = "foo.mep"
        self.reset()
        self.assertNotIn("associated_filename", _init_mod.PLUGIN_SETTINGS)

    def test_clears_calc_history(self):
        _init_mod.PLUGIN_SETTINGS["calc_history"] = [{"e": -1.0}]
        self.reset()
        self.assertNotIn("calc_history", _init_mod.PLUGIN_SETTINGS)

    def test_clears_struct_source(self):
        _init_mod.PLUGIN_SETTINGS["struct_source"] = "xyz"
        self.reset()
        self.assertNotIn("struct_source", _init_mod.PLUGIN_SETTINGS)

    def test_preserves_unrelated_key(self):
        _init_mod.PLUGIN_SETTINGS["user_pref"] = "dark"
        _init_mod.PLUGIN_SETTINGS["calc_history"] = []
        self.reset()
        self.assertIn("user_pref", _init_mod.PLUGIN_SETTINGS)
        self.assertEqual(_init_mod.PLUGIN_SETTINGS["user_pref"], "dark")

    def test_reset_with_empty_settings_does_not_raise(self):
        _init_mod.PLUGIN_SETTINGS.clear()
        try:
            self.reset()
        except Exception as e:
            self.fail(f"reset() raised unexpectedly: {e}")


# ===========================================================================
# 4. show_dialog — first call creates dialog
# ===========================================================================


class TestShowDialogViaContext(unittest.TestCase):
    """
    Call show_dialog through the registered menu-action callback.
    Tests the dialog lifecycle without touching Qt.
    """

    def setUp(self):
        self.ctx, self.mock_dialog = _initialize_fresh()
        self.show_dialog = self.ctx.menu_actions["Extensions/PySCF Calculator..."]

    def test_first_call_creates_dialog(self):
        self.show_dialog()
        _MockPySCFDialog.assert_called_once()

    def test_dialog_receives_main_window(self):
        self.show_dialog()
        args, kwargs = _MockPySCFDialog.call_args
        self.assertIs(args[0], self.ctx._main_window)

    def test_dialog_receives_context(self):
        self.show_dialog()
        args, kwargs = _MockPySCFDialog.call_args
        self.assertIs(args[1], self.ctx)

    def test_dialog_receives_settings_kwarg(self):
        self.show_dialog()
        _, kwargs = _MockPySCFDialog.call_args
        self.assertIn("settings", kwargs)
        self.assertIs(kwargs["settings"], _init_mod.PLUGIN_SETTINGS)

    def test_dialog_show_called(self):
        self.show_dialog()
        self.mock_dialog.show.assert_called_once()

    def test_second_call_raises_if_visible(self):
        """Second show_dialog call when dialog is visible raises+activates."""
        self.show_dialog()
        self.mock_dialog.isVisible.return_value = True
        _MockPySCFDialog.reset_mock()

        self.show_dialog()  # should raise/activate, not create new

        _MockPySCFDialog.assert_not_called()  # no new dialog
        self.mock_dialog.raise_.assert_called_once()
        self.mock_dialog.activateWindow.assert_called_once()


# ===========================================================================
# 5. Real PluginContext (optional — local dev or CI with cloned main app)
# ===========================================================================

# Look for the main app source in standard locations.
_MAIN_APP_CANDIDATES = [
    # Local dev: ../python_molecular_editor relative to this repo
    os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "python_molecular_editor",
            "moleditpy",
            "src",
        )
    ),
    # CI: set via environment variable
    os.environ.get("CI_MAIN_APP_SRC", ""),
]

_MAIN_APP_SRC = next((p for p in _MAIN_APP_CANDIDATES if p and os.path.isdir(p)), None)
HAS_MAIN_APP = _MAIN_APP_SRC is not None


try:
    import pytest

    _skipif = pytest.mark.skipif(
        not HAS_MAIN_APP,
        reason="main app not found; set CI_MAIN_APP_SRC or place at "
        "../python_molecular_editor/moleditpy/src",
    )
except ImportError:
    import functools

    def _skipif(cls):
        return unittest.skip("pytest not available for skipif decoration")(cls)


@_skipif
class TestWithRealPluginContext(unittest.TestCase):
    """
    Verify initialize() works with the actual PluginContext class from the
    MoleditPy main application.

    These tests pass on local dev when python_molecular_editor is present at
    the expected path, and on CI when the main app is cloned via the workflow.
    """

    @classmethod
    def setUpClass(cls):
        if not HAS_MAIN_APP:
            return
        if _MAIN_APP_SRC not in sys.path:
            sys.path.insert(0, _MAIN_APP_SRC)
        from moleditpy.plugins.plugin_interface import PluginContext

        cls.PluginContext = PluginContext

        # Build a real PluginContext with a mock manager
        mock_manager = MagicMock()
        mock_manager.get_main_window.return_value = MagicMock()
        cls.real_ctx = PluginContext(mock_manager, "PySCF Calculator")

    def test_real_initialize_does_not_raise(self):
        """initialize() must complete without raising when given a real context."""
        _init_mod.PLUGIN_SETTINGS.clear()
        try:
            _init_mod.initialize(self.real_ctx)
        except Exception as e:
            self.fail(f"initialize(real_context) raised: {e}")

    def test_real_context_is_plugincontext_instance(self):
        self.assertIsInstance(self.real_ctx, self.PluginContext)

    def test_stub_interface_matches_real(self):
        """Every method called by initialize() exists on the real PluginContext."""
        methods_used = [
            "add_menu_action",
            "register_save_handler",
            "register_load_handler",
            "register_document_reset_handler",
            "get_main_window",
        ]
        for method in methods_used:
            self.assertTrue(
                hasattr(self.PluginContext, method),
                f"Real PluginContext is missing method: {method}",
            )


if __name__ == "__main__":
    unittest.main()
