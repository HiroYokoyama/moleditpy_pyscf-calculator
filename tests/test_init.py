"""
tests/test_init.py
Unit tests for the plugin initialization module.
"""

import os
import sys
import unittest
import importlib.util
from unittest.mock import MagicMock


def _load_module_direct(relpath, module_name):
    src = os.path.join(os.path.dirname(__file__), "..", relpath)
    src = os.path.normpath(src)
    spec = importlib.util.spec_from_file_location(module_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Mock gui to bypass heavy imports under both resolution paths.
# The relative import `from .gui import PySCFDialog` in __init__.py may
# resolve to either "pyscf_calculator.gui" or "<module_name>.gui" depending
# on how Python resolves the package for the dynamically-loaded module.
_mock_gui_module = MagicMock()
sys.modules["pyscf_calculator.gui"] = _mock_gui_module
sys.modules["pyscf_calculator_init_under_test.gui"] = _mock_gui_module

init_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "__init__.py"),
    "pyscf_calculator_init_under_test",
)


class TestInitModule(unittest.TestCase):
    def setUp(self):
        self.context = MagicMock()
        self.context.get_window.return_value = None  # no existing dialog
        init_mod.PLUGIN_SETTINGS.clear()

    def test_metadata(self):
        self.assertEqual(init_mod.PLUGIN_NAME, "PySCF Calculator")
        self.assertIn("pyscf", init_mod.PLUGIN_DEPENDENCIES)

    def test_supported_moleditpy_version_present(self):
        self.assertTrue(hasattr(init_mod, "PLUGIN_SUPPORTED_MOLEDITPY_VERSION"))
        ver = init_mod.PLUGIN_SUPPORTED_MOLEDITPY_VERSION
        self.assertIsInstance(ver, str)
        self.assertGreater(len(ver), 0)

    def test_initialize_registers_handlers(self):
        init_mod.initialize(self.context)
        self.context.register_save_handler.assert_called_once()
        self.context.register_load_handler.assert_called_once()
        self.context.register_document_reset_handler.assert_called_once()
        self.context.add_menu_action.assert_called_once()

    def test_save_and_load_state(self):
        init_mod.initialize(self.context)

        save_handler = self.context.register_save_handler.call_args[0][0]
        load_handler = self.context.register_load_handler.call_args[0][0]

        # Load state
        test_data = {"key1": "value1", "associated_filename": "test.xyz"}
        load_handler(test_data)

        # Check settings
        self.assertEqual(init_mod.PLUGIN_SETTINGS["key1"], "value1")

        # Save state
        saved_data = save_handler()
        self.assertEqual(saved_data, test_data)

    def test_document_reset(self):
        init_mod.initialize(self.context)
        reset_handler = self.context.register_document_reset_handler.call_args[0][0]

        # Pre-seed settings
        init_mod.PLUGIN_SETTINGS.update(
            {
                "associated_filename": "foo",
                "calc_history": [],
                "struct_source": "bar",
                "persist": "keep_me",
            }
        )

        reset_handler()

        # Check cleared keys
        self.assertNotIn("associated_filename", init_mod.PLUGIN_SETTINGS)
        self.assertNotIn("calc_history", init_mod.PLUGIN_SETTINGS)
        self.assertNotIn("struct_source", init_mod.PLUGIN_SETTINGS)
        # Check persisted key
        self.assertEqual(init_mod.PLUGIN_SETTINGS["persist"], "keep_me")

    def test_show_dialog_registers_window(self):
        init_mod.initialize(self.context)
        # Capture the show_dialog callback registered with add_menu_action
        _, show_dialog = self.context.add_menu_action.call_args[0]

        # First call: no existing window → must create and register one
        self.context.get_window.return_value = None
        show_dialog()
        self.context.register_window.assert_called_once()
        window_id = self.context.register_window.call_args[0][0]
        self.assertEqual(window_id, "dialog")

    def test_show_dialog_raises_existing_visible_window(self):
        init_mod.initialize(self.context)
        _, show_dialog = self.context.add_menu_action.call_args[0]

        existing = MagicMock()
        existing.isVisible.return_value = True
        self.context.get_window.return_value = existing

        show_dialog()

        # Should raise the existing window, NOT create a new one
        existing.raise_.assert_called_once()
        existing.activateWindow.assert_called_once()
        self.context.register_window.assert_not_called()

    def test_document_reset_calls_dialog_reset_method(self):
        init_mod.initialize(self.context)
        reset_handler = self.context.register_document_reset_handler.call_args[0][0]

        dialog_mock = MagicMock()
        self.context.get_window.return_value = dialog_mock

        # Seed settings that reset should clear
        init_mod.PLUGIN_SETTINGS["associated_filename"] = "test.xyz"
        reset_handler()

        dialog_mock.on_document_reset.assert_called_once()


if __name__ == "__main__":
    unittest.main()
