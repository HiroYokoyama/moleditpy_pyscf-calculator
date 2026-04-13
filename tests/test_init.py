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

# Mock gui to bypass heavy imports
sys.modules['pyscf_calculator.gui'] = MagicMock()

init_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "__init__.py"),
    "pyscf_calculator_init_under_test",
)

class TestInitModule(unittest.TestCase):
    def setUp(self):
        self.context = MagicMock()
        init_mod.PLUGIN_SETTINGS.clear()
        
    def test_metadata(self):
        self.assertEqual(init_mod.PLUGIN_NAME, "PySCF Calculator")
        self.assertIn("pyscf", init_mod.PLUGIN_DEPENDENCIES)

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
        init_mod.PLUGIN_SETTINGS.update({
            "associated_filename": "foo",
            "calc_history": [],
            "struct_source": "bar",
            "persist": "keep_me"
        })
        
        reset_handler()
        
        # Check cleared keys
        self.assertNotIn("associated_filename", init_mod.PLUGIN_SETTINGS)
        self.assertNotIn("calc_history", init_mod.PLUGIN_SETTINGS)
        self.assertNotIn("struct_source", init_mod.PLUGIN_SETTINGS)
        # Check persisted key
        self.assertEqual(init_mod.PLUGIN_SETTINGS["persist"], "keep_me")

if __name__ == '__main__':
    unittest.main()
