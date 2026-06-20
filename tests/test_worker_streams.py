"""
tests/test_worker_streams.py
Unit tests for the thread-safe stream capture models.
"""

import os
import sys
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


# Mock RDKit and PySCF dependencies
sys.modules["rdkit"] = MagicMock()
sys.modules["rdkit.Chem"] = MagicMock()
sys.modules["rdkit.Chem.rdMolTransforms"] = MagicMock()
sys.modules["pyscf"] = MagicMock()
sys.modules["pyscf.gto"] = MagicMock()
sys.modules["pyscf.scf"] = MagicMock()
sys.modules["pyscf.dft"] = MagicMock()
sys.modules["pyscf.solvent"] = MagicMock()

import types

qt_core = types.ModuleType("PyQt6.QtCore")
qt_core.QThread = MagicMock()
qt_core.pyqtSignal = lambda *a, **k: MagicMock()
pyqt6 = types.ModuleType("PyQt6")
pyqt6.QtCore = qt_core
sys.modules["PyQt6"] = pyqt6
sys.modules["PyQt6.QtCore"] = qt_core

worker_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "worker.py"),
    "pyscf_calculator_worker_under_test",
)
StreamToSignal = worker_mod.StreamToSignal


class TestStreamToSignal(unittest.TestCase):
    def test_write_emits_signal(self):
        mock_signal = MagicMock()
        stream = StreamToSignal(mock_signal)

        stream.write("hello")
        mock_signal.emit.assert_called_once_with("hello")

    def test_write_fallback_stream(self):
        mock_signal = MagicMock()
        fallback = MagicMock()
        stream = StreamToSignal(mock_signal, target_stream=fallback)

        stream.write("fallback test")

        # Should hit both
        mock_signal.emit.assert_called_once_with("fallback test")
        fallback.write.assert_called_once_with("fallback test")
        fallback.flush.assert_called()

    def test_close_stops_emission(self):
        mock_signal = MagicMock()
        stream = StreamToSignal(mock_signal)

        stream.close()
        self.assertTrue(stream._destroyed)

        stream.write("ignored text")
        mock_signal.emit.assert_not_called()

    def test_write_protects_against_destroyed_signal(self):
        mock_signal = MagicMock()
        # Mock that emit raises a RuntimeError when the qt object is dead
        mock_signal.emit.side_effect = RuntimeError(
            "wrapped C/C++ object has been deleted"
        )

        stream = StreamToSignal(mock_signal)

        # Writing should not raise python-level exceptions, it should swallow it and self-destroy
        stream.write("crash attempt")
        self.assertTrue(stream._destroyed)

    def test_encoding_property(self):
        class MockFile:
            encoding = "cp1252"

        stream = StreamToSignal(MagicMock(), target_stream=MockFile())
        self.assertEqual(stream.encoding, "cp1252")

        # Test Default
        stream_default = StreamToSignal(MagicMock())
        self.assertEqual(stream_default.encoding, "utf-8")


if __name__ == "__main__":
    unittest.main()
