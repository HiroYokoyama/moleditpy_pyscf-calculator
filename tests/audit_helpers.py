"""Import production helpers without leaking stubs into other test modules."""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock


def load_module(monkeypatch, name):
    package_name = "_audit_regression_plugin"
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "pyscf_calculator")]
    monkeypatch.setitem(sys.modules, package_name, package)
    core = types.ModuleType("PyQt6.QtCore")

    class Thread:
        def __init__(self, *args):
            pass

        @staticmethod
        def msleep(ms):
            pass

    core.QThread = Thread
    core.pyqtSignal = lambda *args: MagicMock()
    monkeypatch.setitem(sys.modules, "PyQt6.QtCore", core)
    monkeypatch.setitem(sys.modules, "PyQt6.QtGui", types.SimpleNamespace(QColor=MagicMock()))
    for dependency in ("pyscf", "pyscf.scf", "pyscf.dft", "pyscf.gto", "pyscf.solvent", "rdkit", "rdkit.Chem", "pyvista"):
        monkeypatch.setitem(sys.modules, dependency, MagicMock())
    module_name = f"{package_name}.{name}"
    path = Path(package.__path__[0]) / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module
