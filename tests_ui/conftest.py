"""Real Qt lifecycle tests, isolated from tests/' module-level Qt mocks."""
import importlib
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def plugin():
    pytest.importorskip("PyQt6.QtWidgets")
    pytest.importorskip("rdkit")
    pytest.importorskip("pyvista")
    pytest.importorskip("matplotlib")
    package = types.ModuleType("_audit_ui_plugin")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "pyscf_calculator")]
    sys.modules[package.__name__] = package
    return lambda name: importlib.import_module(f"{package.__name__}.{name}")


@pytest.fixture(scope="session")
def app(plugin):
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def context(app):
    from PyQt6.QtWidgets import QMainWindow
    from rdkit import Chem
    mw = QMainWindow()
    mw.state_manager = types.SimpleNamespace(has_unsaved_changes=False, update_window_title=lambda: None)
    ctx = MagicMock()
    ctx.get_main_window.return_value = mw
    ctx.current_molecule = Chem.MolFromXYZBlock("2\nH2\nH 0 0 0\nH 0 0 .74\n")
    ctx.mark_project_modified.side_effect = lambda: setattr(mw.state_manager, "has_unsaved_changes", True)
    yield ctx
    mw.close()


@pytest.fixture
def dialog(plugin, context):
    dlg = plugin("gui").PySCFDialog(context.get_main_window(), context, settings={})
    yield dlg
    dlg.close()


def pump_until(app, predicate, timeout=3):
    import time
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert predicate(), "Qt event did not complete before the deadline"
