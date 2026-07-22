"""
tests/test_utils_coverage.py

Additional coverage for pyscf_calculator/utils.py branches not correctly
exercised by test_utils.py / test_utils_branches.py:

  - real ImportError from `from rdkit.Chem import rdDetermineBonds` (86-88)
  - real generic Exception from rdDetermineBonds calls (89-90)
  - exception while reading has_unsaved_changes from state_manager (102-103)
  - exception while restoring has_unsaved_changes/update_window_title (118-119)
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


sys.modules.setdefault("rdkit", MagicMock())
sys.modules.setdefault("rdkit.Chem", MagicMock())

utils = _load_module_direct(
    os.path.join("pyscf_calculator", "utils.py"),
    "pyscf_calculator_utils_coverage_under_test",
)


class _NoBondsChem:
    """Plain object: no rdDetermineBonds attribute, no __path__ (not a package)."""


class TestBondDeterminationRealErrors(unittest.TestCase):
    def setUp(self):
        self.ctx = MagicMock()
        self.ctx.get_main_window.return_value = None
        utils.Chem.MolFromXYZBlock = MagicMock(return_value=MagicMock())

    def test_real_import_error_is_swallowed(self):
        with patch.dict(sys.modules, {"rdkit.Chem": _NoBondsChem()}):
            # Must not raise
            utils.update_molecule_from_xyz(self.ctx, "2\ncomment\nH 0 0 0\nO 1 0 0")
        # current_molecule was still set despite the swallowed ImportError
        self.assertIsNotNone(self.ctx.current_molecule)

    def test_real_generic_exception_is_swallowed(self):
        bad_rddb = MagicMock()
        bad_rddb.DetermineConnectivity.side_effect = Exception("boom")
        stub_chem = _NoBondsChem()
        stub_chem.rdDetermineBonds = bad_rddb

        with patch.dict(sys.modules, {"rdkit.Chem": stub_chem}):
            utils.update_molecule_from_xyz(self.ctx, "2\ncomment\nH 0 0 0\nO 1 0 0")

        self.assertIsNotNone(self.ctx.current_molecule)


class _RaisingStateManager:
    """state_manager whose has_unsaved_changes property always raises,
    and whose update_window_title raises too."""

    @property
    def has_unsaved_changes(self):
        raise RuntimeError("get boom")

    @has_unsaved_changes.setter
    def has_unsaved_changes(self, value):
        raise RuntimeError("set boom")

    def update_window_title(self):
        raise RuntimeError("title boom")


class TestStateManagerExceptionPaths(unittest.TestCase):
    def test_preserve_and_restore_exceptions_are_swallowed(self):
        ctx = MagicMock()
        mw = MagicMock()
        mw.state_manager = _RaisingStateManager()
        ctx.get_main_window.return_value = mw

        utils.Chem.MolFromXYZBlock = MagicMock(return_value=MagicMock())

        # Must not raise, despite has_unsaved_changes/update_window_title
        # both raising on access.
        utils.update_molecule_from_xyz(
            ctx, "2\ncomment\nH 0 0 0\nO 1 0 0", mark_modified=False
        )
        self.assertIsNotNone(ctx.current_molecule)


if __name__ == "__main__":
    unittest.main()
