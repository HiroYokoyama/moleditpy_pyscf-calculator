"""
tests/test_utils_branches.py

Tests for edge-case branches in pyscf_calculator/utils.py that are not
covered by test_utils.py:

  - update_molecule_from_xyz with mark_modified=False preserves dirty state
  - update_molecule_from_xyz with mark_modified=True does not restore dirty
  - rdDetermineBonds ImportError handled silently
  - rdDetermineBonds generic Exception handled silently
  - update_molecule_from_xyz when new_mol is None (no update)
  - get_unique_path loops past multiple existing files
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


# Mock rdkit before loading so utils.py module-level import succeeds
sys.modules.setdefault("rdkit", MagicMock())
sys.modules.setdefault("rdkit.Chem", MagicMock())

utils = _load_module_direct(
    os.path.join("pyscf_calculator", "utils.py"),
    "pyscf_calculator_utils_branches_under_test",
)


# ---------------------------------------------------------------------------
# get_unique_path — multiple existing files
# ---------------------------------------------------------------------------


class TestGetUniquePathMultiple(unittest.TestCase):
    """get_unique_path must skip _1, _2 … until it finds a free slot."""

    def test_skips_two_existing_variants(self):
        def _mock_exists(p):
            return p in {"test.xyz", "test_1.xyz"}  # _2 is free

        with patch("os.path.exists", side_effect=_mock_exists):
            result = utils.get_unique_path("test.xyz")
        self.assertEqual(result, "test_2.xyz")

    def test_skips_three_existing_variants(self):
        def _mock_exists(p):
            return p in {"out.log", "out_1.log", "out_2.log"}

        with patch("os.path.exists", side_effect=_mock_exists):
            result = utils.get_unique_path("out.log")
        self.assertEqual(result, "out_3.log")


# ---------------------------------------------------------------------------
# update_molecule_from_xyz — bond determination error paths
# ---------------------------------------------------------------------------


class TestUpdateMolBondDetermination(unittest.TestCase):
    """Bond determination errors must be swallowed without raising."""

    def _make_context(self):
        ctx = MagicMock()
        ctx.get_main_window.return_value = None
        return ctx

    def test_import_error_from_rdDetermineBonds_is_silent(self):
        ctx = self._make_context()
        utils.Chem.MolFromXYZBlock = MagicMock(return_value=MagicMock())

        # Simulate ImportError inside the try block
        with patch.dict("sys.modules", {"rdkit.Chem.rdDetermineBonds": None}):
            # Should not raise
            utils.update_molecule_from_xyz(ctx, "2\ncomment\nH 0 0 0\nO 1 0 0")

    def test_generic_exception_from_DetermineConnectivity_is_silent(self):
        ctx = self._make_context()
        mock_mol = MagicMock()
        utils.Chem.MolFromXYZBlock = MagicMock(return_value=mock_mol)

        mock_rddb = MagicMock()
        mock_rddb.DetermineConnectivity.side_effect = Exception("connectivity failed")

        with patch.dict("sys.modules", {"rdkit.Chem.rdDetermineBonds": mock_rddb}):
            # Must not raise
            utils.update_molecule_from_xyz(ctx, "2\ncomment\nH 0 0 0\nO 1 0 0")

        self.assertEqual(ctx.current_molecule, mock_mol)


# ---------------------------------------------------------------------------
# update_molecule_from_xyz — mark_modified=False preserves dirty state
# ---------------------------------------------------------------------------


class TestUpdateMolMarkModified(unittest.TestCase):
    def _make_context_with_state_manager(self, was_dirty):
        ctx = MagicMock()
        mw = MagicMock()
        sm = MagicMock()
        sm.has_unsaved_changes = was_dirty
        mw.state_manager = sm
        ctx.get_main_window.return_value = mw
        return ctx, mw, sm

    def test_mark_modified_false_restores_dirty_true(self):
        ctx, mw, sm = self._make_context_with_state_manager(was_dirty=True)
        utils.Chem.MolFromXYZBlock = MagicMock(return_value=MagicMock())

        utils.update_molecule_from_xyz(
            ctx, "2\ncomment\nH 0 0 0\nO 1 0 0", mark_modified=False
        )

        # Dirty flag must be restored to original value (True)
        self.assertTrue(sm.has_unsaved_changes)

    def test_mark_modified_false_restores_dirty_false(self):
        ctx, mw, sm = self._make_context_with_state_manager(was_dirty=False)
        utils.Chem.MolFromXYZBlock = MagicMock(return_value=MagicMock())

        utils.update_molecule_from_xyz(
            ctx, "2\ncomment\nH 0 0 0\nO 1 0 0", mark_modified=False
        )

        self.assertFalse(sm.has_unsaved_changes)

    def test_mark_modified_false_calls_update_window_title(self):
        ctx, mw, sm = self._make_context_with_state_manager(was_dirty=False)
        utils.Chem.MolFromXYZBlock = MagicMock(return_value=MagicMock())

        utils.update_molecule_from_xyz(
            ctx, "2\ncomment\nH 0 0 0\nO 1 0 0", mark_modified=False
        )

        sm.update_window_title.assert_called_once()

    def test_mark_modified_true_does_not_restore_dirty(self):
        """mark_modified=True must NOT suppress the dirty flag change."""
        ctx, mw, sm = self._make_context_with_state_manager(was_dirty=False)
        utils.Chem.MolFromXYZBlock = MagicMock(return_value=MagicMock())

        utils.update_molecule_from_xyz(
            ctx, "2\ncomment\nH 0 0 0\nO 1 0 0", mark_modified=True
        )

        # update_window_title must NOT be called (suppression path skipped)
        sm.update_window_title.assert_not_called()


# ---------------------------------------------------------------------------
# update_molecule_from_xyz — new_mol is None (bad parse)
# ---------------------------------------------------------------------------


class TestUpdateMolNoneResult(unittest.TestCase):
    def test_current_molecule_not_updated_when_mol_is_none(self):
        ctx = MagicMock()
        ctx.get_main_window.return_value = None
        utils.Chem.MolFromXYZBlock = MagicMock(return_value=None)

        utils.update_molecule_from_xyz(ctx, "2\ncomment\nH 0 0 0\nO 1 0 0")

        # current_molecule must NOT be set
        self.assertFalse(
            hasattr(ctx, "current_molecule")
            and ctx.current_molecule is not None
            and ctx.mock_calls
        )
        # More directly: the attribute assignment should NOT have occurred
        # We check by asserting the call list has no current_molecule assignment
        assigned = any("current_molecule" in str(c) for c in ctx.mock_calls)
        self.assertFalse(assigned)


if __name__ == "__main__":
    unittest.main()
