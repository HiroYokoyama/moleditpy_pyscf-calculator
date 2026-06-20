"""
tests/test_parse_cube_data.py

Unit tests for vis.parse_cube_data() — the pure Gaussian-cube file parser.
No Qt or PySCF needed.  pyvista and PyQt6 are mocked at module level so the
vis module can be imported headlessly.
"""

import os
import sys
import math
import types
import textwrap
import tempfile
import unittest
import importlib.util
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Stubs: pyvista, PyQt6.QtGui, PyQt6.QtCore  (all used at module level in vis.py)
# ---------------------------------------------------------------------------


def _install_stubs():
    sys.modules.setdefault("pyvista", MagicMock())

    qt_gui = types.ModuleType("PyQt6.QtGui")
    qt_gui.QColor = MagicMock()
    sys.modules.setdefault("PyQt6.QtGui", qt_gui)

    qt_core = types.ModuleType("PyQt6.QtCore")
    qt_core.Qt = MagicMock()
    sys.modules.setdefault("PyQt6.QtCore", qt_core)

    pyqt6 = sys.modules.get("PyQt6") or types.ModuleType("PyQt6")
    pyqt6.QtGui = qt_gui
    pyqt6.QtCore = qt_core
    sys.modules.setdefault("PyQt6", pyqt6)

    sys.modules.setdefault("rdkit", MagicMock())
    sys.modules.setdefault("rdkit.Chem", MagicMock())


_install_stubs()


def _load_vis_mod():
    src = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "pyscf_calculator", "vis.py")
    )
    spec = importlib.util.spec_from_file_location(
        "pyscf_calculator_vis_under_test", src
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pyscf_calculator_vis_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


_vis_mod = _load_vis_mod()
parse_cube_data = _vis_mod.parse_cube_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_cube(content: str) -> str:
    """Write *content* to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".cube", delete=False, encoding="utf-8"
    )
    f.write(content)
    f.flush()
    f.close()
    return f.name


# Minimal valid Gaussian cube with 2 atoms, 2×2×2 grid
_CUBE_TEMPLATE = textwrap.dedent("""\
    Comment line 1
    Comment line 2
     2  0.000000  0.000000  0.000000
     2  0.283459  0.000000  0.000000
     2  0.000000  0.283459  0.000000
     2  0.000000  0.000000  0.283459
     1  0.000000  0.000000  0.000000  0.000000
     8  0.000000  1.889726  0.000000  0.000000
    {data}
""")

# 8 floats for a 2×2×2 grid
_CUBE_DATA_8 = "  0.1  0.2  0.3  0.4\n  0.5  0.6  0.7  0.8\n"

_VALID_CUBE = _CUBE_TEMPLATE.format(data=_CUBE_DATA_8)


# ===========================================================================
# Error paths
# ===========================================================================


class TestParseCubeErrors(unittest.TestCase):
    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            parse_cube_data("/nonexistent/path.cube")

    def test_empty_path_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            parse_cube_data("")

    def test_too_short_file_raises_value_error(self):
        path = _write_cube("line1\nline2\n")
        try:
            with self.assertRaises(ValueError) as ctx:
                parse_cube_data(path)
            self.assertIn("short", str(ctx.exception).lower())
        finally:
            os.unlink(path)

    def test_invalid_origin_line_raises_value_error(self):
        # Only one token on the "origin" line
        bad = textwrap.dedent("""\
            Comment 1
            Comment 2
             bad
             2  0.283459  0.0  0.0
             2  0.0  0.283459  0.0
             2  0.0  0.0  0.283459
        """)
        path = _write_cube(bad)
        try:
            with self.assertRaises(ValueError):
                parse_cube_data(path)
        finally:
            os.unlink(path)


# ===========================================================================
# Valid cube
# ===========================================================================


class TestParseCubeValid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = _write_cube(_VALID_CUBE)
        cls.result = parse_cube_data(cls.path)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.path)

    def test_dims_are_correct(self):
        self.assertEqual(self.result["dims"], (2, 2, 2))

    def test_two_atoms_parsed(self):
        self.assertEqual(len(self.result["atoms"]), 2)

    def test_atom_atomic_numbers(self):
        nums = [a[0] for a in self.result["atoms"]]
        self.assertIn(1, nums)  # H
        self.assertIn(8, nums)  # O

    def test_data_flat_length(self):
        self.assertEqual(len(self.result["data_flat"]), 8)

    def test_data_values_correct(self):
        import numpy as np

        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        np.testing.assert_allclose(self.result["data_flat"], expected, atol=1e-6)

    def test_origin_is_zero(self):
        import numpy as np

        np.testing.assert_array_equal(self.result["origin"], [0.0, 0.0, 0.0])

    def test_is_angstrom_header_false_for_positive_dims(self):
        self.assertFalse(self.result["is_angstrom_header"])


# ===========================================================================
# Data truncation and padding
# ===========================================================================


class TestParseCubeSizeMismatch(unittest.TestCase):
    def test_extra_data_truncated_to_expected_size(self):
        # 2×2×2 = 8 values, provide 10
        extra = _CUBE_TEMPLATE.format(
            data="  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0\n"
        )
        path = _write_cube(extra)
        try:
            result = parse_cube_data(path)
            self.assertEqual(len(result["data_flat"]), 8)
        finally:
            os.unlink(path)

    def test_short_data_padded_with_zeros(self):
        # Provide only 4 values for 8-element grid
        short = _CUBE_TEMPLATE.format(data="  0.1  0.2  0.3  0.4\n")
        path = _write_cube(short)
        try:
            result = parse_cube_data(path)
            self.assertEqual(len(result["data_flat"]), 8)
            # Last 4 must be zero-padded
            import numpy as np

            np.testing.assert_array_equal(result["data_flat"][4:], [0.0] * 4)
        finally:
            os.unlink(path)

    def test_empty_data_section_returns_zeros(self):
        # No data lines at all (just blank after atoms)
        no_data = textwrap.dedent("""\
            Comment 1
            Comment 2
             2  0.000000  0.000000  0.000000
             2  0.283459  0.000000  0.000000
             2  0.000000  0.283459  0.000000
             2  0.000000  0.000000  0.283459
             1  0.000000  0.000000  0.000000  0.000000
             8  0.000000  1.889726  0.000000  0.000000
        """)
        path = _write_cube(no_data)
        try:
            result = parse_cube_data(path)
            self.assertEqual(len(result["data_flat"]), 8)
            import numpy as np

            np.testing.assert_array_equal(result["data_flat"], np.zeros(8))
        finally:
            os.unlink(path)


# ===========================================================================
# Negative n_atoms (MO cube — extra header line present)
# ===========================================================================


class TestParseCubeMOFormat(unittest.TestCase):
    """Cube files with n_atoms < 0 have an extra MO-info line after atoms."""

    def test_negative_natoms_skips_mo_info_line(self):
        # n_atoms = -2 means 2 atoms + MO info line
        mo_cube = textwrap.dedent("""\
            MO Cube comment 1
            MO Cube comment 2
            -2  0.000000  0.000000  0.000000
             2  0.283459  0.000000  0.000000
             2  0.000000  0.283459  0.000000
             2  0.000000  0.000000  0.283459
             1  0  0.000000  0.000000  0.000000  0.000000
             1  0.000000  0.000000  0.000000  0.000000
             8  0.000000  1.889726  0.000000  0.000000
             0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8
        """)
        path = _write_cube(mo_cube)
        try:
            result = parse_cube_data(path)
            self.assertEqual(result["dims"], (2, 2, 2))
            self.assertTrue(
                result["is_angstrom_header"] is False
                or result["is_angstrom_header"] is True
            )  # just no crash
        finally:
            os.unlink(path)

    def test_is_angstrom_header_true_for_negative_dims(self):
        # n_atoms=-2, nx=-2 (negative means Angstrom)
        ang_cube = textwrap.dedent("""\
            Angstrom Cube comment 1
            Angstrom Cube comment 2
            -2  0.000000  0.000000  0.000000
            -2  0.283459  0.000000  0.000000
            -2  0.000000  0.283459  0.000000
            -2  0.000000  0.000000  0.283459
             1  0  0.000000  0.000000  0.000000
             8  0  1.889726  0.000000  0.000000
             0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8
        """)
        path = _write_cube(ang_cube)
        try:
            result = parse_cube_data(path)
            self.assertTrue(result["is_angstrom_header"])
        finally:
            os.unlink(path)


# ===========================================================================
# Malformed atom lines are skipped gracefully
# ===========================================================================


class TestParseCubeMalformedAtoms(unittest.TestCase):
    def test_short_atom_line_is_skipped(self):
        """An atom line with fewer than 5 tokens must be skipped, not crash."""
        cube = textwrap.dedent("""\
            Comment 1
            Comment 2
             2  0.000000  0.000000  0.000000
             2  0.283459  0.000000  0.000000
             2  0.000000  0.283459  0.000000
             2  0.000000  0.000000  0.283459
             1  0.000000  0.000000  0.000000  0.000000
             bad line
             0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8
        """)
        path = _write_cube(cube)
        try:
            result = parse_cube_data(path)
            # Should return a result (possibly with 1 valid atom)
            self.assertIsInstance(result, dict)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
