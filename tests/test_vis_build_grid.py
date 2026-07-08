"""
tests/test_vis_build_grid.py

Tests for vis.build_grid_from_meta() — the pure-numpy grid construction
function.  pyvista is mocked so no real VTK rendering occurs.

Coverage targets (vis.py lines 134-181):
  - Bohr→Angstrom unit conversion when is_angstrom_header=False
  - No conversion when is_angstrom_header=True
  - Grid dimensions set correctly
  - point_data["values"] set from data_flat reshaped (C→F order)
  - Non-unit voxel vectors (anisotropic spacing)
"""

import os
import sys
import types
import unittest
import importlib.util
import numpy as np
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Stubs for pyvista and Qt (vis.py imports both at module level)
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
    spec = importlib.util.spec_from_file_location("pyscf_calculator_vis_grid_test", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pyscf_calculator_vis_grid_test"] = mod
    spec.loader.exec_module(mod)
    return mod


_vis_mod = _load_vis_mod()
build_grid_from_meta = _vis_mod.build_grid_from_meta
BOHR_TO_ANG = 0.529177210903


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_meta(
    nx=2,
    ny=2,
    nz=2,
    is_angstrom=False,
    origin=None,
    x_vec=None,
    y_vec=None,
    z_vec=None,
    data=None,
):
    """Build a minimal meta dict for build_grid_from_meta."""
    if origin is None:
        origin = np.zeros(3)
    if x_vec is None:
        x_vec = np.array([0.5, 0.0, 0.0])  # 0.5 Bohr step
    if y_vec is None:
        y_vec = np.array([0.0, 0.5, 0.0])
    if z_vec is None:
        z_vec = np.array([0.0, 0.0, 0.5])
    if data is None:
        data = np.arange(float(nx * ny * nz))
    return {
        "dims": (nx, ny, nz),
        "origin": origin.copy(),
        "x_vec": x_vec.copy(),
        "y_vec": y_vec.copy(),
        "z_vec": z_vec.copy(),
        "data_flat": data.copy(),
        "is_angstrom_header": is_angstrom,
    }


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


class TestBuildGridUnitConversion(unittest.TestCase):
    def _capture_grid_call(self, meta):
        """Run build_grid_from_meta and return the mock StructuredGrid."""
        mock_grid = MagicMock()
        _vis_mod.pv.StructuredGrid.return_value = mock_grid
        build_grid_from_meta(meta)
        return mock_grid

    def test_bohr_origin_converted_to_angstrom(self):
        """When is_angstrom_header=False, origin must be scaled by BOHR_TO_ANG."""
        origin_bohr = np.array([1.0, 2.0, 3.0])
        meta = _make_meta(origin=origin_bohr, is_angstrom=False)
        mock_grid = self._capture_grid_call(meta)
        # The points passed to the grid embed the converted origin.
        # We verify by checking that the first point (ix=iy=iz=0) equals origin_ang.
        points_assigned = mock_grid.points
        expected_origin = origin_bohr * BOHR_TO_ANG
        # First point (all-zero indices) must match the converted origin
        np.testing.assert_allclose(points_assigned[0], expected_origin, atol=1e-10)

    def test_angstrom_header_no_conversion(self):
        """When is_angstrom_header=True, vectors must remain unchanged."""
        x_vec_ang = np.array([0.3, 0.0, 0.0])
        meta = _make_meta(x_vec=x_vec_ang, is_angstrom=True)
        mock_grid = self._capture_grid_call(meta)
        # First two points differ only in ix index → difference = x_vec
        pts = mock_grid.points
        delta = pts[1] - pts[0]  # ix goes from 0 to 1 in F-order for x first
        np.testing.assert_allclose(delta, x_vec_ang, atol=1e-10)

    def test_bohr_x_vec_scaled(self):
        """x_vec in Bohr must be multiplied by BOHR_TO_ANG when converting."""
        x_vec_bohr = np.array([1.0, 0.0, 0.0])
        meta = _make_meta(x_vec=x_vec_bohr, is_angstrom=False)
        mock_grid = self._capture_grid_call(meta)
        pts = mock_grid.points
        delta = pts[1] - pts[0]
        np.testing.assert_allclose(delta, x_vec_bohr * BOHR_TO_ANG, atol=1e-10)

    def test_zero_origin_stays_zero(self):
        """Origin at [0,0,0] must remain [0,0,0] regardless of unit mode."""
        for is_ang in (True, False):
            with self.subTest(is_angstrom=is_ang):
                meta = _make_meta(origin=np.zeros(3), is_angstrom=is_ang)
                mock_grid = self._capture_grid_call(meta)
                np.testing.assert_allclose(mock_grid.points[0], [0, 0, 0], atol=1e-12)


# ---------------------------------------------------------------------------
# Grid dimensions
# ---------------------------------------------------------------------------


class TestBuildGridDimensions(unittest.TestCase):
    def test_dimensions_set_on_grid(self):
        """grid.dimensions must be set to [nx, ny, nz]."""
        nx, ny, nz = 3, 4, 5
        mock_grid = MagicMock()
        _vis_mod.pv.StructuredGrid.return_value = mock_grid
        meta = _make_meta(nx=nx, ny=ny, nz=nz)
        build_grid_from_meta(meta)
        self.assertEqual(mock_grid.dimensions, [nx, ny, nz])

    def test_total_points_equals_nx_times_ny_times_nz(self):
        """Number of point rows in points array must equal nx*ny*nz."""
        nx, ny, nz = 2, 3, 4
        mock_grid = MagicMock()
        _vis_mod.pv.StructuredGrid.return_value = mock_grid
        meta = _make_meta(nx=nx, ny=ny, nz=nz, data=np.zeros(nx * ny * nz))
        build_grid_from_meta(meta)
        pts = mock_grid.points
        self.assertEqual(pts.shape[0], nx * ny * nz)
        self.assertEqual(pts.shape[1], 3)


# ---------------------------------------------------------------------------
# point_data assignment
# ---------------------------------------------------------------------------


class TestBuildGridPointData(unittest.TestCase):
    def test_point_data_values_assigned(self):
        """grid.point_data['values'] must be assigned."""
        nx, ny, nz = 2, 2, 2
        data = np.arange(float(nx * ny * nz))
        mock_grid = MagicMock()
        _vis_mod.pv.StructuredGrid.return_value = mock_grid
        meta = _make_meta(nx=nx, ny=ny, nz=nz, data=data)
        build_grid_from_meta(meta)
        # Check that the assignment happened (mock records item assignment)
        self.assertIn("values", mock_grid.point_data.__setitem__.call_args[0])

    def test_point_data_length_matches_grid_size(self):
        """Flattened point_data must have length nx*ny*nz."""
        nx, ny, nz = 2, 3, 2
        data = np.ones(nx * ny * nz) * 0.5
        mock_grid = MagicMock()
        _vis_mod.pv.StructuredGrid.return_value = mock_grid
        meta = _make_meta(nx=nx, ny=ny, nz=nz, data=data)
        build_grid_from_meta(meta)
        assigned = mock_grid.point_data.__setitem__.call_args[0][1]
        self.assertEqual(len(assigned), nx * ny * nz)

    def test_point_data_order_f_reshape(self):
        """Data is reshaped (C) then flattened (F); round-trip must preserve values."""
        nx, ny, nz = 2, 2, 2
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        mock_grid = MagicMock()
        _vis_mod.pv.StructuredGrid.return_value = mock_grid
        meta = _make_meta(nx=nx, ny=ny, nz=nz, data=data)
        build_grid_from_meta(meta)
        assigned = mock_grid.point_data.__setitem__.call_args[0][1]
        # Check that the round-trip (C reshape → F flatten) produces a permutation of data
        np.testing.assert_allclose(sorted(assigned), sorted(data), atol=1e-12)


# ---------------------------------------------------------------------------
# Non-unit voxel vectors
# ---------------------------------------------------------------------------


class TestBuildGridAnisotropic(unittest.TestCase):
    def test_anisotropic_voxel_vectors(self):
        """Non-cubic voxels: each step in x maps to x_vec displacement."""
        x_vec = np.array([1.5, 0.0, 0.0])  # step size 1.5 Å
        y_vec = np.array([0.0, 2.0, 0.0])
        z_vec = np.array([0.0, 0.0, 0.5])
        meta = _make_meta(
            nx=3,
            ny=2,
            nz=2,
            x_vec=x_vec,
            y_vec=y_vec,
            z_vec=z_vec,
            is_angstrom=True,
            data=np.zeros(3 * 2 * 2),
        )
        mock_grid = MagicMock()
        _vis_mod.pv.StructuredGrid.return_value = mock_grid
        build_grid_from_meta(meta)
        pts = mock_grid.points
        # All x-coordinates span [0, 1.5, 3.0]
        unique_x = sorted(set(np.round(pts[:, 0], 8)))
        self.assertEqual(len(unique_x), 3)
        self.assertAlmostEqual(unique_x[1] - unique_x[0], 1.5, places=8)


if __name__ == "__main__":
    unittest.main()
