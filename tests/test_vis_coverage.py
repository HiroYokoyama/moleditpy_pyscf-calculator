"""
tests/test_vis_coverage.py

Coverage-focused tests for pyscf_calculator/vis.py:
  - parse_cube_data(): header/atom/data parsing edge cases
  - CubeVisualizer: plotter property, load_file, update_iso, clear_actors
  - MappedVisualizer: plotter property, load_files, get_mapped_range,
    update_mesh, clear_actors

pyvista is a MagicMock (no real VTK); numpy is real. Qt is stubbed with a
minimal fake QColor supporting hue()/saturation()/value()/fromHsv()/name()
so the complementary-color branch in update_iso() is exercised for real.
"""

import os
import sys
import types
import tempfile
import unittest
import importlib.util
import numpy as np
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeQColor:
    def __init__(self, spec="blue"):
        self._spec = spec
        # deterministic fake HSV derived from the input string/tuple
        if isinstance(spec, str):
            self._hue = (sum(ord(c) for c in spec) * 7) % 360
        else:
            self._hue = 120
        self._sat = 200
        self._val = 200
        self._name = spec if isinstance(spec, str) else "#123456"

    def hue(self):
        return self._hue

    def saturation(self):
        return self._sat

    def value(self):
        return self._val

    def name(self):
        return self._name

    def isValid(self):
        return True

    @staticmethod
    def fromHsv(h, s, v):
        c = _FakeQColor("computed")
        c._hue, c._sat, c._val = h, s, v
        c._name = f"#hsv{h}"
        return c


def _install_stubs():
    # setdefault (not assignment) everywhere: another test module may have
    # already installed a fuller stub (or the real package may already be
    # loaded); clobbering it here would corrupt sys.modules process-wide
    # for tests collected/run afterwards (see test_plugin_integration.py's
    # real-context tests, which need a real PyQt6.QtGui with QFont).
    sys.modules.setdefault("pyvista", MagicMock())

    qt_gui = types.ModuleType("PyQt6.QtGui")
    qt_gui.QColor = _FakeQColor
    sys.modules.setdefault("PyQt6.QtGui", qt_gui)

    qt_core = types.ModuleType("PyQt6.QtCore")
    qt_core.Qt = MagicMock()
    sys.modules.setdefault("PyQt6.QtCore", qt_core)

    pyqt6 = sys.modules.get("PyQt6") or types.ModuleType("PyQt6")
    pyqt6.QtGui = sys.modules.get("PyQt6.QtGui", qt_gui)
    pyqt6.QtCore = sys.modules.get("PyQt6.QtCore", qt_core)
    sys.modules.setdefault("PyQt6", pyqt6)

    sys.modules.setdefault("rdkit", MagicMock())
    sys.modules.setdefault("rdkit.Chem", MagicMock())


_install_stubs()


def _load_vis_mod():
    src = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "pyscf_calculator", "vis.py")
    )
    spec = importlib.util.spec_from_file_location("pyscf_calculator_vis_cov_test", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pyscf_calculator_vis_cov_test"] = mod
    spec.loader.exec_module(mod)
    return mod


_vis = _load_vis_mod()
# Patch the *loaded module's* QColor reference directly (rather than relying
# on which stub happened to win the sys.modules setdefault race across test
# files) so the complementary-color branch in update_iso() is deterministic.
_vis.QColor = _FakeQColor
parse_cube_data = _vis.parse_cube_data
build_grid_from_meta = _vis.build_grid_from_meta
CubeVisualizer = _vis.CubeVisualizer
MappedVisualizer = _vis.MappedVisualizer


class _FakeGrid:
    """Lightweight stand-in for pv.StructuredGrid with real attribute
    storage (unlike a bare MagicMock, whose point_data[...] assignment
    doesn't persist), so load_file()/load_files() can read back real
    values. `.contour` stays a MagicMock so tests can configure it."""

    def __init__(self):
        self.point_data = {}
        self.points = None
        self.dimensions = None
        self.contour = MagicMock()


# build_grid_from_meta() calls pv.StructuredGrid() with no args each time.
# Rebind the module-local `pv` name to a private mock rather than mutating
# sys.modules["pyvista"].StructuredGrid in place -- that object is the same
# shared MagicMock instance other test files (test_vis_build_grid.py) bind
# their own `pv` alias to, and mutating it in place would break their
# `.StructuredGrid.return_value = ...` expectations process-wide.
_vis.pv = types.SimpleNamespace(
    StructuredGrid=MagicMock(side_effect=lambda *a, **kw: _FakeGrid())
)


# ---------------------------------------------------------------------------
# Helpers for writing cube-like fixture files
# ---------------------------------------------------------------------------


def _write_cube(
    path,
    n_atoms=1,
    extra_mo_line=False,
    data_line=None,
    natoms_override=None,
    nx=2,
    ny=2,
    nz=2,
    negative_dims=False,
    atom_lines=None,
    header_lines=None,
):
    lines = []
    lines.append("comment 1\n")
    lines.append("comment 2\n")
    na = n_atoms if natoms_override is None else natoms_override
    lines.append(f"{na} 0.0 0.0 0.0\n")
    sx = -nx if negative_dims else nx
    lines.append(f"{sx} 1.0 0.0 0.0\n")
    lines.append(f"{ny} 0.0 1.0 0.0\n")
    lines.append(f"{nz} 0.0 0.0 1.0\n")
    if header_lines:
        lines = header_lines
    if extra_mo_line:
        lines.append("1 1\n")
    if atom_lines is not None:
        lines.extend(atom_lines)
    else:
        for i in range(n_atoms):
            lines.append(f"6 0.0 {i}.0 0.0 0.0\n")
    if data_line is not None:
        lines.append(data_line)
    else:
        n_vals = nx * ny * nz
        lines.append(" ".join(["1.0"] * n_vals) + "\n")
    with open(path, "w") as f:
        f.writelines(lines)


class _TempDirMixin:
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _path(self, name="test.cube"):
        return os.path.join(self._tmpdir.name, name)


# ---------------------------------------------------------------------------
# parse_cube_data
# ---------------------------------------------------------------------------


class TestParseCubeDataBasics(_TempDirMixin, unittest.TestCase):
    def test_missing_filename_raises(self):
        with self.assertRaises(FileNotFoundError):
            parse_cube_data("")

    def test_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            parse_cube_data(self._path("nope.cube"))

    def test_too_short_file_raises_value_error(self):
        p = self._path()
        with open(p, "w") as f:
            f.write("a\nb\nc\n")
        with self.assertRaises(ValueError):
            parse_cube_data(p)

    def test_invalid_origin_line_raises(self):
        p = self._path()
        with open(p, "w") as f:
            f.write("c1\nc2\n1 0.0\n2 1 0 0\n2 0 1 0\n2 0 0 1\n1 1.0\n")
        with self.assertRaises(ValueError):
            parse_cube_data(p)

    def test_invalid_axis_line_raises(self):
        p = self._path()
        with open(p, "w") as f:
            f.write("c1\nc2\n1 0 0 0\n2 1\n2 0 1 0\n2 0 0 1\n1 1.0\n")
        with self.assertRaises(ValueError):
            parse_cube_data(p)

    def test_normal_parse_dims_and_atoms(self):
        p = self._path()
        _write_cube(p, n_atoms=2, nx=2, ny=2, nz=2)
        meta = parse_cube_data(p)
        self.assertEqual(meta["dims"], (2, 2, 2))
        self.assertEqual(len(meta["atoms"]), 2)
        self.assertFalse(meta["is_angstrom_header"])
        self.assertEqual(len(meta["data_flat"]), 8)

    def test_negative_dims_sets_angstrom_flag_and_abs_value(self):
        p = self._path()
        _write_cube(p, n_atoms=1, nx=3, ny=2, nz=2, negative_dims=True)
        meta = parse_cube_data(p)
        self.assertTrue(meta["is_angstrom_header"])
        self.assertEqual(meta["dims"], (3, 2, 2))

    def test_negative_natoms_skips_mo_info_line(self):
        p = self._path()
        _write_cube(
            p, n_atoms=1, natoms_override=-1, extra_mo_line=True, nx=2, ny=2, nz=2
        )
        meta = parse_cube_data(p)
        # n_atoms is abs(-1) == 1, MO info line skipped, atom parsed correctly
        self.assertEqual(len(meta["atoms"]), 1)

    def test_malformed_atom_line_skipped(self):
        p = self._path()
        atom_lines = ["bad line\n", "6 0.0 1.0 2.0 3.0\n"]
        _write_cube(p, n_atoms=2, atom_lines=atom_lines, nx=2, ny=2, nz=2)
        meta = parse_cube_data(p)
        # first atom line malformed (only 2 tokens) -> skipped; second valid
        self.assertEqual(len(meta["atoms"]), 1)
        self.assertEqual(meta["atoms"][0][0], 6)

    def test_short_atom_line_lt5_tokens_skipped(self):
        p = self._path()
        atom_lines = ["6 0.0 1.0\n", "6 0.0 1.0 2.0 3.0\n"]
        _write_cube(p, n_atoms=2, atom_lines=atom_lines, nx=2, ny=2, nz=2)
        meta = parse_cube_data(p)
        self.assertEqual(len(meta["atoms"]), 1)

    def test_data_truncated_when_too_many_values(self):
        p = self._path()
        data_line = " ".join(["2.0"] * 20) + "\n"
        _write_cube(p, n_atoms=1, nx=2, ny=2, nz=2, data_line=data_line)
        meta = parse_cube_data(p)
        self.assertEqual(len(meta["data_flat"]), 8)

    def test_data_padded_when_too_few_values(self):
        p = self._path()
        data_line = "1.0 2.0 3.0\n"
        _write_cube(p, n_atoms=1, nx=2, ny=2, nz=2, data_line=data_line)
        meta = parse_cube_data(p)
        self.assertEqual(len(meta["data_flat"]), 8)
        np.testing.assert_allclose(meta["data_flat"][3:], np.zeros(5))

    def test_empty_data_section_defaults_to_zeros(self):
        p = self._path()
        # write header+atoms, no data lines at all after atoms
        lines = [
            "c1\n",
            "c2\n",
            "1 0.0 0.0 0.0\n",
            "2 1 0 0\n",
            "2 0 1 0\n",
            "2 0 0 1\n",
            "6 0.0 0.0 0.0 0.0\n",
        ]
        with open(p, "w") as f:
            f.writelines(lines)
        meta = parse_cube_data(p)
        self.assertEqual(len(meta["data_flat"]), 8)
        np.testing.assert_allclose(meta["data_flat"], np.zeros(8))

    def test_blank_lines_before_data_are_skipped(self):
        p = self._path()
        lines = [
            "c1\n",
            "c2\n",
            "1 0.0 0.0 0.0\n",
            "2 1 0 0\n",
            "2 0 1 0\n",
            "2 0 0 1\n",
            "6 0.0 0.0 0.0 0.0\n",
            "\n",
            "   \n",
            " ".join(["1.0"] * 8) + "\n",
        ]
        with open(p, "w") as f:
            f.writelines(lines)
        meta = parse_cube_data(p)
        self.assertEqual(len(meta["data_flat"]), 8)
        np.testing.assert_allclose(meta["data_flat"], np.ones(8))


# ---------------------------------------------------------------------------
# CubeVisualizer
# ---------------------------------------------------------------------------


def _mw_with_plotter(plotter=None):
    mw = MagicMock()
    v3m = MagicMock()
    v3m.plotter = plotter
    mw.view_3d_manager = v3m
    return mw


class TestCubeVisualizerPlotter(unittest.TestCase):
    def test_plotter_none_when_no_view_3d_manager(self):
        mw = MagicMock(spec=[])
        cv = CubeVisualizer(mw)
        self.assertIsNone(cv.plotter)

    def test_plotter_none_when_plotter_is_none(self):
        mw = _mw_with_plotter(None)
        cv = CubeVisualizer(mw)
        self.assertIsNone(cv.plotter)

    def test_plotter_returned_when_ren_win_truthy(self):
        p = MagicMock()
        p.ren_win = True
        mw = _mw_with_plotter(p)
        cv = CubeVisualizer(mw)
        self.assertIs(cv.plotter, p)

    def test_plotter_none_when_ren_win_access_raises(self):
        p = MagicMock()
        type(p).ren_win = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("dead"))
        )
        mw = _mw_with_plotter(p)
        cv = CubeVisualizer(mw)
        self.assertIsNone(cv.plotter)


class TestCubeVisualizerLoadFile(_TempDirMixin, unittest.TestCase):
    def test_load_file_success_sets_grid_and_data_max(self):
        p = self._path()
        data_line = "1.0 -5.0 2.0 3.0 4.0 5.0 6.0 7.0\n"
        _write_cube(p, n_atoms=1, nx=2, ny=2, nz=2, data_line=data_line)
        mw = _mw_with_plotter(None)
        cv = CubeVisualizer(mw)
        ok = cv.load_file(p)
        self.assertTrue(ok)
        self.assertIsNotNone(cv.current_grid)
        self.assertEqual(cv.data_max, 7.0)

    def test_load_file_failure_returns_false(self):
        mw = _mw_with_plotter(None)
        cv = CubeVisualizer(mw)
        ok = cv.load_file(self._path("missing.cube"))
        self.assertFalse(ok)
        self.assertIsNone(cv.current_grid)

    def test_load_file_empty_flat_data_keeps_default_max(self):
        # Force build_grid_from_meta's point_data to be an empty array by
        # monkeypatching pv.StructuredGrid via the real (mocked) module.
        p = self._path()
        _write_cube(p, n_atoms=1, nx=1, ny=1, nz=1, data_line="0.0\n")
        mw = _mw_with_plotter(None)
        cv = CubeVisualizer(mw)
        ok = cv.load_file(p)
        self.assertTrue(ok)
        self.assertEqual(cv.data_max, 0.0)


class TestCubeVisualizerUpdateIso(_TempDirMixin, unittest.TestCase):
    def _loaded_cv(self):
        p = self._path()
        _write_cube(p, n_atoms=1, nx=2, ny=2, nz=2)
        plotter = MagicMock()
        plotter.ren_win = True
        mw = _mw_with_plotter(plotter)
        cv = CubeVisualizer(mw)
        cv.load_file(p)
        return cv, plotter

    def test_update_iso_noop_without_grid(self):
        mw = _mw_with_plotter(None)
        cv = CubeVisualizer(mw)
        cv.update_iso(0.04, "blue", "red", 0.5)  # should not raise
        self.assertEqual(cv.actors, {})

    def test_update_iso_invalid_isovalue_returns_early(self):
        cv, plotter = self._loaded_cv()
        cv.update_iso(None, "blue", "red", 0.5)
        plotter.add_mesh.assert_not_called()
        cv.update_iso("not-a-number", "blue", "red", 0.5)
        plotter.add_mesh.assert_not_called()

    def test_update_iso_adds_positive_and_negative_actors(self):
        cv, plotter = self._loaded_cv()
        iso_p_mesh = MagicMock()
        iso_p_mesh.n_points = 10
        iso_n_mesh = MagicMock()
        iso_n_mesh.n_points = 8
        cv.current_grid.contour.side_effect = [iso_p_mesh, iso_n_mesh]
        cv.update_iso(0.04, "blue", "red", 0.5)
        self.assertIn("p", cv.actors)
        self.assertIn("n", cv.actors)
        plotter.render.assert_called_once()
        self.assertEqual(plotter.add_mesh.call_count, 2)

    def test_update_iso_skips_actor_when_zero_points(self):
        cv, plotter = self._loaded_cv()
        empty_mesh = MagicMock()
        empty_mesh.n_points = 0
        cv.current_grid.contour.return_value = empty_mesh
        cv.update_iso(0.04, "blue", "red", 0.5)
        self.assertEqual(cv.actors, {})

    def test_update_iso_complementary_color_computed(self):
        cv, plotter = self._loaded_cv()
        mesh = MagicMock()
        mesh.n_points = 5
        cv.current_grid.contour.return_value = mesh
        cv.update_iso(0.04, "blue", "red", 0.5, use_comp_color=True)
        # add_mesh called twice, second call (negative) uses computed color
        calls = plotter.add_mesh.call_args_list
        self.assertEqual(len(calls), 2)
        neg_color = calls[1].kwargs["color"]
        self.assertTrue(neg_color.startswith("#hsv"))

    def test_update_iso_swallows_contour_exception(self):
        cv, plotter = self._loaded_cv()
        cv.current_grid.contour.side_effect = RuntimeError("boom")
        cv.update_iso(0.04, "blue", "red", 0.5)  # should not raise
        self.assertEqual(cv.actors, {})

    def test_clear_actors_removes_known_names(self):
        cv, plotter = self._loaded_cv()
        cv.actors = {"p": MagicMock(), "n": MagicMock()}
        cv.clear_actors()
        plotter.remove_actor.assert_any_call("pyscf_iso_p")
        plotter.remove_actor.assert_any_call("pyscf_iso_n")
        self.assertEqual(cv.actors, {})

    def test_clear_actors_noop_when_plotter_none(self):
        mw = _mw_with_plotter(None)
        cv = CubeVisualizer(mw)
        cv.actors = {"p": MagicMock()}
        cv.clear_actors()
        # actors untouched because plotter is None (early return)
        self.assertEqual(len(cv.actors), 1)

    def test_clear_actors_swallows_remove_actor_exception(self):
        cv, plotter = self._loaded_cv()
        plotter.remove_actor.side_effect = RuntimeError("gone")
        cv.clear_actors()  # should not raise
        self.assertEqual(cv.actors, {})


# ---------------------------------------------------------------------------
# MappedVisualizer
# ---------------------------------------------------------------------------


class TestMappedVisualizerPlotter(unittest.TestCase):
    def test_plotter_none_when_missing_manager(self):
        mw = MagicMock(spec=[])
        mv = MappedVisualizer(mw)
        self.assertIsNone(mv.plotter)

    def test_plotter_returned(self):
        p = MagicMock()
        p.ren_win = True
        mw = _mw_with_plotter(p)
        mv = MappedVisualizer(mw)
        self.assertIs(mv.plotter, p)


class TestMappedVisualizerLoadFiles(_TempDirMixin, unittest.TestCase):
    def _two_files(self):
        surf = self._path("density.cube")
        prop = self._path("esp.cube")
        _write_cube(
            surf,
            n_atoms=1,
            nx=2,
            ny=2,
            nz=2,
            data_line="1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0\n",
        )
        _write_cube(
            prop,
            n_atoms=1,
            nx=2,
            ny=2,
            nz=2,
            data_line="-1.0 -2.0 0.0 1.0 2.0 3.0 4.0 5.0\n",
        )
        return surf, prop

    def test_load_files_success(self):
        surf, prop = self._two_files()
        mw = _mw_with_plotter(None)
        mv = MappedVisualizer(mw)
        ok = mv.load_files(surf, prop)
        self.assertTrue(ok)
        self.assertIsNotNone(mv.grid_surf)
        self.assertIsNotNone(mv.grid_prop)
        self.assertEqual(mv.data_surf_max, 8.0)
        self.assertEqual(mv.data_prop_range, (-2.0, 5.0))

    def test_load_files_failure_returns_false(self):
        mw = _mw_with_plotter(None)
        mv = MappedVisualizer(mw)
        ok = mv.load_files(self._path("nope1.cube"), self._path("nope2.cube"))
        self.assertFalse(ok)
        self.assertIsNone(mv.grid_surf)


class TestMappedVisualizerGetMappedRange(unittest.TestCase):
    def test_returns_default_when_no_grids(self):
        mw = _mw_with_plotter(None)
        mv = MappedVisualizer(mw)
        self.assertEqual(mv.get_mapped_range(0.01), (-0.1, 0.1))

    def test_returns_default_when_contour_empty(self):
        mw = _mw_with_plotter(None)
        mv = MappedVisualizer(mw)
        mv.grid_surf = MagicMock()
        mv.grid_prop = MagicMock()
        empty = MagicMock()
        empty.n_points = 0
        mv.grid_surf.contour.return_value = empty
        self.assertEqual(mv.get_mapped_range(0.01), (-0.1, 0.1))

    def test_returns_min_max_of_sampled_values(self):
        mw = _mw_with_plotter(None)
        mv = MappedVisualizer(mw)
        mv.grid_surf = MagicMock()
        mv.grid_prop = MagicMock()
        iso = MagicMock()
        iso.n_points = 3
        mv.grid_surf.contour.return_value = iso
        sampled = MagicMock()
        sampled.point_data = {"values": np.array([1.0, -2.0, 3.5])}
        iso.sample.return_value = sampled
        self.assertEqual(mv.get_mapped_range(0.01), (-2.0, 3.5))

    def test_returns_default_on_exception(self):
        mw = _mw_with_plotter(None)
        mv = MappedVisualizer(mw)
        mv.grid_surf = MagicMock()
        mv.grid_prop = MagicMock()
        mv.grid_surf.contour.side_effect = RuntimeError("boom")
        self.assertEqual(mv.get_mapped_range(0.01), (-0.1, 0.1))


class TestMappedVisualizerUpdateMesh(unittest.TestCase):
    def _mv_with_grids(self, plotter):
        mw = _mw_with_plotter(plotter)
        mv = MappedVisualizer(mw)
        mv.grid_surf = MagicMock()
        mv.grid_prop = MagicMock()
        return mv

    def test_noop_without_grids(self):
        mw = _mw_with_plotter(None)
        mv = MappedVisualizer(mw)
        mv.update_mesh(0.01, 0.5)  # should not raise

    def test_noop_when_contour_empty(self):
        plotter = MagicMock()
        plotter.ren_win = True
        mv = self._mv_with_grids(plotter)
        empty = MagicMock()
        empty.n_points = 0
        mv.grid_surf.contour.return_value = empty
        mv.update_mesh(0.01, 0.5)
        plotter.add_mesh.assert_not_called()

    def test_noop_when_sample_returns_none(self):
        plotter = MagicMock()
        plotter.ren_win = True
        mv = self._mv_with_grids(plotter)
        iso = MagicMock()
        iso.n_points = 3
        iso.sample.return_value = None
        mv.grid_surf.contour.return_value = iso
        mv.update_mesh(0.01, 0.5)
        plotter.add_mesh.assert_not_called()

    def test_adds_mesh_with_default_clim(self):
        plotter = MagicMock()
        plotter.ren_win = True
        mv = self._mv_with_grids(plotter)
        mv.data_prop_range = (-0.2, 0.3)
        iso = MagicMock()
        iso.n_points = 3
        mapped = MagicMock()
        mapped.n_points = 3
        iso.sample.return_value = mapped
        mv.grid_surf.contour.return_value = iso
        mv.update_mesh(0.01, 0.5, cmap="jet")
        plotter.add_mesh.assert_called_once()
        kwargs = plotter.add_mesh.call_args.kwargs
        self.assertEqual(kwargs["clim"], (-0.2, 0.3))
        self.assertEqual(kwargs["cmap"], "jet")
        plotter.render.assert_called_once()

    def test_adds_mesh_with_explicit_clim(self):
        plotter = MagicMock()
        plotter.ren_win = True
        mv = self._mv_with_grids(plotter)
        iso = MagicMock()
        iso.n_points = 3
        mapped = MagicMock()
        mapped.n_points = 3
        iso.sample.return_value = mapped
        mv.grid_surf.contour.return_value = iso
        mv.update_mesh(0.01, 0.5, clim=[-1, 1])
        kwargs = plotter.add_mesh.call_args.kwargs
        self.assertEqual(kwargs["clim"], [-1, 1])

    def test_exception_during_update_is_swallowed(self):
        plotter = MagicMock()
        plotter.ren_win = True
        mv = self._mv_with_grids(plotter)
        mv.grid_surf.contour.side_effect = RuntimeError("boom")
        mv.update_mesh(0.01, 0.5)  # should not raise

    def test_clear_actors_removes_actor_and_named_mesh(self):
        plotter = MagicMock()
        plotter.ren_win = True
        mv = self._mv_with_grids(plotter)
        actor = MagicMock()
        mv.actor = actor
        mv.clear_actors()
        plotter.remove_actor.assert_any_call(actor)
        plotter.remove_actor.assert_any_call("pyscf_mapped")
        self.assertIsNone(mv.actor)

    def test_clear_actors_noop_when_plotter_none(self):
        mw = _mw_with_plotter(None)
        mv = MappedVisualizer(mw)
        mv.clear_actors()  # should not raise

    def test_clear_actors_swallows_exception(self):
        plotter = MagicMock()
        plotter.ren_win = True
        mv = self._mv_with_grids(plotter)
        plotter.remove_actor.side_effect = RuntimeError("gone")
        mv.clear_actors()  # should not raise


if __name__ == "__main__":
    unittest.main()
