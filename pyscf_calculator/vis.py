import logging
import os

import numpy as np
import pyvista as pv
from PyQt6.QtGui import QColor

logger = logging.getLogger(__name__)


def parse_cube_data(filename):
    """
    Parses a Gaussian Cube file and returns raw data structures.
    Robust version with strict checks for file format.
    """
    if not filename or not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, "r") as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError("File too short to be a Cube file.")

    # --- Header Parsing ---
    try:
        # Line 3: Natoms, Origin
        tokens = lines[2].split()
        if len(tokens) < 4:
            raise ValueError("Invalid Origin line")
        n_atoms_raw = int(tokens[0])
        n_atoms = abs(n_atoms_raw)
        origin_raw = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])

        def parse_vec(line):
            t = line.split()
            if len(t) < 4:
                raise ValueError("Invalid Axis line")
            return int(t[0]), np.array([float(t[1]), float(t[2]), float(t[3])])

        nx, x_vec_raw = parse_vec(lines[3])
        ny, y_vec_raw = parse_vec(lines[4])
        nz, z_vec_raw = parse_vec(lines[5])

        is_angstrom_header = nx < 0 or ny < 0 or nz < 0
        nx, ny, nz = abs(nx), abs(ny), abs(nz)

    except (ValueError, IndexError) as e:
        raise ValueError(f"Header parsing failed: {e}") from e

    # --- Atoms Parsing ---
    atoms = []
    current_line = 6

    for _ in range(n_atoms):
        if current_line >= len(lines):
            break
        line = lines[current_line].split()
        current_line += 1

        try:
            if len(line) < 5:
                # Potentially empty line or malformed
                continue
            atomic_num = int(line[0])
            x, y, z = float(line[2]), float(line[3]), float(line[4])
            atoms.append((atomic_num, np.array([x, y, z])))
        except (ValueError, IndexError):
            continue  # skip a malformed atom line

    # A negative atom count means the cube holds several data sets. The
    # DSET_IDS block (count + that many ids, possibly wrapped) sits *after*
    # the atom lines -- skipping a line before them consumed the first atom.
    n_datasets = 1
    if n_atoms_raw < 0 and current_line < len(lines):
        try:
            parts = lines[current_line].split()
            n_datasets = max(1, int(parts[0]))
            consumed = len(parts) - 1
            current_line += 1
            while consumed < n_datasets and current_line < len(lines):
                consumed += len(lines[current_line].split())
                current_line += 1
        except (ValueError, IndexError):
            n_datasets = 1

    # --- Volumetric Data Parsing ---
    # Find start of data
    while current_line < len(lines):
        line_content = lines[current_line].strip()
        parts = line_content.split()
        if not parts:
            current_line += 1
            continue

        # Check if this line looks like data (float)
        try:
            float(parts[0])
            break  # Start of data found
        except ValueError:
            current_line += 1
            continue

    if current_line >= len(lines):
        # Allow header-only validation if explicitly requested?
        # But for 'data', we need data.
        # Fallback for empty data
        data_values = np.zeros(nx * ny * nz)
    else:
        # (np.fromstring with sep= is deprecated; this also stops at the
        # first non-number, as it did.)
        tokens = " ".join(lines[current_line:]).split()
        try:
            data_values = np.array(tokens, dtype=float)
        except ValueError:
            good = []
            for tok in tokens:
                try:
                    good.append(float(tok))
                except ValueError:
                    break
            data_values = np.array(good)

    n_points = nx * ny * nz
    expected_size = n_points * n_datasets
    actual_size = len(data_values)

    # Correct size mismatches defensively
    if actual_size > expected_size:
        # Truncate
        data_values = data_values[:expected_size]
    elif actual_size < expected_size:
        # Pad with zeros
        pad_size = expected_size - actual_size
        if pad_size > 0:
            pad = np.zeros(pad_size)
            data_values = np.concatenate((data_values, pad))

    if n_datasets > 1:
        # Values are interleaved point by point; the first n_points of the
        # mixed stream belong to no single orbital.
        data_values = data_values[0 : n_points * n_datasets : n_datasets]

    return {
        "atoms": atoms,
        "origin": origin_raw,
        "x_vec": x_vec_raw,
        "y_vec": y_vec_raw,
        "z_vec": z_vec_raw,
        "dims": (nx, ny, nz),
        "data_flat": data_values,
        "is_angstrom_header": is_angstrom_header,
    }


def build_grid_from_meta(meta):
    """
    Reconstructs the PyVista grid.
    """
    nx, ny, nz = meta["dims"]
    origin = meta["origin"].copy()
    x_vec = meta["x_vec"].copy()
    y_vec = meta["y_vec"].copy()
    z_vec = meta["z_vec"].copy()

    # Units Handling
    BOHR_TO_ANGSTROM = 0.529177210903
    convert_to_ang = True
    if meta["is_angstrom_header"]:
        convert_to_ang = False

    if convert_to_ang:
        origin *= BOHR_TO_ANGSTROM
        x_vec *= BOHR_TO_ANGSTROM
        y_vec *= BOHR_TO_ANGSTROM
        z_vec *= BOHR_TO_ANGSTROM

    # Grid Points Generation
    x_range = np.arange(nx)
    y_range = np.arange(ny)
    z_range = np.arange(nz)

    gx, gy, gz = np.meshgrid(x_range, y_range, z_range, indexing="ij")

    gx_f = gx.flatten(order="F")
    gy_f = gy.flatten(order="F")
    gz_f = gz.flatten(order="F")

    points = (
        origin + np.outer(gx_f, x_vec) + np.outer(gy_f, y_vec) + np.outer(gz_f, z_vec)
    )

    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = [nx, ny, nz]

    # Data Mapping
    raw_data = meta["data_flat"]
    vol_3d = raw_data.reshape((nx, ny, nz), order="C")
    grid.point_data["values"] = vol_3d.flatten(order="F")

    return grid


# What tearing down a pyvista actor can raise: the render window is already
# gone (RuntimeError) or the host has no 3D manager (AttributeError).
_GONE = (RuntimeError, AttributeError)
# What reading / gridding a cube can raise.
_CUBE_ERRORS = (OSError, ValueError, KeyError, IndexError)


class _Visualizer:
    """Common plotter lookup. Never cache the plotter: the host can rebuild
    it, and a cached one outlives its render window."""

    def __init__(self, mw):
        self.mw = mw

    @property
    def plotter(self):
        plotter = getattr(getattr(self.mw, "view_3d_manager", None), "plotter", None)
        if plotter is None:
            return None
        try:
            return plotter if plotter.ren_win else None
        except _GONE as exc:  # render window already closed
            logger.debug("plotter unavailable: %s", exc)
            return None


class CubeVisualizer(_Visualizer):
    def __init__(self, mw):
        super().__init__(mw)
        self.current_grid = None
        self.actors = {}  # Store actors by key
        self.data_max = 1.0

    def load_file(self, filename):
        try:
            meta = parse_cube_data(filename)
            self.current_grid = build_grid_from_meta(meta)
        except _CUBE_ERRORS as e:
            logger.warning("Error loading cube: %s", e)
            return False
        flat_data = self.current_grid.point_data["values"]
        self.data_max = float(np.max(np.abs(flat_data))) if len(flat_data) else 1.0
        return True

    def update_iso(self, isovalue, color_p, color_n, opacity, use_comp_color=False):
        if not self.current_grid:
            return
        self.clear_actors()
        if isovalue is None or not isinstance(isovalue, (int, float)):
            return
        plotter = self.plotter
        if plotter is None:
            return

        if use_comp_color:
            c = QColor(color_p)
            color_n = QColor.fromHsv(
                (c.hue() + 180) % 360, c.saturation(), c.value()
            ).name()

        # Driven by the isovalue / opacity controls (Qt slots): a VTK error
        # must be logged, not escape and abort the host application.
        try:
            for key, level, color in (
                ("p", isovalue, color_p),
                ("n", -isovalue, color_n),
            ):
                iso = self.current_grid.contour(isosurfaces=[level])
                if iso.n_points > 0:
                    self.actors[key] = plotter.add_mesh(
                        iso,
                        color=color,
                        opacity=opacity,
                        name=f"pyscf_iso_{key}",
                        reset_camera=False,
                    )
            plotter.render()
        except Exception:
            logger.exception("isosurface update failed")

    def clear_actors(self):
        plotter = self.plotter
        if plotter is not None:
            for name in ("pyscf_iso_p", "pyscf_iso_n"):
                try:
                    plotter.remove_actor(name)
                except _GONE as exc:
                    logger.debug("remove %s skipped: %s", name, exc)
        self.actors.clear()
        # No render here: the caller renders, and rendering while the
        # dialog closes crashes VTK.


class MappedVisualizer(_Visualizer):
    def __init__(self, mw):
        super().__init__(mw)
        self.grid_surf = None
        self.grid_prop = None
        self.actor = None
        self.data_surf_max = 1.0
        self.data_prop_range = (-0.1, 0.1)

    def load_files(self, surf_file, prop_file):
        try:
            self.grid_surf = build_grid_from_meta(parse_cube_data(surf_file))
            self.grid_prop = build_grid_from_meta(parse_cube_data(prop_file))
        except _CUBE_ERRORS as e:
            logger.warning("Error loading mapped cubes: %s", e)
            return False
        flat_s = self.grid_surf.point_data["values"]
        if len(flat_s) > 0:
            self.data_surf_max = float(np.max(np.abs(flat_s)))
        flat_p = self.grid_prop.point_data["values"]
        if len(flat_p) > 0:
            self.data_prop_range = (float(np.min(flat_p)), float(np.max(flat_p)))
        return True

    def _sampled_surface(self, iso_val):
        """The iso_val surface of the density with the property sampled
        onto it (point data "values"), or None when there is no surface."""
        iso = self.grid_surf.contour([iso_val], scalars="values")
        if iso.n_points == 0:
            return None
        mapped = iso.sample(self.grid_prop)
        if mapped is None or mapped.n_points == 0:
            return None
        return mapped

    def get_mapped_range(self, iso_val):
        """(min, max) of the property on the isosurface; (-0.1, 0.1) when
        there is none."""
        if not self.grid_surf or not self.grid_prop:
            return (-0.1, 0.1)
        try:
            mapped = self._sampled_surface(iso_val)
        except (ValueError, RuntimeError, TypeError) as exc:
            logger.warning("mapped range unavailable: %s", exc)
            return (-0.1, 0.1)
        mvals = None if mapped is None else mapped.point_data.get("values")
        if mvals is None or len(mvals) == 0:
            return (-0.1, 0.1)
        return (float(mvals.min()), float(mvals.max()))

    def update_mesh(self, iso_val, opacity, cmap="jet", clim=None):
        if not self.grid_surf or not self.grid_prop:
            return
        self.clear_actors()
        plotter = self.plotter
        if plotter is None:
            return
        # Driven by the ESP mapping controls (Qt slots): log, never raise.
        try:
            mapped = self._sampled_surface(iso_val)
            if mapped is None:
                return
            self.actor = plotter.add_mesh(
                mapped,
                scalars="values",
                cmap=cmap,
                clim=self.data_prop_range if clim is None else clim,
                smooth_shading=True,
                opacity=opacity,
                name="pyscf_mapped",
                reset_camera=False,
            )
            plotter.render()
        except Exception:
            logger.exception("mapped surface update failed")

    def clear_actors(self):
        plotter = self.plotter
        if plotter is None:
            return
        try:
            if self.actor:
                plotter.remove_actor(self.actor)
                self.actor = None
            plotter.remove_actor("pyscf_mapped")
        except _GONE as exc:
            logger.debug("mapped actor removal skipped: %s", exc)
        # No render here (see CubeVisualizer.clear_actors).
