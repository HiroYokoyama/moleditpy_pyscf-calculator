"""Independent affine-field oracle for Cube units and point/value pairing.

Only the rendering container is doubled. The parser and coordinate assembly
are production code; expected samples follow f(x,y,z)=2x-3y+5z+7 directly,
without using the implementation's reshape/flatten mapping as the oracle.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from audit_helpers import load_module


@pytest.mark.parametrize("angstrom", [False, True])
@pytest.mark.parametrize("datasets", [1, 2])
def test_skew_cube_preserves_physical_field(monkeypatch, tmp_path, angstrom, datasets):
    vis = load_module(monkeypatch, "vis")
    grid = SimpleNamespace(point_data={})
    vis.pv.StructuredGrid.return_value = grid
    # Unequal dimensions and fully skew axes make a transpose visible.
    dims = (2, 3, 4)
    origin = np.array([0.7, -1.2, 2.3])
    axes = np.array([[0.4, 0.1, -0.2], [-0.3, 0.6, 0.1], [0.2, -0.1, 0.5]])
    scale = 1.0 if angstrom else 0.529177210903  # CODATA 2018, Angstrom/bohr
    lines = ["synthetic analytic field", "x outermost, z innermost"]
    atom_count = -1 if datasets == 2 else 1
    lines.append(f"{atom_count} " + " ".join(map(str, origin)))
    for count, axis in zip(dims, axes):
        lines.append(f"{-count if angstrom else count} " + " ".join(map(str, axis)))
    lines.append("8 8.0 0.2 0.3 0.4")
    if datasets == 2:
        lines.append("2 17 23")
    values = []
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                x, y, z = (origin + i * axes[0] + j * axes[1] + k * axes[2]) * scale
                value = 2 * x - 3 * y + 5 * z + 7
                values.append(value)
                if datasets == 2:
                    values.append(-1000 - value)  # unmistakably different second MO
    lines.append(" ".join(f"{value:.15e}" for value in values))
    path = tmp_path / "analytic.cube"
    path.write_text("\n".join(lines) + "\n", encoding="ascii")

    meta = vis.parse_cube_data(str(path))
    vis.build_grid_from_meta(meta)

    assert grid.dimensions == list(dims)
    assert len(grid.points) == np.prod(dims)
    x, y, z = np.asarray(grid.points).T
    np.testing.assert_allclose(
        grid.point_data["values"], 2 * x - 3 * y + 5 * z + 7, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        grid.points.min(axis=0), np.array([0.1, -1.5, 2.1]) * scale, atol=1e-12
    )
    np.testing.assert_allclose(
        grid.points.max(axis=0), np.array([1.7, 0.1, 4.0]) * scale, atol=1e-12
    )
    # Geometry conversion must not multiply orbital/density values by bohr^3.
    assert grid.point_data["values"][0] == pytest.approx(values[0], abs=1e-12)
