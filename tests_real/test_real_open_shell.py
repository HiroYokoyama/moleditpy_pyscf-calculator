"""Open-shell results against real PySCF: ROHF/UHF labelling on reload,
spin density and ESP cubes (integrated, so the numbers are checked)."""

import os

import pytest

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("rdkit")
pytest.importorskip("PyQt6.QtCore")

from conftest import XYZ_H2O, XYZ_OH, read_cube


def _integral(path):
    values, vol = read_cube(path)
    return float(values.sum() * vol)


@pytest.mark.parametrize("method", ["ROHF", "UHF"])
def test_spin_density_of_a_doublet_integrates_to_one(
    method, run_job, load_result, run_properties
):
    """Regression: PySCF's ROHF mo_occ is 1-D, which used to be taken for
    closed-shell -- Spin Density was skipped and the result labelled RHF."""
    res = run_job(XYZ_OH, method=method, spin=2)
    assert not res.errors, res.errors
    chk = res.results["chkfile"]

    loaded = load_result(chk)
    assert loaded.results["scf_type"] == ("ROKS" if method == "ROHF" else "UHF")

    props = run_properties(chk, ["SpinDensity"], res.results["out_dir"])
    assert not props.errors, props.errors
    (spin_cube,) = props.results["files"]
    assert os.path.basename(spin_cube).startswith("spin_density")
    # One unpaired electron: integral of (rho_alpha - rho_beta) is 1.
    assert _integral(spin_cube) == pytest.approx(1.0, abs=0.02)


def test_closed_shell_spin_density_is_skipped(run_job, run_properties):
    res = run_job(XYZ_H2O)
    props = run_properties(
        res.results["chkfile"], ["SpinDensity"], res.results["out_dir"]
    )
    assert props.results["files"] == []
    assert "Skipping Spin Density" in props.log


@pytest.mark.parametrize("method,spin", [("RHF", 1), ("ROHF", 2), ("UHF", 2)])
def test_esp_density_counts_every_electron(method, spin, run_job, run_properties):
    xyz = XYZ_H2O if spin == 1 else XYZ_OH
    n_elec = 10 if spin == 1 else 9
    res = run_job(xyz, method=method, spin=spin)
    props = run_properties(res.results["chkfile"], ["ESP"], res.results["out_dir"])
    assert not props.errors, props.errors
    esp, dens = props.results["files"]
    assert os.path.basename(esp).startswith("esp")
    # Loose tolerance: the O 1s core is sharp on the default cube grid.
    assert _integral(dens) == pytest.approx(n_elec, abs=0.2)


def test_homo_cube_is_normalised(run_job, run_properties):
    res = run_job(XYZ_H2O)
    props = run_properties(res.results["chkfile"], ["HOMO"], res.results["out_dir"])
    (cube,) = props.results["files"]
    assert "HOMO" in os.path.basename(cube)
    values, vol = read_cube(cube)
    assert float((values**2).sum() * vol) == pytest.approx(1.0, abs=0.02)


def test_plugin_cube_parser_reads_real_pyscf_cubes(run_job, run_properties):
    """The viewer's parser (vis.parse_cube_data / build_grid_from_meta) on
    genuine PySCF cubes: atoms kept, grid shape and values intact."""
    pytest.importorskip("pyvista")
    pytest.importorskip("PyQt6.QtGui")
    import importlib

    import numpy as np
    from conftest import load_plugin_modules

    load_plugin_modules()
    vis = importlib.import_module("pyscf_calculator.vis")

    res = run_job(XYZ_H2O)
    props = run_properties(res.results["chkfile"], ["HOMO"], res.results["out_dir"])
    (cube,) = props.results["files"]

    meta = vis.parse_cube_data(cube)
    values, _ = read_cube(cube)
    assert [a[0] for a in meta["atoms"]] == [8, 1, 1]
    assert meta["dims"] == values.shape
    assert np.allclose(meta["data_flat"], values.ravel())
    grid = vis.build_grid_from_meta(meta)
    assert grid.n_points == values.size
    assert np.max(np.abs(grid.point_data["values"])) == pytest.approx(
        np.max(np.abs(values))
    )
