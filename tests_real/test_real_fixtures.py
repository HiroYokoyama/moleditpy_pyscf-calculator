"""The plugin's loading / analysis code on real, heavier results that are
too slow to recompute in CI: the SN2 tutorial (CH3Cl + Br-, B3LYP /
ma-def2-SVP), produced through the plugin by data/make_sn2_fixtures.py.
The full tutorial run is test_real_tutorials.py (marked slow)."""

import os
import shutil

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("rdkit")
pytest.importorskip("PyQt6.QtCore")

from conftest import load_plugin_modules, read_cube
from pyscf import lib

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@pytest.fixture
def ts_dir(tmp_path):
    # the viewers write cubes next to the checkpoint: work on a copy
    dst = tmp_path / "sn2_ts"
    shutil.copytree(os.path.join(DATA, "sn2_ts"), dst)
    return str(dst)


def test_reloaded_scan_is_the_tutorial_profile(load_result):
    res = load_result(os.path.join(DATA, "sn2_scan", "pyscf.chk")).results
    pts = res["scan_results"]
    assert [p["step"] for p in pts] == [1, 2, 3, 4, 5]
    assert all(p["converged"] for p in pts)
    assert res["scan_type"] == "Dist"
    values = [p["value"] for p in pts]
    assert values == pytest.approx(np.linspace(2.8, 1.96, 5), abs=1e-3)
    energies = [p["energy"] for p in pts]
    top = int(np.argmax(energies))
    assert 0 < top < 4  # the tutorial's "mountain-shaped" profile

    worker, utils = load_plugin_modules()
    frames = utils.read_xyz_frames(res["scan_trajectory_path"])
    assert len(frames) == 5 and all(f.startswith("6\n") for f in frames)


def test_reloaded_ts_has_one_imaginary_mode_along_the_reaction(load_result, ts_dir):
    res = load_result(os.path.join(ts_dir, "pyscf.chk")).results
    assert res["scf_type"] == "RHF"
    fd = res["freq_data"]
    assert fd["n_imaginary"] == 1
    assert len(fd["freqs"]) == 3 * 6 - 6
    assert "G_tot" in res["thermo_data"]
    assert res["dipole_total_debye"] > 0 and len(res["mulliken_charges"]) == 6
    # the halides carry most of the -1 charge at the TS
    q = dict(zip(res["atom_symbols"], res["mulliken_charges"]))
    assert q["Br"] < 0 and q["Cl"] < 0

    # C moves along the Br-C-Cl axis in the imaginary mode
    mol = lib.chkfile.load_mol(os.path.join(ts_dir, "pyscf.chk"))
    xyz = mol.atom_coords()
    axis = (xyz[5] - xyz[1]) / np.linalg.norm(xyz[5] - xyz[1])
    mode = np.array(fd["modes"][int(np.argmin(fd["freqs"]))])
    assert abs(float(mode[0] / np.linalg.norm(mode[0]) @ axis)) > 0.9


def test_orbital_and_esp_cubes_from_the_ts_checkpoint(run_properties, ts_dir):
    chk = os.path.join(ts_dir, "pyscf.chk")
    props = run_properties(chk, ["HOMO", "LUMO", "ESP"], ts_dir)
    assert not props.errors, props.errors
    names = sorted(os.path.basename(f) for f in props.results["files"])
    assert names[0].endswith("_HOMO.cube") and names[1].endswith("_LUMO.cube")
    dens = next(f for f in props.results["files"] if "density" in f)
    values, vol = read_cube(dens)
    n_elec = 6 + 17 + 3 + 35 + 1  # C Cl 3H Br, charge -1
    assert float(values.sum() * vol) == pytest.approx(n_elec, rel=0.05)


def test_the_saved_input_script_names_the_tutorial_level(ts_dir):
    with open(os.path.join(ts_dir, "pyscf_input.py"), encoding="utf-8") as fh:
        script = fh.read()
    compile(script, "pyscf_input.py", "exec")
    assert "basis='ma-def2-svp'" in script and "charge=-1" in script
    assert "mf.xc = 'b3lyp'" in script
