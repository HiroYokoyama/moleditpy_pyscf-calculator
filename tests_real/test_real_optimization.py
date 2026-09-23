"""Geometry optimisation against real PySCF + geomeTRIC / pyberny."""

import builtins

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("rdkit")
pytest.importorskip("PyQt6.QtCore")

from conftest import XYZ_H2, XYZ_H2O
from pyscf import lib, scf


def _bond(xyz_block, i=0, j=1):
    rows = [ln.split() for ln in xyz_block.strip().splitlines()[2:]]
    a = np.array([float(x) for x in rows[i][1:4]])
    b = np.array([float(x) for x in rows[j][1:4]])
    return float(np.linalg.norm(a - b))


def _check_final_state(res, **mf_kw):
    """The checkpoint must describe the optimised structure: its geometry
    equals optimized_xyz and its energy is the SCF energy there."""
    chk = res.results["chkfile"]
    mol = lib.chkfile.load_mol(chk)
    coords = mol.atom_coords(unit="Ang")
    rows = [ln.split() for ln in res.results["optimized_xyz"].splitlines()[2:]]
    opt = np.array([[float(x) for x in r[1:4]] for r in rows])
    assert np.allclose(coords, opt, atol=1e-5)

    mol.verbose = 0
    ref = scf.RHF(mol)
    if mf_kw.get("solvent"):
        ref = ref.ddCOSMO()
        ref.with_solvent.eps = 78.2
    ref.conv_tol = 1e-10
    ref.run()
    e_chk = float(scf.chkfile.load(chk, "scf/e_tot"))
    assert e_chk == pytest.approx(ref.e_tot, abs=1e-7)


def test_geometric_optimises_h2(run_job):
    pytest.importorskip("geometric")
    res = run_job(XYZ_H2, job_type="Geometry Optimization", max_cycle=77)
    assert not res.errors, res.errors
    # RHF/STO-3G equilibrium bond length of H2 is 0.712 Angstrom.
    assert _bond(res.results["optimized_xyz"]) == pytest.approx(0.712, abs=0.003)
    _check_final_state(res)


def test_berny_fallback_ends_on_the_optimised_geometry(run_job, monkeypatch):
    """Regression: with Berny, the properties SCF and checkpoint came from
    the *starting* geometry."""
    pytest.importorskip("berny")
    real_import = builtins.__import__

    def no_geometric(name, *a, **k):
        if name == "pyscf.geomopt.geometric_solver":
            raise ImportError("blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_geometric)
    res = run_job(XYZ_H2, job_type="Geometry Optimization")
    assert not res.errors, res.errors
    assert "(Berny)" in res.results["optimized_xyz"]
    assert _bond(res.results["optimized_xyz"]) == pytest.approx(0.712, abs=0.005)
    _check_final_state(res)


def test_solvated_optimisation_is_solvated_throughout(run_job):
    pytest.importorskip("geometric")
    vac = run_job(XYZ_H2O, job_type="Geometry Optimization")
    wet = run_job(XYZ_H2O, job_type="Geometry Optimization", solvent="Water")
    assert not wet.errors, wet.errors
    _check_final_state(wet, solvent=True)
    # the solvent reaches the optimiser, so the minimum moves
    assert _bond(wet.results["optimized_xyz"]) != pytest.approx(
        _bond(vac.results["optimized_xyz"]), abs=1e-4
    )
