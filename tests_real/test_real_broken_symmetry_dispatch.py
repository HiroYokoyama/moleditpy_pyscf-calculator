"""Scientific regressions for the requested singlet UHF solution."""
import os

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("PyQt6.QtCore")
pytest.importorskip("rdkit")

XYZ_STRETCHED_H2 = "2\nstretched singlet\nH 0 0 0\nH 0 0 2.5"


def spin_density_norm(chk):
    data = pyscf.scf.chkfile.load(chk, "scf")
    dm = pyscf.scf.uhf.make_rdm1(data["mo_coeff"], data["mo_occ"])
    return np.linalg.norm(dm[0] - dm[1])


@pytest.mark.parametrize("job_type", ["Energy", "TDDFT"])
def test_singlet_jobs_preserve_requested_unrestricted_solution(run_job, job_type):
    result = run_job(XYZ_STRETCHED_H2, method="UHF", spin=1, job_type=job_type, break_symmetry=True, nstates=1)
    assert not result.errors
    assert spin_density_norm(result.results["chkfile"]) > .1


def test_optimizer_receives_broken_symmetry_reference(run_job, monkeypatch):
    from pyscf.geomopt import geometric_solver
    seen = []

    def optimize(mf, **kwargs):
        dm = mf.make_rdm1()
        seen.append(np.linalg.norm(dm[0] - dm[1]))
        return mf.mol

    monkeypatch.setattr(geometric_solver, "optimize", optimize)
    result = run_job(XYZ_STRETCHED_H2, method="UHF", spin=1, job_type="Geometry Optimization", break_symmetry=True)
    assert not result.errors
    assert seen and seen[0] > .1


@pytest.mark.parametrize("kind", ["Rigid", "Relaxed"])
def test_scan_points_remain_on_broken_symmetry_surface(run_job, kind):
    pytest.importorskip("geometric")
    result = run_job(
        XYZ_STRETCHED_H2, method="UHF", spin=1,
        job_type=f"{kind} Surface Scan", break_symmetry=True,
        scan_params={"type": "Dist", "atoms": [0, 1], "start": 2.5, "end": 2.6, "steps": 2},
    )
    assert not result.errors
    assert len(result.results["scan_results"]) == 2
    for step in (1, 2) if kind == "Rigid" else (0, 1):
        assert spin_density_norm(os.path.join(result.results["out_dir"], f"scan_step_{step}.chk")) > .1
