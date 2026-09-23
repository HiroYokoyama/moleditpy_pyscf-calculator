"""Rigid and relaxed scans against real PySCF + geomeTRIC + RDKit."""

import csv
import os

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("rdkit")
pytest.importorskip("PyQt6.QtCore")

from conftest import XYZ_H2, XYZ_H2O, XYZ_OH, load_plugin_modules
from pyscf import gto, scf


def _frame_mol(frame, spin=0):
    atoms = "\n".join(frame.strip().splitlines()[2:])
    return gto.M(atom=atoms, basis="sto-3g", spin=spin, verbose=0)


def _energy(mf_cls, frame, spin=0):
    mf = mf_cls(_frame_mol(frame, spin))
    mf.conv_tol = 1e-10
    return mf.run().e_tot


def _csv_rows(res):
    with open(os.path.join(res.results["out_dir"], "scan_results.csv")) as fh:
        return list(csv.DictReader(fh))


def test_rigid_distance_scan_energies_are_single_points(run_job):
    params = {"type": "Dist", "atoms": [0, 1], "start": 0.6, "end": 0.9, "steps": 3}
    res = run_job(XYZ_H2, job_type="Rigid Surface Scan", scan_params=params)
    assert not res.errors, res.errors
    pts = res.results["scan_results"]
    traj = res.results["scan_trajectory"]
    assert [round(p["value"], 6) for p in pts] == [0.6, 0.75, 0.9]
    for p, frame in zip(pts, traj):
        assert p["converged"]
        assert p["energy"] == pytest.approx(_energy(scf.RHF, frame), abs=1e-7)
    assert all(r["Converged"] == "yes" for r in _csv_rows(res))


def test_relaxed_angle_scan_hits_targets_and_reports_true_energies(run_job):
    """The final SCF of each point used `.mol =` instead of reset()."""
    pytest.importorskip("geometric")
    params = {
        "type": "Angle",
        "atoms": [1, 0, 2],
        "start": 100.0,
        "end": 110.0,
        "steps": 2,
    }
    res = run_job(XYZ_H2O, job_type="Relaxed Surface Scan", scan_params=params)
    assert not res.errors, res.errors
    pts = res.results["scan_results"]
    assert len(pts) == 2
    for p, target, frame in zip(pts, (100.0, 110.0), res.results["scan_trajectory"]):
        assert p["value"] == pytest.approx(target, abs=0.05)
        assert p["energy"] == pytest.approx(_energy(scf.RHF, frame), abs=1e-6)
    rows = _csv_rows(res)
    assert rows[0].keys() >= {"Step", "Value", "Energy", "Converged"}


def test_open_shell_relaxed_scan_keeps_the_uhf_switch(run_job):
    """Regression: relaxed scans re-read the raw method and ran ROHF."""
    pytest.importorskip("geometric")
    params = {"type": "Dist", "atoms": [0, 1], "start": 0.95, "end": 1.05, "steps": 2}
    res = run_job(XYZ_OH, job_type="Relaxed Surface Scan", scan_params=params, spin=2)
    assert not res.errors, res.errors
    for p, frame in zip(res.results["scan_results"], res.results["scan_trajectory"]):
        e_uhf = _energy(scf.UHF, frame, spin=1)
        e_rohf = _energy(scf.ROHF, frame, spin=1)
        assert abs(e_uhf - e_rohf) > 1e-5
        assert p["energy"] == pytest.approx(e_uhf, abs=1e-6)


def test_reloaded_scan_keeps_convergence_flags(run_job):
    worker, _ = load_plugin_modules()
    params = {"type": "Dist", "atoms": [0, 1], "start": 0.7, "end": 0.8, "steps": 2}
    res = run_job(XYZ_H2, job_type="Rigid Surface Scan", scan_params=params)
    rows = worker.LoadWorker._load_scan_csv(
        os.path.join(res.results["out_dir"], "scan_results.csv")
    )
    assert [r["step"] for r in rows] == [1, 2]
    assert all(r["converged"] is True for r in rows)
    assert np.allclose(
        [r["energy"] for r in rows], [p["energy"] for p in res.results["scan_results"]]
    )
