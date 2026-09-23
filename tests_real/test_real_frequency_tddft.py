"""Frequencies (analytic, numerical, solvated) and TDDFT against real PySCF."""

import json
import os

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("rdkit")
pytest.importorskip("PyQt6.QtCore")

from pyscf import gto, scf, tdscf  # noqa: E402
from pyscf.data import nist  # noqa: E402
from pyscf.hessian import thermo  # noqa: E402

from conftest import XYZ_H2O, xyz_atoms  # noqa: E402


def _freqs(res):
    assert not res.errors, res.errors
    assert "freq_data" in res.results, res.log[-2000:]
    return np.array(res.results["freq_data"]["freqs"])


def test_analytic_frequencies_match_direct_pyscf(run_job):
    res = run_job(XYZ_H2O, job_type="Frequency")
    got = _freqs(res)

    mf = scf.RHF(gto.M(atom=xyz_atoms(XYZ_H2O), basis="sto-3g", verbose=0))
    mf.conv_tol = 1e-10
    mf.run()
    ref = thermo.harmonic_analysis(mf.mol, mf.Hessian().kernel())["freq_wavenumber"]
    assert got.shape == (3,)  # 3N - 6 for water
    assert np.allclose(got, np.real(ref), atol=0.5)

    with open(os.path.join(res.results["out_dir"], "freq_analysis.json")) as fh:
        saved = json.load(fh)
    assert np.allclose(saved["freq_data"]["freqs"], got)
    assert "G_tot" in saved["thermo_data"]


def test_numerical_hessian_agrees_with_analytic(run_job):
    analytic = _freqs(run_job(XYZ_H2O, job_type="Frequency"))
    res = run_job(
        XYZ_H2O, job_type="Frequency", hessian="Numerical (finite difference)"
    )
    assert "gradient evaluations" in res.log
    numerical = _freqs(res)
    assert np.allclose(numerical, analytic, atol=3.0)  # cm^-1


def test_solvated_frequencies_via_numerical_hessian(run_job):
    """Solvated frequency jobs used to be skipped with no way out. The
    numerical Hessian (finite differences of ddCOSMO gradients) works."""
    dry = _freqs(run_job(XYZ_H2O, job_type="Frequency"))
    wet = _freqs(
        run_job(
            XYZ_H2O,
            job_type="Frequency",
            solvent="Water",
            hessian="Numerical (finite difference)",
        )
    )
    assert wet.shape == (3,) and np.all(wet > 0)
    assert not np.allclose(wet, dry, atol=0.5)  # the solvent does something


def test_solvated_analytic_hessian_skips_cleanly_or_is_right(run_job):
    """PySCF 2.14's ddCOSMO analytic Hessian fails inside kernel(); the job
    must then skip with a pointer to the numerical option -- never report
    vacuum frequencies. If a later PySCF makes it work, it must agree with
    the numerical one."""
    res = run_job(XYZ_H2O, job_type="Frequency", solvent="Water")
    assert not res.errors, res.errors
    if "freq_data" not in res.results:
        assert "Hessian: Numerical" in res.log
        assert "Frequency analysis failed" not in res.log
        return
    fd = _freqs(
        run_job(
            XYZ_H2O,
            job_type="Frequency",
            solvent="Water",
            hessian="Numerical (finite difference)",
        )
    )
    assert np.allclose(_freqs(res), fd, atol=3.0)


def test_tddft_matches_direct_pyscf(run_job):
    res = run_job(XYZ_H2O, job_type="TDDFT", method="RKS", functional="pbe", nstates=3)
    assert not res.errors, res.errors
    rows = res.results["tddft_data"]
    assert len(rows) == 3

    mol = gto.M(atom=xyz_atoms(XYZ_H2O), basis="sto-3g", verbose=0)
    mf = mol.RKS(xc="pbe")
    mf.conv_tol = 1e-10
    mf.run()
    td = tdscf.TDDFT(mf)
    td.nstates = 3
    td.kernel()
    ev = [r["excitation_energy_ev"] for r in rows]
    assert np.allclose(ev, td.e * nist.HARTREE2EV, atol=1e-4)
    osc = [r["oscillator_strength"] for r in rows]
    assert np.allclose(osc, td.oscillator_strength(), atol=1e-4)


def test_solvated_tddft_runs(run_job):
    res = run_job(
        XYZ_H2O,
        job_type="TDDFT",
        method="RKS",
        functional="pbe",
        nstates=2,
        solvent="Water",
    )
    assert not res.errors, res.errors
    assert "TDDFT calculation failed" not in res.log
    assert len(res.results["tddft_data"]) == 2
