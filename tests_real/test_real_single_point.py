"""Single points against real PySCF: energies, solvent, symmetry, the
broken-symmetry guess, and the pyscf_input.py reproduction script."""

import os
import runpy

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("rdkit")
pytest.importorskip("PyQt6.QtCore")

from pyscf import dft, gto, scf  # noqa: E402

from conftest import XYZ_H2, XYZ_H2O, xyz_atoms  # noqa: E402


def _mol(xyz, **kw):
    kw.setdefault("basis", "sto-3g")
    return gto.M(atom=xyz_atoms(xyz), verbose=0, **kw)


def _chk_energy(chkfile):
    return float(scf.chkfile.load(chkfile, "scf/e_tot"))


def test_rhf_energy_matches_direct_pyscf(run_job, load_result):
    res = run_job(XYZ_H2O)
    assert not res.errors, res.errors
    assert res.finished
    ref = scf.RHF(_mol(XYZ_H2O)).run(conv_tol=1e-10).e_tot
    assert _chk_energy(res.results["chkfile"]) == pytest.approx(ref, abs=1e-8)

    loaded = load_result(res.results["chkfile"])
    assert loaded.results["scf_type"] == "RHF"


def test_solvent_is_applied_to_an_energy_job(run_job):
    """Regression: Energy jobs used to run in vacuum while logging ddCOSMO."""
    res = run_job(XYZ_H2O, solvent="Water")
    assert not res.errors, res.errors
    e_job = _chk_energy(res.results["chkfile"])

    ref = scf.RHF(_mol(XYZ_H2O)).ddCOSMO()
    ref.with_solvent.eps = 78.2
    ref.conv_tol = 1e-10
    ref.run()
    vac = scf.RHF(_mol(XYZ_H2O)).run(conv_tol=1e-10).e_tot

    assert e_job == pytest.approx(ref.e_tot, abs=1e-7)
    assert abs(e_job - vac) > 1e-4  # really solvated


def test_symmetry_keeps_energy_and_frame(run_job, load_result):
    plain = run_job(XYZ_H2O)
    symm = run_job(XYZ_H2O, symmetry=True)
    assert not symm.errors, symm.errors
    assert "Point-group symmetry: C2v" in symm.log
    assert _chk_energy(symm.results["chkfile"]) == pytest.approx(
        _chk_energy(plain.results["chkfile"]), abs=1e-8
    )
    # The input frame is kept, so cubes/modes line up with the editor.
    a = load_result(plain.results["chkfile"]).results["loaded_xyz"]
    b = load_result(symm.results["chkfile"]).results["loaded_xyz"]
    assert a.splitlines()[2:] == b.splitlines()[2:]


def test_broken_symmetry_guess_finds_the_lower_uhf_state(run_job):
    """Stretched H2: a spin-restricted guess relaxes back onto RHF."""
    stretched = XYZ_H2.replace("0.740000", "2.500000")
    plain = run_job(stretched, method="UHF", break_symmetry=False)
    broken = run_job(stretched, method="UHF", break_symmetry=True)
    assert "symmetry-broken initial guess" in broken.log
    e_plain = _chk_energy(plain.results["chkfile"])
    e_bs = _chk_energy(broken.results["chkfile"])
    assert e_bs < e_plain - 1e-3

    m = _mol(stretched)
    mf = scf.UHF(m)
    mf.__dict__.update(scf.chkfile.load(broken.results["chkfile"], "scf"))
    s2 = mf.spin_square()[0]
    assert s2 > 0.5  # a genuinely spin-polarised (broken-symmetry) state


def test_input_script_reproduces_the_job(run_job):
    """pyscf_input.py used to embed the XYZ header, which gto.M rejects."""
    res = run_job(XYZ_H2O, method="RKS", functional="pbe", grid_level=4)
    assert not res.errors, res.errors
    script = os.path.join(res.results["out_dir"], "pyscf_input.py")
    glb = runpy.run_path(script, run_name="__main__")
    assert glb["mf"].grids.level == 4
    assert glb["mf"].e_tot == pytest.approx(
        _chk_energy(res.results["chkfile"]), abs=1e-7
    )


def test_open_shell_rks_is_switched_to_uks(run_job, load_result):
    from conftest import XYZ_OH

    res = run_job(XYZ_OH, method="RKS", functional="pbe", spin=2)
    assert "Switching to UKS" in res.log
    assert load_result(res.results["chkfile"]).results["scf_type"] == "UHF"
    ref = dft.UKS(_mol(XYZ_OH, spin=1), xc="pbe").run(conv_tol=1e-10).e_tot
    # OH's hole sits in one of two degenerate pi orbitals and the DFT grid
    # is not rotationally invariant: which one the SCF picks (it varies with
    # the OpenMP thread count) moves the energy by ~2e-6 Ha.
    assert _chk_energy(res.results["chkfile"]) == pytest.approx(ref, abs=5e-6)
    assert np.isfinite(ref)
