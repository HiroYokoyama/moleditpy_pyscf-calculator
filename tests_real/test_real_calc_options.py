"""Every calculation option on the Calculation tab, run through the plugin
on real PySCF and checked against an independent PySCF reference with the
same setting -- the smallest molecule each option allows."""

import ast
import os
import pathlib
import re

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("rdkit")
pytest.importorskip("PyQt6.QtCore")

from pyscf import dft, gto, lib, scf  # noqa: E402
from pyscf.hessian import thermo  # noqa: E402

from conftest import (  # noqa: E402
    XYZ_H2,
    XYZ_H2O,
    XYZ_H2O2,
    XYZ_NH3_PLANAR,
    XYZ_OH,
    atom_xyz,
    xyz_atoms,
)

CALC_TAB = pathlib.Path(__file__).resolve().parent.parent / "pyscf_calculator" / "calc_tab.py"


def _combo_items(attr):
    tree = ast.parse(CALC_TAB.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "addItems"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == attr
        ):
            return ast.literal_eval(node.args[0])
    raise AssertionError(attr)


def _chk(res):
    assert not res.errors, res.errors
    assert res.finished
    return res.results["chkfile"]


def _chk_energy(res):
    return float(scf.chkfile.load(_chk(res), "scf/e_tot"))


def _chk_mol(res):
    return lib.chkfile.load_mol(_chk(res))


def _ref(xyz, method="RHF", spin=0, charge=0, basis="sto-3g", xc="pbe", **opts):
    """Direct PySCF reference energy, built without the plugin."""
    mol = gto.M(
        atom=xyz_atoms(xyz), basis=basis, spin=spin, charge=charge, verbose=0
    )
    mf = getattr(dft if "KS" in method else scf, method)(mol)
    if "KS" in method:
        mf.xc = xc
        if "grid" in opts:
            mf.grids.level = opts["grid"]
            if opts["grid"] >= 4:
                mf.grids.prune = False
    if opts.get("disp"):
        mf.disp = opts["disp"]
    if opts.get("eps"):
        mf = mf.ddCOSMO()
        mf.with_solvent.eps = opts["eps"]
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    return mf.run()


# ---------------------------------------------------------------------------
# Method x spin state
# ---------------------------------------------------------------------------

# (UI method, molecule, multiplicity, method PySCF must actually run)
METHOD_CASES = [
    ("RHF", XYZ_H2O, 1, "RHF"),
    ("UHF", XYZ_H2O, 1, "UHF"),
    ("ROHF", XYZ_H2O, 1, "ROHF"),
    ("RKS", XYZ_H2O, 1, "RKS"),
    ("UKS", XYZ_H2O, 1, "UKS"),
    ("ROKS", XYZ_H2O, 1, "ROKS"),
    ("RHF", XYZ_OH, 2, "UHF"),  # open shell: switched to UHF
    ("UHF", XYZ_OH, 2, "UHF"),
    ("ROHF", XYZ_OH, 2, "ROHF"),
    ("RKS", XYZ_OH, 2, "UKS"),  # open shell: switched to UKS
    ("UKS", XYZ_OH, 2, "UKS"),
    ("ROKS", XYZ_OH, 2, "ROKS"),
]


@pytest.mark.parametrize("ui_method,xyz,mult,runs_as", METHOD_CASES)
def test_every_method(ui_method, xyz, mult, runs_as, run_job):
    res = run_job(xyz, method=ui_method, functional="pbe", spin=mult)
    ref = _ref(xyz, method=runs_as, spin=mult - 1)
    # Open-shell KS on OH: the hole sits in one of two degenerate pi
    # orbitals and the DFT grid is not rotationally invariant, so even
    # direct PySCF spreads by ~2e-6 Ha across initial guesses.
    tol = 5e-6 if (mult > 1 and "KS" in runs_as) else 1e-6
    assert _chk_energy(res) == pytest.approx(ref.e_tot, abs=tol)


# Multiplicity 1..6 on the smallest system that can hold it.
SPIN_CASES = [
    (atom_xyz("He"), 1, "sto-3g"),
    (atom_xyz("H"), 2, "sto-3g"),
    (atom_xyz("O"), 3, "sto-3g"),
    (atom_xyz("N"), 4, "sto-3g"),
    (atom_xyz("C"), 5, "sto-3g"),
    (atom_xyz("N"), 6, "6-31g"),
]


@pytest.mark.parametrize("xyz,mult,basis", SPIN_CASES)
def test_every_multiplicity(xyz, mult, basis, run_job):
    res = run_job(xyz, method="UHF", spin=mult, basis=basis)
    assert _chk_mol(res).spin == mult - 1
    ref = _ref(xyz, method="UHF", spin=mult - 1, basis=basis)
    assert _chk_energy(res) == pytest.approx(ref.e_tot, abs=1e-6)


@pytest.mark.parametrize(
    "xyz,charge,mult",
    [(XYZ_OH, -1, 1), (XYZ_H2, 1, 2), (XYZ_H2O, 2, 1)],
    ids=["OH-", "H2+", "H2O2+"],
)
def test_charge(xyz, charge, mult, run_job):
    res = run_job(xyz, method="UHF", charge=charge, spin=mult)
    assert _chk_mol(res).charge == charge
    ref = _ref(xyz, method="UHF", charge=charge, spin=mult - 1)
    assert _chk_energy(res) == pytest.approx(ref.e_tot, abs=1e-6)


def test_impossible_charge_multiplicity_is_a_clear_error(run_job):
    res = run_job(XYZ_H2, spin=2)  # 2 electrons cannot be a doublet
    assert res.errors and "Molecule Build Failed" in res.errors[0]


# ---------------------------------------------------------------------------
# Solvents
# ---------------------------------------------------------------------------

# Literature static dielectric constants at 25 C (CRC Handbook).
LITERATURE_EPS = {
    "Water": 78.36,
    "Ethanol": 24.85,
    "Methanol": 32.61,
    "Acetone": 20.49,
    "THF": 7.43,
    "Chloroform": 4.71,
    "Dichloromethane": 8.93,
    "Toluene": 2.37,
    "Benzene": 2.27,
}
SOLVENTS = [s for s in _combo_items("solvent_combo") if s != "None (Vacuum)"]


def test_solvent_list_is_covered():
    assert set(SOLVENTS) == set(LITERATURE_EPS)


@pytest.mark.parametrize("solvent", SOLVENTS)
def test_every_solvent(solvent, run_job):
    res = run_job(XYZ_H2O, solvent=solvent)
    eps = float(re.search(r"eps=([0-9.]+)", res.log).group(1))
    assert eps == pytest.approx(LITERATURE_EPS[solvent], rel=0.05)
    ref = _ref(XYZ_H2O, eps=eps)
    assert _chk_energy(res) == pytest.approx(ref.e_tot, abs=1e-7)


def test_solvation_grows_with_polarity(run_job):
    """Physics check independent of the eps table: a polar molecule is
    stabilised more the more polar the solvent."""
    order = sorted(LITERATURE_EPS, key=LITERATURE_EPS.get)
    energies = [_chk_energy(run_job(XYZ_H2O, solvent=s)) for s in order]
    vac = _chk_energy(run_job(XYZ_H2O))
    assert all(e < vac for e in energies)
    assert all(b <= a + 1e-7 for a, b in zip(energies, energies[1:]))


# ---------------------------------------------------------------------------
# Numerical settings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", range(10))
def test_every_grid_level(level, run_job):
    res = run_job(XYZ_H2O, method="RKS", functional="pbe", grid_level=level)
    ref = _ref(XYZ_H2O, method="RKS", grid=level)
    assert _chk_energy(res) == pytest.approx(ref.e_tot, abs=1e-8)


def test_max_cycle_and_conv_tol_reach_pyscf(run_job):
    res = run_job(XYZ_H2O, max_cycle=37, conv_tol="1e-7", memory=1234, threads=2)
    with open(os.path.join(res.results["out_dir"], "pyscf.out")) as fh:
        out = fh.read()
    assert "SCF max_cycles = 37" in out
    assert "SCF conv_tol = 1e-07" in out
    assert "max_memory 1234 MB" in out
    assert "PySCF running with 2 OpenMP threads" in res.log


def test_too_few_cycles_is_reported(run_job):
    res = run_job(XYZ_H2O, max_cycle=1)
    assert "SCF did not converge within 1 cycles" in res.log


def test_symmetry_option(run_job):
    on = run_job(XYZ_H2O, symmetry=True)
    off = run_job(XYZ_H2O, symmetry=False)
    assert "Point-group symmetry: C2v" in on.log
    assert "Point-group symmetry" not in off.log
    assert _chk_energy(on) == pytest.approx(_chk_energy(off), abs=1e-8)


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("choice,key", [("D3(BJ)", "d3bj"), ("D3(zero)", "d3zero"), ("D4", "d4")])
def test_every_dispersion_choice(choice, key, run_job):
    pytest.importorskip("pyscf.dispersion")
    res = run_job(XYZ_H2O, method="RKS", functional="pbe", dispersion=choice)
    ref = _ref(XYZ_H2O, method="RKS", disp=key)
    plain = _ref(XYZ_H2O, method="RKS")
    assert _chk_energy(res) == pytest.approx(ref.e_tot, abs=1e-8)
    assert abs(ref.e_tot - plain.e_tot) > 1e-7  # the correction is really on


def test_dispersion_reaches_the_optimizer(run_job):
    pytest.importorskip("pyscf.dispersion")
    pytest.importorskip("geometric")
    res = run_job(
        XYZ_H2,
        job_type="Geometry Optimization",
        method="RKS",
        functional="pbe",
        dispersion="D3(BJ)",
    )
    mol = _chk_mol(res)
    mol.verbose = 0
    mf = dft.RKS(mol, xc="pbe")
    mf.disp = "d3bj"
    mf.conv_tol = 1e-10
    assert _chk_energy(res) == pytest.approx(mf.run().e_tot, abs=1e-7)


def test_vv10_functional_with_dispersion_is_refused(run_job):
    res = run_job(XYZ_H2, method="RKS", functional="wb97x-v", dispersion="D3(BJ)")
    assert res.errors and "VV10" in res.errors[0]


# ---------------------------------------------------------------------------
# Job types not covered elsewhere
# ---------------------------------------------------------------------------


def test_optimization_plus_frequency(run_job):
    pytest.importorskip("geometric")
    res = run_job(XYZ_H2, job_type="Optimization + Frequency")
    freqs = res.results["freq_data"]["freqs"]
    assert len(freqs) == 1 and freqs[0] > 0  # H2: one stretch
    assert res.results["freq_data"]["n_imaginary"] == 0
    assert "consistent with a minimum" in res.log

    mol = _chk_mol(res)
    mol.verbose = 0
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.run()
    ref = thermo.harmonic_analysis(mol, mf.Hessian().kernel())["freq_wavenumber"]
    assert freqs[0] == pytest.approx(float(np.real(ref[0])), abs=1.0)


def _planar(res, tol=1e-3):
    rows = [ln.split() for ln in res.results["optimized_xyz"].splitlines()[2:]]
    xyz = np.array([[float(x) for x in r[1:4]] for r in rows])
    centred = xyz - xyz.mean(axis=0)
    return np.linalg.svd(centred, compute_uv=False)[-1] < tol


def test_transition_state_optimization(run_job):
    pytest.importorskip("geometric")
    res = run_job(XYZ_NH3_PLANAR, job_type="Transition State Optimization")
    assert "TS Optimization" in res.results["optimized_xyz"]
    assert _planar(res)  # the NH3 inversion TS is planar


def test_ts_optimization_plus_frequency_has_one_imaginary_mode(run_job):
    pytest.importorskip("geometric")
    res = run_job(XYZ_NH3_PLANAR, job_type="TS Optimization + Frequency")
    fd = res.results["freq_data"]
    assert fd["n_imaginary"] == 1
    assert sum(f < 0 for f in fd["freqs"]) == 1
    assert "consistent with a transition state" in res.log


@pytest.mark.parametrize("temperature,pressure_atm", [(298.15, 1.0), (500.0, 2.0)])
def test_thermochemistry_temperature_and_pressure(temperature, pressure_atm, run_job):
    res = run_job(
        XYZ_H2O,
        job_type="Frequency",
        temperature=temperature,
        pressure=pressure_atm * 101325.0,
    )
    th = res.results["thermo_data"]
    assert th["temperature"][0] == pytest.approx(temperature)
    assert th["pressure"][0] == pytest.approx(pressure_atm * 101325.0)

    mf = _ref(XYZ_H2O)
    freq = thermo.harmonic_analysis(mf.mol, mf.Hessian().kernel())
    ref = thermo.thermo(mf, freq["freq_au"], temperature, pressure_atm * 101325.0)
    assert th["G_tot"][0] == pytest.approx(ref["G_tot"][0], abs=1e-6)


@pytest.mark.parametrize("nstates", [1, 4])
def test_tddft_nstates(nstates, run_job):
    res = run_job(XYZ_H2O, job_type="TDDFT", method="RKS", functional="pbe", nstates=nstates)
    assert len(res.results["tddft_data"]) == nstates


def test_tdhf_for_hartree_fock(run_job):
    from pyscf import tdscf
    from pyscf.data import nist

    res = run_job(XYZ_H2O, job_type="TDDFT", method="RHF", nstates=2)
    td = tdscf.TDHF(_ref(XYZ_H2O))
    td.nstates = 2
    td.kernel()
    ev = [r["excitation_energy_ev"] for r in res.results["tddft_data"]]
    assert np.allclose(ev, td.e * nist.HARTREE2EV, atol=1e-4)


# ---------------------------------------------------------------------------
# Scans: every coordinate type, rigid and relaxed
# ---------------------------------------------------------------------------


def _scan_energy_check(res, spin=0):
    assert not res.errors, res.errors
    for p, frame in zip(res.results["scan_results"], res.results["scan_trajectory"]):
        mol = gto.M(atom="\n".join(frame.splitlines()[2:]), basis="sto-3g", spin=spin, verbose=0)
        ref = scf.RHF(mol)
        ref.conv_tol = 1e-10
        assert p["energy"] == pytest.approx(ref.run().e_tot, abs=1e-6)


def test_rigid_angle_scan(run_job):
    params = {"type": "Angle", "atoms": [1, 0, 2], "start": 100.0, "end": 110.0, "steps": 2}
    res = run_job(XYZ_H2O, job_type="Rigid Surface Scan", scan_params=params)
    _scan_energy_check(res)
    assert res.results["scan_type"] == "Angle"


def test_rigid_dihedral_scan(run_job):
    params = {"type": "Dihedral", "atoms": [2, 0, 1, 3], "start": 90.0, "end": 180.0, "steps": 2}
    res = run_job(XYZ_H2O2, job_type="Rigid Surface Scan", scan_params=params)
    _scan_energy_check(res)
    assert [round(p["value"]) for p in res.results["scan_results"]] == [90, 180]


def test_relaxed_distance_scan(run_job):
    pytest.importorskip("geometric")
    params = {"type": "Dist", "atoms": [0, 1], "start": 0.7, "end": 0.8, "steps": 2}
    res = run_job(XYZ_H2, job_type="Relaxed Surface Scan", scan_params=params)
    _scan_energy_check(res)
    vals = [p["value"] for p in res.results["scan_results"]]
    assert vals == pytest.approx([0.7, 0.8], abs=1e-3)


def test_relaxed_dihedral_scan(run_job):
    pytest.importorskip("geometric")
    params = {"type": "Dihedral", "atoms": [2, 0, 1, 3], "start": 100.0, "end": 140.0, "steps": 2}
    res = run_job(XYZ_H2O2, job_type="Relaxed Surface Scan", scan_params=params)
    _scan_energy_check(res)
    vals = [p["value"] for p in res.results["scan_results"]]
    assert vals == pytest.approx([100.0, 140.0], abs=0.1)
    assert os.path.exists(os.path.join(res.results["out_dir"], "scan_info.json"))


# ---------------------------------------------------------------------------
# Post-SCF properties and orbital cubes
# ---------------------------------------------------------------------------


def test_dipole_and_mulliken_match_pyscf(run_job, load_result):
    res = run_job(XYZ_H2O)
    ref = _ref(XYZ_H2O)
    dip = ref.dip_moment(unit="Debye", verbose=0)
    _, chg = ref.mulliken_pop(verbose=0)
    assert np.allclose(res.results["dipole_debye"], dip, atol=1e-5)
    assert np.allclose(res.results["mulliken_charges"], chg, atol=1e-5)
    assert "Dipole moment (Debye)" in res.log
    loaded = load_result(res.results["chkfile"]).results
    assert loaded["dipole_total_debye"] == pytest.approx(np.linalg.norm(dip), abs=1e-5)


# (task string, spin channel, expected 0-based MO index) for RHF H2O
# (10 electrons: HOMO = 4, LUMO = 5) and UHF OH (alpha HOMO 4, beta HOMO 3).
RHF_TASKS = [
    ("HOMO", 4),
    ("HOMO-1", 3),
    ("LUMO", 5),
    ("LUMO+1", 6),
    ("MO 2", 1),
    ("#3", 3),
    ("MO 4_HOMO-1", 3),
]
UHF_TASKS = [("HOMO_A", 0, 4), ("HOMO_B", 1, 3), ("LUMO_B", 1, 4), ("MO 5_A", 0, 4)]


def _same_cube(path, mol, coeff, tmp_path):
    from pyscf.tools import cubegen

    from conftest import read_cube

    ref_path = str(tmp_path / "ref.cube")
    cubegen.orbital(mol, ref_path, coeff)
    got, _ = read_cube(path)
    ref, _ = read_cube(ref_path)
    return np.allclose(got, ref, atol=1e-8)


@pytest.mark.parametrize("task,idx", RHF_TASKS)
def test_orbital_cube_is_the_requested_mo_rhf(task, idx, run_job, run_properties, tmp_path):
    res = run_job(XYZ_H2O)
    chk = res.results["chkfile"]
    props = run_properties(chk, [task], res.results["out_dir"])
    (cube,) = props.results["files"]
    assert os.path.basename(cube).startswith(f"{idx + 1:03d}_")
    mol = lib.chkfile.load_mol(chk)
    coeff = scf.chkfile.load(chk, "scf/mo_coeff")
    assert _same_cube(cube, mol, coeff[:, idx], tmp_path)


@pytest.mark.parametrize("task,spin,idx", UHF_TASKS)
def test_orbital_cube_is_the_requested_mo_uhf(task, spin, idx, run_job, run_properties, tmp_path):
    res = run_job(XYZ_OH, method="UHF", spin=2)
    chk = res.results["chkfile"]
    props = run_properties(chk, [task], res.results["out_dir"])
    (cube,) = props.results["files"]
    assert os.path.basename(cube).startswith(f"{idx + 1:03d}{'ab'[spin]}_")
    mol = lib.chkfile.load_mol(chk)
    coeff = scf.chkfile.load(chk, "scf/mo_coeff")
    assert _same_cube(cube, mol, coeff[spin][:, idx], tmp_path)
