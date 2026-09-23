"""The four tutorials (tutorial/*.html), run through the plugin with the
tutorial's exact settings, checking the result each tutorial tells the
reader to expect. The molecule *is* the claim here, so these are not
minimal systems; the SN2 tutorial is the slow one (marked `slow`)."""

import itertools
import os

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("rdkit")
pytest.importorskip("PyQt6.QtCore")
pytest.importorskip("geometric")

from conftest import read_cube
from pyscf import lib, scf

HARTREE_TO_KJMOL = 2625.4996

# s-trans 1,3-butadiene, planar (xy plane)
XYZ_BUTADIENE = """10
s-trans-1,3-butadiene
C -1.847 -0.387 0.000
C -0.616 0.160 0.000
C 0.616 -0.160 0.000
C 1.847 0.387 0.000
H -2.736 0.232 0.000
H -1.971 -1.465 0.000
H -0.524 1.244 0.000
H 0.524 -1.244 0.000
H 1.971 1.465 0.000
H 2.736 -0.232 0.000"""

XYZ_ACETONE = """10
acetone
C 0.000 0.000 0.183
O 0.000 0.000 1.400
C 0.000 1.286 -0.608
C 0.000 -1.286 -0.608
H 0.000 2.143 0.068
H 0.883 1.330 -1.254
H -0.883 1.330 -1.254
H 0.000 -2.143 0.068
H 0.883 -1.330 -1.254
H -0.883 -1.330 -1.254"""

# staggered ethane; H3-C1-C2-H6 is the scanned dihedral
XYZ_ETHANE = """8
ethane
C 0.000 0.000 0.765
C 0.000 0.000 -0.765
H 1.018 0.000 1.160
H -0.509 0.882 1.160
H -0.509 -0.882 1.160
H -1.018 0.000 -1.160
H 0.509 -0.882 -1.160
H 0.509 0.882 -1.160"""

# CH3Cl + Br- with Br on the backside at C-Br = 2.8 A (tutorial step 1)
XYZ_SN2 = """6
CH3Cl + Br- backside
C 0.000 0.000 0.000
Cl 0.000 0.000 -1.800
H 1.028 0.000 0.357
H -0.514 0.890 0.357
H -0.514 -0.890 0.357
Br 0.000 0.000 2.800"""


def _ok(res):
    assert not res.errors, res.errors
    assert res.finished
    return res.results


# ---------------------------------------------------------------------------
# Tutorial 1: butadiene pi orbitals, nodes increase with energy
# ---------------------------------------------------------------------------


def _pi_sign_changes(mol, coeff):
    """Sign changes of the out-of-plane (z) p coefficient along C1..C4,
    or None when the orbital is not a pi orbital."""
    labels = mol.ao_labels(fmt=False)
    pz = {}
    total = float(np.sum(coeff**2))
    for i, (atom, sym, shell, comp) in enumerate(labels):
        if sym == "C" and shell.endswith("p") and comp == "z":
            pz[atom] = coeff[i]
    if sum(v**2 for v in pz.values()) < 0.5 * total:
        return None  # sigma orbital
    vals = [pz[a] for a in sorted(pz)]
    signs = [np.sign(v) for v in vals if abs(v) > 1e-3]
    return sum(1 for a, b in itertools.pairwise(signs) if a != b)


def test_tutorial_1_butadiene_nodes(run_job, run_properties):
    r = _ok(
        run_job(
            XYZ_BUTADIENE,
            job_type="Geometry Optimization",
            method="RKS",
            functional="b3lyp",
            basis="sto-3g",
            threads=8,
        )
    )
    chk = r["chkfile"]
    mol = lib.chkfile.load_mol(chk)
    coeff = scf.chkfile.load(chk, "scf/mo_coeff")
    occ = scf.chkfile.load(chk, "scf/mo_occ")
    homo = int(np.max(np.nonzero(occ > 0)))
    assert homo + 1 == 15  # "HOMO (Orb 15)"

    # HOMO-1, HOMO, LUMO, LUMO+1 = Orb 14..17 with 0, 1, 2, 3 nodes
    nodes = [_pi_sign_changes(mol, coeff[:, i]) for i in range(13, 17)]
    assert nodes == [0, 1, 2, 3]

    # the tutorial's step 2: generate those four cubes via the plugin
    props = run_properties(chk, ["HOMO-1", "HOMO", "LUMO", "LUMO+1"], r["out_dir"])
    names = sorted(os.path.basename(f) for f in props.results["files"])
    assert names == [
        "014_HOMO-1.cube",
        "015_HOMO.cube",
        "016_LUMO.cube",
        "017_LUMO+1.cube",
    ]


# ---------------------------------------------------------------------------
# Tutorial 2: acetone ESP, negative at O, positive at the hydrogens
# ---------------------------------------------------------------------------


def _cube_value_at(path, point_bohr):
    values, _ = read_cube(path)
    with open(path) as fh:
        lines = fh.readlines()
    origin = np.array([float(x) for x in lines[2].split()[1:4]])
    axes = np.array([[float(x) for x in lines[3 + i].split()[1:4]] for i in range(3)])
    idx = np.rint(np.linalg.solve(axes.T, point_bohr - origin)).astype(int)
    idx = np.clip(idx, 0, np.array(values.shape) - 1)
    return float(values[tuple(idx)])


def test_tutorial_2_acetone_esp(run_job, run_properties):
    r = _ok(
        run_job(
            XYZ_ACETONE,
            job_type="Geometry Optimization",
            method="RKS",
            functional="b3lyp",
            basis="sto-3g",
            threads=8,
        )
    )
    props = run_properties(r["chkfile"], ["ESP"], r["out_dir"])
    esp, _dens = props.results["files"]

    mol = lib.chkfile.load_mol(r["chkfile"])
    xyz = mol.atom_coords()  # Bohr
    sym = [mol.atom_symbol(i) for i in range(mol.natm)]
    c_o = xyz[sym.index("O")] - xyz[0]
    o_out = xyz[sym.index("O")] + 2.5 * c_o / np.linalg.norm(
        c_o
    )  # beyond the lone pairs
    assert _cube_value_at(esp, o_out) < 0  # red: negative at the oxygen

    h_vals = []
    for i, s in enumerate(sym):
        if s == "H":
            c = xyz[
                int(
                    np.argmin(
                        [
                            np.linalg.norm(xyz[i] - xyz[j]) if sym[j] == "C" else 1e9
                            for j in range(mol.natm)
                        ]
                    )
                )
            ]
            out = xyz[i] + 2.0 * (xyz[i] - c) / np.linalg.norm(xyz[i] - c)
            h_vals.append(_cube_value_at(esp, out))
    assert all(v > 0 for v in h_vals)  # blue: positive at the hydrogens


# ---------------------------------------------------------------------------
# Tutorial 3: ethane rotation, maxima 0/120, minima 60/180, ~12 kJ/mol
# ---------------------------------------------------------------------------


def test_tutorial_3_ethane_rotation(run_job):
    params = {
        "type": "Dihedral",
        "atoms": [2, 0, 1, 5],
        "start": 0.0,
        "end": 180.0,
        "steps": 10,
    }
    r = _ok(
        run_job(
            XYZ_ETHANE,
            job_type="Relaxed Surface Scan",
            method="RKS",
            functional="b3lyp",
            basis="sto-3g",
            scan_params=params,
            threads=8,
        )
    )
    pts = r["scan_results"]
    assert len(pts) == 10
    ang = np.array([p["value"] for p in pts])
    e = np.array([p["energy"] for p in pts])
    rel = (e - e.min()) * HARTREE_TO_KJMOL
    # the profile runs 0 -> 180 on one branch (180 must not come back as -180)
    assert np.allclose(ang, np.linspace(0.0, 180.0, 10), atol=0.1)

    def at(deg):
        return rel[int(np.argmin(np.abs(ang - deg)))]

    # eclipsed maxima at 0 and 120, staggered minima at 60 and 180
    assert at(0) > at(60) and at(120) > at(60) and at(120) > at(180)
    assert at(60) < 0.5 and at(180) < 0.5  # both staggered minima
    assert at(0) == pytest.approx(at(120), abs=0.3)  # equivalent eclipsed maxima
    assert rel.max() == pytest.approx(12.0, abs=2.5)  # "approximately 12 kJ/mol"


# ---------------------------------------------------------------------------
# Tutorial 4: SN2 CH3Cl + Br-, scan then TS with one imaginary mode
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_tutorial_4_sn2_scan_and_ts(run_job, tmp_path):
    params = {"type": "Dist", "atoms": [0, 5], "start": 2.8, "end": 1.96, "steps": 5}
    scan = _ok(
        run_job(
            XYZ_SN2,
            job_type="Relaxed Surface Scan",
            method="RKS",
            functional="b3lyp",
            basis="ma-def2-svp",
            charge=-1,
            scan_params=params,
            threads=16,
        )
    )
    pts = scan["scan_results"]
    assert len(pts) == 5
    e = np.array([p["energy"] for p in pts])
    top = int(np.argmax(e))
    assert 0 < top < 4  # "mountain-shaped": the maximum is inside the scan

    # step 4: TS Optimization + Frequency from the highest scan point
    frame = scan["scan_trajectory"][top]
    ts = _ok(
        run_job(
            frame,
            job_type="TS Optimization + Frequency",
            method="RKS",
            functional="b3lyp",
            basis="ma-def2-svp",
            charge=-1,
            threads=16,
        )
    )
    fd = ts["freq_data"]
    assert fd["n_imaginary"] == 1  # "exactly one imaginary frequency"

    # the imaginary mode moves C along the Br-C-Cl axis (reaction coordinate)
    i_imag = int(np.argmin(fd["freqs"]))
    mode = np.array(fd["modes"][i_imag])  # (natm, 3)
    mol = lib.chkfile.load_mol(ts["chkfile"])
    xyz = mol.atom_coords()
    axis = xyz[5] - xyz[1]
    axis /= np.linalg.norm(axis)
    c_disp = mode[0] / np.linalg.norm(mode[0])
    assert abs(float(c_disp @ axis)) > 0.9

    # symmetric TS: C-Br and C-Cl both stretched relative to the reactant
    d_cbr = np.linalg.norm(xyz[0] - xyz[5]) * lib.param.BOHR
    d_ccl = np.linalg.norm(xyz[0] - xyz[1]) * lib.param.BOHR
    assert 2.2 < d_cbr < 2.8 and 1.9 < d_ccl < 2.6
