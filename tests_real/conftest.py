"""
Shared harness for the real-PySCF tests.

These tests run the plugin's workers against a real PySCF (and, where
needed, geomeTRIC / pyberny / RDKit / PyQt6). They live outside tests/
because that suite replaces sys.modules["pyscf"] with a MagicMock; the
two must never share a process.

Every test module skips itself when a dependency is missing, so this
directory is safe to run anywhere:

    python -m pytest tests_real -v

PySCF has no native Windows build -- run these on Linux or macOS.
"""

import os
import sys
import types
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
PKG_DIR = REPO / "pyscf_calculator"

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OMP_NUM_THREADS", "4")


def load_plugin_modules():
    """Import pyscf_calculator.worker / .utils without running the package
    __init__ (which pulls in the GUI, pyvista and matplotlib)."""
    if "pyscf_calculator" not in sys.modules:
        pkg = types.ModuleType("pyscf_calculator")
        pkg.__path__ = [str(PKG_DIR)]
        sys.modules["pyscf_calculator"] = pkg
    import pyscf_calculator.worker as worker  # noqa: PLC0415
    import pyscf_calculator.utils as utils  # noqa: PLC0415

    return worker, utils


@pytest.fixture(scope="session")
def qcore():
    from PyQt6.QtCore import QCoreApplication

    return QCoreApplication.instance() or QCoreApplication([])


class JobResult:
    def __init__(self):
        self.logs = []
        self.errors = []
        self.results = None
        self.finished = False

    @property
    def log(self):
        return "".join(self.logs)


def _collect(worker_obj, res, with_result=True):
    worker_obj.log_signal.connect(res.logs.append)
    worker_obj.error_signal.connect(res.errors.append)
    worker_obj.finished_signal.connect(lambda *a: setattr(res, "finished", True))
    if with_result:
        worker_obj.result_signal.connect(lambda d: setattr(res, "results", d))


XYZ_H2O = """3
water
O 0.000000 0.000000 0.117790
H 0.000000 0.755453 -0.471161
H 0.000000 -0.755453 -0.471161"""

XYZ_H2 = """2
hydrogen
H 0.000000 0.000000 0.000000
H 0.000000 0.000000 0.740000"""

XYZ_OH = """2
hydroxyl
O 0.000000 0.000000 0.000000
H 0.000000 0.000000 0.970000"""


# Planar NH3: the inversion transition state (exactly one imaginary mode).
XYZ_NH3_PLANAR = """4
ammonia, planar
N 0.000000 0.000000 0.000000
H 1.000000 0.000000 0.000000
H -0.500000 0.866025 0.000000
H -0.500000 -0.866025 0.000000"""

# Smallest molecule with a dihedral to scan.
XYZ_H2O2 = """4
hydrogen peroxide
O 0.000000 0.737500 -0.052800
O 0.000000 -0.737500 -0.052800
H 0.819000 0.817000 0.422000
H -0.819000 -0.817000 0.422000"""


def atom_xyz(symbol):
    return f"1\n{symbol} atom\n{symbol} 0.000000 0.000000 0.000000"


def base_config(tmp_path, **overrides):
    cfg = {
        "job_type": "Energy",
        "method": "RHF",
        "functional": "b3lyp",
        "basis": "sto-3g",
        "charge": 0,
        "spin": 1,
        "nstates": 3,
        "threads": 4,
        "memory": 2000,
        "symmetry": False,
        "break_symmetry": False,
        "max_cycle": 200,
        "conv_tol": "1e-10",
        "grid_level": 3,
        "solvent": "None (Vacuum)",
        "hessian": "Analytic",
        "out_dir": str(tmp_path),
        "plugin_version": "test",
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def run_job(qcore, tmp_path):
    """Run PySCFWorker synchronously; returns a JobResult."""
    worker, _ = load_plugin_modules()

    def _run(xyz, **overrides):
        res = JobResult()
        w = worker.PySCFWorker(xyz, base_config(tmp_path, **overrides))
        _collect(w, res)
        w.run()  # synchronous: signals are delivered directly
        return res

    return _run


@pytest.fixture
def load_result(qcore):
    worker, _ = load_plugin_modules()

    def _load(chkfile):
        res = JobResult()
        lw = worker.LoadWorker(chkfile)
        lw.error_signal.connect(res.errors.append)
        lw.finished_signal.connect(lambda d: setattr(res, "results", d))
        lw.run()
        return res

    return _load


@pytest.fixture
def run_properties(qcore):
    worker, _ = load_plugin_modules()

    def _props(chkfile, tasks, out_dir):
        res = JobResult()
        pw = worker.PropertyWorker(chkfile, tasks, out_dir)
        _collect(pw, res)
        pw.run()
        return res

    return _props


def xyz_atoms(xyz):
    """Header-stripped 'Sym x y z' lines, as PySCF's atom= takes them."""
    lines = xyz.strip().splitlines()
    if lines[0].strip().isdigit():
        lines = lines[2:]
    return "\n".join(lines)


def read_cube(path):
    """(values[nx,ny,nz], voxel_volume_bohr3) of a Gaussian cube file."""
    import numpy as np

    with open(path) as fh:
        lines = fh.readlines()
    natm = abs(int(lines[2].split()[0]))
    axes = [lines[3 + i].split() for i in range(3)]
    dims = [int(a[0]) for a in axes]
    vecs = np.array([[float(x) for x in a[1:4]] for a in axes])
    vol = abs(np.linalg.det(vecs))
    data = np.array(" ".join(lines[6 + natm :]).split(), dtype=float)
    return data.reshape(dims), vol
