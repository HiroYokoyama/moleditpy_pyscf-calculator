from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from audit_helpers import load_module


def make_mf():
    mf = SimpleNamespace(e_tot=0, converged=True, max_cycle=100, mo_energy=[-.5, .1], mo_occ=[1, 0], mo_coeff=np.eye(2))

    def kernel(dm0=None):
        mf.e_tot = -1.0
        return mf.e_tot

    mf.kernel = MagicMock(side_effect=kernel)
    mf.make_rdm1 = lambda: "converged-density"
    return mf


@pytest.mark.parametrize("job", ["Energy", "TDDFT", "Geometry Optimization"])
def test_each_job_initializes_requested_broken_symmetry(monkeypatch, tmp_path, job):
    mod = load_module(monkeypatch, "worker")
    worker = mod.PySCFWorker("H 0 0 0", {"job_type": job, "method": "UHF", "spin": 1, "break_symmetry": True})
    worker.out_dir = str(tmp_path)
    mol = SimpleNamespace(natm=2)
    references = []

    def new_mf(*args):
        mf = make_mf()
        references.append(mf)
        return mf

    monkeypatch.setattr(worker, "_build_molecule", lambda stream: (mol, "H 0 0 0"))
    monkeypatch.setattr(worker, "_new_job_mf", new_mf)
    monkeypatch.setattr(worker, "_write_input_script", lambda *args: None)
    monkeypatch.setattr(worker, "_broken_symmetry_guess", lambda *args: "broken-density")
    monkeypatch.setattr(worker, "_optimize", lambda *args: mol)
    monkeypatch.setattr(worker, "_run_tddft", lambda *args: None)
    monkeypatch.setattr(worker, "_finish", lambda *args: None)
    worker._run_job(MagicMock())
    for mf in references:
        mf.kernel.assert_called_once_with(dm0="broken-density")


def test_rigid_scan_seeds_first_point_and_then_uses_converged_neighbor(monkeypatch, tmp_path):
    mod = load_module(monkeypatch, "worker")
    worker = mod.PySCFWorker("2\nH2\nH 0 0 0\nH 0 0 .74", {"method": "UHF", "spin": 1, "break_symmetry": True})
    worker.out_dir = str(tmp_path)
    mol = SimpleNamespace(natm=2, basis="sto-3g", charge=0, spin=0)
    rd_mol = MagicMock()
    rd_mol.GetAtoms.return_value = [SimpleNamespace(GetSymbol=lambda: "H")] * 2
    rd_mol.GetConformer.return_value.GetAtomPosition.return_value = SimpleNamespace(x=0, y=0, z=.74)
    mod.Chem.MolFromXYZBlock.return_value = rd_mol
    mod.Chem.RWMol.return_value = rd_mol
    mod.gto.M.return_value = MagicMock()
    references = []

    def new_mf(*args):
        mf = make_mf()
        references.append(mf)
        return mf

    monkeypatch.setattr(worker, "_new_step_mf", new_mf)
    monkeypatch.setattr(worker, "_broken_symmetry_guess", lambda *args: "broken-density")
    params = {"type": "Dist", "atoms": [0, 1], "start": .7, "end": .9, "steps": 2}
    results = {}
    worker.run_rigid_scan(mol, make_mf(), params, results)
    references[0].kernel.assert_called_once_with(dm0="broken-density")
    references[1].kernel.assert_called_once_with(dm0="converged-density")
    assert len(results["scan_results"]) == 2


def test_scf_callback_honors_cancellation(monkeypatch):
    mod = load_module(monkeypatch, "worker")
    worker = mod.PySCFWorker("", {})
    mf = make_mf()
    worker._apply_mf_settings(mf)
    worker._stop_requested = True
    with pytest.raises(InterruptedError):
        mf.callback({})
