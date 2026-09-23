"""Everything a job reports must be in its pyscf.out, not only in the GUI
log: the worker's own summaries (which used to go to the GUI signal only)
and output written by C code (which sat in C stdio buffers when the file
descriptors were switched back)."""

import ctypes
import os
import sys

import pytest

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("rdkit")
pytest.importorskip("PyQt6.QtCore")

from conftest import XYZ_H2, XYZ_H2O, XYZ_OH, load_plugin_modules


def _out(res):
    with open(
        os.path.join(res.results["out_dir"], "pyscf.out"), encoding="utf-8"
    ) as fh:
        return fh.read()


def test_worker_summaries_reach_the_log_file(run_job):
    res = run_job(XYZ_OH, method="RHF", spin=2, job_type="Frequency")
    out = _out(res)
    for text in (
        "Switching to UHF",
        "PySCF running with",
        "converged SCF energy",  # PySCF's own output
        "Imaginary modes",
        "Frequency Analysis Completed",
        "SCF Properties",
        "Mulliken charges",
        "Checkpoint saved to",
    ):
        assert text in out, text
        assert text in res.log, text  # and still in the GUI log


def test_tddft_table_reaches_the_log_file(run_job):
    res = run_job(XYZ_H2O, job_type="TDDFT", method="RKS", functional="pbe", nstates=2)
    out = _out(res)
    assert "===== TDDFT Results =====" in out
    assert "Osc. Str." in out


def test_scan_progress_reaches_the_log_file(run_job):
    params = {"type": "Dist", "atoms": [0, 1], "start": 0.7, "end": 0.8, "steps": 2}
    res = run_job(XYZ_H2, job_type="Rigid Surface Scan", scan_params=params)
    out = _out(res)
    assert "Step 1/2" in out and "Step 2/2" in out
    assert "Scan results saved to" in out


def test_property_generation_reaches_the_log_file(run_job, run_properties):
    res = run_job(XYZ_H2O)
    run_properties(res.results["chkfile"], ["HOMO"], res.results["out_dir"])
    assert "Generating 005_HOMO.cube" in _out(res)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX libc printf")
def test_c_level_output_is_flushed_into_the_log_file(qcore, tmp_path):
    worker, _ = load_plugin_modules()

    class _Signal:
        def __init__(self):
            self.text = []

        def emit(self, t):
            self.text.append(t)

    class _Worker:
        log_signal = _Signal()
        _stream = None

    log_file = tmp_path / "pyscf.out"
    libc = ctypes.CDLL(None)
    with worker.redirected_output(_Worker(), str(log_file)):
        # stdout is a file here, so C stdio buffers this line
        libc.printf(b"C-LEVEL-MARKER\n")
    assert "C-LEVEL-MARKER" in log_file.read_text()
