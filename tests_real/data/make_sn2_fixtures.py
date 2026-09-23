"""Regenerate tests_real/data/sn2_* : the SN2 tutorial run through the plugin.

    cd tests_real && python data/make_sn2_fixtures.py

Tutorial 4 (CH3Cl + Br-, B3LYP/ma-def2-SVP): a relaxed C-Br scan 2.8 ->
1.96 A in 5 steps, then TS Optimization + Frequency from its highest
point. Takes a few minutes, so CI tests the plugin's loading / analysis
of these real results (test_real_fixtures.py) instead of recomputing.
Only the files the plugin reads back are kept.
"""

import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from conftest import base_config, load_plugin_modules  # noqa: E402

XYZ_SN2 = """6
CH3Cl + Br- backside
C 0.000 0.000 0.000
Cl 0.000 0.000 -1.800
H 1.028 0.000 0.357
H -0.514 0.890 0.357
H -0.514 -0.890 0.357
Br 0.000 0.000 2.800"""

KEEP = (
    "pyscf.chk",
    "scan_results.csv",
    "scan_trajectory.xyz",
    "scan_info.json",
    "freq_analysis.json",
    "properties.json",
    "pyscf_input.py",
)
LEVEL = {"method": "RKS", "functional": "b3lyp", "basis": "ma-def2-svp", "charge": -1}


def run(worker, xyz, tmp, **cfg):
    from PyQt6.QtCore import QCoreApplication

    QCoreApplication.instance() or QCoreApplication([])
    out = {}
    w = worker.PySCFWorker(xyz, base_config(tmp, threads=16, **LEVEL, **cfg))
    w.error_signal.connect(lambda e: sys.exit(f"job failed: {e}"))
    w.result_signal.connect(out.update)
    w.run()
    return out


def keep(src_dir, name):
    dst = os.path.join(HERE, name)
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for f in KEEP:
        if os.path.exists(os.path.join(src_dir, f)):
            shutil.copy(os.path.join(src_dir, f), dst)
    print("wrote", dst, sorted(os.listdir(dst)))


def main():
    worker, _ = load_plugin_modules()
    tmp = tempfile.mkdtemp()
    params = {"type": "Dist", "atoms": [0, 5], "start": 2.8, "end": 1.96, "steps": 5}
    scan = run(
        worker, XYZ_SN2, tmp, job_type="Relaxed Surface Scan", scan_params=params
    )
    keep(scan["out_dir"], "sn2_scan")

    top = int(np.argmax([p["energy"] for p in scan["scan_results"]]))
    ts = run(
        worker,
        scan["scan_trajectory"][top],
        tmp,
        job_type="TS Optimization + Frequency",
    )
    keep(ts["out_dir"], "sn2_ts")


if __name__ == "__main__":
    main()
