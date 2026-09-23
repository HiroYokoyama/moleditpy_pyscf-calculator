"""The plugin's real Qt widgets (offscreen) wired to real PySCF runs.

The mocked suite fakes every widget and worker, so it could not see e.g.
that Stop connected a signal PyQt6's QThread does not have. These tests
build the actual dialog, set options through the widgets, run jobs, stop
a running one, round-trip the settings and open the result viewers."""

import importlib
import json
import os
import time

import pytest

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("rdkit")
pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pyvista")
pytest.importorskip("matplotlib")

from conftest import XYZ_H2, XYZ_H2O, load_plugin_modules
from PyQt6.QtWidgets import QApplication
from rdkit import Chem
from rdkit.Chem import AllChem


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def plugin():
    load_plugin_modules()
    return {
        name: importlib.import_module(f"pyscf_calculator.{name}")
        for name in (
            "gui",
            "worker",
            "energy_diag",
            "scan_results",
            "tddft_table",
            "freq_vis",
        )
    }


def _water():
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    return mol


def _h2():
    mol = Chem.AddHs(Chem.MolFromSmiles("[H][H]"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    return mol


class _Context:
    """The PluginContext surface the dialog uses, without a main window."""

    def __init__(self, mol):
        self.current_molecule = mol
        self.modified = 0

    def get_main_window(self):
        return None

    def mark_project_modified(self):
        self.modified += 1

    def draw_molecule_3d(self, mol):
        pass

    def reset_3d_camera(self):
        pass


def _pump(qapp, seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        qapp.processEvents()
        time.sleep(0.01)


@pytest.fixture
def dialog(qapp, plugin, tmp_path, monkeypatch):
    # never pick up a developer's saved defaults
    monkeypatch.setattr(
        plugin["gui"], "_defaults_path", lambda: str(tmp_path / "none.json")
    )

    def make(mol, settings=None):
        dlg = plugin["gui"].PySCFDialog(
            parent=None, context=_Context(mol), settings=settings or {}, version="test"
        )
        dlg.calc_tab.out_dir_edit.setText(str(tmp_path / "results"))
        return dlg

    return make


def _run_sync(plugin, monkeypatch):
    """Workers run in the calling thread; returns the list of workers."""
    started = []

    def start(self):
        started.append(self)
        self.run()

    monkeypatch.setattr(plugin["worker"].PySCFWorker, "start", start)
    monkeypatch.setattr(plugin["worker"].LoadWorker, "start", start)
    return started


def test_options_set_in_the_widgets_reach_the_result(qapp, plugin, dialog, monkeypatch):
    workers = _run_sync(plugin, monkeypatch)
    dlg = dialog(_water())
    tab = dlg.calc_tab
    tab.job_type_combo.setCurrentText("Frequency")
    tab.method_combo.setCurrentText("RKS")
    tab.functional_combo.setCurrentText("pbe")
    tab.basis_combo.setCurrentText("sto-3g")
    tab.hessian_combo.setCurrentText("Numerical (finite difference)")
    tab.spin_temperature.setValue(400.0)
    tab.spin_pressure.setValue(2.0)
    tab.check_symmetry.setChecked(True)

    # the Frequency-only rows are visible for this job type
    assert tab.hessian_combo.isVisibleTo(tab)
    assert tab.spin_temperature.isVisibleTo(tab)

    tab.run_calculation()
    _pump(qapp)
    job = workers[0]
    assert job.config["hessian"].startswith("Numerical")
    assert job.config["pressure"] == pytest.approx(2.0 * 101325.0)

    with open(os.path.join(job.out_dir, "freq_analysis.json")) as fh:
        thermo = json.load(fh)["thermo_data"]
    assert thermo["temperature"][0] == pytest.approx(400.0)
    assert thermo["pressure"][0] == pytest.approx(2.0 * 101325.0)
    with open(os.path.join(job.out_dir, "pyscf.out")) as fh:
        out = fh.read()
    assert "Point-group symmetry" in out and "Numerical Hessian" in out
    # the dialog recorded the result and re-enabled Run
    assert dlg.calc_history and tab.run_btn.isEnabled()


def test_stop_a_running_job(qapp, plugin, dialog):
    """A long relaxed scan on H2 (cheap steps, so the cooperative stop is
    honoured between points) is stopped from the GUI while it runs."""
    pytest.importorskip("geometric")
    dlg = dialog(_h2())
    tab = dlg.calc_tab
    tab.job_type_combo.setCurrentText("Relaxed Surface Scan")
    tab.method_combo.setCurrentText("RHF")
    tab.scan_params = {
        "type": "Dist",
        "atoms": [0, 1],
        "start": 0.6,
        "end": 1.6,
        "steps": 200,
    }
    tab.run_calculation()
    _pump(qapp, 1.0)
    assert tab.worker is not None and tab.worker.isRunning()

    tab.stop_calculation()  # raised AttributeError ('terminated') before
    _pump(qapp)
    assert tab.worker is None
    assert tab.run_btn.isEnabled() and not tab.stop_btn.isEnabled()


def test_closing_the_dialog_mid_job(qapp, plugin, dialog):
    pytest.importorskip("geometric")
    dlg = dialog(_h2())
    tab = dlg.calc_tab
    tab.job_type_combo.setCurrentText("Relaxed Surface Scan")
    tab.method_combo.setCurrentText("RHF")
    tab.scan_params = {
        "type": "Dist",
        "atoms": [0, 1],
        "start": 0.6,
        "end": 1.6,
        "steps": 200,
    }
    tab.run_calculation()
    _pump(qapp, 1.0)
    dlg.close()  # closeEvent -> stop_calculation
    _pump(qapp)
    assert tab.worker is None


def test_settings_round_trip_through_real_widgets(qapp, dialog):
    first = dialog(_water())
    tab = first.calc_tab
    tab.method_combo.setCurrentText("UKS")
    tab.functional_combo.setCurrentText("r2scan")
    tab.check_break_sym.setChecked(True)
    tab.hessian_combo.setCurrentText("Numerical (finite difference)")
    tab.dispersion_combo.setCurrentText("D4")
    tab.spin_temperature.setValue(350.0)
    tab.spin_pressure.setValue(3.5)
    tab.spin_grid_level.setValue(5)
    tab.solvent_combo.setCurrentText("THF")
    first.update_internal_state()
    saved = dict(first.settings)

    second = dialog(_water(), settings=saved)
    t2 = second.calc_tab
    assert t2.method_combo.currentText() == "UKS"
    assert t2.functional_combo.currentText() == "r2scan"
    assert t2.check_break_sym.isChecked()
    assert t2.hessian_combo.currentText().startswith("Numerical")
    assert t2.dispersion_combo.currentText() == "D4"
    assert t2.spin_temperature.value() == pytest.approx(350.0)
    assert t2.spin_pressure.value() == pytest.approx(3.5)
    assert t2.spin_grid_level.value() == 5
    assert t2.solvent_combo.currentText() == "THF"


def test_result_viewers_on_real_results(qapp, plugin, run_job):
    energy = run_job(
        XYZ_H2O, method="RKS", functional="pbe", job_type="TDDFT", nstates=3
    )
    r = energy.results

    # orbital energy diagram paints the real MO levels
    diag = plugin["energy_diag"].EnergyDiagramDialog(
        {"energies": r["mo_energy"], "occupations": r["mo_occ"], "type": r["scf_type"]},
        result_dir=r["out_dir"],
    )
    diag.resize(450, 600)
    assert not diag.grab().isNull()
    assert diag.hit_zones  # levels were drawn and are clickable

    # TDDFT table shows every state
    table = plugin["tddft_table"].TddftTable(None, r["tddft_data"])
    assert table.table.rowCount() == 3

    # scan profile + frame stepping on a real scan
    params = {"type": "Dist", "atoms": [0, 1], "start": 0.6, "end": 0.9, "steps": 3}
    scan = run_job(XYZ_H2, job_type="Rigid Surface Scan", scan_params=params).results
    dlg = plugin["scan_results"].ScanResultDialog(
        results=scan["scan_results"],
        trajectory=scan["scan_trajectory"],
        context=_Context(_h2()),
        scan_type=scan["scan_type"],
    )
    assert dlg.canvas.axes.get_xlabel() == "Bond Length (Å)"
    dlg.set_frame(2)
    assert dlg.frame_idx == 2

    # frequency viewer on real normal modes
    freq = run_job(XYZ_H2O, job_type="Frequency").results["freq_data"]
    fv = plugin["freq_vis"].FreqVisualizer(
        None, _water(), freq["freqs"], freq["modes"], context=_Context(_water())
    )
    assert fv.list_freq.topLevelItemCount() == 3
    fv.list_freq.setCurrentItem(fv.list_freq.topLevelItem(2))  # select a mode
    fv.cleanup()
