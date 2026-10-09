from conftest import pump_until


def test_property_error_restores_analysis_without_clearing_calculation(dialog, plugin, app, monkeypatch, tmp_path):
    from PyQt6.QtCore import QThread, pyqtSignal
    from PyQt6.QtWidgets import QMessageBox
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args))

    class Worker(QThread):
        log_signal = pyqtSignal(str)
        finished_signal = pyqtSignal()
        error_signal = pyqtSignal(str)
        result_signal = pyqtSignal(dict)

        def __init__(self, *args):
            super().__init__()

        def run(self):
            self.error_signal.emit("Cube generation failed")

    monkeypatch.setattr(plugin("vis_tab"), "PropertyWorker", Worker)
    chk = tmp_path / "pyscf.chk"
    chk.touch()
    tab = dialog.vis_tab
    tab.chkfile_path = str(chk)
    tab.last_out_dir = str(tmp_path)
    tab.run_specific_analysis(["ESP"])
    pump_until(app, lambda: tab.prop_worker is None)
    assert errors
    assert tab.btn_run_analysis.isEnabled()
    assert dialog.calc_tab.worker is None
