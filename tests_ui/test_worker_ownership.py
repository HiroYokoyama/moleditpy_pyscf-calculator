import threading

from conftest import pump_until


def test_custom_success_does_not_release_running_property_worker(dialog, app):
    from PyQt6.QtCore import QThread, pyqtSignal
    gate = threading.Event()

    class Worker(QThread):
        finished_signal = pyqtSignal()

        def run(self):
            self.finished_signal.emit()
            gate.wait(3)

    tab = dialog.vis_tab
    worker = Worker()
    tab.prop_worker = worker
    worker.finished_signal.connect(tab.on_prop_finished)
    worker.finished.connect(tab._on_prop_thread_finished)
    tab.btn_run_analysis.setEnabled(False)
    worker.start()
    try:
        pump_until(app, worker.isRunning)
        app.processEvents()
        assert tab.prop_worker is worker
        assert not tab.btn_run_analysis.isEnabled()
    finally:
        gate.set()
        assert worker.wait(3000)
        pump_until(app, lambda: tab.prop_worker is None)
