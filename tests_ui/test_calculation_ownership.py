import threading

from conftest import pump_until


def test_calculation_success_retains_thread_until_native_exit(dialog, app):
    from PyQt6.QtCore import QThread, pyqtSignal
    gate = threading.Event()
    delivered = []

    class Worker(QThread):
        finished_signal = pyqtSignal()
        _stop_requested = False

        def run(self):
            self.finished_signal.emit()
            gate.wait(3)

    tab = dialog.calc_tab
    worker = Worker()
    tab.worker = worker
    tab.run_btn.setEnabled(False)
    worker.finished_signal.connect(tab.on_finished)
    worker.finished_signal.connect(lambda: delivered.append(True))
    worker.finished.connect(tab._on_worker_stopped)
    worker.start()
    try:
        pump_until(app, lambda: bool(delivered))
        assert tab.worker is worker
        assert not tab.run_btn.isEnabled()
    finally:
        gate.set()
        assert worker.wait(3000)
        pump_until(app, lambda: tab.worker is None)
    assert tab.run_btn.isEnabled()
