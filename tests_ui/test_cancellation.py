import threading

from conftest import pump_until


def test_stop_keeps_native_worker_until_exit(dialog, plugin, app):
    from PyQt6.QtCore import QThread, pyqtSignal

    gate = threading.Event()

    class Worker(QThread):
        log_signal = pyqtSignal(str)
        result_signal = pyqtSignal(dict)
        error_signal = pyqtSignal(str)
        finished_signal = pyqtSignal()
        _stop_requested = False
        _stream = None

        def run(self):
            gate.wait(3)

    tab = dialog.calc_tab
    worker = Worker()
    tab.worker = worker
    worker.finished.connect(tab._on_worker_stopped)
    tab.run_btn.setEnabled(False)
    worker.start()
    try:
        pump_until(app, worker.isRunning)
        tab.stop_calculation()
        assert tab.worker is worker
        assert worker.isRunning()
        assert not tab.run_btn.isEnabled()
        assert worker._stop_requested
    finally:
        gate.set()
        assert worker.wait(3000)
        pump_until(app, lambda: tab.worker is None)
    assert tab.run_btn.isEnabled()


def test_close_defers_until_worker_returns(dialog, app):
    from PyQt6.QtCore import QThread
    gate = threading.Event()

    class Worker(QThread):
        _stop_requested = False
        _stream = None

        def run(self):
            gate.wait(3)

    worker = Worker()
    dialog.vis_tab.load_worker = worker
    worker.finished.connect(dialog.vis_tab._on_load_thread_finished)
    worker.start()
    try:
        pump_until(app, worker.isRunning)
        dialog.show()
        assert dialog.close() is False
        assert dialog.vis_tab.load_worker is worker
        assert worker._stop_requested
    finally:
        gate.set()
        assert worker.wait(3000)
        pump_until(app, lambda: dialog.vis_tab.load_worker is None)
        pump_until(app, lambda: not dialog.isVisible())
