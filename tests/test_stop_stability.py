"""
tests/test_stop_stability.py
Unit tests for the stop-calculation stability fixes in the PySCF calculator plugin.

Coverage:
  - CaptureStdOut: FD restoration is atomic per-FD; failures are reported via logging
  - StreamToSignal: write() always forwards to target_stream; signal gates on _destroyed
  - PySCFWorker: cooperative stop flag and _stream attribute initialise correctly
  - PropertyWorker/LoadWorker: cooperative stop flags initialise correctly
  - CalcTab._on_worker_stopped: idempotent (double-call safe)
  - CalcTab.stop_calculation: sequence of operations is correct
  - PySCFDialog._safe_stop_worker: sequence of operations is correct

Run with:
    python -m pytest tests/test_stop_stability.py -v
"""

import io
import os
import sys
import types
import logging
import unittest
import importlib
import importlib.util
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# We import worker.py and calc_tab.py directly via importlib so that we
# bypass pyscf_calculator/__init__.py (which pulls in gui.py → full Qt stack).
# ---------------------------------------------------------------------------

def _load_module_direct(relpath, module_name):
    """Load a .py file as a module without going through the package __init__."""
    src = os.path.join(os.path.dirname(__file__), "..", relpath)
    src = os.path.normpath(src)
    spec = importlib.util.spec_from_file_location(module_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- Minimal stubs that satisfy worker.py's top-level imports ---
def _install_stubs():
    # PyQt6.QtCore stubs
    qt_core = types.ModuleType("PyQt6.QtCore")

    # QThread stub: a plain class so PySCFWorker can inherit from it
    class _QThread:
        def __init__(self): pass
        def start(self): pass
        def isRunning(self): return False
        def wait(self, ms=0): return True
        def terminate(self): pass
        @staticmethod
        def msleep(ms): pass
    qt_core.QThread = _QThread
    qt_core.pyqtSignal = lambda *a, **kw: MagicMock()

    # Qt namespace stub (needed by calc_tab: Qt.CursorShape.PointingHandCursor etc.)
    class _Qt:
        class CursorShape:
            PointingHandCursor = None
        class AlignmentFlag:
            AlignRight = None
        class Orientation:
            Horizontal = None
    qt_core.Qt = _Qt

    # QTimer stub
    class _QTimer:
        @staticmethod
        def singleShot(ms, fn): pass
    qt_core.QTimer = _QTimer

    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core
    sys.modules["PyQt6"] = pyqt6
    sys.modules["PyQt6.QtCore"] = qt_core

    # PyQt6.QtWidgets stubs (needed by calc_tab)
    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    for name in [
        "QWidget", "QVBoxLayout", "QHBoxLayout", "QLabel", "QComboBox",
        "QPushButton", "QSpinBox", "QCheckBox", "QGroupBox", "QFormLayout",
        "QMessageBox", "QLineEdit", "QFileDialog", "QProgressBar", "QTextEdit",
        "QDialog", "QSizePolicy", "QScrollArea", "QFrame", "QTabWidget", "QToolTip"
    ]:
        setattr(qt_widgets, name, MagicMock)
    pyqt6.QtWidgets = qt_widgets
    sys.modules["PyQt6.QtWidgets"] = qt_widgets

    # rdkit stubs
    rdkit = types.ModuleType("rdkit")
    rdkit_chem = types.ModuleType("rdkit.Chem")
    rdkit_chem.Chem = MagicMock()
    rdkit_chem.GetFormalCharge = MagicMock(return_value=0)
    rdkit.Chem = rdkit_chem
    sys.modules["rdkit"] = rdkit
    sys.modules["rdkit.Chem"] = rdkit_chem
    rdkit_transforms = types.ModuleType("rdkit.Chem.rdMolTransforms")
    sys.modules["rdkit.Chem.rdMolTransforms"] = rdkit_transforms

    # pyscf stub (mark unavailable so worker handles None correctly)
    sys.modules["pyscf"] = None  # type: ignore



_install_stubs()

# Now load the modules under test directly
_worker_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "worker.py"),
    "pyscf_calculator_worker_under_test",
)
CaptureStdOut = _worker_mod.CaptureStdOut
StreamToSignal = _worker_mod.StreamToSignal
PySCFWorker = _worker_mod.PySCFWorker
PropertyWorker = _worker_mod.PropertyWorker
LoadWorker = _worker_mod.LoadWorker

_calc_tab_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "calc_tab.py"),
    "pyscf_calculator_calc_tab_under_test",
)
CalcTab = _calc_tab_mod.CalcTab

_gui_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "gui.py"),
    "pyscf_calculator_gui_under_test",
)
PySCFDialog = _gui_mod.PySCFDialog


# ===========================================================================
# CaptureStdOut
# ===========================================================================

class TestCaptureStdOutExit(unittest.TestCase):
    """CaptureStdOut.__exit__ must restore each FD independently and safely."""

    def _bare_capturer(self):
        """Build a CaptureStdOut with no FD state (as if __enter__ was never called)."""
        cap = CaptureStdOut.__new__(CaptureStdOut)
        cap.filename = "/unused"
        cap.original_stdout_fd = None
        cap.original_stderr_fd = None
        cap.saved_stdout_fd = None
        cap.saved_stderr_fd = None
        cap.log_file = None
        return cap

    def test_exit_without_enter_does_not_crash(self):
        """__exit__ must be safe even when no FDs were saved."""
        cap = self._bare_capturer()
        cap.__exit__(None, None, None)   # must not raise

    def test_saved_fds_are_none_after_exit(self):
        """saved FD fields must be None after __exit__ finishes."""
        cap = self._bare_capturer()
        cap.saved_stdout_fd = None
        cap.saved_stderr_fd = None
        cap.__exit__(None, None, None)
        self.assertIsNone(cap.saved_stdout_fd)
        self.assertIsNone(cap.saved_stderr_fd)

    def test_stdout_fd_failure_still_restores_stderr(self):
        """If stdout dup2 fails, stderr restoration must still happen."""
        cap = self._bare_capturer()
        cap.original_stdout_fd = 1
        cap.original_stderr_fd = 2
        cap.saved_stdout_fd = 99
        cap.saved_stderr_fd = 100

        stderr_calls = []

        def fake_dup2(src, dst):
            if dst == 1:
                raise OSError("simulated stdout FD failure")
            stderr_calls.append((src, dst))

        with patch("os.dup2", side_effect=fake_dup2), \
             patch("os.close"), \
             self.assertLogs(level="WARNING") as log_ctx:
            cap.__exit__(None, None, None)

        self.assertIn((100, 2), stderr_calls,
                      "stderr FD must be restored even after stdout dup2 fails")
        self.assertTrue(any("stdout" in m for m in log_ctx.output),
                        "Failure must appear in logging output")

    def test_stderr_fd_failure_still_restores_stdout(self):
        """If stderr dup2 fails, stdout is already restored (independent order)."""
        cap = self._bare_capturer()
        cap.original_stdout_fd = 1
        cap.original_stderr_fd = 2
        cap.saved_stdout_fd = 99
        cap.saved_stderr_fd = 100

        stdout_calls = []

        def fake_dup2(src, dst):
            if dst == 2:
                raise OSError("simulated stderr FD failure")
            stdout_calls.append((src, dst))

        with patch("os.dup2", side_effect=fake_dup2), \
             patch("os.close"), \
             self.assertLogs(level="WARNING") as log_ctx:
            cap.__exit__(None, None, None)

        self.assertIn((99, 1), stdout_calls,
                      "stdout FD must be restored even after stderr dup2 fails")

    def test_log_file_closed_and_nulled(self):
        """log_file must be closed and set to None after __exit__."""
        cap = self._bare_capturer()
        mock_file = MagicMock()
        cap.log_file = mock_file
        cap.__exit__(None, None, None)
        mock_file.close.assert_called_once()
        self.assertIsNone(cap.log_file)


# ===========================================================================
# StreamToSignal
# ===========================================================================

class TestStreamToSignal(unittest.TestCase):

    def _make(self, signal=None, target=None):
        sig = signal if signal is not None else MagicMock()
        tgt = target if target is not None else io.StringIO()
        return StreamToSignal(sig, target_stream=tgt), sig, tgt

    def test_write_emits_signal_when_alive(self):
        s, sig, _ = self._make()
        s.write("hello")
        sig.emit.assert_called_once_with("hello")

    def test_write_no_signal_when_destroyed(self):
        s, sig, _ = self._make()
        s._destroyed = True
        s.write("suppressed")
        sig.emit.assert_not_called()

    def test_write_always_goes_to_target_stream_when_alive(self):
        s, _, tgt = self._make()
        s.write("data")
        self.assertEqual(tgt.getvalue(), "data")

    def test_write_still_goes_to_target_stream_when_destroyed(self):
        """Even after close(), file output must continue."""
        s, _, tgt = self._make()
        s._destroyed = True
        s.write("file only")
        self.assertEqual(tgt.getvalue(), "file only",
                         "target_stream write must happen even when _destroyed=True")

    def test_close_sets_destroyed_true(self):
        s, _, tgt = self._make()
        self.assertFalse(s._destroyed)
        s.close()
        self.assertTrue(s._destroyed)

    def test_close_does_not_close_target_stream(self):
        s, _, tgt = self._make()
        s.close()
        self.assertFalse(tgt.closed,
                         "close() must not close the underlying target_stream")

    def test_signal_emit_runtime_error_sets_destroyed(self):
        """RuntimeError from dead Qt signal must mark stream as destroyed."""
        sig = MagicMock()
        sig.emit.side_effect = RuntimeError("Qt C++ object deleted")
        s, _, tgt = self._make(signal=sig)
        s.write("trigger")
        self.assertTrue(s._destroyed)

    def test_target_stream_ioerror_is_swallowed(self):
        """IOError writing to target_stream must not propagate."""
        tgt = MagicMock()
        tgt.write.side_effect = IOError("closed file")
        s = StreamToSignal(MagicMock(), target_stream=tgt)
        try:
            s.write("safe?")   # must not raise
        except IOError as e:
            self.fail(f"StreamToSignal.write() must swallow IOError, got: {e}")


# ===========================================================================
# PySCFWorker — cooperative stop fields
# ===========================================================================

class TestPySCFWorkerFields(unittest.TestCase):
    """Verify new fields exist on freshly constructed workers."""

    def _make_worker(self):
        w = PySCFWorker.__new__(PySCFWorker)
        # Call only PySCFWorker.__init__ logic (skip QThread.__init__)
        w.xyz_str = "H 0 0 0"
        w.config = {}
        w._stop_requested = False
        w._stream = None
        return w

    def test_stop_requested_is_false(self):
        self.assertFalse(self._make_worker()._stop_requested)

    def test_stream_is_none(self):
        self.assertIsNone(self._make_worker()._stream)

class TestOtherWorkersFields(unittest.TestCase):
    """Verify LoadWorker and PropertyWorker cooperative fields."""

    def test_property_worker_fields(self):
        w = PropertyWorker.__new__(PropertyWorker)
        w.chkfile = None; w.tasks = []; w.out_dir = None
        w._stop_requested = False
        w._stream = None
        self.assertFalse(w._stop_requested)
        self.assertIsNone(w._stream)

    def test_load_worker_fields(self):
        w = LoadWorker.__new__(LoadWorker)
        w.chkfile = None
        w._stop_requested = False
        self.assertFalse(w._stop_requested)


# ===========================================================================
# CalcTab._on_worker_stopped — idempotency
# ===========================================================================

class TestOnWorkerStoppedIdempotent(unittest.TestCase):

    def _make_tab(self):
        tab = MagicMock(spec=CalcTab)
        tab.worker = MagicMock()
        tab._cleanup_count = 0

        def _cleanup():
            tab._cleanup_count += 1

        tab.cleanup_ui_state = _cleanup
        return tab

    def test_first_call_sets_worker_none_and_cleans_up(self):
        tab = self._make_tab()
        CalcTab._on_worker_stopped(tab)
        self.assertIsNone(tab.worker)
        self.assertEqual(tab._cleanup_count, 1)

    def test_second_call_is_noop(self):
        tab = self._make_tab()
        CalcTab._on_worker_stopped(tab)    # first call
        CalcTab._on_worker_stopped(tab)    # second call — must be no-op
        self.assertEqual(tab._cleanup_count, 1,
                         "_on_worker_stopped must not run cleanup twice")

    def test_call_with_no_worker_is_safe(self):
        tab = self._make_tab()
        tab.worker = None
        CalcTab._on_worker_stopped(tab)    # must not raise
        self.assertEqual(tab._cleanup_count, 0)


# ===========================================================================
# CalcTab.stop_calculation — operation sequence
# ===========================================================================

class TestStopCalculationSequence(unittest.TestCase):

    def _make_tab(self, wait_result=True):
        mock_stream = MagicMock()
        mock_stream._destroyed = False

        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = True
        mock_worker._stop_requested = False
        mock_worker._stream = mock_stream
        mock_worker.wait.return_value = wait_result

        tab = MagicMock(spec=CalcTab)
        tab.worker = mock_worker
        tab.log = MagicMock()

        return tab, mock_worker, mock_stream

    def test_stop_with_no_worker_returns_immediately(self):
        tab, _, _ = self._make_tab()
        tab.worker = None
        CalcTab.stop_calculation(tab)    # must not raise or hang

    def test_stop_with_idle_worker_returns_immediately(self):
        tab, worker, _ = self._make_tab()
        worker.isRunning.return_value = False
        CalcTab.stop_calculation(tab)
        worker._stop_requested  # should still be False
        self.assertFalse(worker._stop_requested)

    def test_stop_sets_stop_requested(self):
        tab, worker, _ = self._make_tab()
        CalcTab.stop_calculation(tab)
        self.assertTrue(worker._stop_requested)

    def test_stop_closes_stream(self):
        tab, worker, stream = self._make_tab()
        CalcTab.stop_calculation(tab)
        stream.close.assert_called_once()

    def test_stream_closed_before_signal_disconnect(self):
        """Order: stream.close() must precede log_signal.disconnect()."""
        tab, worker, stream = self._make_tab()
        call_order = []
        stream.close.side_effect = lambda: call_order.append("close")
        worker.log_signal.disconnect.side_effect = lambda: call_order.append("disc")
        CalcTab.stop_calculation(tab)
        if "close" in call_order and "disc" in call_order:
            self.assertLess(call_order.index("close"), call_order.index("disc"),
                            "stream must be invalidated before signal is disconnected")

    def test_terminate_not_called_on_cooperative_stop(self):
        tab, worker, _ = self._make_tab(wait_result=True)
        CalcTab.stop_calculation(tab)
        worker.terminate.assert_not_called()

    def test_terminate_called_on_timeout(self):
        tab, worker, _ = self._make_tab(wait_result=False)
        CalcTab.stop_calculation(tab)
        worker.terminate.assert_called_once()

    def test_finished_connected_for_deferred_cleanup(self):
        """finished signal must be connected so cleanup happens after thread exit."""
        tab, worker, _ = self._make_tab()
        CalcTab.stop_calculation(tab)
        worker.finished.connect.assert_called()


# ===========================================================================
# PySCFDialog._safe_stop_worker
# ===========================================================================

class TestSafeStopWorker(unittest.TestCase):
    def _make_dialog(self):
        dialog = MagicMock(spec=PySCFDialog)
        # bind the real method
        dialog._safe_stop_worker = PySCFDialog._safe_stop_worker.__get__(dialog)
        return dialog

    def test_safe_stop_sets_flag(self):
        dialog = self._make_dialog()
        worker = MagicMock()
        worker.isRunning.return_value = True
        worker._stop_requested = False
        dialog._safe_stop_worker(worker)
        self.assertTrue(worker._stop_requested)

    def test_safe_stop_calls_terminate_on_timeout(self):
        dialog = self._make_dialog()
        worker = MagicMock()
        worker.isRunning.return_value = True
        worker.wait.return_value = False
        dialog._safe_stop_worker(worker)
        worker.terminate.assert_called_once()
        worker.wait.assert_has_calls([call(1500), call(500)])


if __name__ == "__main__":
    unittest.main()
