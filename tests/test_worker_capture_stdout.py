"""
tests/test_worker_capture_stdout.py

Tests for CaptureStdOut.__enter__ / __exit__ and StreamToSignal.isatty.

Coverage targets (worker.py):
  - CaptureStdOut.__init__ (lines 30-34): attribute init
  - CaptureStdOut.__enter__ (lines 37-66): FD duplication, log-file redirect
  - CaptureStdOut.__exit__ error branches (lines 71, 73, 77): swallowed exceptions
  - CaptureStdOut.__exit__ restore-FD failure (lines 89-90, 100-101)
  - CaptureStdOut.__exit__ log-file close (line 106)
  - StreamToSignal.isatty (lines 152-154)

Strategy:
  CaptureStdOut is tested with a real temp file so __enter__ actually
  redirects FDs and __exit__ restores them.  FD-restore failures are
  triggered by closing the saved FD before __exit__ runs, causing os.dup2
  to raise on restore and exercising the warning branches.
"""

import os
import sys
import types
import tempfile
import unittest
import importlib.util
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Qt / pyscf stubs
# ---------------------------------------------------------------------------


def _install_stubs(force=False):
    qt_core = types.ModuleType("PyQt6.QtCore")

    class _QThread:
        def __init__(self):
            pass

    qt_core.QThread = _QThread
    qt_core.pyqtSignal = lambda *a, **kw: MagicMock()
    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core

    def _set(k, v):
        if force:
            sys.modules[k] = v
        else:
            sys.modules.setdefault(k, v)

    _set("PyQt6", pyqt6)
    _set("PyQt6.QtCore", qt_core)
    _set("rdkit", MagicMock())
    _set("rdkit.Chem", MagicMock())
    _set("rdkit.Chem.rdMolTransforms", MagicMock())


_install_stubs()


def _load_worker_mod():
    _install_stubs(force=True)
    module_name = "_worker_capstdout_test"
    sys.modules.setdefault("pyscf", MagicMock())
    for sub in ("pyscf.gto", "pyscf.scf", "pyscf.dft", "pyscf.solvent"):
        sys.modules.setdefault(sub, MagicMock())
    src = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "pyscf_calculator", "worker.py")
    )
    spec = importlib.util.spec_from_file_location(module_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_worker_mod()
CaptureStdOut = _mod.CaptureStdOut
StreamToSignal = _mod.StreamToSignal


# ===========================================================================
# CaptureStdOut.__init__
# ===========================================================================


class TestCaptureStdOutInit(unittest.TestCase):
    def test_filename_stored(self):
        cap = CaptureStdOut("/tmp/test.log")
        self.assertEqual(cap.filename, "/tmp/test.log")

    def test_initial_fds_none(self):
        cap = CaptureStdOut("/tmp/test.log")
        self.assertIsNone(cap.original_stdout_fd)
        self.assertIsNone(cap.original_stderr_fd)
        self.assertIsNone(cap.saved_stdout_fd)
        self.assertIsNone(cap.saved_stderr_fd)


# ===========================================================================
# CaptureStdOut.__enter__
# ===========================================================================


class TestCaptureStdOutEnter(unittest.TestCase):
    def test_enter_returns_log_file(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            cap = CaptureStdOut(log_path)
            result = cap.__enter__()
            cap.__exit__(None, None, None)
            self.assertIsNotNone(result)
        finally:
            os.unlink(log_path)

    def test_enter_sets_saved_fds(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            cap = CaptureStdOut(log_path)
            cap.__enter__()
            # After __enter__, saved FDs should be set (not None)
            self.assertIsNotNone(cap.saved_stdout_fd)
            cap.__exit__(None, None, None)
        finally:
            os.unlink(log_path)

    def test_enter_sets_original_fds(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            cap = CaptureStdOut(log_path)
            cap.__enter__()
            self.assertIsNotNone(cap.original_stdout_fd)
            self.assertIsNotNone(cap.original_stderr_fd)
            cap.__exit__(None, None, None)
        finally:
            os.unlink(log_path)

    def test_context_manager_usage(self):
        """Full with-statement round-trip: enter and exit without error."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            with CaptureStdOut(log_path) as log_file:
                self.assertIsNotNone(log_file)
        finally:
            os.unlink(log_path)

    def test_enter_opens_log_file_in_append_mode(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            cap = CaptureStdOut(log_path)
            cap.__enter__()
            mode = cap.log_file.mode
            cap.__exit__(None, None, None)
            self.assertIn("a", mode)
        finally:
            os.unlink(log_path)


# ===========================================================================
# CaptureStdOut.__exit__ — error branches
# ===========================================================================


class TestCaptureStdOutExitErrors(unittest.TestCase):
    """
    Trigger the warning/swallow branches in __exit__ by making OS calls fail.
    """

    def test_exit_with_already_closed_saved_fd_logs_warning(self):
        """
        If saved_stdout_fd is closed before __exit__, dup2 fails.
        __exit__ must log a warning and not raise.
        """
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            cap = CaptureStdOut(log_path)
            cap.__enter__()
            # Close the saved FD so dup2(saved_fd, orig_fd) fails in __exit__
            saved = cap.saved_stdout_fd
            os.close(saved)
            # saved_stdout_fd is still set on the object → __exit__ tries to dup2

            with self.assertLogs(level="WARNING") as cm:
                cap.__exit__(None, None, None)

            warning_text = " ".join(cm.output)
            self.assertIn("stdout", warning_text.lower())
        finally:
            try:
                os.unlink(log_path)
            except OSError:
                pass

    def test_exit_does_not_raise_on_flush_error(self):
        """Flushing a closed stream in __exit__ must be swallowed."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            cap = CaptureStdOut(log_path)
            cap.__enter__()
            # Pre-close the log file to trigger the flush-swallow branch
            cap.log_file.close()
            try:
                cap.__exit__(None, None, None)
            except Exception as e:
                self.fail(f"__exit__ raised unexpectedly: {e}")
        finally:
            try:
                os.unlink(log_path)
            except OSError:
                pass

    def test_exit_clears_log_file_reference(self):
        """After __exit__, log_file attribute must be None."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            cap = CaptureStdOut(log_path)
            cap.__enter__()
            cap.__exit__(None, None, None)
            self.assertIsNone(cap.log_file)
        finally:
            try:
                os.unlink(log_path)
            except OSError:
                pass


# ===========================================================================
# CaptureStdOut.__enter__ fileno() fallback (lines 46-47, 51-52)
# ===========================================================================


class TestCaptureStdOutEnterFileFallback(unittest.TestCase):
    """When sys.__stdout__.fileno() raises, FD falls back to 1 (stdout)."""

    def test_fileno_raises_uses_fallback_fd(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            cap = CaptureStdOut(log_path)
            with patch.object(
                sys,
                "__stdout__",
                create=True,
                new_callable=lambda: (
                    lambda: type(
                        "FakeStdout",
                        (),
                        {
                            "fileno": staticmethod(
                                lambda: (_ for _ in ()).throw(OSError("no fd"))
                            )
                        },
                    )()
                ),
            ):
                pass
            # Simpler approach: temporarily give __stdout__ a fileno that raises
            orig = sys.__stdout__

            class _NoFD:
                def fileno(self):
                    raise OSError("no fd")

                def flush(self):
                    pass

            sys.__stdout__ = _NoFD()
            try:
                cap.__enter__()
                self.assertEqual(cap.original_stdout_fd, 1)
            finally:
                sys.__stdout__ = orig
                try:
                    cap.__exit__(None, None, None)
                except Exception:
                    pass
        finally:
            try:
                os.unlink(log_path)
            except OSError:
                pass


# ===========================================================================
# CaptureStdOut.__exit__ flush-error swallowing (lines 71, 73)
# ===========================================================================


class TestCaptureStdOutExitFlushErrors(unittest.TestCase):
    def test_exit_swallows_stdout_flush_error(self):
        """sys.stdout.flush() raising in __exit__ must be swallowed."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            cap = CaptureStdOut(log_path)
            cap.__enter__()
            with patch("sys.stdout") as mock_stdout, patch("sys.stderr") as mock_stderr:
                mock_stdout.flush.side_effect = OSError("flush failed")
                mock_stderr.flush.side_effect = OSError("flush failed")
                try:
                    cap.__exit__(None, None, None)
                except Exception as e:
                    self.fail(f"__exit__ raised on flush error: {e}")
        finally:
            try:
                os.unlink(log_path)
            except OSError:
                pass


# ===========================================================================
# StreamToSignal.flush
# ===========================================================================


class TestStreamToSignalFlush(unittest.TestCase):
    def test_flush_delegates_to_target(self):
        target = MagicMock()
        s = StreamToSignal(MagicMock(), target_stream=target)
        s.flush()
        target.flush.assert_called_once()

    def test_flush_with_no_target_does_not_raise(self):
        s = StreamToSignal(MagicMock())
        try:
            s.flush()
        except Exception as e:
            self.fail(f"flush() raised: {e}")

    def test_flush_swallows_target_error(self):
        target = MagicMock()
        target.flush.side_effect = OSError("broken pipe")
        s = StreamToSignal(MagicMock(), target_stream=target)
        try:
            s.flush()
        except Exception as e:
            self.fail(f"flush() raised on target error: {e}")


# ===========================================================================
# StreamToSignal.isatty
# ===========================================================================


class TestStreamToSignalIsatty(unittest.TestCase):
    def test_isatty_returns_false_when_no_target(self):
        s = StreamToSignal(MagicMock())
        self.assertFalse(s.isatty())

    def test_isatty_delegates_to_target_stream(self):
        target = MagicMock()
        target.isatty.return_value = True
        s = StreamToSignal(MagicMock(), target_stream=target)
        self.assertTrue(s.isatty())

    def test_isatty_returns_false_when_target_has_no_isatty(self):
        target = MagicMock(spec=[])  # no isatty attribute
        s = StreamToSignal(MagicMock(), target_stream=target)
        self.assertFalse(s.isatty())


if __name__ == "__main__":
    unittest.main()
