import contextlib
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from audit_helpers import load_module


def test_capture_serializes_workers_and_restores_streams(monkeypatch, tmp_path):
    mod = load_module(monkeypatch, "worker")
    import sys
    streams = (sys.stdout, sys.stderr)
    entered = threading.Event()
    release = threading.Event()
    order = []
    errors = []

    # Keep OS descriptor changes out of pytest itself while exercising the
    # real shared lock and Python-stream restoration with real log files.
    class Capture:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            self.file = open(self.path, "w", encoding="utf-8")
            return self.file

        def __exit__(self, *args):
            self.file.close()

    monkeypatch.setattr(mod, "CaptureStdOut", Capture)

    def run(name):
        try:
            worker = SimpleNamespace(_stop_requested=False, log_signal=MagicMock(), _stream=None)
            with mod.redirected_output(worker, tmp_path / name) as stream:
                order.append(name)
                stream.write(name)
                if name == "first":
                    entered.set()
                    assert release.wait(3)
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=run, args=("first",))
    second = threading.Thread(target=run, args=("second",))
    first.start()
    try:
        assert entered.wait(3)
        second.start()
        assert not mod._OUTPUT_LOCK.acquire(blocking=False)
    finally:
        release.set()
        first.join(3)
        if second.ident is not None:
            second.join(3)
    assert not errors
    assert order == ["first", "second"]
    assert (sys.stdout, sys.stderr) == streams
    assert (tmp_path / "first").read_text() == "first"
    assert (tmp_path / "second").read_text() == "second"


def test_cancelled_waiter_does_not_change_streams(monkeypatch, tmp_path):
    mod = load_module(monkeypatch, "worker")
    worker = SimpleNamespace(_stop_requested=True)
    with mod._OUTPUT_LOCK:
        with pytest.raises(InterruptedError):
            with mod.redirected_output(worker, tmp_path / "cancelled"):
                pytest.fail("Cancelled worker entered capture")
    assert not (tmp_path / "cancelled").exists()


def test_partial_descriptor_setup_is_rolled_back(monkeypatch, tmp_path):
    mod = load_module(monkeypatch, "worker")
    monkeypatch.setattr(mod, "_flush_c_stdio", lambda: None)
    monkeypatch.setattr(mod.os, "dup", MagicMock(side_effect=[101, OSError("stderr unavailable")]))
    restore = MagicMock()
    close = MagicMock()
    monkeypatch.setattr(mod.os, "dup2", restore)
    monkeypatch.setattr(mod.os, "close", close)
    capture = mod.CaptureStdOut(tmp_path / "capture.log")
    with pytest.raises(OSError):
        capture.__enter__()
    restore.assert_called_once_with(101, capture.original_stdout_fd)
    close.assert_called_once_with(101)
    assert capture.log_file is None
