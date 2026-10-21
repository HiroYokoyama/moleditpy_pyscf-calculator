"""Keep a test run's temporary files in one place and remove them afterwards.

Many tests hand the workers a directory from ``tempfile.mkdtemp()`` and never
remove it, and the workers write their inputs and results into it; a single run
left a few hundred ``tmpXXXXXXXX`` directories in the system temp directory.
Everything a run creates -- in this process and in the processes it starts --
goes under one directory that is removed when the session ends.
"""

import os
import shutil
import tempfile

# Python falls back to os.getcwd() -- the repository -- when no temp directory
# is usable; keep a run out of the source tree in that case too.
if os.path.abspath(tempfile.gettempdir()) == os.path.abspath(os.getcwd()):
    tempfile.tempdir = os.path.abspath(
        os.environ.get("RUNNER_TEMP") or os.path.expanduser("~/.cache/moleditpy_pyscf_calculator_tests")
    )
    os.makedirs(tempfile.tempdir, exist_ok=True)

_SESSION_TMP = tempfile.mkdtemp(prefix="moleditpy_pyscf_calculator_tests_")
tempfile.tempdir = _SESSION_TMP
for _name in ("TMP", "TEMP", "TMPDIR"):
    os.environ[_name] = _SESSION_TMP


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_SESSION_TMP, ignore_errors=True)


# Unit suites use synchronous Qt doubles; native delivery is tested in subprocesses.
import sys as _callback_sys
import types as _callback_types


def _connect_callback_double(signal, callback, owner=None, signature=(), cleanup=()):
    signal.connect(callback)
    return callback


_callback_module = _callback_types.ModuleType("_pyscf_callback_unit._qt_callbacks")
_callback_module.connect_queued = _connect_callback_double
_callback_sys.modules[_callback_module.__name__] = _callback_module
