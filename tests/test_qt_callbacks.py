"""Real Qt regressions run in child processes, outside any unit-test Qt stubs."""

import os
from pathlib import Path
import subprocess
import sys
import unittest

HELPER = Path(__file__).resolve().parents[1] / "pyscf_calculator/_qt_callbacks.py"


class QueuedCallbackTests(unittest.TestCase):
    def run_qt(self, body):
        prefix = f"""
import sys
try:
    import PySide6
    from PySide6.QtCore import QCoreApplication, QEvent, QObject, QRunnable, QThreadPool, QThread, Signal
    from shiboken6 import isValid, delete
except ImportError:
    sys.exit(77)
import gc, importlib.util, os
required = os.environ.get('REQUIRED_QT_VERSION')
if required:
    assert PySide6.__version__ == required, PySide6.__version__
spec = importlib.util.spec_from_file_location('qt_callbacks_under_test', {str(HELPER)!r})
callbacks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(callbacks)
def drain(app):
    for _ in range(10):
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        gc.collect()
"""
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", prefix + body],
            capture_output=True, text=True, encoding="utf-8", timeout=40,
        )
        if result.returncode == 77 and os.environ.get("REQUIRE_REAL_QT") != "1":
            self.skipTest("Real PySide6 is exercised by the dedicated Qt regression workflow")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def stress(self, app_owned):
        self.run_qt("""
app = QCoreApplication([])
owner = QObject()
delivered, wrong_threads, receivers = [], [], []
class Emitter(QObject):
    result = Signal(int)
class Task(QRunnable):
    def __init__(self, value):
        super().__init__()
        self.value, self.signals = value, Emitter()
    def run(self):
        self.signals.result.emit(self.value)
def receive(value):
    delivered.append(value)
    if QThread.currentThread() != app.thread():
        wrong_threads.append(value)
pool = QThreadPool()
for value in range(300):
    task = Task(value)
    receivers.append(callbacks.connect_queued(task.signals.result, receive, OWNER,
        (int,), cleanup=(task.signals.result,)))
    pool.start(task)
del task
assert pool.waitForDone(10000)
gc.collect()
assert not delivered, 'Callback ran before the GUI event loop processed it'
drain(app)
assert sorted(delivered) == list(range(300)), len(delivered)
assert not wrong_threads, wrong_threads
assert all(not isValid(receiver) for receiver in receivers), 'Terminal receivers leaked'
""".replace("OWNER", "None" if app_owned else "owner"))

    def test_results_survive_sender_destruction(self):
        self.stress(app_owned=False)

    def test_ownerless_callbacks_are_owned_by_the_application(self):
        self.stress(app_owned=True)

    def test_typed_slots_preserve_plugin_payloads(self):
        self.run_qt("""
app = QCoreApplication([])
owner = QObject()
class Emitter(QObject):
    text = Signal(str)
    mapping = Signal(dict)
    sequence = Signal(list)
    status = Signal(bool, str)
    ellipsoid = Signal(object, object, object, object, str)
emitter = Emitter()
answers = []
for name, signature, values in (
    ('text', (str,), ('answer',)),
    ('mapping', (dict,), ({'answer': 42},)),
    ('sequence', (list,), ([1, 2],)),
    ('status', (bool, str), (True, 'cube.cube')),
    ('ellipsoid', (object, object, object, object, str), (None, [1], {'x':2}, None, '')),
):
    def receive(*args):
        assert QThread.currentThread() == app.thread()
        answers.append(args)
    callbacks.connect_queued(getattr(emitter, name), receive, owner, signature)
    getattr(emitter, name).emit(*values)
    drain(app)
    assert answers[-1] == values, (answers[-1], values)
""")

    def test_destroyed_window_drops_pending_updates(self):
        self.run_qt("""
app = QCoreApplication([])
owner = QObject()
class Emitter(QObject):
    result = Signal(int)
emitter = Emitter()
received = []
callbacks.connect_queued(emitter.result, received.append, owner, (int,))
emitter.result.emit(1)
delete(owner)
drain(app)
assert received == [], received
""")

    def test_receiver_is_a_native_typed_qobject_slot(self):
        self.run_qt("""
app = QCoreApplication([])
owner = QObject()
class Emitter(QObject):
    result = Signal(int)
emitter = Emitter()
receiver = callbacks.connect_queued(emitter.result, lambda value: None, owner, (int,))
assert receiver.parent() is owner
assert receiver.metaObject().indexOfSlot('dispatch(int)') >= 0
assert receiver.thread() == app.thread()
""")

    def test_missing_application_is_reported(self):
        self.run_qt("""
assert QCoreApplication.instance() is None
class Emitter(QObject):
    result = Signal(int)
emitter = Emitter()
try:
    callbacks.connect_queued(emitter.result, lambda value: None, signature=(int,))
except RuntimeError as error:
    assert 'Qt application' in str(error)
else:
    raise AssertionError('A parentless receiver was created without an application')
""")


if __name__ == "__main__":
    unittest.main()
