# SPDX-License-Identifier: LGPL-3.0-or-later
"""Deliver worker callbacks through owned QObject slots on the GUI thread."""

from functools import lru_cache

from PySide6.QtCore import QCoreApplication, QObject, Qt, Slot


@lru_cache(maxsize=None)
def _receiver_type(signature):
    class CallbackReceiver(QObject):
        def __init__(self, owner, callback):
            super().__init__(owner)
            self._callback = callback

        @Slot(*signature)
        def dispatch(self, *args):
            self._callback(*args)

    return CallbackReceiver


def connect_queued(signal, callback, owner=None, signature=(), cleanup=()):
    """Route a callback through a GUI-owned typed slot until terminal cleanup."""
    if owner is None:
        owner = QCoreApplication.instance()
    if owner is None:
        raise RuntimeError("Worker callbacks require a running Qt application")
    receiver = _receiver_type(tuple(signature))(owner, callback)
    signal.connect(receiver.dispatch, Qt.ConnectionType.QueuedConnection)
    for terminal in cleanup:
        terminal.connect(receiver.deleteLater)
    return receiver
