import pytest


@pytest.mark.parametrize("action", ["accept", "reject", "close"])
def test_every_exit_stops_selection_polling(plugin, context, app, action):
    dlg = plugin("scan_dialog").ScanDialog(context=context)
    dlg.show()
    app.processEvents()
    assert dlg.sel_timer.isActive()
    getattr(dlg, action)()
    app.processEvents()
    assert not dlg.sel_timer.isActive()
