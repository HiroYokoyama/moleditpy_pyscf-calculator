def test_state_count_round_trips_through_project_settings(plugin, dialog, context):
    dialog.calc_tab.nstates_input.setValue(30)
    dialog.save_settings()
    restored = plugin("gui").PySCFDialog(context.get_main_window(), context, settings=dict(dialog.settings))
    try:
        assert restored.calc_tab.nstates_input.value() == 30
    finally:
        restored.close()
