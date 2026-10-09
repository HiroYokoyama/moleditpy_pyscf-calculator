def test_energy_result_clears_previous_thermochemistry(dialog, plugin, monkeypatch, tmp_path):
    tab = dialog.vis_tab
    tab.thermo_data = {"G_tot": [-1.0, "Eh"]}
    tab.freq_data = {"freqs": [100]}
    tab.optimized_xyz = "old geometry"
    tab.mo_data = {"type": "RHF", "energies": [-.5], "occupations": [2]}
    tab.btn_show_thermo.setEnabled(True)
    monkeypatch.setattr(plugin("vis_tab").QTimer, "singleShot", lambda *args: None)
    tab.on_load_finished({"out_dir": str(tmp_path)})
    assert tab.thermo_data is None
    assert tab.freq_data is None
    assert tab.optimized_xyz is None
    assert tab.mo_data is None
    assert not tab.btn_show_thermo.isEnabled()
    assert not tab.btn_load_geom.isEnabled()
    assert not tab.btn_run_analysis.isEnabled()


def test_document_reset_invalidates_delayed_geometry_load(dialog, plugin, monkeypatch, tmp_path):
    callbacks = []
    monkeypatch.setattr(plugin("vis_tab").QTimer, "singleShot", lambda ms, fn: callbacks.append(fn))
    tab = dialog.vis_tab
    tab.on_load_finished({"out_dir": str(tmp_path), "loaded_xyz": "2\nnew\nH 0 0 0\nH 0 0 2"})
    before = dialog.context.current_molecule
    dialog.on_document_reset()
    for fn in callbacks:
        fn()
    assert dialog.context.current_molecule is before
