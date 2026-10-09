def test_regenerated_esp_uses_matching_density_surface(plugin, dialog, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QListWidgetItem
    tab = dialog.vis_tab
    for name in ("esp_2.cube", "density_2.cube"):
        (tmp_path / name).touch()
    item = QListWidgetItem("esp_2.cube")
    item.setToolTip(str(tmp_path / "esp_2.cube"))
    pairs = []
    monkeypatch.setattr(tab, "switch_to_mapped_mode", lambda surf, prop: pairs.append((surf, prop)))
    tab.on_file_selected(item)
    assert pairs == [(str(tmp_path / "density_2.cube"), str(tmp_path / "esp_2.cube"))]
