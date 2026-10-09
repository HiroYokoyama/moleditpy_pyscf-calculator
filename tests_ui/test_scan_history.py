def test_history_scan_does_not_replace_editor_molecule(plugin, context, dialog, tmp_path):
    folder = tmp_path / "scan"
    folder.mkdir()
    (folder / "scan_results.csv").write_text("Step,Value,Energy,Converged\n1,.7,-1,yes\n2,.9,-.9,yes", encoding="utf-8")
    (folder / "scan_trajectory.xyz").write_text("2\nfirst\nH 0 0 0\nH 0 0 .7\n2\nsecond\nH 0 0 0\nH 0 0 .9", encoding="utf-8")
    before = context.current_molecule
    dialog.vis_tab.load_result_folder(str(folder), update_structure=False)
    scan = dialog.vis_tab.scan_dlg
    assert context.current_molecule is before
    # Explicit selection may install the viewed geometry.
    scan.set_frame(1)
    assert context.current_molecule is scan.base_mol
    scan.close()
