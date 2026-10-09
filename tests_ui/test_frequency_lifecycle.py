import numpy as np


def test_closing_dock_stops_animation_and_restores_coordinates(plugin, context, app):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QDockWidget
    mw = context.get_main_window()
    mol = context.current_molecule
    base = mol.GetConformer().GetPositions().copy()
    vis = plugin("freq_vis").FreqVisualizer(mw, mol, [100], np.ones((1, 2, 3)) * .1, context=context)
    dock = QDockWidget("Frequencies", mw)
    dock.setWidget(vis)
    mw.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    mw.show()
    app.processEvents()
    vis.list_freq.setCurrentItem(vis.list_freq.topLevelItem(0))
    vis.toggle_play()
    vis.animate_frame()
    assert not np.allclose(mol.GetConformer().GetPositions(), base)
    dock.close()
    app.processEvents()
    assert not vis.timer.isActive()
    assert not vis.is_playing
    np.testing.assert_allclose(mol.GetConformer().GetPositions(), base)
