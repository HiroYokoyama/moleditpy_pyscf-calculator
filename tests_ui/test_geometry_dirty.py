import pytest


@pytest.mark.parametrize("persistent", [True, False])
def test_geometry_dirty_state_matches_import_policy(plugin, context, persistent):
    plugin("utils").update_molecule_from_xyz(context, "2\nupdated\nH 0 0 0\nH 0 0 1", mark_modified=persistent)
    assert context.get_main_window().state_manager.has_unsaved_changes is persistent
