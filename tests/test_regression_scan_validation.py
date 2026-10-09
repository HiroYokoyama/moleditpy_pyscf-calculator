import pytest

from audit_helpers import load_module


@pytest.mark.parametrize("changes", [
    {"start": float("nan")}, {"end": float("inf")}, {"start": 0},
    {"atoms": [0, 0]}, {"atoms": [0, 2]}, {"atoms": [-1, 1]},
    {"steps": 1}, {"steps": 2.5}, {"type": "Angle", "atoms": [0, 1, 2], "end": 180},
])
def test_invalid_scan_config_is_rejected(monkeypatch, changes):
    mod = load_module(monkeypatch, "utils")
    params = {"type": "Dist", "atoms": [0, 1], "start": .7, "end": .9, "steps": 2}
    params.update(changes)
    with pytest.raises(ValueError):
        mod.validate_scan_params(params, 2)


def test_wrapped_dihedral_and_backward_scan_are_valid(monkeypatch):
    mod = load_module(monkeypatch, "utils")
    mod.validate_scan_params({"type": "Dihedral", "atoms": [0, 1, 2, 3], "start": 360, "end": -360, "steps": 10}, 4)
