import pytest

from audit_helpers import load_module


@pytest.mark.parametrize("dimensions,data", [
    ((0, 1, 1), ""), ((1000, 1000, 1000), "0"),
    ((2, 1, 1), "0"), ((1, 1, 1), "nan"), ((1, 1, 1), "broken"),
])
def test_invalid_cube_rejected_before_rendering(monkeypatch, tmp_path, dimensions, data):
    mod = load_module(monkeypatch, "vis")
    path = tmp_path / "invalid.cube"
    nx, ny, nz = dimensions
    path.write_text(f"comment\ncomment\n0 0 0 0\n{nx} 1 0 0\n{ny} 0 1 0\n{nz} 0 0 1\n{data}", encoding="utf-8")
    with pytest.raises(ValueError):
        mod.parse_cube_data(str(path))


@pytest.mark.parametrize("origin,atom,dataset,data", [
    ("nan 0 0", "1 0 0 0 0", "1 1", "0"),
    ("0 0 0", "1 0 inf 0 0", "1 1", "0"),
    ("0 0 0", "1 0 0 0 0", "1 bogus", "0"),
    ("0 0 0", "1 0 0 0 0", "1 1", "garbage\n0"),
])
def test_malformed_cube_metadata_is_rejected(monkeypatch, tmp_path, origin, atom, dataset, data):
    mod = load_module(monkeypatch, "vis")
    path = tmp_path / "metadata.cube"
    path.write_text(f"comment\ncomment\n-1 {origin}\n1 1 0 0\n1 0 1 0\n1 0 0 1\n{atom}\n{dataset}\n{data}", encoding="utf-8")
    with pytest.raises(ValueError):
        mod.parse_cube_data(str(path))
