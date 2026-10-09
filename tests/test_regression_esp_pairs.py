from unittest.mock import MagicMock

import numpy as np

from audit_helpers import load_module


def test_incomplete_existing_pairs_do_not_split_new_esp_pair(monkeypatch, tmp_path):
    mod = load_module(monkeypatch, "worker")
    (tmp_path / "esp.cube").touch()
    (tmp_path / "density_1.cube").touch()
    worker = mod.PropertyWorker("unused.chk", ["ESP"], str(tmp_path))
    monkeypatch.setattr(worker, "_spin_density_matrices", lambda *args: (np.eye(2), np.eye(2)))
    files = worker._make_esp(MagicMock(), MagicMock(), np.eye(2), [2, 0])
    assert files == [str(tmp_path / "esp_2.cube"), str(tmp_path / "density_2.cube")]
