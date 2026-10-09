import ast
from types import SimpleNamespace

from audit_helpers import load_module


def test_reproduction_script_carries_guess_threads_and_escaped_literals(monkeypatch, tmp_path):
    mod = load_module(monkeypatch, "worker")
    config = {"method": "UKS", "functional": "functional'with\\escape\nraise RuntimeError('injected')", "basis": "basis'quoted", "spin": 1, "break_symmetry": True, "threads": 4}
    worker = mod.PySCFWorker("", config)
    worker.out_dir = str(tmp_path)
    worker._write_input_script("H 0 0 0\nH 0 0 1", "UKS", config["functional"], SimpleNamespace(), 4, 0)
    script = (tmp_path / "pyscf_input.py").read_text(encoding="utf-8")
    ast.parse(script)
    assert not any(isinstance(node, ast.Raise) for node in ast.parse(script).body)
    assert "lib.num_threads(4)" in script
    assert "mf.kernel(dm0=dm0)" in script
    assert "dm0[1, ao_start:ao_end, ao_start:ao_end] = 0.0" in script
    assert repr(config["basis"]) in script
    assert repr(config["functional"]) in script
