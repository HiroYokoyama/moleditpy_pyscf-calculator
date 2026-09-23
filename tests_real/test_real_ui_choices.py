"""Every functional and basis set the Calculation tab offers must run
through the plugin on real PySCF. PySCF 2.14 rejected two UI entries
(m11 by name, wb97x-d outright) -- a user could pick them and only get a
traceback. The lists are read from calc_tab.py, so a new entry is covered
automatically."""

import ast
import pathlib

import pytest

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("rdkit")
pytest.importorskip("PyQt6.QtCore")

from conftest import XYZ_H2

CALC_TAB = (
    pathlib.Path(__file__).resolve().parent.parent / "pyscf_calculator" / "calc_tab.py"
)


def _combo_items(attr):
    """Literal list passed to self.<attr>.addItems([...]) in calc_tab.py."""
    tree = ast.parse(CALC_TAB.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "addItems"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == attr
        ):
            return ast.literal_eval(node.args[0])
    raise AssertionError(f"{attr}.addItems not found")


FUNCTIONALS = _combo_items("functional_combo")
BASES = _combo_items("basis_combo")


@pytest.mark.parametrize("functional", FUNCTIONALS)
def test_every_offered_functional_runs(functional, run_job):
    res = run_job(XYZ_H2, method="RKS", functional=functional, grid_level=1)
    assert not res.errors, res.errors
    assert res.finished


@pytest.mark.parametrize("basis", BASES)
def test_every_offered_basis_runs(basis, run_job):
    res = run_job(XYZ_H2, basis=basis)
    assert not res.errors, res.errors
    assert res.finished
