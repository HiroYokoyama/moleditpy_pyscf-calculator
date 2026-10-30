"""XYZ coordinates must survive a comma-decimal numeric locale.

RDKit's Chem.MolFromXYZBlock parses numbers through the C runtime locale and,
under e.g. de_DE, truncates every coordinate to its integer part
(HiroYokoyama/python_molecular_editor#185). utils.mol_from_xyz_block parses the
numbers in Python instead. Other test modules replace rdkit in sys.modules with
mocks, so the real-RDKit check runs in a fresh interpreter.
"""

import os
import subprocess
import sys
import textwrap

import pytest

UTILS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "pyscf_calculator", "utils.py")
)

SCRIPT = textwrap.dedent(
    """
    import importlib.util, locale, sys
    try:
        from rdkit import Chem
    except ImportError:
        sys.exit(77)

    spec = importlib.util.spec_from_file_location("utils_under_test", sys.argv[1])
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)

    for name in ("de_DE.UTF-8", "de_DE.utf8", "de_DE", "de-DE", "German"):
        try:
            locale.setlocale(locale.LC_NUMERIC, name)
        except locale.Error:
            continue
        if locale.localeconv()["decimal_point"] == ",":
            break

    atoms = "O 0.0000 0.0000 0.1173\\nH 0.0000 0.7572 -0.4692\\nH 0.0000 -0.7572 -0.4692"
    for text in ("3\\nwater\\n" + atoms, atoms):
        mol = utils.mol_from_xyz_block(text)
        assert mol is not None and mol.GetNumAtoms() == 3
        y = mol.GetConformer().GetAtomPosition(1).y
        assert abs(y - 0.7572) < 1e-9, y
    assert utils.mol_from_xyz_block("3\\nbad\\nO 0 0") is None
    assert utils.mol_from_xyz_block("Xx 0 0 0") is None
    print("ok")
    """
)


def test_mol_from_xyz_block_keeps_fractional_coordinates_under_comma_locale():
    proc = subprocess.run(
        [sys.executable, "-I", "-c", SCRIPT, UTILS],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    if proc.returncode == 77:
        pytest.skip("rdkit is not installed")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "ok"
