"""
tests/test_utils.py
Unit tests for pyscf_calculator utilities.
"""
import os
import sys
import unittest
import importlib.util
from unittest.mock import MagicMock, patch

def _load_module_direct(relpath, module_name):
    src = os.path.join(os.path.dirname(__file__), "..", relpath)
    src = os.path.normpath(src)
    spec = importlib.util.spec_from_file_location(module_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Mock rdkit heavily because we want tests to pass purely standalone
sys.modules["rdkit"] = MagicMock()
sys.modules["rdkit.Chem"] = MagicMock()

utils = _load_module_direct(
    os.path.join("pyscf_calculator", "utils.py"),
    "pyscf_calculator_utils_under_test",
)

class TestUtils(unittest.TestCase):
    def test_get_unique_path_new(self):
        with patch('os.path.exists', return_value=False):
            self.assertEqual(utils.get_unique_path("test.xyz"), "test.xyz")

    def test_get_unique_path_existing(self):
        def _mock_exists(p):
            # simulate "test.xyz" exists, but "test_1.xyz" does not
            if p == "test.xyz": return True
            return False
            
        with patch('os.path.exists', side_effect=_mock_exists):
            self.assertEqual(utils.get_unique_path("test.xyz"), "test_1.xyz")

    def test_rdkit_to_xyz_none(self):
        self.assertEqual(utils.rdkit_to_xyz(None), "")

    def test_rdkit_to_xyz_valid_mol(self):
        # Create a mock rdkit Mol object
        mol_mock = MagicMock()
        mol_mock.GetNumAtoms.return_value = 2
        
        conf_mock = MagicMock()
        
        pos0 = MagicMock(); pos0.x, pos0.y, pos0.z = 0.0, 0.0, 0.0
        pos1 = MagicMock(); pos1.x, pos1.y, pos1.z = 1.0, 0.0, 0.0
        conf_mock.GetAtomPosition.side_effect = [pos0, pos1]
        
        mol_mock.GetConformer.return_value = conf_mock
        
        atom0 = MagicMock(); atom0.GetSymbol.return_value = "H"
        atom1 = MagicMock(); atom1.GetSymbol.return_value = "O"
        mol_mock.GetAtomWithIdx.side_effect = [atom0, atom1]
        
        xyz = utils.rdkit_to_xyz(mol_mock)
        lines = xyz.split("\n")
        self.assertEqual(lines[0], "2")
        self.assertIn("H 0.000000 0.000000 0.000000", xyz)
        self.assertIn("O 1.000000 0.000000 0.000000", xyz)

if __name__ == '__main__':
    unittest.main()
