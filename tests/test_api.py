import sys
import unittest
import importlib.util
from pathlib import Path

# Paths
_TESTS_DIR = Path(__file__).resolve().parent
_PLUGIN_ROOT = _TESTS_DIR.parent
_WORKSPACE_ROOT = _PLUGIN_ROOT.parent

# Find the main app
_DEFAULT_APP_CANDIDATES = [
    _WORKSPACE_ROOT / "python_molecular_editor",
    _PLUGIN_ROOT / "python_molecular_editor",  # CI path
]
_APP_PATH = next((p for p in _DEFAULT_APP_CANDIDATES if p and (p / "moleditpy").exists()), None)

# Skip conditions
HAS_APP = _APP_PATH is not None

def _load_checker():
    checker_path = _TESTS_DIR / "plugin_api_checker.py"
    spec = importlib.util.spec_from_file_location("plugin_api_checker", checker_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(checker_path)
    spec.loader.exec_module(mod)
    return mod

class TestAPIChecker(unittest.TestCase):
    @unittest.skipUnless(HAS_APP, "Main application repository (python_molecular_editor) not found")
    def test_no_unknown_api_accesses(self):
        checker_mod = _load_checker()
        
        # Build the APIInfo from the main app
        extractor = checker_mod.AppAPIExtractor(_APP_PATH, verbose=False)
        api = extractor.extract()
        
        # Merge allowlists (same as running with --default-allowlist --mw-allowlist + site allowlist)
        site_allowlist = checker_mod._load_site_allowlist(_PLUGIN_ROOT)
        allowlist = checker_mod._merge_allowlists(
            checker_mod._MANAGER_ALLOWLIST,
            checker_mod._MW_ALLOWLIST,
            site_allowlist
        )
        
        # Collect all .py files in the plugin repo, excluding tests and hidden folders
        plugin_files = []
        for p in _PLUGIN_ROOT.rglob("*.py"):
            # Skip files in 'tests' directory or any hidden folders like '.git' or '__pycache__'
            if "tests" in p.parts or any(part.startswith(".") for part in p.parts) or "__pycache__" in p.parts:
                continue
            # Also skip this test file itself and the checker script
            if p.name in ("test_api.py", "plugin_api_checker.py"):
                continue
            plugin_files.append(p)
            
        all_issues = []
        for pf in plugin_files:
            checker = checker_mod.PluginFileChecker(
                pf,
                api,
                check_context=True,  # Check context.X methods too
                allowlist=allowlist
            )
            issues = checker.check()
            if issues:
                all_issues.extend(issues)
        
        # Format and fail if any issues found
        if all_issues:
            lines = [
                f"  [{i.code}] {Path(i.file).relative_to(_PLUGIN_ROOT)} line {i.line}: {i.message}"
                for i in all_issues
            ]
            self.fail(
                f"{len(all_issues)} unknown API access(es) found in {Path(_PLUGIN_ROOT).name}:\n"
                + "\n".join(lines)
            )

if __name__ == "__main__":
    unittest.main()
