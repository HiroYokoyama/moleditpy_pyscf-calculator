# PySCF Calculator — Test Suite

## Overview

All tests run headlessly without a display server and without requiring PySCF or
the real Qt library to be installed.  PyQt6, PySCF, and RDKit are fully mocked
at the Python module level.

Tests that exercise the *real* PySCF library are decorated with
`@pytest.mark.skipif(not HAS_PYSCF, reason="pyscf not installed")` and are
skipped automatically in environments where pyscf is absent.

## Running

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=pyscf_calculator --cov-report=term-missing

# Run a single file
python -m pytest tests/test_worker_pyscf_availability.py -v
```

## Test Files

### test_worker_pyscf_availability.py
**PySCF optional — worker with/without pyscf**

| Class | What is tested |
|---|---|
| `TestWorkerWithoutPySCF` | `PySCFWorker.run()` and `PropertyWorker.run()` emit `error_signal` and do **not** emit `finished_signal` when `pyscf is None` |
| `TestWorkerMolBuildFailure` | `PySCFWorker.run()` emits `error_signal("Molecule Build Failed: …")` for both `ValueError` and `RuntimeError` from `gto.M()`, and does not emit `finished_signal` |
| `TestWorkerSpinParsing` | Spin multiplicity string `"N (Label)"` is converted to `2S = N-1`; plain integer strings and invalid strings fall back correctly |
| `TestWorkerWithRealPySCF` *(skipped if pyscf absent)* | Real `gto.M` builds H₂; real RHF energy converges near −1.117 Hartree |

**Design note:** The helper `_load_worker_with_pyscf_mock(pyscf_value)` force-reinstalls Qt stubs before each load to prevent `test_worker_streams.py`'s `QThread=MagicMock()` from corrupting the base class of `PySCFWorker`.

---

### test_init_show_dialog.py
**Plugin dialog lifecycle**

| Class | What is tested |
|---|---|
| `TestShowDialogNewInstance` | First call creates a `PySCFDialog`, calls `show()`, passes the main window as first arg and the shared settings dict as `settings=` kwarg |
| `TestShowDialogExistingVisible` | Second call when dialog `isVisible()` calls `raise_()` + `activateWindow()` and does **not** create a new dialog |
| `TestShowDialogReplaceHidden` | When existing dialog is not visible, it is `close()`d + `deleteLater()`d and a fresh dialog is created |
| `TestShowDialogRuntimeError` | `RuntimeError` from `isVisible()` (C++ object deleted) is swallowed and a new dialog is created |

---

### test_utils_branches.py
**Utility edge cases**

| Class | What is tested |
|---|---|
| `TestGetUniquePathMultiple` | `get_unique_path` loops past `_1`, `_2` … until a free slot is found |
| `TestUpdateMolBondDetermination` | `ImportError` from `rdDetermineBonds` and generic `Exception` from `DetermineConnectivity` are swallowed silently |
| `TestUpdateMolMarkModified` | `mark_modified=False` reads and restores `state_manager.has_unsaved_changes`; calls `update_window_title()`; `mark_modified=True` skips the suppression path entirely |
| `TestUpdateMolNoneResult` | When `MolFromXYZBlock` returns `None`, `context.current_molecule` is **not** assigned |

---

### test_worker_streams.py
**StreamToSignal — basic signal/stream routing**

Covers `write()` signal emission, fallback stream forwarding, `close()` setting `_destroyed`, RuntimeError swallowing, and the `encoding` property.

---

### test_stop_stability.py
**Stop-calculation robustness**

| Class | What is tested |
|---|---|
| `TestCaptureStdOutExit` | FD restoration is independent per-FD; failures are logged without raising |
| `TestStreamToSignal` | `write()` always forwards to target stream; signal gate on `_destroyed` |
| `TestPySCFWorkerFields` | `_stop_requested=False` and `_stream=None` at construction |
| `TestOtherWorkersFields` | `PropertyWorker` and `LoadWorker` cooperative stop flags initialise correctly |
| `TestOnWorkerStoppedIdempotent` | Double-call of `_on_worker_stopped` is safe (idempotent) |
| `TestStopCalculationSequence` | Stream is closed before signal disconnect; `terminate()` only called on timeout |
| `TestSafeStopWorker` | `_safe_stop_worker` sets stop flag and calls `terminate()` on timeout |

---

### test_init.py
**Plugin initialization & persistence**

Covers plugin metadata constants, `initialize()` registering all handlers, save/load round-trip, and `handle_reset()` clearing the correct keys.

---

### test_calc_tab.py
**Calculation tab config builder**

Covers `build_config()` for standard DFT, solvent, and scan jobs.

---

### test_gui.py
**Main dialog internal state**

Covers `update_internal_state()` reading from CalcTab widgets and `apply_defaults()` pushing defaults to the UI.

---

### test_scan_dialog.py
**Surface scan configuration dialog**

Covers valid/invalid scan parameter acceptance and UI enable/disable state.

---

### test_scan_results.py
**Scan result analysis**

Covers absolute Hartree and relative kJ/mol plot data and CSV export.

---

### test_energy_diag.py
**Orbital energy diagram**

Covers RHF and UHF data initialization.

---

### test_tddft_table.py
**TDDFT results table** — 100% coverage

Covers `__init__`, `populate()` with scaling, and `save_csv()` including empty and exception paths.

---

### test_utils.py
**Core utility functions**

Covers `get_unique_path`, `rdkit_to_xyz`, and `update_molecule_from_xyz` for PDB, standard XYZ, and raw atom formats.

---

### test_worker_numeric_hessian.py
**Finite-difference Hessian computation**

| Class | What is tested |
|---|---|
| `TestNumericHessianStop` | `_stop_requested=True` at entry raises `InterruptedError`; log message contains "stopped" |
| `TestNumericHessianCompute` | Shape `(n,3,n,3)`, symmetry `H[i,j,k,l]==H[k,l,i,j]`, zero-gradient → zero Hessian, progress log per atom, start log message |
| `TestNumericHessianFallback` | When `as_scanner()` raises, manual fallback branch returns a valid-shaped Hessian |

---

### test_worker_property_worker.py
**PropertyWorker HOMO/LUMO detection and task loop**

| Class | What is tested |
|---|---|
| `TestPropertyWorkerNoPySCF` | `pyscf=None` → `error_signal` emitted, `finished_signal` **not** emitted |
| `TestPropertyWorkerEmptyTasks` | Empty task list → `finished_signal` emitted, `result_signal` dict contains `"files": []`, no `error_signal` |
| `TestHomoLumoDetection` | RHF 1-D array, UHF tuple `(alpha, beta)`, ROKS 2-D array, plain list, and all-occupied (lumo_idx=-1 guard) all complete without error |
| `TestPropertyWorkerStop` | `_stop_requested=True` before loop → loop exits early, `files=[]`, no error |

---

### test_plugin_integration.py
**Main-app PluginContext contract**

Tests the integration boundary between the plugin and the MoleditPy host application.
Two execution modes (see [Integration Test Strategy](#integration-test-strategy) below).

| Class | What is tested |
|---|---|
| `TestInitializeRegistrations` | `initialize(context)` registers menu action at `"Extensions/PySCF Calculator..."`, plus save/load/reset handlers |
| `TestSaveLoadState` | Save handler returns live `PLUGIN_SETTINGS` reference; load handler clears-and-updates; round-trip preserves data; empty load clears all |
| `TestDocumentReset` | Reset handler clears `associated_filename`, `calc_history`, `struct_source`; preserves unrelated keys; safe on empty settings |
| `TestShowDialogViaContext` | First call creates `PySCFDialog` with correct args; second call (visible dialog) raises/activates without creating new instance |
| `TestWithRealPluginContext` *(skipped unless main app present)* | `initialize()` does not raise with the real `PluginContext`; real class has all methods used by the plugin |

---

## Integration Test Strategy

### Goal

Verify that the plugin's `initialize(context)` correctly honours the
**PluginContext contract** defined by the MoleditPy host application, so that
breaking changes in either side are caught automatically.

### Two-tier approach

```
Tier 1 — Stub mode (always runs, including GitHub CI)
  StubPluginContext mirrors the PluginContext API surface.
  No main app required.  Catches interface mismatches immediately.

Tier 2 — Real-context mode (runs when main app source is available)
  Uses the actual PluginContext class from python_molecular_editor.
  Catches subtle incompatibilities not visible from the stub.
```

### Local development

The real-context tests activate automatically when the main app is found at:

```
../python_molecular_editor/moleditpy/src      ← relative to this repo
```

If the plugin repo and the main app are siblings in the same parent folder:
```
<parent>/
    moleditpy_pyscf-calculator/   ← this plugin
    python_molecular_editor/      ← main app (sibling folder)
```

All 25 integration tests including the 3 real-context tests will run.

### GitHub Actions CI

The workflow has two jobs:

| Job | Triggers | What runs |
|---|---|---|
| `test` | every push/PR | Full suite, stub-only integration tests |
| `test-integration` | every push/PR | Clones main app from GitHub, runs integration tests with real `PluginContext` |

The `test-integration` job clones:
```
https://github.com/HiroYokoyama/python_molecular_editor
```
and sets `CI_MAIN_APP_SRC` so `test_plugin_integration.py` finds the real
`PluginContext`.

### StubPluginContext design

The stub captures all registrations:

```python
ctx.menu_actions   # dict: path → callback
ctx.save_handlers  # list of callables
ctx.load_handlers  # list of callables
ctx.reset_handlers # list of callables
```

Every method called by `initialize()` is implemented on the stub.  All other
`PluginContext` methods are present as no-ops so the stub can also be passed to
code that calls less-common APIs.

### Detecting regressions

If the plugin changes `initialize()` to call a new `PluginContext` method:
- `TestInitializeRegistrations` will catch missing registrations
- `TestWithRealPluginContext.test_stub_interface_matches_real` will catch methods
  missing from the real class

If the main app renames or removes a `PluginContext` method:
- `TestWithRealPluginContext.test_stub_interface_matches_real` will fail in the
  integration CI job, giving an early warning before a user upgrade would break

---

## Coverage Summary (as of last run)

| Module | Coverage |
|---|---|
| `tddft_table.py` | **100%** |
| `__init__.py` | **95%** |
| `utils.py` | **88%** |
| `worker.py` | ~25% |
| `gui.py` | 31% |
| `scan_results.py` | 22% |
| `calc_tab.py` | 22% |
| `scan_dialog.py` | 21% |
| `energy_diag.py` | ~40% |
| `vis.py` | ~35% |
| `vis_tab.py` | 0% |
| `freq_vis.py` | 0% |
| **Total** | **~22%** |

The low overall coverage is expected: `vis.py`, `vis_tab.py`, and `freq_vis.py` contain Qt widget code that requires a live display to instantiate meaningfully, and `worker.py`'s calculation body requires a functioning PySCF installation with real molecule data.

## Mocking Strategy

Each test file installs its own stubs at **module level** so that collection order
does not matter.  Files that load worker classes call `_install_qt_stubs(force=True)`
immediately before `spec.loader.exec_module()` to guarantee a proper `QThread`
base class even when another test file has installed `QThread = MagicMock()`.
