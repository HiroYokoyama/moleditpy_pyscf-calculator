# PySCF Calculator — Test Suite

277 tests across 24 files. All run headlessly — no display server, no PySCF
installation, no real Qt required. PyQt6, PySCF, and RDKit are fully mocked
at module level in each file that needs them.

Tests that exercise the *real* PySCF library are decorated with
`@pytest.mark.skipif(not HAS_PYSCF, ...)` and are skipped automatically when
pyscf is absent.

---

## Running the tests

```bash
# Full suite
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=pyscf_calculator --cov-report=term-missing

# Single file
python -m pytest tests/test_worker_single_point.py -v

# Single test by name
python -m pytest tests/ -k "TestSinglePointCompletion"
```

---

## Test file summary

| File | Tests | Area |
|---|---|---|
| `test_plugin_integration.py` | 25 | PluginContext contract (stub + real) |
| `test_worker_single_point.py` | ~20 | End-to-end single-point calculation |
| `test_worker_pyscf_availability.py` | ~20 | Worker with/without PySCF |
| `test_worker_numeric_hessian.py` | ~15 | Finite-difference Hessian |
| `test_worker_property_worker.py` | ~15 | HOMO/LUMO detection, task loop |
| `test_worker_capture_stdout.py` | ~20 | CaptureStdOut + StreamToSignal |
| `test_worker_streams.py` | ~10 | StreamToSignal signal routing |
| `test_worker_load_worker.py` | — | LoadWorker lifecycle |
| `test_stop_stability.py` | ~15 | Stop-calculation robustness |
| `test_init.py` | ~10 | Plugin init, save/load persistence |
| `test_init_show_dialog.py` | ~10 | Dialog lifecycle |
| `test_calc_tab.py` | ~10 | Calculation tab config builder |
| `test_gui.py` | ~10 | Main dialog internal state |
| `test_scan_dialog.py` | ~10 | Scan config dialog |
| `test_scan_results.py` | ~10 | Scan result analysis, CSV export |
| `test_energy_diag.py` | ~5 | Orbital energy diagram |
| `test_energy_diag_branches.py` | ~5 | Energy diagram edge cases |
| `test_tddft_table.py` | ~10 | TDDFT results table (100% coverage) |
| `test_utils.py` | ~10 | Core utility functions |
| `test_utils_branches.py` | ~10 | Utility edge cases |
| `test_parse_cube_data.py` | — | Cube file parsing |
| `test_vis_build_grid.py` | ~15 | Grid construction from cube metadata |
| `test_freq_vis_spectrum.py` | ~15 | Gaussian/Lorentzian spectrum convolution |
| `test_freq_vis_normalizer.py` | ~15 | Frequency normalisation, SpectrumWidget |

---

## Test files — detailed

### `test_plugin_integration.py` — PluginContext contract (25 tests)

Tests the integration boundary between the plugin and the MoleditPy host.
Uses a two-tier strategy — see [Integration Test Strategy](#integration-test-strategy).

| Class | What it covers |
|---|---|
| `TestInitializeRegistrations` | `initialize(context)` registers menu action `"Extensions/PySCF Calculator..."`, save/load/reset handlers |
| `TestSaveLoadState` | Save handler returns live `PLUGIN_SETTINGS` reference; load clears-and-updates; round-trip preserves data; empty load clears all |
| `TestDocumentReset` | Reset clears `associated_filename`, `calc_history`, `struct_source`; preserves unrelated keys; safe on empty settings |
| `TestShowDialogViaContext` | First call creates `PySCFDialog` with correct args; second call (visible) raises/activates without creating a new instance |
| `TestWithRealPluginContext` *(skipped unless main app present)* | `initialize()` does not raise with real `PluginContext`; real class has all methods used by the plugin |

---

### `test_worker_pyscf_availability.py` — Worker with/without PySCF

| Class | What it covers |
|---|---|
| `TestWorkerWithoutPySCF` | `PySCFWorker.run()` and `PropertyWorker.run()` emit `error_signal`, do **not** emit `finished_signal` when `pyscf is None` |
| `TestWorkerMolBuildFailure` | `gto.M()` raising `ValueError` or `RuntimeError` → `error_signal("Molecule Build Failed: …")`; no `finished_signal` |
| `TestWorkerSpinParsing` | `"N (Label)"` → `2S = N-1`; plain int strings and invalid strings fall back correctly |
| `TestWorkerMolBuildSuccessPath` | `mol.build()` called; `mol.stdout` assigned; exception after build → `error_signal`; `job_1` taken → `job_2`; `threads=4` → `pyscf.lib.num_threads(4)` |
| `TestWorkerWithRealPySCF` *(skipped if pyscf absent)* | Real `gto.M` builds H₂; real RHF energy converges near −1.117 Hartree |

**Design note:** `_load_worker_with_pyscf_mock(pyscf_value)` force-reinstalls Qt
stubs before each load so `test_worker_streams.py`'s `QThread=MagicMock()`
cannot corrupt `PySCFWorker`'s base class.

---

### `test_worker_single_point.py` — End-to-end single-point run

All PySCF mocked; uses a real temp directory so file writes succeed.

| Class | What it covers |
|---|---|
| `TestSinglePointCompletion` | `finished_signal` + `result_signal` emitted; no `error_signal`; result has `chkfile`, `out_dir`, `cube_files=[]`; `pyscf_input.py` written in `job_1/` |
| `TestSinglePointMOExtraction` | RHF 1-D `mo_energy` → `scf_type="RHF"`; UHF tuple → `scf_type="UHF"` + nested lists; `mo_energy=None` → empty lists + warning |
| `TestMethodSelection` | RHF + `spin="1"` stays RHF; RHF + `spin="2"` auto-switches to UHF; `method="RKS"` dispatches `dft.RKS` |
| `TestSolventSetup` | `solvent="Water"` uses `eps=78.2`, logs name; vacuum default leaves `ddCOSMO` uncalled |
| `TestOuterExceptionHandler` | `mol.build()` raising plain `Exception` propagates to outer handler → `error_signal` |

---

### `test_worker_numeric_hessian.py` — Finite-difference Hessian

| Class | What it covers |
|---|---|
| `TestNumericHessianStop` | `_stop_requested=True` at entry → `InterruptedError`; log contains "stopped" |
| `TestNumericHessianCompute` | Shape `(n,3,n,3)`; symmetry `H[i,j,k,l]==H[k,l,i,j]`; zero-gradient → zero Hessian; progress log per atom |
| `TestNumericHessianFallback` | `as_scanner()` raises → manual fallback returns valid-shaped Hessian |
| `TestNumericHessianAtomCoordsFallback` | `mol.atom_coords(unit='Bohr')` raises `TypeError` (older PySCF) → fallback `* 1.8897` still symmetric |

---

### `test_worker_property_worker.py` — HOMO/LUMO detection and task loop

| Class | What it covers |
|---|---|
| `TestPropertyWorkerNoPySCF` | `pyscf=None` → `error_signal` emitted; `finished_signal` **not** emitted |
| `TestPropertyWorkerEmptyTasks` | Empty task list → `finished_signal`; `result["files"]=[]`; no `error_signal` |
| `TestHomoLumoDetection` | RHF 1-D, UHF tuple `(alpha, beta)`, ROKS 2-D, plain list, all-occupied (`lumo_idx=-1` guard) all complete without error |
| `TestPropertyWorkerStop` | `_stop_requested=True` before loop → exits early; `files=[]`; no error |

---

### `test_worker_capture_stdout.py` — CaptureStdOut and StreamToSignal

| Class | What it covers |
|---|---|
| `TestCaptureStdOutInit` | Filename stored; `_saved_fd1`/`_saved_fd2` initialised to `None` |
| `TestCaptureStdOutEnter` | Returns open log file; OS-level FDs saved and redirected; file opened in append mode |
| `TestCaptureStdOutExitErrors` | Closed FD during restore → `WARNING` logged, no raise; flush error swallowed; `log_file=None` after exit |
| `TestCaptureStdOutEnterFileFallback` | `sys.__stdout__.fileno()` raises `UnsupportedOperation` → fallback FD=1 |
| `TestCaptureStdOutExitFlushErrors` | `sys.stdout.flush()` raising in `__exit__` is silently swallowed |
| `TestStreamToSignalFlush` | `flush()` delegates to target; no-target case does not raise; errors swallowed |
| `TestStreamToSignalIsatty` | Returns `False` with no target; delegates to target's `isatty()`; `False` when target has no `isatty` |

---

### `test_worker_streams.py` — StreamToSignal basics

Covers `write()` signal emission, fallback stream forwarding, `close()` setting
`_destroyed`, RuntimeError swallowing, and the `encoding` property.

---

### `test_stop_stability.py` — Stop-calculation robustness

| Class | What it covers |
|---|---|
| `TestCaptureStdOutExit` | FD restoration is independent per-FD; failures logged without raising |
| `TestStreamToSignal` | `write()` always forwards to target stream; signal gate on `_destroyed` |
| `TestPySCFWorkerFields` | `_stop_requested=False` and `_stream=None` at construction |
| `TestOtherWorkersFields` | `PropertyWorker` and `LoadWorker` stop flags initialise correctly |
| `TestOnWorkerStoppedIdempotent` | Double-call of `_on_worker_stopped` is safe |
| `TestStopCalculationSequence` | Stream closed before signal disconnect; `terminate()` only called on timeout |
| `TestSafeStopWorker` | `_safe_stop_worker` sets stop flag, calls `terminate()` on timeout |

---

### `test_init.py` — Plugin initialisation and persistence

Covers plugin metadata constants, `initialize()` registering all handlers,
save/load round-trip, and `handle_reset()` clearing the correct keys.

---

### `test_init_show_dialog.py` — Dialog lifecycle

| Class | What it covers |
|---|---|
| `TestShowDialogNewInstance` | First call creates `PySCFDialog`, calls `show()`, passes correct args |
| `TestShowDialogExistingVisible` | Second call when visible → `raise_()` + `activateWindow()`; no new dialog |
| `TestShowDialogReplaceHidden` | Hidden dialog → `close()` + `deleteLater()`; fresh dialog created |
| `TestShowDialogRuntimeError` | `RuntimeError` from `isVisible()` (C++ deleted) swallowed; new dialog created |

---

### `test_calc_tab.py` — Calculation tab config builder

Covers `build_config()` for standard DFT, solvent, and scan jobs.

---

### `test_gui.py` — Main dialog internal state

Covers `update_internal_state()` reading from CalcTab widgets and
`apply_defaults()` pushing defaults to the UI.

---

### `test_scan_dialog.py` — Scan configuration dialog

Covers valid/invalid scan parameter acceptance and UI enable/disable state.

---

### `test_scan_results.py` — Scan result analysis

Covers absolute Hartree and relative kJ/mol plot data, and CSV export.

---

### `test_energy_diag.py` / `test_energy_diag_branches.py` — Orbital energy diagram

Covers RHF and UHF data initialisation, and edge-case branches.

---

### `test_tddft_table.py` — TDDFT results table (100% coverage)

Covers `__init__`, `populate()` with scaling, and `save_csv()` including
empty-data and exception paths.

---

### `test_utils.py` — Core utility functions

Covers `get_unique_path`, `rdkit_to_xyz`, and `update_molecule_from_xyz` for
PDB, standard XYZ, and raw atom formats.

---

### `test_utils_branches.py` — Utility edge cases

| Class | What it covers |
|---|---|
| `TestGetUniquePathMultiple` | `get_unique_path` loops past `_1`, `_2` … until a free slot |
| `TestUpdateMolBondDetermination` | `ImportError` from `rdDetermineBonds` and generic `Exception` from `DetermineConnectivity` swallowed silently |
| `TestUpdateMolMarkModified` | `mark_modified=False` reads/restores `state_manager.has_unsaved_changes`; `mark_modified=True` skips suppression |
| `TestUpdateMolNoneResult` | `MolFromXYZBlock` returns `None` → `context.current_molecule` is **not** assigned |

---

### `test_vis_build_grid.py` — Grid construction from cube metadata

| Class | What it covers |
|---|---|
| `TestBuildGridUnitConversion` | Origin converted Bohr → Å; `is_angstrom_header=True` skips conversion; `x_vec` scaled by cell size; zero origin stays zero |
| `TestBuildGridDimensions` | `grid.dimensions` set to `[nx, ny, nz]`; total points = `nx×ny×nz` |
| `TestBuildGridPointData` | `"values"` key present; point count matches `nx×ny×nz`; F-order reshape |
| `TestBuildGridAnisotropic` | Non-cubic voxels produce correct point spacing per axis |

---

### `test_freq_vis_spectrum.py` — Gaussian / Lorentzian spectrum convolution

Uses `SpectrumWidget.__new__` + manual attribute initialisation to bypass
`QWidget.__init__` (no running `QApplication` needed).

| Class | What it covers |
|---|---|
| `TestRecalcCurveEmpty` | Empty `freqs` → early return; `curve_x`/`curve_y` unchanged |
| `TestRecalcCurveGaussian` | X-grid length 1000; peak within 10 cm⁻¹ of target; non-negative; two separated peaks both visible; FWHM at exp(−0.5) point (3% tolerance) |
| `TestRecalcCurveLorentzian` | Peak at target; height equals intensity; non-negative; half-width-at-half-max verified (2% tolerance) |
| `TestSetParams` | Wider sigma → wider half-max; Gaussian ↔ Lorentzian switch; different `max_wn` changes grid endpoint |

**Design note:** Qt stubs are force-installed (`sys.modules[k] = v`) to prevent
cross-file stub pollution when test files share a process.

---

### `test_freq_vis_normalizer.py` — Frequency normalisation and SpectrumWidget

| Class | What it covers |
|---|---|
| `TestFreqVisualizerNormalization` | Float list; list/tuple/ndarray single-element unwrapping; empty inner list → `0.0`; complex zero-imag → real; complex imag-only → `-(abs(imag))`; all outputs are `float`; empty input; `modes` stored as ndarray |
| `TestSpectrumWidgetInit` | Real constructor sets `freqs`, `intensities`, `width_val=20.0`, `max_wn=4000.0`, `use_gaussian=True`; `recalc_curve()` called → `curve_x`/`curve_y` length 1000 |
| `TestSpectrumWidgetSetParams` | `set_params()` updates all params; triggers recalculation; both line shapes produce non-zero peaks |

**Design note:** `QPalette` requires a proper class stub
(`class _QPalette: class ColorRole: Base = 0`) — not a plain `MagicMock` —
because `freq_vis.py` accesses `QPalette.ColorRole.Base` as a class attribute.

---

## Integration Test Strategy

### Two-tier approach

```
Tier 1 — Stub mode (always runs, no main app required)
  StubPluginContext mirrors the PluginContext API surface.
  Catches interface mismatches without any external dependency.

Tier 2 — Real-context mode (runs when main app source is available)
  Uses the actual PluginContext class from python_molecular_editor.
  Catches subtle incompatibilities not visible from the stub.
```

### Local development

Real-context tests activate automatically when the repos are siblings:

```
<parent>/
    moleditpy_pyscf-calculator/   ← this plugin
    python_molecular_editor/      ← main app (sibling)
```

### CI

| Job | Triggers | Main app | Tests run |
|---|---|---|---|
| `test` | every push / PR | No | Full suite; `TestWithRealPluginContext` skipped |
| `test-integration` | every push / PR | Cloned `--depth 1` | Full suite including `TestWithRealPluginContext` |

---

## Mocking strategy

Each test file installs its own stubs at **module level** (before any plugin
import) so collection order does not matter. Files that load worker classes call
`_install_qt_stubs(force=True)` immediately before `spec.loader.exec_module()`
to guarantee a proper `QThread` base class even when another file has replaced
`QThread` with a `MagicMock`.

---

## Coverage summary

| Module | Coverage |
|---|---|
| `tddft_table.py` | **100%** |
| `__init__.py` | ~95% |
| `utils.py` | ~88% |
| `freq_vis.py` | ~47% |
| `vis.py` | ~47% |
| `worker.py` | ~33% |
| `gui.py` | ~31% |
| `scan_results.py` | ~22% |
| `calc_tab.py` | ~22% |
| `scan_dialog.py` | ~21% |
| `energy_diag.py` | ~22% |
| `vis_tab.py` | 0% |
| **Overall** | **~25%** |

Low overall coverage is expected: `vis_tab.py`, `vis.py`, and `freq_vis.py`
contain Qt widget code that requires a live display; `worker.py`'s calculation
body requires a functioning PySCF installation with real molecule data.
