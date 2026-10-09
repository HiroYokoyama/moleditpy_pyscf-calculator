# PySCF Calculator bug audit

Reviewed 2026-10-09, version 4.0.3, commit `d89c65ea9fb8d76b49ff8280f626fac32d3b2bba`.

Scope: all 12 plugin Python modules, worker dispatch and scientific options, Qt lifecycle, result persistence/loading, cube parsing/rendering, host API interactions, tests, and CI configuration. Production code was not changed. This is a source review with targeted reproductions, not a guarantee that every defect has been found.

Validation: **903 passed, 3 skipped, 0 failures/errors** in `tests/` (906 collected; JUnit record in `audit_junit.xml`). Initial runs stalled using the default temporary location; rerunning with TEMP/TMP under D:/DEVELOPMENT/DEV_MAIN completed. `tests_real/` skipped all 11 modules because PySCF is unavailable in this Windows environment. Compile checks passed. Real offscreen PyQt6 and RDKit were used for the targeted UI reproductions below. No full quantum-chemistry numerical validation was possible.

P1 = high priority (crash, incorrect scientific workflow, or cross-job corruption). P2 = normal priority (broken behavior, persistence, or lifecycle).

## Findings

### 1. [P1] Forced cancellation can leave process-wide output and native state corrupted

Locations: `calc_tab.py:746`, `gui.py:231`, `worker.py:276`.

Stop waits two seconds, then calls QThread.terminate(); dialog teardown has a similar path. Outside scan/task boundaries, `_stop_requested` is not checked during SCF, optimization, or Hessian computation. Ordinary long calculations therefore reach forced termination. Output restoration depends entirely on Python `finally`/context-manager cleanup, while OS descriptors and `sys.stdout`/`sys.stderr` have been changed for the entire host process. Forced native-thread termination does not guarantee that cleanup or lock release occurs. Subsequent logging/calculations can fail or hang; closing the dialog also proceeds after a bounded wait whose failure is not handled.

Recommendation: run calculations in child processes with controlled cancellation, or implement cooperative callbacks and retain the worker until it actually exits. Do not use forced thread termination as the standard stop mechanism. Qt explicitly documents that terminate is dangerous and prevents normal cleanup: https://doc.qt.io/qt-6/qthread.html#terminate

Evidence: source/API review; destructive forced termination was not reproduced inside the host.

### 2. [P1] Concurrent calculation and property workers race on global output redirection

Locations: `worker.py:276`, `calc_tab.py:699`, `vis_tab.py:897`.

Calculation and property generation can run simultaneously: their guards only inspect their own worker. Both redirect the same process-wide streams/descriptors. If A starts, B starts, then A finishes before B, A restores the original streams while B is active; B subsequently restores A's closed StreamToSignal and descriptors pointing at A's log. Jobs can write into each other's logs and leave the host routed to a completed job. This happens during normal completion, without Stop.

Recommendation: isolate each calculation's logging in a subprocess, or prevent all overlapping redirected-output contexts with a shared job coordinator.

Evidence: source review of the permitted overlap and restoration order.

### 3. [P1] Broken-symmetry option is bypassed for TDDFT, scans, and optimizer initialization

Locations: `worker.py:646`, `worker.py:844`, `worker.py:1026`, `worker.py:1281`, `worker.py:1366`.

The symmetry-broken density is only supplied by `_run_scf`. TDDFT dispatch skips that helper and calls `mf.kernel()` directly. Rigid/relaxed scans also run kernels directly, and optimization starts before `_run_scf` is invoked. A singlet UHF/UKS job with Break Initial Guess Symmetry enabled can therefore calculate TDDFT or an entire scan on the restricted solution, or optimize on that surface before evaluating a broken-symmetry final energy.

Recommendation: use one initialization policy for every job type; seed optimization/scan scanners with the requested broken-symmetry solution.

Evidence: source control-flow review; real numerical reproduction requires PySCF.

### 4. [P1] Thermochemistry from the previous job remains attached to a new energy result

Location: `vis_tab.py:501`.

`on_load_finished` replaces `thermo_data` only when the new result contains it. Load a frequency result, then an energy-only result: Show Properties remains enabled and displays the old job's Gibbs energy, ZPE, and conditions. Missing MO/geometry keys can similarly retain earlier state. These values appear under the newly selected result path.

Recommendation: clear all per-result data and controls before populating the new result; associate displayed data explicitly with its source job.

Evidence: reproduced using real Qt VisTab widgets; a new energy payload retained `{'G_tot': [-1.0, 'Eh']}` from the preceding result.

### 5. [P1] Worker references are released on a custom signal before thread exit

Locations: `calc_tab.py:802`, `vis_tab.py:940`, `worker.py:1691`.

Success/error signals trigger UI cleanup that sets the worker reference to None. Those signals are emitted from within `run()`, before return and, for PropertyWorker success, before redirected-output cleanup. The GUI can process the signal while the thread is still running. Dropping its last owning reference can destroy a running QThread, or enable a new calculation while old cleanup is still active. A property/load error can also call CalcTab cleanup while a separate calculation is active (see finding 6).

Recommendation: keep a strong worker reference until Qt's actual `QThread.finished`; use custom result/error signals only to report outcomes. Make cleanup specific to the worker instance.

Evidence: source review; race depends on GUI scheduling and is not reproduced deterministically here.

### 6. [P2] Property-generation errors leave analysis disabled and clean up the wrong tab

Locations: `vis_tab.py:927`, `gui.py:205`.

Property errors connect to the parent dialog's `on_error`, which cleans up CalcTab. The analysis button is re-enabled only on successful `finished_signal`, which PropertyWorker does not emit after an exception. An invalid checkpoint or cube-write failure leaves Run Analysis disabled. If a calculation is running concurrently, that same handler clears its worker reference and enables Run prematurely.

Recommendation: give each worker a dedicated error/outcome handler and perform lifecycle cleanup on its actual thread-finished signal.

Evidence: source review of signal wiring and error paths.

### 7. [P2] Cancel/OK hides ScanDialog without stopping its polling timer

Locations: `scan_dialog.py:76`, `scan_dialog.py:244`, `scan_dialog.py:282`.

Timer shutdown and measurement-mode restoration live in `closeEvent`, but Cancel calls `reject` and OK calls `accept`. These paths hide the dialog without invoking that cleanup. The hidden dialog continues polling and changing selection-dependent state; Cancel can leave measurement mode enabled.

Recommendation: centralize cleanup in `done`/accepted/rejected handling, including restoring the previous measurement mode.

Evidence: reproduced with real Qt: after show -> reject, visible=False and sel_timer.isActive()=True.

### 8. [P2] Closing the frequency dock leaves animation running

Locations: `freq_vis.py:77`, `freq_vis.py:534`, `vis_tab.py:606`.

The dock's close/hide event is not connected to FreqVisualizer.cleanup. Its unparented timer keeps firing and modifying the molecule after the user closes the visible controls. `close_freq_window` also discards references without explicitly calling cleanup.

Recommendation: stop animation and restore base geometry when the dock closes; parent the timer and explicitly clean up before dropping references.

Evidence: reproduced with real Qt and RDKit: after playing -> dock.close(), visible=False, timer active=True, is_playing=True.

### 9. [P2] Geometry imports do not mark an existing project dirty

Locations: `utils.py:145`, `vis_tab.py:1167`.

`mark_modified=True` only means the helper does not restore the old dirty flag; it never marks the project modified. The host's PluginContext.current_molecule setter only replaces/redraws the 3D molecule. Loading an optimized geometry into a saved project can change coordinates while leaving the project clean, so closing it may not prompt to save. Earlier result-loading history changes do not cover clicking Load Geometry later after saving.

Recommendation: explicitly call context.mark_project_modified for persistent geometry updates, and preserve dirty state only for transient viewing.

Evidence: real RDKit helper reproduction left dirty=False; checked the host setter in `python_molecular_editor/moleditpy/src/moleditpy/plugins/plugin_interface.py:226`.

### 10. [P2] Reopening a project whose latest history entry is a scan overwrites its molecule

Locations: `gui.py:409`, `vis_tab.py:322`, `scan_results.py:372`.

History auto-load requests `update_structure=False`, but the scan branch ignores it. ScanResultDialog construction unconditionally installs a reconstructed first-frame molecule and marks the project modified. Merely opening the calculator for a saved project can replace its current geometry/topology with scan frame 1.

Recommendation: propagate the update_structure policy to scan loading; defer editor replacement until the user explicitly selects/views a frame.

Evidence: source review of the auto-load -> scan-dialog constructor path.

### 11. [P2] Regenerated ESP files no longer enter mapped-density visualization

Locations: `worker.py:1730`, `vis_tab.py:1007`.

PropertyWorker avoids overwrites by producing `esp_1.cube` and `density_1.cube` on repeat generation. VisTab recognizes only the exact basename `esp.cube`, so selecting the regenerated ESP uses ordinary positive/negative isosurfaces rather than mapping ESP onto its matching density surface. Independently unique filenames also make matching by suffix insufficient when existing pairs are incomplete.

Recommendation: return/store explicit surface-property pairs and use that metadata when selecting files.

Evidence: source review of generated names and exact-name dispatch.

### 12. [P2] Saved input does not reproduce a broken-symmetry SCF

Location: `worker.py:802`.

The input script always calls `mf.kernel()` and omits the configured broken-symmetry initial density. A singlet UHF/UKS energy job can produce one electronic solution in the plugin and another when rerunning its advertised reproducible input. The Threads option is also only written as a comment, not applied.

Recommendation: serialize the actual initial-guess policy and num_threads call. Document that optimization/scan/frequency workflows are not reproduced by this SCF-only script.

Evidence: compared `_write_input_script` with `_run_scf`.

### 13. [P2] TDDFT state count is not persisted

Locations: `gui.py:40`, `calc_tab.py:663`.

`nstates_input` is used to execute TDDFT, but nstates is absent from the shared `_FIELDS` table used for saving, restoring, and Save as Default. A user choosing 30 states gets the default 10 after reopening the dialog/project.

Recommendation: add nstates to the persisted field table and verify a round trip.

Evidence: source review of configuration and persistence.

## Additional observations

- `_make_job_dir` uses check-then-create with exist_ok=True. Two processes sharing an output root can claim the same job directory. Use atomic mkdir with FileExistsError retry.
- Malformed cube data is silently truncated/padded with zeros; zero-sized grid axes and very large dimensions are not rejected before allocation. This can hide truncated output or exhaust memory while loading a file.
- Scan setup accepts NaN/inf and physically invalid coordinate values; worker entry points do not revalidate atom indices against the current molecule. Configuration can survive changes to the editor molecule within the same document.
- No direct shell execution or Python eval of result-file contents was found in the plugin. Generated Python scripts interpolate editable text without Python-literal escaping; use repr when writing atom/basis/functional strings. This review did not exercise hostile-input scenarios.
- Mocked coverage is useful but missed the real Qt accept/reject and dock-close behaviors reproduced here. Add small real Qt lifecycle tests and real PySCF option-dispatch tests, particularly broken-symmetry scans/TDDFT.

Suggested repair order: cancellation/output isolation and worker ownership; broken-symmetry dispatch and stale scientific data; error recovery and dock/dialog lifecycle; persistence and file pairing.
