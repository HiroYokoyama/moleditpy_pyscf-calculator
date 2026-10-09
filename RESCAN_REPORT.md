# PySCF Calculator follow-up audit

Reviewed 2026-10-09 after fixing the findings in [AUDIT_REPORT.md](AUDIT_REPORT.md).

The second source scan covered all plugin modules, calculation dispatch, thread ownership and cancellation, result loading, dialog/timer teardown, persistence, input generation, cube parsing, tests, and CI. All 13 original findings and the three additional robustness observations have fixes. The second scan also found and corrected missing validation of nonfinite cube coordinates, malformed dataset identifiers, and garbage before volumetric data. No further confirmed actionable defects remained in the reviewed scope. This is not a guarantee that all possible defects have been found.

## Separate fix commits

| Fix | Commit |
| --- | --- |
| Persist TDDFT state count | `b87062f` |
| Mark imported geometry as modified | `3ca74ea` |
| Stop scan selection timer on every exit | `d3860c9` |
| Stop frequency animation on dock close | `c094e29` |
| Respect scan history geometry import policy | `8f43623` |
| Pair regenerated ESP/density files | `0ee0a6e` |
| Reproduce and quote generated input settings | `faaea1a` |
| Allocate job directories atomically | `d0eb198` |
| Reject corrupt and oversized cube files | `1dbe0ea` |
| Validate scan parameters and trajectory lengths | `1848993` |
| Apply broken symmetry throughout job dispatch | `28ff4da` |
| Serialize output capture and restore partial setup | `302443d` |
| Retain calculation thread ownership | `1e49c70` |
| Recover property controls after failure | `624f999` |
| Cancel cooperatively and defer teardown | `3402f7e` |
| Clear stale results and delayed geometry callbacks | `a16b997` |
| Support log sinks without cancellation state | `8d4510d` |
| Emulate thread-finished signals in synchronous GUI tests | `bf88e6f` |

Every fix commit includes `Assisted-by: GPT-6.1 Sol`. Regression tests accompany the fixes, including real Qt thread, dock, timer, and project-state tests. Tests use real RDKit molecules for geometry lifecycle checks. Linux/macOS CI runs the new real PySCF broken-symmetry regressions and runs Qt tests in a separate process from the mocked suite.

## Local validation

- `tests/`: **934 passed, 3 skipped, 0 failures/errors** (937 collected).
- `tests_ui/`: **16 passed, 0 failures/errors**, using real offscreen PyQt6.
- `tests_real/`: **12 modules skipped**, because PySCF is not installed in this Windows environment. Scientific regression outcomes require Linux/macOS CI.
- Python compilation passed. Diff whitespace checks passed with the repository's existing CRLF convention.

Each suite runs in its own process. TEMP/TMP were redirected to a workspace directory because the default temporary location caused initial test runs to stall.

The first Linux/macOS CI run found two additional failures: a lightweight output-capture sink did not define `_stop_requested`, and the synchronous GUI harness bypassed the native `QThread.finished` notification. The helper now treats the cancellation flag as optional, with a dedicated regression test. The synchronous harness emits the native exit signal in `finally`, and cancellation tests wait for actual completion instead of using a fixed delay. Both follow-up fixes have separate commits. The scientific regressions added for broken-symmetry dispatch passed in the first CI run.

## Practical limits

Cancellation is cooperative: SCF and optimization callbacks stop at safe boundaries, and other native operations may need to finish before the worker exits. The dialog retains the worker while waiting. Output redirection remains process-wide, but plugin capture contexts are serialized and partial setup is rolled back. Full interactive validation inside a running MoleditPy host was not available; host interactions were exercised with test contexts.
