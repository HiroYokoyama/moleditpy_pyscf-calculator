import contextlib
import csv
import ctypes
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import traceback

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms
except ImportError:
    pass

# We import pyscf inside the thread or check availability
try:
    import pyscf
    from pyscf import (
        dft,
        gto,
        scf,
        solvent,  # noqa: F401 # Ensure ddCOSMO mixin is available
    )
except ImportError:
    pyscf = None

logger = logging.getLogger(__name__)

_HC_EV_NM = 1239.84193  # hc in eV·nm, for excitation wavelength conversion
# PySCF's own factor, so tables agree with its logs; CODATA 2018 fallback.
try:
    from pyscf.data.nist import HARTREE2EV as _HARTREE_TO_EV

    _HARTREE_TO_EV = float(_HARTREE_TO_EV)
except (ImportError, TypeError, ValueError):
    _HARTREE_TO_EV = 27.211386245988


# UI functional names PySCF/libxc does not know by that name.
_XC_ALIASES = {
    "m11": "hyb_mgga_x_m11,mgga_c_m11",
}


# Imaginary modes smaller than this (cm^-1) are treated as numerical noise.
_IMAG_NOISE_CM = 20.0

# Dispersion choices in the UI -> PySCF's mf.disp keyword (pyscf-dispersion).
_DISPERSION = {
    "None": None,
    "D3(BJ)": "d3bj",
    "D3(zero)": "d3zero",
    "D4": "d4",
}


def _unwrap_angle(measured: float, target: float) -> float:
    """measured +/- k*360 closest to target (degrees)."""
    return target + ((measured - target + 180.0) % 360.0 - 180.0)


def _as_list(value):
    """Nested plain lists from numpy arrays / tuples (JSON- and Qt-safe)."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [
            _as_list(v) if hasattr(v, "tolist") or isinstance(v, (list, tuple)) else v
            for v in value
        ]
    return value


def classify_mo_data(mo_energy, mo_occ):
    """(scf_type, mo_energy, mo_occ) with plain lists.

    scf_type is what the viewers distinguish: "UHF" (two spin channels),
    "ROKS" (restricted open shell -- PySCF writes one 1-D mo_occ of 0/1/2,
    a SOMO carrying 1) or "RHF".
    """
    energy, occ = _as_list(mo_energy), _as_list(mo_occ)
    is_uhf = (
        isinstance(energy, list)
        and len(energy) == 2
        and all(isinstance(x, list) for x in energy)
    )
    if is_uhf:
        return "UHF", energy, occ
    flat = []
    for o in occ if isinstance(occ, list) else []:
        flat.extend(o if isinstance(o, list) else [o])
    has_somo = any(isinstance(o, (int, float)) and 0.5 < o < 1.5 for o in flat)
    return ("ROKS" if has_somo else "RHF"), energy, occ


def _json_safe(obj):
    """obj with numpy values turned into plain Python and NaN/inf -> None."""
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


def resolve_xc(functional: str) -> str:
    """The xc string PySCF needs for a functional name shown in the UI."""
    return _XC_ALIASES.get(str(functional).strip().lower(), functional)


def _flush_c_stdio():
    """fflush(NULL): C libraries buffer stdout when it is a file. Without
    this, output still in those buffers when the descriptors are swapped
    back reaches the terminal instead of pyscf.out."""
    try:
        if sys.platform == "win32":
            ctypes.cdll.msvcrt.fflush(None)
        else:
            ctypes.CDLL(None).fflush(None)
    except (OSError, AttributeError) as exc:
        logger.debug("fflush unavailable: %s", exc)


class CaptureStdOut:
    def __init__(self, filename):
        self.filename = filename
        self.original_stdout_fd = None
        self.original_stderr_fd = None
        self.saved_stdout_fd = None
        self.saved_stderr_fd = None

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        _flush_c_stdio()  # earlier C output belongs to the old target
        # Open log file
        self.log_file = open(self.filename, "a", buffering=1, encoding="utf-8")

        # Get FDs
        try:
            # Use __stdout__ to ensure we get the real OS FD, even if sys.stdout was MonkeyPatched/Wrapper
            self.original_stdout_fd = sys.__stdout__.fileno()
        except Exception:
            self.original_stdout_fd = 1  # Fallback to standard FD 1

        try:
            self.original_stderr_fd = sys.__stderr__.fileno()
        except Exception:
            self.original_stderr_fd = 2  # Fallback

        # Save Original FDs
        if self.original_stdout_fd is not None:
            self.saved_stdout_fd = os.dup(self.original_stdout_fd)
        if self.original_stderr_fd is not None:
            self.saved_stderr_fd = os.dup(self.original_stderr_fd)

        # Redirect FDs to Log File
        if self.original_stdout_fd is not None:
            os.dup2(self.log_file.fileno(), self.original_stdout_fd)
        if self.original_stderr_fd is not None:
            os.dup2(self.log_file.fileno(), self.original_stderr_fd)

        return self.log_file

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush — safe to swallow: stream may already be dead after thread kill
        try:
            sys.stdout.flush()
        except Exception:
            pass  # safe: stdout may be redirected or closed
        try:
            sys.stderr.flush()
        except Exception:
            pass  # safe: stderr may be redirected or closed
        try:
            if getattr(self, "log_file", None) is not None:
                self.log_file.flush()
        except (OSError, ValueError):
            pass  # safe: log file may already be closed
        _flush_c_stdio()  # C output of this job still goes to the file

        # Restore stdout FD — each step is independently guarded so that
        # a partial failure (e.g. after QThread.terminate()) does not leave
        # the other FD permanently redirected.
        if self.saved_stdout_fd is not None and self.original_stdout_fd is not None:
            try:
                os.dup2(self.saved_stdout_fd, self.original_stdout_fd)
            except Exception as _e:
                logger.warning(
                    "[worker.py] CaptureStdOut: failed to restore stdout FD: %s", _e
                )
            finally:
                try:
                    os.close(self.saved_stdout_fd)
                except Exception as _e:
                    logger.warning(
                        "[worker.py] CaptureStdOut: failed to close saved stdout FD: %s",
                        _e,
                    )
                self.saved_stdout_fd = None

        if self.saved_stderr_fd is not None and self.original_stderr_fd is not None:
            try:
                os.dup2(self.saved_stderr_fd, self.original_stderr_fd)
            except Exception as _e:
                logger.warning(
                    "[worker.py] CaptureStdOut: failed to restore stderr FD: %s", _e
                )
            finally:
                try:
                    os.close(self.saved_stderr_fd)
                except Exception as _e:
                    logger.warning(
                        "[worker.py] CaptureStdOut: failed to close saved stderr FD: %s",
                        _e,
                    )
                self.saved_stderr_fd = None

        if getattr(self, "log_file", None) is not None:
            try:
                self.log_file.close()
            except Exception:
                pass  # safe: file may already be closed by OS after terminate()
            self.log_file = None


class StreamToSignal(io.TextIOBase):
    def __init__(self, signal, target_stream=None):
        self.signal = signal
        self.target_stream = target_stream
        self._destroyed = False  # Track if we should stop emitting

    def write(self, text):
        # Safety: Only emit if signal is still valid
        if not self._destroyed and self.signal:
            try:
                self.signal.emit(text)
            except (RuntimeError, AttributeError):
                # Signal connection destroyed or Worker deleted
                # Mark as destroyed to prevent future attempts
                self._destroyed = True

        # Always try to write to target stream as fallback
        if self.target_stream:
            try:
                self.target_stream.write(text)
                self.target_stream.flush()
            except Exception:
                pass

    def flush(self):
        if self.target_stream:
            try:
                self.target_stream.flush()
            except Exception:
                pass

    def close(self):
        # Mark as destroyed to stop signal emissions
        self._destroyed = True
        # Do not close system stdout here

    @property
    def encoding(self):
        if self.target_stream and hasattr(self.target_stream, "encoding"):
            return self.target_stream.encoding
        return "utf-8"

    def isatty(self):
        if self.target_stream and hasattr(self.target_stream, "isatty"):
            return self.target_stream.isatty()
        return False


@contextlib.contextmanager
def redirected_output(worker, log_file):
    """Route C-level and Python stdout/stderr into log_file and the worker's
    log signal for the duration of a job; always restore them."""
    capturer = CaptureStdOut(log_file)
    f_log = capturer.__enter__()
    saved = (sys.stdout, sys.stderr)
    stream = StreamToSignal(worker.log_signal, target_stream=f_log)
    worker._stream = stream  # the GUI may close() it before terminate()
    sys.stdout = sys.stderr = stream
    try:
        yield stream
    finally:
        # Close before restoring: a late print() must not emit on a worker
        # that is being torn down.
        stream.close()
        worker._stream = None
        sys.stdout, sys.stderr = saved
        capturer.__exit__(None, None, None)


def _log_to_file(worker, text):
    """Worker messages go to the GUI log *and* pyscf.out.

    Emitting on log_signal alone reached only the GUI, so the job's own
    summaries (method switch, SCF properties, TDDFT table, imaginary-mode
    check, scan progress, ...) were missing from the log file."""
    stream = getattr(worker, "_stream", None)
    if stream is not None and not stream._destroyed:
        stream.write(text)  # emits the signal and writes the file
    else:
        worker.log_signal.emit(text)


class _FrequencySkipped(Exception):
    """Frequency analysis deliberately not run (reported, not an error)."""


class PySCFWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)  # Pass back data like XYZ, Cube paths

    _log = _log_to_file

    def __init__(self, xyz_str, config):
        super().__init__()
        self.xyz_str = xyz_str
        self.config = config
        # Cooperative stop flag — set to True from the GUI thread to request
        # clean termination between SCF steps / scan iterations.
        self._stop_requested = False
        self._stream = None  # Will hold StreamToSignal so GUI can invalidate it

    def _parse_spin_2s(self) -> int:
        """Return 2S from the spin multiplicity stored in config."""
        try:
            spin_str = str(self.config.get("spin", "1"))
            spin_mult = (
                int(spin_str.split(" ")[0]) if " " in spin_str else int(spin_str)
            )
            return max(0, spin_mult - 1)
        except Exception:
            return 0

    def _resolve_solvent_eps(self, solvent_name: str) -> float:
        """Return the dielectric constant for solvent_name (water fallback on failure)."""
        _HARDCODED = {
            "Water": 78.2,
            "Ethanol": 24.5,
            "Methanol": 32.7,
            "Acetone": 20.7,
            "THF": 7.58,
            "Chloroform": 4.81,
            "Dichloromethane": 8.93,
            "Toluene": 2.38,
            "Benzene": 2.27,
        }
        if solvent_name in _HARDCODED:
            return _HARDCODED[solvent_name]
        try:
            from pyscf.solvent import ddcosmo

            if hasattr(ddcosmo, "param") and hasattr(ddcosmo.param, "EPSILON"):
                pyscf_eps = ddcosmo.param.EPSILON
            elif hasattr(ddcosmo, "EPSILON"):
                pyscf_eps = ddcosmo.EPSILON
            else:
                import pyscf.solvent.ddcosmo.param as dd_param

                pyscf_eps = dd_param.EPSILON
            lookup = solvent_name if solvent_name in pyscf_eps else solvent_name.lower()
            return pyscf_eps.get(lookup, 78.2)
        except Exception:
            return 78.2

    def _apply_solvent(self, mf, solvent_name: str):
        """Wrap mf with ddCOSMO using the resolved eps for solvent_name."""
        mf = mf.ddCOSMO()
        mf.with_solvent.eps = self._resolve_solvent_eps(solvent_name)
        return mf

    def _apply_mf_settings(self, mf):
        """Apply max_cycle and conv_tol from config to mf."""
        mf.max_cycle = self.config.get("max_cycle", 100)
        try:
            mf.conv_tol = float(self.config.get("conv_tol", "1e-9"))
        except Exception as _e:
            logger.warning("[worker.py] _apply_mf_settings silenced: %s", _e)

    def _new_step_mf(self, mol, method_name, functional):
        """A fresh, fully configured mf (solvent + SCF settings) for a scan point.

        Scan points used to share a shallow copy of the job's mf, whose grids
        and solvent objects reset() mutates in place.
        """
        mf = self._build_mf(mol, method_name, functional)
        solvent_name = self.config.get("solvent", "None (Vacuum)")
        if solvent_name and "None" not in solvent_name:
            mf = self._apply_solvent(mf, solvent_name)
        self._apply_mf_settings(mf)
        return mf

    def _wants_numerical_hessian(self) -> bool:
        return str(self.config.get("hessian", "Analytic")).startswith("Numerical")

    def _numerical_hessian_obj(self, mf, mol):
        """PySCF's finite-difference Hessian (central differences of analytic
        gradients, 6 N gradient evaluations). Returns the same (natm, natm,
        3, 3) layout as the analytic one, and works wherever gradients do --
        solvent models and functionals without an analytic Hessian included.
        """
        try:
            from pyscf.tools import finite_diff
        except ImportError as exc:
            raise RuntimeError(
                "Numerical Hessian needs pyscf.tools.finite_diff; please update PySCF."
            ) from exc
        self._log(
            f"Numerical Hessian: {6 * mol.natm} gradient evaluations "
            "(central finite differences)...\n"
        )
        return finite_diff.Hessian(mf.nuc_grad_method())

    def _report_imaginary_modes(self, freqs, job_type):
        """Count imaginary modes and say whether that fits the stationary
        point the job was after: a minimum has none, a TS exactly one."""
        imag = [f for f in freqs if f < -_IMAG_NOISE_CM]
        small = [f for f in freqs if -_IMAG_NOISE_CM <= f < 0]
        want_ts = "Transition State" in job_type or "TS Optimization" in job_type
        expected = 1 if want_ts else 0
        kind = "transition state" if want_ts else "minimum"
        listing = ", ".join(f"{f:.1f}i" for f in (-x for x in imag))
        if len(imag) == expected:
            msg = f"Imaginary modes: {len(imag)} -- consistent with a {kind}"
            if listing:
                msg += f" ({listing} cm^-1)"
            self._log(msg + ".\n")
        else:
            self._log(
                f"WARNING: {len(imag)} imaginary mode(s) "
                f"({listing or 'none'} cm^-1); a {kind} should have "
                f"{expected}. This structure is not the intended stationary "
                "point.\n"
            )
        if small:
            self._log(
                f"Note: {len(small)} small imaginary mode(s) below "
                f"{_IMAG_NOISE_CM:.0f} cm^-1, usually numerical noise "
                "(grid / convergence).\n"
            )
        return len(imag)

    @staticmethod
    def _scf_properties(mf):
        """Dipole moment (Debye) and Mulliken charges of a finished SCF, or {}."""
        try:
            if mf.mo_coeff is None:
                return {}
            dip = np.asarray(mf.dip_moment(unit="Debye", verbose=0), dtype=float)
            _, charges = mf.mulliken_pop(verbose=0)
            mol = mf.mol
            return {
                "dipole_debye": dip.tolist(),
                "dipole_total_debye": float(np.linalg.norm(dip)),
                "mulliken_charges": np.asarray(charges, dtype=float).tolist(),
                "atom_symbols": [mol.atom_symbol(i) for i in range(mol.natm)],
            }
        except Exception as exc:
            logger.warning("[worker.py] SCF properties unavailable: %s", exc)
            return {}

    def _report_scf_properties(self, props):
        dx, dy, dz = props["dipole_debye"]
        lines = [
            "\n===== SCF Properties =====\n",
            f"Dipole moment (Debye): X={dx:.4f} Y={dy:.4f} Z={dz:.4f}  "
            f"Total={props['dipole_total_debye']:.4f}\n",
            "Mulliken charges:\n",
        ]
        for i, (sym, q) in enumerate(
            zip(props["atom_symbols"], props["mulliken_charges"])
        ):
            lines.append(f"  {i + 1:>3} {sym:<3} {q:+.4f}\n")
        self._log("".join(lines))
        try:
            with open(
                os.path.join(self.out_dir, "properties.json"), "w", encoding="utf-8"
            ) as fh:
                json.dump(props, fh, indent=2)
        except Exception as exc:
            self._log(f"Warning: Failed to save properties.json: {exc}\n")

    @staticmethod
    def _is_solvent_hessian(h_obj):
        """True when the Hessian object carries the solvent response
        (e.g. pyscf.solvent.hessian.pcm.ddCOSMOHessian)."""
        return any(
            getattr(cls, "__module__", "").startswith("pyscf.solvent")
            for cls in type(h_obj).__mro__
        )

    def _broken_symmetry_guess(self, mf, mol):
        """A spin-asymmetric initial density matrix for unrestricted SCF.

        A spin-restricted guess makes UHF/UKS relax straight back onto the
        RHF/RKS solution, which is the wrong answer for a singlet diradical or
        an antiferromagnetically coupled pair. Draining the beta density from
        the first atom's basis functions is PySCF's standard recipe; the SCF
        then relaxes into the broken-symmetry state.
        """
        dm = np.array(mf.get_init_guess(key="minao"), copy=True)
        if dm.ndim != 3 or dm.shape[0] != 2:
            raise ValueError("initial guess is not spin-resolved")

        ao_start, ao_end = mol.aoslice_by_atom()[0][2:4]
        if ao_end <= ao_start:
            raise ValueError("first atom contributes no basis functions")

        dm[1][ao_start:ao_end, ao_start:ao_end] = 0.0
        return dm

    def _dispersion(self):
        """PySCF `disp` keyword for the chosen correction, or None."""
        return _DISPERSION.get(str(self.config.get("dispersion", "None")))

    def _build_mf(self, mol, method_name, functional):
        """Create a mean-field object for the given mol, method, and functional."""
        grid_level = self.config.get("grid_level", 3)
        if method_name == "RHF":
            mf = scf.RHF(mol)
        elif method_name == "UHF":
            mf = scf.UHF(mol)
        elif method_name == "ROHF":
            mf = scf.ROHF(mol)
        elif method_name == "RKS":
            mf = dft.RKS(mol)
        elif method_name == "UKS":
            mf = dft.UKS(mol)
        elif method_name == "ROKS":
            mf = dft.ROKS(mol)
        else:
            raise ValueError(f"Unknown method: {method_name}")
        if "KS" in method_name:
            mf.xc = resolve_xc(functional)
            try:
                mf.grids.level = grid_level
                if grid_level >= 4:
                    mf.grids.prune = False
            except Exception as _e:
                logger.warning("[worker.py] _build_mf grid silenced: %s", _e)
        # Set at construction: PySCF caches the dispersion object on the mf.
        disp = self._dispersion()
        if disp:
            mf.disp = disp
        return mf

    def _check_dispersion_setup(self, method_name, functional):
        """An error message when the dispersion choice cannot run, else None."""
        if not self._dispersion():
            return None
        if "KS" in method_name and str(functional).lower().endswith("-v"):
            return (
                f"{functional} already contains VV10 dispersion; "
                "set Dispersion to None."
            )
        try:
            import pyscf.dispersion  # noqa: F401
        except ImportError:
            return (
                "Dispersion corrections need the pyscf-dispersion package "
                "(pip install pyscf-dispersion)."
            )
        return None

    # ------------------------------------------------------------------
    # Job driver
    # ------------------------------------------------------------------

    def run(self):
        if pyscf is None:
            self.error_signal.emit("PySCF is not installed in the python environment.")
            return

        try:
            self.out_dir = self._make_job_dir()
            log_file = os.path.join(self.out_dir, "pyscf.out")
            with redirected_output(self, log_file) as stream:
                self._run_job(stream)
        except Exception as e:  # noqa: BLE001 -- thread boundary: report, never raise
            self.error_signal.emit(str(e) + "\n" + traceback.format_exc())

    def _make_job_dir(self):
        """A fresh job_<n> directory under the configured output root."""
        root_dir = self.config.get("out_dir") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output"
        )
        n = 1
        while os.path.exists(os.path.join(root_dir, f"job_{n}")):
            n += 1
        out_dir = os.path.join(root_dir, f"job_{n}")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _run_job(self, stream):
        n_threads = self.config.get("threads", 0)
        if n_threads > 0:
            pyscf.lib.num_threads(n_threads)

        built = self._build_molecule(stream)
        if built is None:
            return
        mol, clean_atom_str = built

        try:
            self._log(f"PySCF running with {pyscf.lib.num_threads()} OpenMP threads.\n")
        except Exception as _e:  # noqa: BLE001 -- informational only
            logger.warning("thread count unavailable: %s", _e)

        method_name = self._resolve_method()
        functional = self.config.get("functional", "b3lyp")
        # The scans rebuild mf per point and must use the same (possibly
        # open-shell-switched) method as the rest of the job.
        self._method_name = method_name

        disp_error = self._check_dispersion_setup(method_name, functional)
        if disp_error:
            self.error_signal.emit(disp_error)
            return
        if self._dispersion():
            self._log(f"Dispersion correction: {self.config.get('dispersion')}\n")

        solvent = self.config.get("solvent", "None (Vacuum)")
        use_solvent = solvent != "None (Vacuum)"
        eps_value = self._resolve_solvent_eps(solvent) if use_solvent else 0.0
        if use_solvent:
            self._log(f"Solvent Model: ddCOSMO ({solvent}) eps={eps_value}\n")

        self._write_input_script(
            clean_atom_str, method_name, functional, mol, n_threads, eps_value
        )

        chk_path = os.path.join(self.out_dir, "pyscf.chk")
        mf = self._new_job_mf(mol, method_name, functional, chk_path)

        from pyscf import lib

        lib.logger.TIMER_LEVEL = 0

        job_type = self.config.get("job_type", "Energy")
        try:
            if "Scan" in job_type:
                self._run_scan_job(mol, mf, job_type)
                return

            results = {}
            if "Optimization" in job_type:
                mol = self._optimize(mf, job_type, method_name, results)
                if mol is None:
                    return
                # The optimizers drive a scanner copy and leave `mf` at the
                # starting geometry: the properties SCF needs a fresh mf.
                mf = self._new_job_mf(mol, method_name, functional, chk_path)

            if any(k in job_type for k in ("Optimization", "Energy", "Frequency")):
                self._run_scf(mf, mol, method_name)
            if "Frequency" in job_type:
                self._run_frequency(mf, mol, job_type, method_name, results)
            if "TDDFT" in job_type:
                self._run_tddft(mf, method_name, stream, results)

            self._finish(mf, chk_path, results)

        except Exception as e:  # noqa: BLE001 -- thread boundary: report, never raise
            traceback.print_exc()
            # A user stop has already disconnected the signals; an
            # InterruptedError dialog would be wrong.
            if not self._stop_requested:
                self.error_signal.emit(str(e))
            else:
                logger.info("Calculation stopped by user: %s", e)

    def _build_molecule(self, stream):
        """(mol, header-stripped atom block), or None after reporting."""
        # PySCF's atom= takes raw atom lines, not an XYZ file's header.
        lines = self.xyz_str.strip().split("\n")
        if len(lines) > 2 and lines[0].strip().isdigit():
            clean_atom_str = "\n".join(lines[2:])
        else:
            clean_atom_str = self.xyz_str.strip()
        try:
            mol = gto.M(
                atom=clean_atom_str,
                basis=self.config.get("basis", "sto-3g"),
                charge=self.config.get("charge", 0),
                spin=self._parse_spin_2s(),
                verbose=4,
                output=None,
                max_memory=self.config.get("memory", 4000),
                # Point-group symmetry. PySCF keeps the input frame, so
                # cubes and modes still line up with the editor geometry.
                symmetry=bool(self.config.get("symmetry", False)),
            )
            mol.stdout = stream
            mol.verbose = 4
            mol.build()
        except (RuntimeError, ValueError) as e_mol:
            # e.g. a charge / multiplicity that the electron count forbids
            self.error_signal.emit(
                f"Molecule Build Failed: {e_mol}\nCheck Charge and Multiplicity settings."
            )
            return None
        if mol.symmetry:
            self._log(f"Point-group symmetry: {mol.topgroup} (using {mol.groupname})\n")
        return mol, clean_atom_str

    def _resolve_method(self):
        """The configured method, switched to UHF/UKS for an open shell."""
        method_name = self.config.get("method", "RHF")
        if self._parse_spin_2s() != 0:
            if method_name == "RHF":
                self._log("Switching to UHF due to spin != 0.\n")
                return "UHF"
            if method_name == "RKS":
                self._log("Switching to UKS due to spin != 0.\n")
                return "UKS"
        return method_name

    def _new_job_mf(self, mol, method_name, functional, chk_path):
        """The job's mean-field object: solvent, SCF settings, checkpoint.
        Every job type starts from it (the optimizer included)."""
        mf = self._new_step_mf(mol, method_name, functional)
        mf.chkfile = chk_path
        return mf

    def _write_input_script(
        self, clean_atom_str, method_name, functional, mol, n_threads, eps_value
    ):
        """pyscf_input.py: a standalone script that reproduces the SCF."""
        cfg = self.config
        job_type = cfg.get("job_type", "")
        spin_2s = self._parse_spin_2s()
        solvent = cfg.get("solvent", "None (Vacuum)")
        header = [
            "# PySCF Input for MoleditPy PySCF Calculator plugin",
            f"# Plugin Version: {cfg.get('plugin_version', '0.0.0')}",
            f"# Job Type: {cfg.get('job_type')}",
            f"# Method: {method_name}",
        ]
        if "KS" in method_name:
            header.append(f"# Functional: {functional}")
        header += [
            f"# Basis: {cfg.get('basis')}",
            f"# Charge: {cfg.get('charge', 0)}",
            f"# Multiplicity: {spin_2s + 1}",
            f"# Threads: {n_threads}",
            f"# Memory: {cfg.get('memory')} MB",
        ]
        if "TDDFT" in job_type:
            header.append(f"# TDN States: {cfg.get('nstates')}")
        header += [
            f"# Max Cycle: {cfg.get('max_cycle')}",
            f"# Conv Tol: {cfg.get('conv_tol')}",
        ]
        if cfg.get("scan_params"):
            header.append("# Scan Parameters:")
            header += [f"#   {k}: {v}" for k, v in cfg["scan_params"].items()]
        if solvent != "None (Vacuum)":
            header.append(f"# Solvent: {solvent} (eps={eps_value})")

        body = [
            "",
            "from pyscf import gto, scf, dft",
            f"mol = gto.M(atom='''{clean_atom_str}''', ",
            f"    basis='{cfg.get('basis')}', ",
            f"    charge={cfg.get('charge', 0)}, ",
            f"    spin={spin_2s}, ",
            f"    max_memory={cfg.get('memory', 4000)}, ",
        ]
        if cfg.get("symmetry", False):
            body.append("    symmetry=True, ")
        body.append("    verbose=4)")
        if "KS" in method_name:
            grid_level = cfg.get("grid_level", 3)
            body += [
                f"mf = dft.{method_name}(mol)",
                f"mf.xc = '{resolve_xc(functional)}'",
                f"mf.grids.level = {grid_level}",
            ]
            if grid_level >= 4:
                body.append("mf.grids.prune = False")
        else:
            body.append(f"mf = scf.{method_name}(mol)")
        if self._dispersion():
            body.append(f"mf.disp = '{self._dispersion()}'")
        body.append(f"mf.max_cycle = {cfg.get('max_cycle', 100)}")
        try:
            body.append(f"mf.conv_tol = {float(cfg.get('conv_tol', '1e-9'))}")
        except (TypeError, ValueError):
            logger.warning("conv_tol %r is not a number", cfg.get("conv_tol"))
        if solvent != "None (Vacuum)":
            body += ["mf = mf.ddCOSMO()", f"mf.with_solvent.eps = {eps_value}"]
        body.append("mf.kernel()")
        if "TDDFT" in job_type:
            body += [
                "",
                "# TDDFT Calculation",
                "from pyscf import tdscf",
                f"td = tdscf.TDDFT(mf) if 'KS' in '{method_name}' else tdscf.TDHF(mf)",
                f"td.nstates = {cfg.get('nstates', 10)}",
                "td.verbose = 4",
                "td.kernel()",
            ]
        with open(
            os.path.join(self.out_dir, "pyscf_input.py"), "w", encoding="utf-8"
        ) as f:
            f.write("\n".join(header + body) + "\n")

    def _run_scan_job(self, mol, mf, job_type):
        scan_params = self.config.get("scan_params")
        if not scan_params:
            self.error_signal.emit("Scan parameters missing.")
            return
        results = {}
        if "Rigid" in job_type:
            self.run_rigid_scan(mol, mf, scan_params, results)
        elif "Relaxed" in job_type:
            self.run_relaxed_scan(mol, mf, scan_params, results)

        results["out_dir"] = self.out_dir
        # What was scanned, so the profile plot can label its axis (also
        # after a reload).
        results["scan_type"] = scan_params.get("type", "Coordinate")
        try:
            with open(
                os.path.join(self.out_dir, "scan_info.json"), "w", encoding="utf-8"
            ) as fh:
                json.dump(scan_params, fh, indent=2)
        except OSError as e_info:
            logger.warning("scan_info.json not written: %s", e_info)

        self.result_signal.emit(results)
        self.finished_signal.emit()

    def _optimize(self, mf, job_type, method_name, results):
        """Optimise (TS or minimum): the optimised mol, or None after reporting."""
        is_ts = "Transition State" in job_type or "TS Optimization" in job_type
        kind = "Transition State Optimization" if is_ts else "Geometry Optimization"
        self._log(f"Starting {kind} using {method_name}...\n")

        try:
            from pyscf.geomopt.geometric_solver import optimize

            mol_eq = optimize(mf, **({"transition": True} if is_ts else {}))
            header = (
                "Generated by PySCF TS Optimization"
                if is_ts
                else "Generated by PySCF Optimization"
            )
        except ImportError:
            if is_ts:
                self.error_signal.emit(
                    "Transition State optimization REQUIRES 'geometric' library. Please install it (pip install geometric)."
                )
                return None
            self._log(
                "\nWARNING: geometric-lib not found. Trying internal optimizer (Berny)...\n"
            )
            try:
                from pyscf.geomopt.berny_solver import optimize as optimize_berny

                mol_eq = optimize_berny(mf)
                header = "Generated by PySCF Optimization (Berny)"
            except ImportError:
                self.error_signal.emit(
                    "Neither 'geometric' nor 'berny' optimizer found. Please install 'geometric' (pip install geometric)."
                )
                return None

        coords = mol_eq.atom_coords(unit="Ang")
        xyz_lines = [f"{mol_eq.natm}", header]
        for i, c in enumerate(coords):
            xyz_lines.append(
                f"{mol_eq.atom_symbol(i)} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}"
            )
        results["optimized_xyz"] = "\n".join(xyz_lines)
        return mol_eq

    def _run_scf(self, mf, mol, method_name):
        """The job's SCF (skipped if already run), broken-symmetry guess
        for a closed-shell UHF/UKS when asked, convergence warning."""
        if mf.e_tot:
            return
        self._log(f"Running partial energy calculation using {method_name}...\n")
        # Only a spin-restricted guess needs breaking. With 2S > 0 the alpha
        # and beta occupations already differ.
        if (
            self.config.get("break_symmetry", True)
            and method_name in ("UHF", "UKS")
            and self._parse_spin_2s() == 0
        ):
            try:
                dm0 = self._broken_symmetry_guess(mf, mol)
            except Exception as e:  # noqa: BLE001 -- fall back to the standard guess
                self._log(
                    f"WARNING: Symmetry breaking failed ({e}). Proceeding with standard initial guess.\n"
                )
                mf.kernel()
            else:
                self._log(
                    "Applying symmetry-broken initial guess "
                    "(beta density removed from atom 1)...\n"
                )
                mf.kernel(dm0=dm0)
        else:
            mf.kernel()

        # Energy / Optimization jobs used to report an unconverged SCF
        # energy without a word.
        if not getattr(mf, "converged", True):
            self._log(
                f"WARNING: SCF did not converge within {mf.max_cycle} cycles; "
                "the energy and orbitals are not reliable.\n"
            )

    def _hessian(self, mf, mol):
        """Analytic or (by choice) finite-difference Hessian. Raises
        _FrequencySkipped when a solvated job has no usable solvent Hessian."""
        if self._wants_numerical_hessian():
            return self._numerical_hessian_obj(mf, mol).kernel()
        h_obj = mf.Hessian()
        if not (self._use_solvent() or hasattr(mf, "with_solvent")):
            return h_obj.kernel()
        # Only a Hessian carrying the solvent response is acceptable --
        # never vacuum frequencies for a solvated structure. PySCF 2.14 has
        # one for PCM, but its ddCOSMO class fails inside kernel(), so it is
        # tried and a failure becomes a clean skip.
        if self._is_solvent_hessian(h_obj):
            try:
                return h_obj.kernel()
            except Exception as e_sh:  # noqa: BLE001 -- any failure means "unavailable"
                logger.info("analytic solvent Hessian failed: %s", e_sh)
        self._log(
            "NOTE: Frequency analysis is skipped: no working analytic Hessian "
            "for this solvent model in this PySCF. Choose 'Hessian: Numerical' "
            "to compute it by finite differences.\n"
        )
        raise _FrequencySkipped("Frequency Analysis Skipped (Solvent Not Supported)")

    def _use_solvent(self):
        return self.config.get("solvent", "None (Vacuum)") != "None (Vacuum)"

    def _run_frequency(self, mf, mol, job_type, method_name, results):
        self._log(f"Starting Frequency Analysis using {method_name}...\n")
        if not mf.e_tot:
            self._log("Running SCF for Frequency Analysis...\n")
            mf.kernel()
        if not mf.converged:
            self._log(
                "WARNING: SCF did not converge before Frequency Analysis. Results may be inaccurate.\n"
            )

        self._log("Calculating Hessian...\n")
        try:
            hessian = self._hessian(mf, mol)

            from pyscf.hessian import thermo

            self._log("Performing Harmonic Analysis...\n")
            freq_res = thermo.harmonic_analysis(mol, hessian)

            self._log("Calculating Thermodynamic Properties...\n")
            t_data = thermo.thermo(
                mf,
                freq_res["freq_au"],
                temperature=float(self.config.get("temperature", 298.15)),
                pressure=float(self.config.get("pressure", 101325)),
            )

            # Imaginary frequencies come back complex: store them as negative
            freqs = []
            raw = freq_res["freq_wavenumber"]
            for f in raw.tolist() if hasattr(raw, "tolist") else raw:
                if isinstance(f, complex):
                    freqs.append(-abs(f.imag) if f.imag != 0 else f.real)
                else:
                    freqs.append(float(f))

            intensities = freq_res.get("infra_red_intensity")
            results["freq_data"] = {
                "freqs": freqs,
                "modes": freq_res["norm_mode"].tolist(),
                "intensities": _as_list(intensities)
                if intensities is not None
                else None,
                "n_imaginary": self._report_imaginary_modes(freqs, job_type),
            }
            self._log("Frequency Analysis Completed.\n")
            if t_data:
                results["thermo_data"] = _json_safe(t_data)

            freq_json_path = os.path.join(self.out_dir, "freq_analysis.json")
            save_data = {
                k: results[k] for k in ("freq_data", "thermo_data") if k in results
            }
            try:
                with open(freq_json_path, "w", encoding="utf-8") as f:
                    json.dump(_json_safe(save_data), f, indent=2)
                self._log(f"Frequency data saved to: {freq_json_path}\n")
            except (OSError, TypeError, ValueError) as e_save:
                self._log(f"Warning: Failed to save frequency JSON: {e_save}\n")

        except _FrequencySkipped as e_skip:
            self._log(f"Note: {e_skip}\n")
        except Exception as e_freq:  # noqa: BLE001 -- a failed Hessian must not lose the SCF result
            self._log(
                f"Frequency analysis failed: {e_freq}\n{traceback.format_exc()}\n"
            )
            if not self._wants_numerical_hessian() and isinstance(
                e_freq, (NotImplementedError, AttributeError)
            ):
                self._log(
                    "HINT: no analytic Hessian for this method; "
                    "choose 'Hessian: Numerical' and rerun.\n"
                )

    def _run_tddft(self, mf, method_name, stream, results):
        self._log("Starting TDDFT Calculation...\n")
        if not mf.e_tot:
            self._log("Running SCF for TDDFT...\n")
            mf.kernel()
        if not mf.converged:
            self._log(
                "WARNING: SCF did not converge before TDDFT. Results may be inaccurate.\n"
            )

        try:
            from pyscf import tdscf

            # TDDFT for KS references (RKS/UKS/ROKS), TDHF for HF ones
            td_obj = tdscf.TDDFT(mf) if "KS" in method_name else tdscf.TDHF(mf)
            nstates = int(self.config.get("nstates", 10))
            td_obj.nstates = nstates
            td_obj.verbose = 4
            td_obj.stdout = stream

            self._log(f"Calculating {nstates} Excited States...\n")
            td_obj.kernel()

            tddft_list = self._tddft_rows(td_obj, mf.e_tot)
            results["tddft_data"] = tddft_list
            self._save_tddft(tddft_list)
        except Exception as e_td:  # noqa: BLE001 -- a failed TDDFT must not lose the SCF result
            self._log(f"TDDFT calculation failed: {e_td}\n{traceback.format_exc()}\n")

    _TDDFT_HEADER = (
        f"{'State':<6} {'Energy (eV)':<12} {'Wavelen (nm)':<12} {'Osc. Str.':<10}"
    )

    def _tddft_rows(self, td_obj, e_ground):
        """Per-state rows (logged as a table): eV, nm, oscillator strength."""
        energies_exc = _as_list(td_obj.e_tot)
        if isinstance(energies_exc, float):
            energies_exc = [energies_exc]
        try:
            oscs = _as_list(td_obj.oscillator_strength())
            if isinstance(oscs, float):
                oscs = [oscs]
        except Exception:  # noqa: BLE001 -- oscillator strengths are optional
            oscs = [0.0] * len(energies_exc)

        # td.e holds the excitation energies directly; fall back to
        # differencing total energies only without it.
        exc_au = getattr(td_obj, "e", None)
        if exc_au is None or np.ndim(exc_au) != 1:
            exc_au = [e - e_ground for e in energies_exc]
        exc_au = [float(x) for x in exc_au]

        td_conv = getattr(td_obj, "converged", None)
        if td_conv is not None and not np.all(td_conv):
            self._log(
                "WARNING: not all excited states converged; "
                "treat the affected states with caution.\n"
            )

        self._log("\n===== TDDFT Results =====\n")
        self._log(self._TDDFT_HEADER + "\n" + "-" * 45 + "\n")
        rows = []
        for i, e_exc_tot in enumerate(energies_exc):
            exc_ev = exc_au[i] * _HARTREE_TO_EV
            exc_nm = _HC_EV_NM / exc_ev if abs(exc_ev) > 1e-6 else float("inf")
            osc = oscs[i] if i < len(oscs) else 0.0
            self._log(f"{i + 1:<6} {exc_ev:<12.4f} {exc_nm:<12.2f} {osc:<10.4f}\n")
            rows.append(
                {
                    "state": i + 1,
                    "energy_total": e_exc_tot,
                    "excitation_energy_ev": exc_ev,
                    "wavelength_nm": exc_nm,
                    "oscillator_strength": osc,
                }
            )
        self._log("-" * 45 + "\n")
        return rows

    def _save_tddft(self, rows):
        """tddft_results.txt (table) and tddft_results.json (for reloading)."""
        res_file = os.path.join(self.out_dir, "tddft_results.txt")
        try:
            with open(res_file, "w", encoding="utf-8") as f:
                f.write(self._TDDFT_HEADER + "\n" + "-" * 45 + "\n")
                f.writelines(
                    f"{r['state']:<6} {r['excitation_energy_ev']:<12.4f} "
                    f"{r['wavelength_nm']:<12.2f} {r['oscillator_strength']:<10.4f}\n"
                    for r in rows
                )
            self._log(f"TDDFT results saved to: {res_file}\n")
        except OSError as e_save:
            self._log(f"Warning: Failed to save TDDFT text result: {e_save}\n")

        json_file = os.path.join(self.out_dir, "tddft_results.json")
        try:
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump({"tddft_data": rows}, f, indent=2)
            self._log(f"TDDFT results saved to: {json_file}\n")
        except (OSError, TypeError, ValueError) as e_json:
            self._log(f"Warning: Failed to save TDDFT JSON: {e_json}\n")

    def _finish(self, mf, chk_path, results):
        """SCF properties, MO data and the checkpoint path; emit the result."""
        scf_props = self._scf_properties(mf)
        if scf_props:
            results.update(scf_props)
            self._report_scf_properties(scf_props)

        if mf.mo_energy is None or mf.mo_occ is None:
            self._log("Warning: No MO energy/occupancy data found.\n")
        # Same classification LoadWorker applies to the checkpoint.
        scf_type, mo_energy, mo_occ = classify_mo_data(mf.mo_energy, mf.mo_occ)
        results.update(
            {
                "mo_energy": mo_energy,
                "mo_occ": mo_occ,
                "scf_type": scf_type,
                "chkfile": chk_path,
                "out_dir": self.out_dir,
                "cube_files": [],
            }
        )
        self._log(f"Checkpoint saved to: {chk_path}\n")
        self.result_signal.emit(results)
        self.finished_signal.emit()

    def run_rigid_scan(self, mol, mf, params, results):
        self._log("\n===== Rigid Surface Scan =====\n")

        # Parse Params
        stype = params["type"]
        atoms = [int(a) for a in params["atoms"]]
        start_val = float(params["start"])
        end_val = float(params["end"])
        steps = int(params["steps"])

        # Setup RDKit Mol for Geometry Manipulation (Thread-safe local copy)
        rd_mol = Chem.MolFromXYZBlock(self.xyz_str)
        if not rd_mol:
            self.error_signal.emit(
                "Failed to create RDKit molecule from XYZ for scanning."
            )
            return

        # Ensure connectivity exists for the scan atoms (needed for Set*Deg/Length)
        rw_mol = Chem.RWMol(rd_mol)

        # 1. Attempt to reconstruct ALL bonds (crucial for group rotation)
        try:
            from rdkit.Chem import rdDetermineBonds

            rdDetermineBonds.DetermineConnectivity(rw_mol)
        except ImportError:
            self._log(
                "Warning: rdDetermineBonds not found. Group rotation might fail.\n"
            )
        except Exception as e:
            self._log(f"Warning deriving connectivity: {e}\n")

        # 2. Force-add bonds specifically needed for the scan metric
        # (in case DetermineConnectivity missed them due to bond stretching)
        needed_bonds = []
        if stype == "Dist":
            needed_bonds = [(atoms[0], atoms[1])]
        elif stype == "Angle":
            needed_bonds = [(atoms[0], atoms[1]), (atoms[1], atoms[2])]
        elif stype == "Dihedral":
            needed_bonds = [
                (atoms[0], atoms[1]),
                (atoms[1], atoms[2]),
                (atoms[2], atoms[3]),
            ]

        for a1, a2 in needed_bonds:
            if not rw_mol.GetBondBetweenAtoms(a1, a2):
                rw_mol.AddBond(a1, a2, Chem.BondType.SINGLE)

        # Initialize Ring Info (Critical for rdMolTransforms)
        try:
            Chem.SanitizeMol(rw_mol)
        except Exception as e:
            # If sanitization fails (e.g. valence), try to just compute rings
            self._log(f"Sanitization warning: {e}. Attempting partial update.\n")
            try:
                rw_mol.UpdatePropertyCache(strict=False)
                Chem.GetSymmSSSR(rw_mol)
            except Exception as _e:
                logger.warning("[worker.py] silenced: %s", _e)

        # Use the explicit connectivity molecule
        rd_mol = rw_mol
        conf = rd_mol.GetConformer()

        scan_results = []
        trajectory = []  # List of XYZ strings

        scan_values = np.linspace(start_val, end_val, steps)

        csv_lines = ["Step,Value,Energy,Converged"]

        method_name = getattr(self, "_method_name", self.config.get("method", "RHF"))
        functional = self.config.get("functional", "b3lyp")
        dm_prev = None  # density of the last converged point

        for i, val in enumerate(scan_values):
            # Cooperative stop check — avoids force-kill between steps
            if self._stop_requested:
                self._log(f"Rigid scan stopped by user after {i} step(s).\n")
                break
            self._log(f"Step {i + 1}/{steps}: {stype} = {val:.4f} ... ")

            # 1. Modify Geometry using RDKit
            # Note: RDKit uses Degrees for angles
            try:
                if stype == "Dist":
                    rdMolTransforms.SetBondLength(conf, atoms[0], atoms[1], val)
                elif stype == "Angle":
                    rdMolTransforms.SetAngleDeg(conf, atoms[0], atoms[1], atoms[2], val)
                elif stype == "Dihedral":
                    rdMolTransforms.SetDihedralDeg(
                        conf, atoms[0], atoms[1], atoms[2], atoms[3], val
                    )
            except Exception as e:
                self._log(f"Geometry set failed: {e}\n")
                continue

            # 2. Rebuild PySCF Mol
            # Get new coords
            new_xyz = []
            symbols = [a.GetSymbol() for a in rd_mol.GetAtoms()]
            for idx, atom in enumerate(rd_mol.GetAtoms()):
                pos = conf.GetAtomPosition(idx)
                new_xyz.append(f"{symbols[idx]} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")

            xyz_block = f"{mol.natm}\nStep {i + 1}\n" + "\n".join(new_xyz)
            trajectory.append(xyz_block)

            # Build new PySCF mol
            mol_step = gto.M(
                atom="\n".join(new_xyz),
                basis=mol.basis,
                charge=mol.charge,
                spin=mol.spin,
                verbose=0,
                max_memory=self.config.get("memory", 4000),
            )
            mol_step.build()

            # 3. Single-point energy on a fresh mf, seeded from the last
            # converged neighbour so the whole profile stays on one SCF
            # solution instead of each point picking its own from minao.
            mf_step = self._new_step_mf(mol_step, method_name, functional)
            mf_step.chkfile = os.path.join(self.out_dir, f"scan_step_{i + 1}.chk")
            mf_step.verbose = 0

            mf_step.kernel(dm0=dm_prev)
            e_tot = mf_step.e_tot

            # An unconverged point can sit many kcal/mol off and would
            # otherwise enter the profile looking like a real barrier.
            converged = bool(getattr(mf_step, "converged", True))
            if converged:
                try:
                    dm_prev = mf_step.make_rdm1()
                except Exception:
                    dm_prev = None
                self._log(f"E = {e_tot:.6f} Ha\n")
            else:
                self._log(f"E = {e_tot:.6f} Ha  ** SCF NOT CONVERGED **\n")

            scan_results.append(
                {
                    "step": i + 1,
                    "value": val,
                    "energy": e_tot,
                    "converged": converged,
                }
            )
            csv_lines.append(
                f"{i + 1},{val:.6f},{e_tot:.8f},{'yes' if converged else 'NO'}"
            )

            # Keep UI responsive-ish
            QThread.msleep(10)

        # Compile Results
        results["scan_results"] = scan_results
        results["scan_trajectory"] = trajectory  # Should be parsed by viewer

        # Save CSV
        csv_path = os.path.join(self.out_dir, "scan_results.csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(csv_lines))
        self._log(f"Scan results saved to {csv_path}\n")

        # Save Trajectory XYZ
        traj_path = os.path.join(self.out_dir, "scan_trajectory.xyz")
        with open(traj_path, "w") as f:
            f.write("\n".join(trajectory))

    def run_relaxed_scan(self, mol, mf, params, results):
        self._log("\n===== Relaxed Surface Scan (Constrained Optimization) =====\n")

        stype = params["type"]
        atoms = [int(a) for a in params["atoms"]]
        start_val = float(params["start"])
        end_val = float(params["end"])
        steps = int(params["steps"])

        scan_values = np.linspace(start_val, end_val, steps)

        scan_results = []
        trajectory = []
        csv_lines = ["Step,Value,Energy,Converged"]

        # Store method info for reconstruction
        method_name = getattr(self, "_method_name", self.config.get("method", "RHF"))
        functional = self.config.get("functional", "b3lyp")
        basis = self.config.get("basis", "sto-3g")
        charge = self.config.get("charge", 0)

        # Extract spin (2S)
        spin_2s = self._parse_spin_2s()

        # Current geometry as starting point
        current_coords = mol.atom_coords(unit="Ang")
        current_symbols = [mol.atom_symbol(k) for k in range(mol.natm)]

        # Ensure geometric is available
        try:
            from pyscf.geomopt.geometric_solver import optimize
        except ImportError:
            self.error_signal.emit(
                "Relaxed scan requires 'geometric' library. Install with: pip install geometric"
            )
            return

        # Ensure initial molecule is converged so we have a good starting checkpoint for Step 0
        if not mf.e_tot:
            self._log("Ensuring initial SCF convergence before scanning...\n")
            mf.kernel()

        for i, val in enumerate(scan_values):
            # Cooperative stop check
            if self._stop_requested:
                self._log(f"Relaxed scan stopped by user after {i} step(s).\n")
                break
            self._log(f"\nStep {i + 1}/{steps}: Constrained {stype} = {val:.4f}\n")

            # 1. Create Constraints File for geometric
            # geometric format:
            # $set
            # bond 0 1 1.5
            const_str = "$set\n"

            # Geometric indices are 1-based
            # syntax: type a1 a2 [a3 a4] value
            atom_str = " ".join([str(a + 1) for a in atoms])

            # Type mapping
            g_type = "distance"
            if stype == "Angle":
                g_type = "angle"
            elif stype == "Dihedral":
                g_type = "dihedral"

            # Note: geometric angles are in degrees
            const_str += f"{g_type} {atom_str} {val:.6f}\n"

            const_file = os.path.join(self.out_dir, f"constraints_step_{i}.txt")
            with open(const_file, "w") as f:
                f.write(const_str)

            self._log(f"  Constraint: {g_type} {atom_str} = {val:.6f}\n")

            # 2. Build molecule from current geometry
            try:
                # Create atom string from current coordinates
                atom_str_pyscf = ""
                for sym, coord in zip(current_symbols, current_coords):
                    atom_str_pyscf += (
                        f"{sym} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}; "
                    )

                # Create new molecule object
                step_mol = gto.M(
                    atom=atom_str_pyscf,
                    basis=basis,
                    charge=charge,
                    spin=spin_2s,
                    verbose=0,
                    max_memory=self.config.get("memory", 4000),
                )

                # 3. Create new mean field object for this step
                step_mf = self._new_step_mf(step_mol, method_name, functional)

                # Set checkpoint
                step_chk = os.path.join(self.out_dir, f"scan_step_{i}.chk")
                step_mf.chkfile = step_chk

                # --- Initialization Strategy: Seed from previous step ---
                try:
                    src_chk = None
                    if i == 0:
                        # Use the main mf checkpoint (which we ensured is converged above)
                        if os.path.exists(mf.chkfile):
                            src_chk = mf.chkfile
                    else:
                        # Use previous step's checkpoint
                        prev_chk = os.path.join(self.out_dir, f"scan_step_{i - 1}.chk")
                        if os.path.exists(prev_chk):
                            src_chk = prev_chk

                    if src_chk:
                        shutil.copyfile(src_chk, step_chk)
                        step_mf.init_guess = "chkfile"
                        # self._log(f"  > Seeding guess from {os.path.basename(src_chk)}\n")
                except Exception as e_seed:
                    self._log(f"  Warning: Failed to seed initial guess: {e_seed}\n")
                # --------------------------------------------------------

                mol_eq = optimize(step_mf, constraints=const_file)

                # Force an explicit SCF calculation on the final optimized structure
                # to ensure the energy is 100% accurate and matches the mol_eq coordinates.
                self._log("  Calculating final energy for optimized structure...\n")
                step_converged = True
                try:
                    # reset(), not `.mol =`: the DFT grids and the solvent
                    # cavity are only rebuilt for the new geometry by reset().
                    step_mf.reset(mol_eq)
                    e_tot = step_mf.kernel()
                    step_converged = bool(getattr(step_mf, "converged", True))
                    self._log(f"  ✓ Final optimized energy: {e_tot:.8f} Ha\n")
                except Exception as e:
                    self._log(f"  ⚠ Failed final SCF, attempting fallback... {e}\n")
                    step_converged = False
                    if hasattr(step_mf, "e_tot") and step_mf.e_tot is not None:
                        e_tot = step_mf.e_tot
                    else:
                        # Recording 0.0 Ha put a ~76 Hartree spike in the
                        # energy profile that reads as a real barrier.
                        self._log(
                            "  ⚠ No usable energy for this point; dropping it "
                            "from the scan.\n"
                        )
                        continue

                # Capture optimized geometry
                current_coords = mol_eq.atom_coords(unit="Ang")
                current_symbols = [mol_eq.atom_symbol(k) for k in range(mol_eq.natm)]

                # Measure the actual coordinate value from optimized geometry
                # (may differ slightly from target constraint)
                actual_val = val  # Default to target
                try:
                    if stype == "Dist":
                        # Calculate bond length
                        p1 = current_coords[atoms[0]]
                        p2 = current_coords[atoms[1]]
                        actual_val = np.linalg.norm(p1 - p2)
                    elif stype == "Angle":
                        # Calculate angle
                        p1 = current_coords[atoms[0]]
                        p2 = current_coords[atoms[1]]
                        p3 = current_coords[atoms[2]]
                        v1 = p1 - p2
                        v2 = p3 - p2
                        cos_angle = np.dot(v1, v2) / (
                            np.linalg.norm(v1) * np.linalg.norm(v2)
                        )
                        actual_val = np.degrees(
                            np.arccos(np.clip(cos_angle, -1.0, 1.0))
                        )
                    elif stype == "Dihedral":
                        # Use RDKit to calculate dihedral angle (ensures correct convention)

                        # Create temporary RDKit molecule from current coords
                        temp_mol = Chem.RWMol()
                        for sym in current_symbols:
                            atom = Chem.Atom(sym)
                            temp_mol.AddAtom(atom)

                        # Add conformer with current coordinates
                        conf = Chem.Conformer(len(current_symbols))
                        for atom_idx, coord in enumerate(current_coords):
                            conf.SetAtomPosition(atom_idx, tuple(coord))
                        temp_mol.AddConformer(conf)

                        # Calculate dihedral using RDKit
                        measured = rdMolTransforms.GetDihedralDeg(
                            temp_mol.GetConformer(),
                            atoms[0],
                            atoms[1],
                            atoms[2],
                            atoms[3],
                        )
                        # RDKit reports (-180, 180]; put it on the branch of
                        # the target, or a 180 deg point comes back as -180
                        # and jumps to the other end of the profile.
                        actual_val = _unwrap_angle(measured, val)
                except Exception as e:
                    self._log(f"  Warning: Could not measure actual value: {e}\n")

                if abs(actual_val - val) > 0.01:  # Log if difference is significant
                    self._log(f"  Target: {val:.4f}, Actual: {actual_val:.4f}\n")

                xyz_lines = [
                    f"{len(current_symbols)}",
                    f"Step {i + 1} {stype}={actual_val:.4f} E={e_tot:.6f} Ha",
                ]
                for s, c in zip(current_symbols, current_coords):
                    xyz_lines.append(f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")

                xyz_frame = "\n".join(xyz_lines)
                trajectory.append(xyz_frame)

                if step_converged:
                    self._log(f"  ✓ Converged: E = {e_tot:.8f} Ha\n")
                else:
                    self._log(f"  ** SCF NOT CONVERGED **: E = {e_tot:.8f} Ha\n")

                scan_results.append(
                    {
                        "step": i + 1,
                        "value": actual_val,  # Use actual measured value
                        "energy": e_tot,
                        "converged": step_converged,
                    }
                )
                csv_lines.append(
                    f"{i + 1},{actual_val:.6f},{e_tot:.8f},"
                    f"{'yes' if step_converged else 'NO'}"
                )

            except Exception as e:
                self._log(f"  ✗ Optimization step {i + 1} failed: {e}\n")
                self._log(traceback.format_exc())
                # Break scan on failure
                break

        # Compile Results
        results["scan_results"] = scan_results
        results["scan_trajectory"] = trajectory

        # Save CSV
        csv_path = os.path.join(self.out_dir, "scan_results.csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(csv_lines))
        self._log(f"\nScan results saved to {csv_path}\n")

        # Save Trajectory XYZ
        traj_path = os.path.join(self.out_dir, "scan_trajectory.xyz")
        with open(traj_path, "w") as f:
            f.write("\n".join(trajectory))
        self._log(f"Scan trajectory saved to {traj_path}\n")


class PropertyWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)  # {"files": [cube paths]}

    _log = _log_to_file

    def __init__(self, chkfile, tasks, out_dir):
        """tasks: "ESP", "SpinDensity" or orbital strings ("HOMO", "LUMO+1",
        "MO 15", "#14", ...; see orbital_index)."""
        super().__init__()
        self.chkfile = chkfile
        self.tasks = tasks
        self.out_dir = out_dir
        self._stop_requested = False
        self._stream = None

    @staticmethod
    def _is_uhf_coeff(mo_coeff) -> bool:
        return isinstance(mo_coeff, (tuple, list)) or (
            isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 3
        )

    @staticmethod
    def _unpack_uhf_coeff(mo_coeff, mo_occ):
        return mo_coeff[0], mo_coeff[1], mo_occ[0], mo_occ[1]

    @classmethod
    def _spin_density_matrices(cls, mo_coeff, mo_occ):
        """(dm_alpha, dm_beta) for UHF, ROHF/ROKS and RHF checkpoints.

        PySCF stores a restricted open-shell mo_occ as one 1-D array of
        0/1/2 values: a SOMO is alpha-only, a doubly occupied MO is both.
        """
        from pyscf.scf import hf, uhf

        if cls._is_uhf_coeff(mo_coeff):
            c_a, c_b, o_a, o_b = cls._unpack_uhf_coeff(mo_coeff, mo_occ)
            dm_ab = uhf.make_rdm1((c_a, c_b), (o_a, o_b))
            return dm_ab[0], dm_ab[1]
        occ = np.asarray(mo_occ, dtype=float)
        if occ.ndim == 2:
            return hf.make_rdm1(mo_coeff, occ[0]), hf.make_rdm1(mo_coeff, occ[1])
        occ_a = (occ > 0.5).astype(float)
        occ_b = (occ > 1.5).astype(float)
        return hf.make_rdm1(mo_coeff, occ_a), hf.make_rdm1(mo_coeff, occ_b)

    @staticmethod
    def _is_open_shell_occ(mo_occ):
        occ = np.asarray(mo_occ, dtype=float)
        return occ.ndim == 2 or bool(np.any((occ > 0.5) & (occ < 1.5)))

    @staticmethod
    def _find_homo_lumo_1d(occs, threshold=0.1):
        homo_idx = -1
        for i, occ_val in enumerate(occs):
            if occ_val > threshold:
                homo_idx = i
        return homo_idx, homo_idx + 1

    @staticmethod
    def orbital_index(task, homo_idx, lumo_idx):
        """0-based MO index for a task string (spin suffix already removed):
        "HOMO", "LUMO+1", "HOMO-2", "MO 15" / "15" (1-based), "#14"
        (0-based) or "MO 15_HOMO-1" (1-based, label informational)."""
        m = re.search(r"MO\s+(\d+)_([A-Za-z0-9+-]+)", task)
        if m:
            return int(m.group(1)) - 1
        for name, base in (("HOMO", homo_idx), ("LUMO", lumo_idx)):
            if name in task:
                if "+" in task:
                    return base + int(task.split("+")[1])
                if "-" in task:
                    return base - int(task.split("-")[1])
                return base
        if "MO" in task or task.isdigit() or task.startswith("#"):
            digits = re.sub(r"\D", "", task)
            if digits:
                return int(digits) if task.startswith("#") else int(digits) - 1
        raise ValueError(f"Unknown task format: {task}")

    @staticmethod
    def relative_label(idx, homo_idx, lumo_idx):
        if idx <= homo_idx:
            diff = homo_idx - idx
            return "HOMO" if diff == 0 else f"HOMO-{diff}"
        if idx >= lumo_idx:
            diff = idx - lumo_idx
            return "LUMO" if diff == 0 else f"LUMO+{diff}"
        return f"MO_{idx}"

    def run(self):
        if pyscf is None:
            self.error_signal.emit("PySCF not found.")
            return

        try:
            log_file = os.path.join(self.out_dir, "pyscf.out")
            with redirected_output(self, log_file) as stream:
                results = self._generate(stream)
                self.result_signal.emit(results)
                self.finished_signal.emit()
        except Exception as e:  # noqa: BLE001 -- thread boundary: report, never raise
            if not self._stop_requested:
                self.error_signal.emit(str(e) + "\n" + traceback.format_exc())
            else:
                logger.info("PropertyWorker stopped by user: %s", e)

    def _generate(self, stream):
        from pyscf import lib, scf, tools

        mol = lib.chkfile.load_mol(self.chkfile)
        mol.output = None
        mol.stdout = stream
        mol.verbose = 4

        scf_data = scf.chkfile.load(self.chkfile, "scf")
        mo_coeff = scf_data["mo_coeff"]
        mo_occ = scf_data["mo_occ"]

        files = []
        for task in self.tasks:
            if self._stop_requested:
                self._log("Property generation stopped by user.\n")
                break
            if task == "ESP":
                files += self._make_esp(tools, mol, mo_coeff, mo_occ)
            elif task == "SpinDensity":
                files += self._make_spin_density(tools, mol, mo_coeff, mo_occ)
            elif isinstance(task, str):
                files += self._make_orbital(tools, mol, mo_coeff, mo_occ, task)
        return {"files": files}

    @staticmethod
    def _unique_path(path):
        # deferred: the tests load worker.py outside its package
        from .utils import get_unique_path

        return get_unique_path(path)

    def _make_esp(self, tools, mol, mo_coeff, mo_occ):
        f_esp = self._unique_path(os.path.join(self.out_dir, "esp.cube"))
        f_dens = self._unique_path(os.path.join(self.out_dir, "density.cube"))
        # Total density for the MEP, for RHF / UHF / ROHF alike
        dm_a, dm_b = self._spin_density_matrices(mo_coeff, mo_occ)
        dm = dm_a + dm_b
        self._log(f"Generating ESP ({os.path.basename(f_esp)})...\n")
        tools.cubegen.mep(mol, f_esp, dm)
        self._log(f"Generating Density ({os.path.basename(f_dens)})...\n")
        tools.cubegen.density(mol, f_dens, dm)
        return [f_esp, f_dens]

    def _make_spin_density(self, tools, mol, mo_coeff, mo_occ):
        if not (self._is_uhf_coeff(mo_coeff) or self._is_open_shell_occ(mo_occ)):
            self._log(
                "Skipping Spin Density (Not an open-shell calculation or format unknown).\n"
            )
            return []
        dm_a, dm_b = self._spin_density_matrices(mo_coeff, mo_occ)
        f_spin = self._unique_path(os.path.join(self.out_dir, "spin_density.cube"))
        self._log(f"Generating Spin Density ({os.path.basename(f_spin)})...\n")
        tools.cubegen.density(mol, f_spin, dm_a - dm_b)
        return [f_spin]

    def _make_orbital(self, tools, mol, mo_coeff, mo_occ, task):
        """Cube of one MO. Task forms: see orbital_index(), each with an
        optional "_A" / "_B" spin suffix (UHF; alpha by default)."""
        is_uhf = self._is_uhf_coeff(mo_coeff)
        use_beta = "_B" in task or "Beta" in task
        if use_beta:
            task = task.replace("_B", "").replace("Beta", "").strip()
            spin_suffix = "_B"
        elif "_A" in task or "Alpha" in task:
            task = task.replace("_A", "").replace("Alpha", "").strip()
            spin_suffix = "_A"
        else:
            spin_suffix = "_A" if is_uhf else ""

        if is_uhf:
            channel = 1 if use_beta else 0
            coeff, occ = mo_coeff[channel], mo_occ[channel]
        else:
            coeff = mo_coeff
            # a 2-D (alpha, beta) occupation: HOMO/LUMO follow alpha
            occ = mo_occ[0] if np.ndim(mo_occ) == 2 else mo_occ
        homo_idx, lumo_idx = self._find_homo_lumo_1d(occ, 0.1)

        try:
            idx = self.orbital_index(task, homo_idx, lumo_idx)
        except (ValueError, IndexError) as e:
            self._log(f"Error parsing orbital: {task} ({e})\n")
            return []
        if idx < 0 or idx >= coeff.shape[1]:
            self._log(f"Orbital index {idx} out of bounds for {task}\n")
            return []

        label = self.relative_label(idx, homo_idx, lumo_idx)
        spin_char = ("b" if use_beta else "a") if is_uhf else ""
        fname = f"{idx + 1:03d}{spin_char}_{label}.cube"  # 1-based in the name
        f_path = self._unique_path(os.path.join(self.out_dir, fname))
        self._log(
            f"Generating {os.path.basename(f_path)} (Index {idx}{spin_suffix})...\n"
        )
        tools.cubegen.orbital(mol, f_path, coeff[:, idx])
        return [f_path]


class LoadWorker(QThread):
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, chkfile):
        super().__init__()
        self.chkfile = chkfile
        self._stop_requested = False

    @staticmethod
    def load_scan_type(result_dir):
        """Scanned coordinate type ("Dist" / "Angle" / "Dihedral") recorded
        in scan_info.json, or None for older results."""
        try:
            with open(
                os.path.join(result_dir, "scan_info.json"), encoding="utf-8"
            ) as fh:
                return json.load(fh).get("type")
        except Exception:
            return None

    @staticmethod
    def _load_scan_csv(path):
        """Reload a scan from its CSV.

        Keys are lower-cased to match what the live scan emits ("energy",
        "value", ...); the CSV header is capitalised, so a reloaded scan used
        to raise KeyError as soon as the viewer asked for r["energy"].
        """
        scan_res = []
        with open(path) as f:
            for row in csv.DictReader(f):
                item = {}
                for k, v in row.items():
                    if k is None:
                        continue
                    key = k.strip().lower()
                    if key == "converged":
                        item[key] = str(v).strip().lower() in ("yes", "true", "1")
                        continue
                    if key == "step":
                        try:
                            item[key] = int(float(v))
                            continue
                        except (TypeError, ValueError):
                            pass
                    try:
                        item[key] = float(v)
                    except (TypeError, ValueError):
                        item[key] = v
                scan_res.append(item)
        return scan_res

    # Auxiliary result files next to the checkpoint.
    _AUX_FILES = (
        "scan_results.csv",
        "tddft_results.json",
        "freq_analysis.json",
        "properties.json",
    )

    def _load_checkpoint(self, lib, scf):
        """MO data and geometry from pyscf.chk, or None if stopped."""
        mol = lib.chkfile.load_mol(self.chkfile)
        if self._stop_requested:
            return None
        scf_data = scf.chkfile.load(self.chkfile, "scf")
        scf_type, mo_energy, mo_occ = classify_mo_data(
            scf_data.get("mo_energy"), scf_data.get("mo_occ")
        )
        coords = mol.atom_coords(unit="Ang")
        xyz_lines = [f"{mol.natm}", "Loaded from Checkpoint"]
        for i, c in enumerate(coords):
            xyz_lines.append(f"{mol.atom_symbol(i)} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
        return {
            "mo_energy": mo_energy,
            "mo_occ": mo_occ,
            "scf_type": scf_type,
            "loaded_xyz": "\n".join(xyz_lines),
            "chkfile": self.chkfile,
        }

    @staticmethod
    def _read_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_aux_files(self, base_dir, results):
        """Merge scan / TDDFT / frequency / property files into results.
        A damaged file is logged and skipped, never fatal."""
        loaders = {
            "scan_results.csv": lambda p: {
                "scan_results": self._load_scan_csv(p),
                "scan_type": self.load_scan_type(base_dir),
            },
            "tddft_results.json": lambda p: {
                k: v for k, v in self._read_json(p).items() if k == "tddft_data"
            },
            # the viewer reads freq_data["freqs"]: unpack the file's two keys
            "freq_analysis.json": lambda p: {
                k: v
                for k, v in self._read_json(p).items()
                if k in ("freq_data", "thermo_data")
            },
            "properties.json": self._read_json,
        }
        for name, loader in loaders.items():
            path = os.path.join(base_dir, name)
            if not os.path.exists(path):
                continue
            try:
                results.update(loader(path))
            except Exception as exc:
                logger.warning(
                    "[worker.py] LoadWorker: failed to load %s: %s", name, exc
                )
        traj = os.path.join(base_dir, "scan_trajectory.xyz")
        if os.path.exists(traj):
            results["scan_trajectory_path"] = traj

    def run(self):
        if pyscf is None:
            self.error_signal.emit("PySCF not found.")
            return

        try:
            from pyscf import lib, scf

            base_dir = os.path.dirname(self.chkfile)
            results = {"out_dir": base_dir}
            has_aux = any(
                os.path.exists(os.path.join(base_dir, n)) for n in self._AUX_FILES
            )
            # A scan / TDDFT / frequency folder may have no checkpoint; with
            # neither, loading the checkpoint raises the error the user sees.
            if os.path.exists(self.chkfile) or not has_aux:
                chk = self._load_checkpoint(lib, scf)
                if chk is None:
                    return
                results.update(chk)
            self._load_aux_files(base_dir, results)

            if self._stop_requested:
                return
            self.finished_signal.emit(results)

        except Exception as e:
            if self._stop_requested:
                return
            self.error_signal.emit(str(e) + "\n" + traceback.format_exc())
