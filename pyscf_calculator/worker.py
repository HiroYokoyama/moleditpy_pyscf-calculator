import csv
import sys
import os
import io
import json
import traceback
import numpy as np
import math
import re
import shutil
from PyQt6.QtCore import QThread, pyqtSignal
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms
except ImportError:
    pass

# We import pyscf inside the thread or check availability
try:
    import pyscf
    from pyscf import gto, scf, dft
    from pyscf import solvent  # noqa: F401 # Ensure ddCOSMO mixin is available
except ImportError:
    pyscf = None

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


def resolve_xc(functional: str) -> str:
    """The xc string PySCF needs for a functional name shown in the UI."""
    return _XC_ALIASES.get(str(functional).strip().lower(), functional)


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
        except Exception:
            pass  # safe: log file may already be closed

        # Restore stdout FD — each step is independently guarded so that
        # a partial failure (e.g. after QThread.terminate()) does not leave
        # the other FD permanently redirected.
        if self.saved_stdout_fd is not None and self.original_stdout_fd is not None:
            try:
                os.dup2(self.saved_stdout_fd, self.original_stdout_fd)
            except Exception as _e:
                logging.warning(
                    "[worker.py] CaptureStdOut: failed to restore stdout FD: %s", _e
                )
            finally:
                try:
                    os.close(self.saved_stdout_fd)
                except Exception as _e:
                    logging.warning(
                        "[worker.py] CaptureStdOut: failed to close saved stdout FD: %s",
                        _e,
                    )
                self.saved_stdout_fd = None

        if self.saved_stderr_fd is not None and self.original_stderr_fd is not None:
            try:
                os.dup2(self.saved_stderr_fd, self.original_stderr_fd)
            except Exception as _e:
                logging.warning(
                    "[worker.py] CaptureStdOut: failed to restore stderr FD: %s", _e
                )
            finally:
                try:
                    os.close(self.saved_stderr_fd)
                except Exception as _e:
                    logging.warning(
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


class PySCFWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)  # Pass back data like XYZ, Cube paths

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
            logging.warning("[worker.py] _apply_mf_settings silenced: %s", _e)

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
                "Numerical Hessian needs pyscf.tools.finite_diff; "
                "please update PySCF."
            ) from exc
        self.log_signal.emit(
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
            self.log_signal.emit(msg + ".\n")
        else:
            self.log_signal.emit(
                f"WARNING: {len(imag)} imaginary mode(s) "
                f"({listing or 'none'} cm^-1); a {kind} should have "
                f"{expected}. This structure is not the intended stationary "
                "point.\n"
            )
        if small:
            self.log_signal.emit(
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
            logging.warning("[worker.py] SCF properties unavailable: %s", exc)
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
        self.log_signal.emit("".join(lines))
        try:
            with open(
                os.path.join(self.out_dir, "properties.json"), "w", encoding="utf-8"
            ) as fh:
                json.dump(props, fh, indent=2)
        except Exception as exc:
            self.log_signal.emit(f"Warning: Failed to save properties.json: {exc}\n")

    @staticmethod
    def _is_solvent_hessian(h_obj):
        """True when the Hessian object carries the solvent response
        (e.g. pyscf.solvent.hessian.pcm.ddCOSMOHessian)."""
        return any(
            getattr(cls, "__module__", "").startswith("pyscf.solvent")
            for cls in type(h_obj).__mro__
        )

    @staticmethod
    def _to_list(arr):
        if arr is None:
            return []
        return arr.tolist() if hasattr(arr, "tolist") else list(arr)

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
                logging.warning("[worker.py] _build_mf grid silenced: %s", _e)
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
            import pyscf.dispersion  # noqa: F401, PLC0415
        except ImportError:
            return (
                "Dispersion corrections need the pyscf-dispersion package "
                "(pip install pyscf-dispersion)."
            )
        return None

    def run(self):
        if pyscf is None:
            self.error_signal.emit("PySCF is not installed in the python environment.")
            return

        try:
            # Prepare Output Root Directory
            root_dir = self.config.get("out_dir", None)
            if not root_dir:
                root_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "output"
                )

            # Create Job Subdirectory (job_1, job_2...) within root
            n = 1
            while True:
                out_dir = os.path.join(root_dir, f"job_{n}")
                if not os.path.exists(out_dir):
                    break
                n += 1

            os.makedirs(out_dir, exist_ok=True)
            self.out_dir = out_dir

            # Notify user of new location
            # Note: signal might not be connected yet? No, it's defined.
            # But GUI connection happens before start().

            # Setup Logging (Standard Name)
            log_file = os.path.join(out_dir, "pyscf.out")

            # C-Level Redirection (CaptureStdOut)
            capturer = CaptureStdOut(log_file)
            f_log = capturer.__enter__()

            # Python Redirection
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            stream = StreamToSignal(self.log_signal, target_stream=f_log)
            self._stream = (
                stream  # Exposed so GUI can call stream.close() before terminate()
            )
            sys.stdout = stream
            sys.stderr = stream

            # Configure Threads
            n_threads = self.config.get("threads", 0)
            if n_threads > 0:
                pyscf.lib.num_threads(n_threads)

            # Prepare Charge and Spin (Convert M -> 2S)
            charge = self.config.get("charge", 0)
            spin_2s = self._parse_spin_2s()

            # Setup Molecule
            try:
                # Compatibility: PySCF atom= expects raw atoms, not XYZ file format with headers.
                # Strip header if present.
                raw_xyz = self.xyz_str.strip()
                lines = raw_xyz.split("\n")
                if len(lines) > 2 and lines[0].strip().isdigit():
                    # Standard XYZ: Skip Count and Comment
                    clean_atom_str = "\n".join(lines[2:])
                else:
                    clean_atom_str = raw_xyz

                mol = gto.M(
                    atom=clean_atom_str,
                    basis=self.config.get("basis", "sto-3g"),
                    charge=charge,
                    spin=spin_2s,
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
                if mol.symmetry:
                    self.log_signal.emit(
                        f"Point-group symmetry: {mol.topgroup} "
                        f"(using {mol.groupname})\n"
                    )
            except (RuntimeError, ValueError) as e_mol:
                # Catch specific PySCF errors (e.g. Spin/Charge mismatch)
                msg = str(e_mol)
                self.error_signal.emit(
                    f"Molecule Build Failed: {msg}\nCheck Charge and Multiplicity settings."
                )
                return

            # Log Parallelism Info
            try:
                n_threads = pyscf.lib.num_threads()
                self.log_signal.emit(
                    f"PySCF running with {n_threads} OpenMP threads.\n"
                )
            except Exception as _e:
                logging.warning("[worker.py] silenced: %s", _e)

            # --- Parameters Setup ---
            scan_params = self.config.get("scan_params", None)
            # ------------------------

            # Select Method
            method_name = self.config.get("method", "RHF")
            functional = self.config.get("functional", "b3lyp")

            # Auto-adjust method for Open Shell if needed
            if spin_2s != 0:
                if method_name == "RHF":
                    method_name = "UHF"
                    self.log_signal.emit("Switching to UHF due to spin != 0.\n")
                elif method_name == "RKS":
                    method_name = "UKS"
                    self.log_signal.emit("Switching to UKS due to spin != 0.\n")

            # The scans rebuild mf per point and must use the same (possibly
            # open-shell-switched) method as the rest of the job.
            self._method_name = method_name

            disp_error = self._check_dispersion_setup(method_name, functional)
            if disp_error:
                self.error_signal.emit(disp_error)
                return
            if self._dispersion():
                self.log_signal.emit(
                    f"Dispersion correction: {self.config.get('dispersion')}\n"
                )

            # --- Solvent Setup ---
            selected_solvent = self.config.get("solvent", "None (Vacuum)")
            use_solvent = selected_solvent != "None (Vacuum)"
            eps_value = 0.0
            if use_solvent:
                eps_value = self._resolve_solvent_eps(selected_solvent)
                self.log_signal.emit(
                    f"Solvent Model: ddCOSMO ({selected_solvent}) eps={eps_value}\n"
                )
            # ---------------------

            inp_file = os.path.join(out_dir, "pyscf_input.py")
            with open(inp_file, "w", encoding="utf-8") as f:
                f.write("# PySCF Input for MoleditPy PySCF Calculator plugin\n")
                f.write(
                    f"# Plugin Version: {self.config.get('plugin_version', '0.0.0')}\n"
                )
                f.write(f"# Job Type: {self.config.get('job_type')}\n")
                f.write(f"# Method: {method_name}\n")
                if "KS" in method_name:
                    f.write(f"# Functional: {functional}\n")
                f.write(f"# Basis: {self.config.get('basis')}\n")
                f.write(f"# Charge: {charge}\n")
                f.write(f"# Multiplicity: {spin_2s + 1}\n")
                f.write(f"# Threads: {n_threads}\n")
                f.write(f"# Memory: {self.config.get('memory')} MB\n")
                if "TDDFT" in self.config.get("job_type", ""):
                    f.write(f"# TDN States: {self.config.get('nstates')}\n")
                f.write(f"# Max Cycle: {self.config.get('max_cycle')}\n")
                f.write(f"# Conv Tol: {self.config.get('conv_tol')}\n")

                if scan_params:
                    f.write("# Scan Parameters:\n")
                    for k, v in scan_params.items():
                        f.write(f"#   {k}: {v}\n")

                if use_solvent:
                    f.write(f"# Solvent: {selected_solvent} (eps={eps_value})\n")

                f.write("\n")
                f.write("from pyscf import gto, scf, dft\n")
                # The header-stripped atoms: gto.M does not take the XYZ
                # count/comment lines, so the script must not embed them.
                f.write(f"mol = gto.M(atom='''{clean_atom_str}''', \n")
                f.write(f"    basis='{self.config.get('basis')}', \n")
                f.write(f"    charge={charge}, \n")
                f.write(f"    spin={spin_2s}, \n")
                f.write(f"    max_memory={self.config.get('memory', 4000)}, \n")
                if self.config.get("symmetry", False):
                    f.write("    symmetry=True, \n")
                f.write("    verbose=4)\n")

                if "KS" in method_name:
                    grid_level = self.config.get("grid_level", 3)
                    f.write(f"mf = dft.{method_name}(mol)\n")
                    f.write(f"mf.xc = '{resolve_xc(functional)}'\n")
                    f.write(f"mf.grids.level = {grid_level}\n")
                    if grid_level >= 4:
                        f.write("mf.grids.prune = False\n")
                else:
                    f.write(f"mf = scf.{method_name}(mol)\n")

                if self._dispersion():
                    f.write(f"mf.disp = '{self._dispersion()}'\n")
                f.write(f"mf.max_cycle = {self.config.get('max_cycle', 100)}\n")
                try:
                    tol = float(self.config.get("conv_tol", "1e-9"))
                    f.write(f"mf.conv_tol = {tol}\n")
                except Exception as _e:
                    logging.warning("[worker.py] silenced: %s", _e)

                if use_solvent:
                    f.write("mf = mf.ddCOSMO()\n")
                    f.write(f"mf.with_solvent.eps = {eps_value}\n")

                f.write("mf.kernel()\n")

                if "TDDFT" in self.config.get("job_type", ""):
                    f.write("\n# TDDFT Calculation\n")
                    f.write("from pyscf import tdscf\n")
                    f.write(
                        f"td = tdscf.TDDFT(mf) if 'KS' in '{method_name}' else tdscf.TDHF(mf)\n"
                    )
                    f.write(f"td.nstates = {self.config.get('nstates', 10)}\n")
                    f.write("td.verbose = 4\n")
                    f.write("td.kernel()\n")

            mf = self._build_mf(mol, method_name, functional)
            # Every job type starts from this mf (the optimizer included), so
            # the solvent has to go on here -- otherwise Energy / Optimization /
            # TDDFT silently ran in vacuum while the log reported ddCOSMO.
            if use_solvent:
                mf = self._apply_solvent(mf, selected_solvent)

            # Ensure Checkpoint is in the new job folder
            chk_path = os.path.join(self.out_dir, "pyscf.chk")
            mf.chkfile = chk_path
            self._apply_mf_settings(mf)

            # Explicitly set pyscf logger stream too for global usage
            from pyscf import lib

            lib.logger.TIMER_LEVEL = 0  # Reduces some noise, or keeps it standard
            # Note: sys.stdout/stderr are already redirected to `stream` above.
            # The original values are saved in `original_stdout`/`original_stderr`.
            # Do NOT re-assign original_stdout here — that would corrupt the
            # reference needed by the outer finally block to restore real stdout.

            try:
                # Prepare job type
                job_type = self.config.get("job_type", "Energy")
                results = {}

                # --- SCAN DISPATCH ---
                if "Scan" in job_type:
                    scan_params = self.config.get("scan_params", None)
                    if not scan_params:
                        self.error_signal.emit("Scan parameters missing.")
                        return

                    if "Rigid" in job_type:
                        self.run_rigid_scan(mol, mf, scan_params, results)
                    elif "Relaxed" in job_type:
                        # Relaxed scan uses constraint optimization at each step
                        self.run_relaxed_scan(mol, mf, scan_params, results)

                    # Ensure out_dir is included for history
                    results["out_dir"] = self.out_dir
                    # What was scanned, so the profile plot can label its
                    # axis (also after a reload).
                    results["scan_type"] = scan_params.get("type", "Coordinate")
                    try:
                        with open(
                            os.path.join(self.out_dir, "scan_info.json"),
                            "w",
                            encoding="utf-8",
                        ) as fh:
                            json.dump(scan_params, fh, indent=2)
                    except Exception as e_info:
                        logging.warning("[worker.py] scan_info.json: %s", e_info)

                    self.result_signal.emit(results)
                    self.finished_signal.emit()
                    return

                if "Optimization" in job_type:
                    is_ts = (
                        "Transition State" in job_type or "TS Optimization" in job_type
                    )

                    if is_ts:
                        self.log_signal.emit(
                            f"Starting Transition State Optimization using {method_name}...\n"
                        )
                    else:
                        self.log_signal.emit(
                            f"Starting Geometry Optimization using {method_name}...\n"
                        )

                    try:
                        from pyscf.geomopt.geometric_solver import optimize

                        # Prepare kwargs for optimize
                        opt_params = {}
                        if is_ts:
                            opt_params["transition"] = True

                        mol_eq = optimize(mf, **opt_params)
                        header_comment = (
                            "Generated by PySCF TS Optimization"
                            if is_ts
                            else "Generated by PySCF Optimization"
                        )

                    except ImportError:
                        if is_ts:
                            self.error_signal.emit(
                                "Transition State optimization REQUIRES 'geometric' library. Please install it (pip install geometric)."
                            )
                            return

                        self.log_signal.emit(
                            "\nWARNING: geometric-lib not found. Trying internal optimizer (Berny)...\n"
                        )
                        try:
                            from pyscf.geomopt.berny_solver import (
                                optimize as optimize_berny,
                            )

                            mol_eq = optimize_berny(mf)
                            header_comment = "Generated by PySCF Optimization (Berny)"
                        except ImportError:
                            self.error_signal.emit(
                                "Neither 'geometric' nor 'berny' optimizer found. Please install 'geometric' (pip install geometric)."
                            )
                            return

                    # Convert optimized geometry to XYZ string (in Angstroms)
                    coords = mol_eq.atom_coords(unit="Ang")
                    symbols = [mol_eq.atom_symbol(i) for i in range(mol_eq.natm)]
                    xyz_lines = [f"{len(symbols)}", header_comment]
                    for s, c in zip(symbols, coords):
                        xyz_lines.append(f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
                    results["optimized_xyz"] = "\n".join(xyz_lines)

                    # The optimizers drive a scanner copy and leave `mf` bound
                    # to the starting geometry, so the properties SCF needs a
                    # fresh mf at mol_eq -- with the same settings and solvent
                    # the optimization used.
                    mf = self._build_mf(mol_eq, method_name, functional)
                    if use_solvent:
                        mf = self._apply_solvent(mf, selected_solvent)
                    mf.chkfile = chk_path
                    self._apply_mf_settings(mf)
                    mol = mol_eq

                # Ensure Energy is calculated (if not done by Opt or if detached)
                # Optimization updates mf but we need to ensure kernel is run for properties
                if (
                    "Optimization" in job_type
                    or "Energy" in job_type
                    or "Frequency" in job_type
                ):
                    if not mf.e_tot:
                        self.log_signal.emit(
                            f"Running partial energy calculation using {method_name}...\n"
                        )

                        should_break = self.config.get("break_symmetry", True)

                        # Only a spin-restricted guess needs breaking. With
                        # spin_2s > 0 the alpha and beta occupations already
                        # differ, so there is no symmetry left to break.
                        if (
                            should_break
                            and method_name in ["UHF", "UKS"]
                            and spin_2s == 0
                        ):
                            try:
                                dm0 = self._broken_symmetry_guess(mf, mol)
                                self.log_signal.emit(
                                    "Applying symmetry-broken initial guess "
                                    "(beta density removed from atom 1)...\n"
                                )
                                mf.kernel(dm0=dm0)
                            except Exception as e:
                                self.log_signal.emit(
                                    f"WARNING: Symmetry breaking failed ({str(e)}). Proceeding with standard initial guess.\n"
                                )
                                mf.kernel()
                        else:
                            mf.kernel()

                if "Frequency" in job_type:
                    self.log_signal.emit(
                        f"Starting Frequency Analysis using {method_name}...\n"
                    )

                    # Ensure we have a converged SCF on the current molecule
                    if not mf.e_tot:
                        self.log_signal.emit("Running SCF for Frequency Analysis...\n")
                        mf.kernel()

                    if not mf.converged:
                        self.log_signal.emit(
                            "WARNING: SCF did not converge before Frequency Analysis. Results may be inaccurate.\n"
                        )

                    self.log_signal.emit("Calculating Hessian...\n")
                    try:
                        hessian = None

                        solvated = use_solvent or hasattr(mf, "with_solvent")
                        if self._wants_numerical_hessian():
                            h_obj = self._numerical_hessian_obj(mf, mol)
                            hessian = h_obj.kernel()
                        elif solvated:
                            # Only a Hessian carrying the solvent response is
                            # acceptable -- never vacuum frequencies for a
                            # solvated structure. PySCF 2.14 has one for PCM,
                            # but its ddCOSMO class fails inside kernel(), so
                            # it is tried and a failure becomes a clean skip.
                            # The numerical Hessian stays an explicit choice.
                            h_obj = mf.Hessian()
                            hessian = None
                            if self._is_solvent_hessian(h_obj):
                                try:
                                    hessian = h_obj.kernel()
                                except Exception as e_sh:
                                    logging.info(
                                        "[worker.py] analytic solvent Hessian "
                                        "failed: %s",
                                        e_sh,
                                    )
                            if hessian is None:
                                self.log_signal.emit(
                                    "NOTE: Frequency analysis is skipped: no "
                                    "working analytic Hessian for this solvent "
                                    "model in this PySCF. Choose 'Hessian: "
                                    "Numerical' to compute it by finite "
                                    "differences.\n"
                                )
                                raise Exception(
                                    "Frequency Analysis Skipped (Solvent Not Supported)"
                                )
                        else:
                            hessian = mf.Hessian().kernel()

                        from pyscf.hessian import thermo

                        self.log_signal.emit("Performing Harmonic Analysis...\n")
                        freq_res = thermo.harmonic_analysis(mol, hessian)

                        # Calculate Thermo
                        self.log_signal.emit(
                            "Calculating Thermodynamic Properties...\n"
                        )
                        # Ensure temp/pressure are floats
                        T = float(self.config.get("temperature", 298.15))
                        P = float(self.config.get("pressure", 101325))
                        t_data = thermo.thermo(
                            mf, freq_res["freq_au"], temperature=T, pressure=P
                        )

                        # Store data for GUI Visualizer
                        # Check for IR intensity (not always available in standard harmonic_analysis)
                        intensities = freq_res.get("infra_red_intensity", None)

                        # Process Frequencies: Handle imaginary (complex) values -> negative reals
                        raw_freqs = freq_res["freq_wavenumber"]
                        processed_freqs = []
                        if hasattr(raw_freqs, "tolist"):
                            raw_freqs = raw_freqs.tolist()

                        for f in raw_freqs:
                            if isinstance(f, complex):
                                if f.imag != 0:
                                    processed_freqs.append(-abs(f.imag))
                                else:
                                    processed_freqs.append(f.real)
                            else:
                                processed_freqs.append(float(f))

                        n_imag = self._report_imaginary_modes(
                            processed_freqs, job_type
                        )
                        results["freq_data"] = {
                            "freqs": processed_freqs,
                            "modes": freq_res["norm_mode"].tolist(),
                            "intensities": intensities.tolist()
                            if hasattr(intensities, "tolist")
                            else intensities,
                            "n_imaginary": n_imag,
                        }
                        self.log_signal.emit("Frequency Analysis Completed.\n")

                        # Store Thermo
                        if t_data:
                            # Robust conversion function for JSON serialization
                            def make_json_safe(obj):
                                if obj is None:
                                    return None
                                if isinstance(obj, (bool, int, str)):
                                    return obj
                                if isinstance(obj, float):
                                    if math.isnan(obj) or math.isinf(obj):
                                        return None
                                    return obj
                                if hasattr(obj, "tolist"):  # numpy array
                                    obj = obj.tolist()
                                if isinstance(obj, (list, tuple)):
                                    return [make_json_safe(item) for item in obj]
                                if isinstance(obj, dict):
                                    return {
                                        k: make_json_safe(v) for k, v in obj.items()
                                    }
                                # Fallback: convert to string
                                return str(obj)

                            results["thermo_data"] = make_json_safe(t_data)

                        # Save freq_data and thermo_data to JSON file
                        freq_json_path = os.path.join(
                            self.out_dir, "freq_analysis.json"
                        )
                        try:
                            # Custom JSON encoder to handle all edge cases
                            class SafeEncoder(json.JSONEncoder):
                                def default(self, obj):
                                    if hasattr(obj, "tolist"):
                                        return obj.tolist()
                                    if isinstance(obj, (np.integer, np.floating)):
                                        return obj.item()
                                    if isinstance(obj, float):
                                        if math.isnan(obj) or math.isinf(obj):
                                            return None
                                        return obj
                                    if isinstance(obj, tuple):
                                        return list(obj)
                                    return str(obj)

                            save_data = {}
                            if "freq_data" in results:
                                save_data["freq_data"] = results["freq_data"]
                            if "thermo_data" in results:
                                save_data["thermo_data"] = results["thermo_data"]

                            with open(freq_json_path, "w") as f:
                                json.dump(save_data, f, indent=2, cls=SafeEncoder)
                            self.log_signal.emit(
                                f"Frequency data saved to: {freq_json_path}\n"
                            )
                        except Exception as e_save:
                            self.log_signal.emit(
                                f"Warning: Failed to save frequency JSON: {e_save}\n"
                            )
                            self.log_signal.emit(traceback.format_exc())

                    except Exception as e_freq:
                        if "Skipped" in str(e_freq):
                            self.log_signal.emit(f"Note: {e_freq}\n")
                        else:
                            self.log_signal.emit(
                                f"Frequency analysis failed: {e_freq}\n{traceback.format_exc()}\n"
                            )
                            if not self._wants_numerical_hessian() and isinstance(
                                e_freq, (NotImplementedError, AttributeError)
                            ):
                                self.log_signal.emit(
                                    "HINT: no analytic Hessian for this method; "
                                    "choose 'Hessian: Numerical' and rerun.\n"
                                )

                if "TDDFT" in job_type:
                    self.log_signal.emit("Starting TDDFT Calculation...\n")
                    if not mf.e_tot:
                        self.log_signal.emit("Running SCF for TDDFT...\n")
                        mf.kernel()

                    if not mf.converged:
                        self.log_signal.emit(
                            "WARNING: SCF did not converge before TDDFT. Results may be inaccurate.\n"
                        )

                    try:
                        from pyscf import tdscf

                        # Select TDDFT Method
                        # For RHF/UHF -> TDHF
                        # For RKS/UKS -> TDDFT (or TDA)

                        td_obj = None

                        # Simple dispatch based on MF class is usually safer if unsure
                        # But explicit class usage allows TDA control if we add it later

                        if "KS" in method_name:  # RKS, UKS or ROKS
                            # Default to full TDDFT
                            # Could use TDA if we add an option later: tdscf.TDA(mf)
                            td_obj = tdscf.TDDFT(mf)
                        else:  # HF
                            td_obj = tdscf.TDHF(mf)

                        nstates = int(self.config.get("nstates", 10))
                        td_obj.nstates = nstates
                        td_obj.verbose = 4
                        # Redirect output? td_obj uses lib.logger which respects global stream we set?
                        # Or explicitly set stdout
                        try:
                            td_obj.stdout = stream
                        except Exception as _e:
                            logging.warning("[worker.py] silenced: %s", _e)

                        self.log_signal.emit(
                            f"Calculating {nstates} Excited States...\n"
                        )
                        td_obj.kernel()

                        self.log_signal.emit("\n===== TDDFT Results =====\n")
                        self.log_signal.emit(
                            f"{'State':<6} {'Energy (eV)':<12} {'Wavelen (nm)':<12} {'Osc. Str.':<10}\n"
                        )
                        self.log_signal.emit("-" * 45 + "\n")

                        # Results Extraction
                        # td_obj.e_tot are total energies of Excited States
                        # Excitation Energy = E_exc_state - E_ground_state

                        energies_exc = td_obj.e_tot
                        # e_tot can be a list or numpy array
                        if hasattr(energies_exc, "tolist"):
                            energies_exc = energies_exc.tolist()
                        elif isinstance(energies_exc, float):
                            energies_exc = [energies_exc]

                        # Oscillator Strengths
                        try:
                            oscs = td_obj.oscillator_strength()
                            if hasattr(oscs, "tolist"):
                                oscs = oscs.tolist()
                            if isinstance(oscs, float):
                                oscs = [oscs]
                        except Exception:
                            oscs = [0.0] * len(energies_exc)

                        e_ground = mf.e_tot
                        # td.e holds the excitation energies directly; fall
                        # back to differencing total energies only without it.
                        exc_au = getattr(td_obj, "e", None)
                        if exc_au is None or np.ndim(exc_au) != 1:
                            exc_au = [e - e_ground for e in energies_exc]
                        exc_au = [float(x) for x in exc_au]

                        td_conv = getattr(td_obj, "converged", None)
                        if td_conv is not None and not np.all(td_conv):
                            self.log_signal.emit(
                                "WARNING: not all excited states converged; "
                                "treat the affected states with caution.\n"
                            )

                        tddft_list = []

                        for i, e_exc_tot in enumerate(energies_exc):
                            exc_ev = exc_au[i] * _HARTREE_TO_EV

                            if abs(exc_ev) > 1e-6:
                                exc_nm = _HC_EV_NM / exc_ev
                            else:
                                exc_nm = float("inf")

                            osc = oscs[i] if i < len(oscs) else 0.0

                            self.log_signal.emit(
                                f"{i + 1:<6} {exc_ev:<12.4f} {exc_nm:<12.2f} {osc:<10.4f}\n"
                            )

                            tddft_list.append(
                                {
                                    "state": i + 1,
                                    "energy_total": e_exc_tot,
                                    "excitation_energy_ev": exc_ev,
                                    "wavelength_nm": exc_nm,
                                    "oscillator_strength": osc,
                                }
                            )

                        results["tddft_data"] = tddft_list
                        self.log_signal.emit("-" * 45 + "\n")

                        # --- Persist Results to Files ---
                        # Save as text
                        try:
                            res_file = os.path.join(self.out_dir, "tddft_results.txt")
                            with open(res_file, "w") as f:
                                f.write(
                                    f"{'State':<6} {'Energy (eV)':<12} {'Wavelen (nm)':<12} {'Osc. Str.':<10}\n"
                                )
                                f.write("-" * 45 + "\n")
                                for item in tddft_list:
                                    f.write(
                                        f"{item['state']:<6} {item['excitation_energy_ev']:<12.4f} {item['wavelength_nm']:<12.2f} {item['oscillator_strength']:<10.4f}\n"
                                    )
                            self.log_signal.emit(
                                f"TDDFT results saved to: {res_file}\n"
                            )
                        except Exception as e_save:
                            self.log_signal.emit(
                                f"Warning: Failed to save TDDFT text result: {e_save}\n"
                            )

                        # Save as JSON for reloading
                        try:
                            json_file = os.path.join(self.out_dir, "tddft_results.json")
                            with open(json_file, "w") as f:
                                json.dump({"tddft_data": tddft_list}, f, indent=2)
                            self.log_signal.emit(
                                f"TDDFT results saved to: {json_file}\n"
                            )
                        except Exception as e_json:
                            self.log_signal.emit(
                                f"Warning: Failed to save TDDFT JSON: {e_json}\n"
                            )

                    except Exception as e_td:
                        self.log_signal.emit(
                            f"TDDFT calculation failed: {e_td}\n{traceback.format_exc()}\n"
                        )

                if "Energy" == job_type:  # Only Energy
                    # Already handled by top block but ensuring...
                    if not mf.e_tot:
                        mf.kernel()

                # Dipole moment and Mulliken charges of the final SCF
                scf_props = self._scf_properties(mf)
                if scf_props:
                    results.update(scf_props)
                    self._report_scf_properties(scf_props)

                # --- SAVE CHECKPOINT (ALWAYS) ---
                # Checkpoint is already set to self.out_dir/pyscf.chk and written by mf.kernel()

                chk_path = os.path.join(self.out_dir, "pyscf.chk")

                results.update({"chkfile": chk_path, "out_dir": self.out_dir})

                is_uhf = False
                # Match LoadWorker's labels: restricted open-shell (ROHF/ROKS)
                # is reported as "ROKS" since consumers only distinguish
                # RHF / UHF / ROKS.
                restricted_type = "ROKS" if method_name in ("ROHF", "ROKS") else "RHF"

                # Defensive check for None
                if mf.mo_energy is None or mf.mo_occ is None:
                    self.log_signal.emit(
                        "Warning: No MO energy/occupancy data found.\n"
                    )
                    results["mo_energy"] = []
                    results["mo_occ"] = []
                    results["scf_type"] = restricted_type
                else:
                    if isinstance(mf.mo_energy, tuple):
                        is_uhf = True
                    elif isinstance(mf.mo_energy, list) and len(mf.mo_energy) == 2:
                        # Check content types?
                        if hasattr(mf.mo_energy[0], "__len__"):
                            is_uhf = True
                    elif hasattr(mf.mo_energy, "ndim") and mf.mo_energy.ndim == 2:
                        is_uhf = True

                    try:
                        if is_uhf:
                            if isinstance(mf.mo_energy, tuple):
                                e_a, e_b = mf.mo_energy
                                o_a, o_b = mf.mo_occ
                            else:
                                e_a = mf.mo_energy[0]
                                e_b = mf.mo_energy[1] if len(mf.mo_energy) > 1 else []
                                o_a = mf.mo_occ[0]
                                o_b = mf.mo_occ[1] if len(mf.mo_occ) > 1 else []

                            results["mo_energy"] = [
                                self._to_list(e_a),
                                self._to_list(e_b),
                            ]
                            results["mo_occ"] = [self._to_list(o_a), self._to_list(o_b)]
                            results["scf_type"] = "UHF"
                        else:
                            results["mo_energy"] = self._to_list(mf.mo_energy)
                            results["mo_occ"] = self._to_list(mf.mo_occ)
                            results["scf_type"] = restricted_type
                    except Exception as e_process:
                        self.log_signal.emit(f"Error processing MO data: {e_process}\n")
                        results["mo_energy"] = []
                        results["mo_occ"] = []

                self.log_signal.emit(f"Checkpoint saved to: {chk_path}\n")

                results["cube_files"] = []
                self.result_signal.emit(results)
                self.finished_signal.emit()

            except Exception as e:
                traceback.print_exc()
                # Do not emit error_signal for user-initiated stops — the signal
                # is already disconnected, but emitting an InterruptedError as an
                # error dialog would be confusing/wrong.
                if not self._stop_requested:
                    self.error_signal.emit(str(e))
                else:
                    logging.info("[worker.py] Calculation stopped by user: %s", e)
            finally:
                # CRITICAL: Close/destroy the StreamToSignal BEFORE restoring streams.
                # This prevents delayed print() calls from trying to emit signals
                # after Worker cleanup, which causes segmentation faults.
                if "stream" in locals() and hasattr(stream, "close"):
                    try:
                        stream.close()  # Marks _destroyed = True
                    except Exception as _e:
                        logging.warning("[worker.py] silenced stream.close: %s", _e)
                self._stream = None  # Release GUI-facing reference

        except Exception as e:
            self.error_signal.emit(str(e) + "\n" + traceback.format_exc())

        finally:
            # Restore Python streams (Crucial for preventing threads crashing on reuse)
            if "original_stdout" in locals():
                sys.stdout = original_stdout
            if "original_stderr" in locals():
                sys.stderr = original_stderr

            # Restore C-Level FDs
            if "capturer" in locals():
                try:
                    capturer.__exit__(None, None, None)
                except Exception as _e:
                    logging.warning("[worker.py] silenced: %s", _e)

    def run_rigid_scan(self, mol, mf, params, results):
        self.log_signal.emit("\n===== Rigid Surface Scan =====\n")

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
            self.log_signal.emit(
                "Warning: rdDetermineBonds not found. Group rotation might fail.\n"
            )
        except Exception as e:
            self.log_signal.emit(f"Warning deriving connectivity: {e}\n")

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
            self.log_signal.emit(
                f"Sanitization warning: {e}. Attempting partial update.\n"
            )
            try:
                rw_mol.UpdatePropertyCache(strict=False)
                Chem.GetSymmSSSR(rw_mol)
            except Exception as _e:
                logging.warning("[worker.py] silenced: %s", _e)

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
                self.log_signal.emit(f"Rigid scan stopped by user after {i} step(s).\n")
                break
            self.log_signal.emit(f"Step {i + 1}/{steps}: {stype} = {val:.4f} ... ")

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
                self.log_signal.emit(f"Geometry set failed: {e}\n")
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
                self.log_signal.emit(f"E = {e_tot:.6f} Ha\n")
            else:
                self.log_signal.emit(f"E = {e_tot:.6f} Ha  ** SCF NOT CONVERGED **\n")

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
        self.log_signal.emit(f"Scan results saved to {csv_path}\n")

        # Save Trajectory XYZ
        traj_path = os.path.join(self.out_dir, "scan_trajectory.xyz")
        with open(traj_path, "w") as f:
            f.write("\n".join(trajectory))

    def run_relaxed_scan(self, mol, mf, params, results):
        self.log_signal.emit(
            "\n===== Relaxed Surface Scan (Constrained Optimization) =====\n"
        )

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
            self.log_signal.emit(
                "Ensuring initial SCF convergence before scanning...\n"
            )
            mf.kernel()

        for i, val in enumerate(scan_values):
            # Cooperative stop check
            if self._stop_requested:
                self.log_signal.emit(
                    f"Relaxed scan stopped by user after {i} step(s).\n"
                )
                break
            self.log_signal.emit(
                f"\nStep {i + 1}/{steps}: Constrained {stype} = {val:.4f}\n"
            )

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

            self.log_signal.emit(f"  Constraint: {g_type} {atom_str} = {val:.6f}\n")

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
                        # self.log_signal.emit(f"  > Seeding guess from {os.path.basename(src_chk)}\n")
                except Exception as e_seed:
                    self.log_signal.emit(
                        f"  Warning: Failed to seed initial guess: {e_seed}\n"
                    )
                # --------------------------------------------------------

                mol_eq = optimize(step_mf, constraints=const_file)

                # Force an explicit SCF calculation on the final optimized structure
                # to ensure the energy is 100% accurate and matches the mol_eq coordinates.
                self.log_signal.emit(
                    "  Calculating final energy for optimized structure...\n"
                )
                step_converged = True
                try:
                    # reset(), not `.mol =`: the DFT grids and the solvent
                    # cavity are only rebuilt for the new geometry by reset().
                    step_mf.reset(mol_eq)
                    e_tot = step_mf.kernel()
                    step_converged = bool(getattr(step_mf, "converged", True))
                    self.log_signal.emit(
                        f"  ✓ Final optimized energy: {e_tot:.8f} Ha\n"
                    )
                except Exception as e:
                    self.log_signal.emit(
                        f"  ⚠ Failed final SCF, attempting fallback... {e}\n"
                    )
                    step_converged = False
                    if hasattr(step_mf, "e_tot") and step_mf.e_tot is not None:
                        e_tot = step_mf.e_tot
                    else:
                        # Recording 0.0 Ha put a ~76 Hartree spike in the
                        # energy profile that reads as a real barrier.
                        self.log_signal.emit(
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
                        actual_val = rdMolTransforms.GetDihedralDeg(
                            temp_mol.GetConformer(),
                            atoms[0],
                            atoms[1],
                            atoms[2],
                            atoms[3],
                        )
                except Exception as e:
                    self.log_signal.emit(
                        f"  Warning: Could not measure actual value: {e}\n"
                    )

                if abs(actual_val - val) > 0.01:  # Log if difference is significant
                    self.log_signal.emit(
                        f"  Target: {val:.4f}, Actual: {actual_val:.4f}\n"
                    )

                xyz_lines = [
                    f"{len(current_symbols)}",
                    f"Step {i + 1} {stype}={actual_val:.4f} E={e_tot:.6f} Ha",
                ]
                for s, c in zip(current_symbols, current_coords):
                    xyz_lines.append(f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")

                xyz_frame = "\n".join(xyz_lines)
                trajectory.append(xyz_frame)

                if step_converged:
                    self.log_signal.emit(f"  ✓ Converged: E = {e_tot:.8f} Ha\n")
                else:
                    self.log_signal.emit(
                        f"  ** SCF NOT CONVERGED **: E = {e_tot:.8f} Ha\n"
                    )

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
                self.log_signal.emit(f"  ✗ Optimization step {i + 1} failed: {e}\n")
                self.log_signal.emit(traceback.format_exc())
                # Break scan on failure
                break

        # Compile Results
        results["scan_results"] = scan_results
        results["scan_trajectory"] = trajectory

        # Save CSV
        csv_path = os.path.join(self.out_dir, "scan_results.csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(csv_lines))
        self.log_signal.emit(f"\nScan results saved to {csv_path}\n")

        # Save Trajectory XYZ
        traj_path = os.path.join(self.out_dir, "scan_trajectory.xyz")
        with open(traj_path, "w") as f:
            f.write("\n".join(trajectory))
        self.log_signal.emit(f"Scan trajectory saved to {traj_path}\n")


class PropertyWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)  # { "type": "mo"|"esp", "files": [...] }

    def __init__(self, chkfile, tasks, out_dir):
        """
        tasks: list of dicts, e.g. [{"type": "mo", "indices": [HOMO, LUMO]}, {"type": "esp"}]
        indices can be relative strings "HOMO", "HOMO-1" or integers.
        """
        super().__init__()
        self.chkfile = chkfile
        self.tasks = (
            tasks  # expecting list of orbital names like "HOMO", "LUMO+1" etc. or "ESP"
        )
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

    def run(self):
        if pyscf is None:
            self.error_signal.emit("PySCF not found.")
            return

        try:
            # Load SCF from checkpoint
            from pyscf import lib, scf, tools

            # Setup logging for this worker too
            log_file = os.path.join(self.out_dir, "pyscf.out")

            # C-Level Redirection
            capturer = CaptureStdOut(log_file)
            f_log = capturer.__enter__()

            # Python Redirection
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            stream = StreamToSignal(self.log_signal, target_stream=f_log)
            self._stream = stream
            sys.stdout = stream
            sys.stderr = stream

            # We need to reload the molecule and SCF object
            mol = lib.chkfile.load_mol(self.chkfile)
            mol.output = None  # Ensure no StreamToSignal assignment causing stat errors
            mol.stdout = stream  # Capture PySCF specifics
            mol.verbose = 4

            # Read SCF data
            scf_data = scf.chkfile.load(self.chkfile, "scf")
            mo_coeff = scf_data["mo_coeff"]
            mo_occ = scf_data["mo_occ"]

            # Determine HOMO/LUMO indices
            homo_idx = -1
            lumo_idx = -1

            # Robust HOMO/LUMO initialization for RHF/UHF/ROHF
            # Use threshold 0.1 to avoid numerical precision issues (e.g., 1e-12)
            occ_threshold = 0.1

            try:
                occs = (
                    mo_occ[0]
                    if isinstance(mo_occ, tuple)
                    or (hasattr(mo_occ, "ndim") and mo_occ.ndim == 2)
                    else mo_occ
                )
                homo_idx, lumo_idx = self._find_homo_lumo_1d(occs, occ_threshold)
            except Exception as e:
                self.log_signal.emit(f"Warning: Failed to auto-detect HOMO/LUMO: {e}\n")

            results = {"files": []}

            for task in self.tasks:
                if self._stop_requested:
                    self.log_signal.emit("Property generation stopped by user.\n")
                    break
                from .utils import get_unique_path  # noqa: PLC0415 — deferred to keep relative import out of module-level test context

                if task == "ESP":
                    # Generate Unique Paths
                    f_esp = get_unique_path(os.path.join(self.out_dir, "esp.cube"))
                    f_dens = get_unique_path(
                        os.path.join(self.out_dir, "density.cube")
                    )

                    # Total density for the MEP, for RHF / UHF / ROHF alike
                    dm_a, dm_b = self._spin_density_matrices(mo_coeff, mo_occ)
                    dm = dm_a + dm_b

                    self.log_signal.emit(
                        f"Generating ESP ({os.path.basename(f_esp)})...\n"
                    )
                    tools.cubegen.mep(mol, f_esp, dm)

                    self.log_signal.emit(
                        f"Generating Density ({os.path.basename(f_dens)})...\n"
                    )
                    tools.cubegen.density(mol, f_dens, dm)

                    results["files"].append(f_esp)
                    results["files"].append(f_dens)

                elif task == "SpinDensity":
                    if self._is_uhf_coeff(mo_coeff) or self._is_open_shell_occ(
                        mo_occ
                    ):
                        dm_a, dm_b = self._spin_density_matrices(mo_coeff, mo_occ)
                        f_spin = get_unique_path(
                            os.path.join(self.out_dir, "spin_density.cube")
                        )
                        self.log_signal.emit(
                            f"Generating Spin Density ({os.path.basename(f_spin)})...\n"
                        )
                        tools.cubegen.density(mol, f_spin, dm_a - dm_b)
                        results["files"].append(f_spin)
                    else:
                        self.log_signal.emit(
                            "Skipping Spin Density (Not an open-shell calculation or format unknown).\n"
                        )

                elif isinstance(task, str):
                    # Parse offset or absolute index
                    idx = -1
                    spin_suffix = ""
                    target_coeff = mo_coeff

                    # Detect Spin Request in label (internal convention)
                    # "15_HOMO_A" ? NO, task string is likely "HOMO" or "#5"
                    # But if we want to differentiate Alpha/Beta, the GUI must pass it.
                    # Currently strict GUI implementation doesn't pass suffix yet.
                    # But we can try to guess or handle it if we add it to the call.

                    # Assume task might be "HOMO_A" or "HOMO_B" logic?
                    # Or we just assume Alpha for now unless specified?

                    is_uhf = self._is_uhf_coeff(mo_coeff)

                    # Standard logic: if UHF, we need to know A or B.
                    # If not specified, maybe generate both? Or just Alpha?
                    # Let's check if task has specific format.

                    # NOTE: EnergyDiagramDialog generates "MO <n>" or "HOMO".
                    # We need to support "MO <n> A" or simple mapping.
                    # Let's look at the label logic in GUI later.
                    # For now, handle existing logic + suffix if present.

                    use_beta = False
                    if "_B" in task or "Beta" in task:
                        use_beta = True
                        task = task.replace("_B", "").replace("Beta", "").strip()
                        spin_suffix = "_B"
                    elif "_A" in task or "Alpha" in task:
                        task = task.replace("_A", "").replace("Alpha", "").strip()
                        spin_suffix = "_A"
                    elif is_uhf:
                        # Default to Alpha if UHF but not specified?
                        # Or maybe A is default suffix
                        spin_suffix = "_A"

                    if is_uhf:
                        c_a, c_b, o_a, o_b = self._unpack_uhf_coeff(mo_coeff, mo_occ)
                        if use_beta:
                            target_coeff = c_b
                            target_occ = o_b
                        else:
                            target_coeff = c_a
                            target_occ = o_a
                        homo_idx, lumo_idx = self._find_homo_lumo_1d(
                            target_occ, occ_threshold
                        )
                    else:
                        target_coeff = mo_coeff
                        # homo_idx already calc for RHF

                    try:
                        # Improved Task Parsing for "MO <idx>_<Label>" format
                        # Explicit regex for "MO <index>_<Label>" (e.g. MO 15_HOMO)
                        mo_lbl_match = re.search(r"MO\s+(\d+)_([A-Za-z0-9+-]+)", task)
                        clean_lbl = "MO"  # Default

                        if mo_lbl_match:
                            # e.g. "MO 15_HOMO" -> idx=14, lbl="HOMO"
                            idx = (
                                int(mo_lbl_match.group(1)) - 1
                            )  # Convert 1-based to 0-based
                            clean_lbl = mo_lbl_match.group(2)

                        # Case 1: Relative to HOMO/LUMO (Legacy/Manual)
                        elif "HOMO" in task:
                            base = homo_idx
                            clean_lbl = "HOMO"
                            if "+" in task:
                                offset = int(task.split("+")[1])
                                idx = base + offset
                                clean_lbl = f"HOMO+{offset}"
                            elif "-" in task:
                                offset = int(task.split("-")[1])
                                idx = base - offset
                                clean_lbl = f"HOMO-{offset}"
                            else:
                                idx = base
                        elif "LUMO" in task:
                            base = lumo_idx
                            clean_lbl = "LUMO"
                            if "+" in task:
                                offset = int(task.split("+")[1])
                                idx = base + offset
                                clean_lbl = f"LUMO+{offset}"
                            elif "-" in task:
                                offset = int(task.split("-")[1])
                                idx = base - offset
                                clean_lbl = f"LUMO-{offset}"
                            else:
                                idx = base

                        # Case 2: Explicit "MO <n>" or just numbers
                        elif "MO" in task or task.isdigit() or task.startswith("#"):
                            # "MO 15", "15", "#15", "#15_HOMO", etc.
                            # Robust digit extraction using Regex
                            # This handles "11SO" bug by ignoring all non-digits
                            clean_task = re.sub(
                                r"\D", "", task
                            )  # \D matches non-digits
                            if clean_task:
                                val = int(clean_task)
                                if task.startswith("#"):
                                    idx = val  # Internal 0-based index
                                else:
                                    idx = val - 1  # User 1-based index (MO 1 = Index 0)
                            else:
                                raise ValueError(f"Unknown task format: {task}")
                        else:
                            pass  # ... same error handling ...

                    except Exception as e:
                        self.log_signal.emit(f"Error parsing orbital: {task} ({e})\n")
                        continue

                    if idx < 0 or idx >= target_coeff.shape[1]:
                        self.log_signal.emit(
                            f"Orbital index {idx} out of bounds for {task}\n"
                        )
                        continue

                    clean_lbl = f"MO_{idx}"  # Default fallback
                    if idx <= homo_idx:
                        diff = homo_idx - idx
                        clean_lbl = "HOMO" if diff == 0 else f"HOMO-{diff}"
                    elif idx >= lumo_idx:
                        diff = idx - lumo_idx
                        clean_lbl = "LUMO" if diff == 0 else f"LUMO+{diff}"

                    rel_label = clean_lbl

                    prefix_idx = idx + 1  # 1-based for user-facing filename
                    if is_uhf:
                        s_char = "a" if "_A" in spin_suffix else "b"
                        fname = f"{prefix_idx:03d}{s_char}_{rel_label}.cube"
                    else:
                        fname = f"{prefix_idx:03d}_{rel_label}.cube"

                    # Sanitization: Ensure safe filenames but keep readable
                    # fname = fname.replace(" ", "") # User requested spaces in name
                    f_path_base = os.path.join(self.out_dir, fname)

                    f_path = get_unique_path(f_path_base)

                    if getattr(self, "_stop_requested", False):
                        break

                    self.log_signal.emit(
                        f"Generating {os.path.basename(f_path)} (Index {idx}{spin_suffix})...\n"
                    )
                    tools.cubegen.orbital(mol, f_path, target_coeff[:, idx])
                    results["files"].append(f_path)

            self.result_signal.emit(results)
            self.finished_signal.emit()

        except Exception as e:
            if not getattr(self, "_stop_requested", False):
                self.error_signal.emit(str(e) + "\n" + traceback.format_exc())
            else:
                logging.info("[worker.py] PropertyWorker stopped by user: %s", e)

        finally:
            if getattr(self, "_stream", None) is not None and hasattr(
                self._stream, "close"
            ):
                try:
                    self._stream.close()
                except Exception:
                    pass
            self._stream = None

            if "original_stdout" in locals():
                sys.stdout = original_stdout
            if "original_stderr" in locals():
                sys.stderr = original_stderr

            # Restore C-Level FDs
            if "capturer" in locals():
                try:
                    capturer.__exit__(None, None, None)
                except Exception:
                    pass


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

    def run(self):
        if pyscf is None:
            self.error_signal.emit("PySCF not found.")
            return

        try:
            from pyscf import lib, scf

            results = {"out_dir": os.path.dirname(self.chkfile)}
            base_dir = os.path.dirname(self.chkfile)

            # Check if this is a scan/tddft/freq-only folder (no checkpoint needed)
            has_scan = os.path.exists(os.path.join(base_dir, "scan_results.csv"))
            has_tddft = os.path.exists(os.path.join(base_dir, "tddft_results.json"))
            has_freq = os.path.exists(os.path.join(base_dir, "freq_analysis.json"))

            # If only auxiliary data exists (no checkpoint), load it and return
            if (has_scan or has_tddft or has_freq) and not os.path.exists(self.chkfile):
                # Load scan data
                if has_scan:
                    try:
                        scan_csv = os.path.join(base_dir, "scan_results.csv")
                        results["scan_results"] = self._load_scan_csv(scan_csv)
                        results["scan_type"] = self.load_scan_type(base_dir)
                        scan_traj = os.path.join(base_dir, "scan_trajectory.xyz")
                        if os.path.exists(scan_traj):
                            results["scan_trajectory_path"] = scan_traj
                    except Exception as e:
                        logging.warning(
                            "[worker.py] LoadWorker: failed to load scan: %s", e
                        )

                # Load TDDFT data
                if has_tddft:
                    try:
                        with open(
                            os.path.join(base_dir, "tddft_results.json"), "r"
                        ) as f:
                            tddft_data = json.load(f)
                            if "tddft_data" in tddft_data:
                                results["tddft_data"] = tddft_data["tddft_data"]
                    except Exception as e:
                        logging.warning(
                            "[worker.py] LoadWorker: failed to load TDDFT: %s", e
                        )

                # Load frequency data
                if has_freq:
                    try:
                        with open(
                            os.path.join(base_dir, "freq_analysis.json"), "r"
                        ) as f:
                            freq_json = json.load(f)
                        # Same unpacking as the checkpoint path below: the
                        # viewer reads freq_data["freqs"], not the file root.
                        if "freq_data" in freq_json:
                            results["freq_data"] = freq_json["freq_data"]
                        if "thermo_data" in freq_json:
                            results["thermo_data"] = freq_json["thermo_data"]
                    except Exception as e:
                        logging.warning(
                            "[worker.py] LoadWorker: failed to load freq: %s", e
                        )

                if self._stop_requested:
                    return
                self.finished_signal.emit(results)
                return

            # Original checkpoint loading logic
            # Load Molecule
            mol = lib.chkfile.load_mol(self.chkfile)
            if self._stop_requested:
                return

            # Load SCF Data
            scf_data = scf.chkfile.load(self.chkfile, "scf")
            mo_energy = scf_data.get("mo_energy", None)
            mo_occ = scf_data.get("mo_occ", None)

            # Identify Type (Enhanced: UHF, RHF, ROKS, ROHF)
            scf_type = "RHF"

            # Step 1: Check for Unrestricted (UHF/UKS)
            is_uhf = False
            if isinstance(mo_energy, tuple):
                is_uhf = True
            elif (
                isinstance(mo_energy, list)
                and len(mo_energy) == 2
                and isinstance(mo_energy[0], (list, np.ndarray))
            ):
                is_uhf = True
            elif isinstance(mo_energy, np.ndarray) and mo_energy.ndim == 2:
                is_uhf = True

            # Step 2: Check for Restricted Open-shell (ROKS/ROHF)
            # ROKS has 2D mo_occ: shape (2, N) for Alpha/Beta occupancies
            # and contains partial occupancy (values near 1.0)
            if not is_uhf:  # Only check if not already identified as UHF
                try:
                    if isinstance(mo_occ, np.ndarray) and mo_occ.ndim == 2:
                        # Check for partial occupancy (SOMO signature: occ ≈ 1.0)
                        has_partial_occ = False
                        for occ_val in mo_occ.flatten():
                            if 0.5 < occ_val < 1.5:  # Near 1.0 (SOMO)
                                has_partial_occ = True
                                break
                        if has_partial_occ:
                            scf_type = "ROKS"
                    elif isinstance(mo_occ, np.ndarray) and mo_occ.ndim == 1:
                        # What PySCF actually writes for ROHF/ROKS: one 1-D
                        # array of 0/1/2, a SOMO carrying occupation 1.
                        if np.any((mo_occ > 0.5) & (mo_occ < 1.5)):
                            scf_type = "ROKS"
                    elif isinstance(mo_occ, list):
                        # Handle list of lists case
                        if len(mo_occ) == 2 and all(
                            isinstance(x, (list, np.ndarray)) for x in mo_occ
                        ):
                            has_partial_occ = False
                            for sublist in mo_occ:
                                for occ_val in (
                                    sublist
                                    if isinstance(sublist, list)
                                    else sublist.tolist()
                                ):
                                    if 0.5 < occ_val < 1.5:
                                        has_partial_occ = True
                                        break
                                if has_partial_occ:
                                    break
                            if has_partial_occ:
                                scf_type = "ROKS"
                except Exception as _e:
                    # If ROKS detection fails, default to RHF (safe fallback)
                    logging.warning("[worker.py] PySCF ROKS detection silenced: %s", _e)

            if is_uhf:
                scf_type = "UHF"
                # Convert to lists for JSON/Qt safety
                try:
                    if isinstance(mo_energy, tuple):
                        mo_energy = [
                            e.tolist() if hasattr(e, "tolist") else list(e)
                            for e in mo_energy
                        ]
                        mo_occ = [
                            o.tolist() if hasattr(o, "tolist") else list(o)
                            for o in mo_occ
                        ]
                    else:
                        # Numpy 2D case
                        if hasattr(mo_energy, "tolist"):
                            mo_energy = mo_energy.tolist()
                        if hasattr(mo_occ, "tolist"):
                            mo_occ = mo_occ.tolist()
                except Exception as _e:
                    logging.warning(
                        "[worker.py] Array conversion failed on UHF: %s", _e
                    )
            else:
                # RHF/ROKS Case
                try:
                    if hasattr(mo_energy, "tolist"):
                        mo_energy = mo_energy.tolist()
                    if hasattr(mo_occ, "tolist"):
                        mo_occ = mo_occ.tolist()
                except Exception as _e:
                    logging.warning(
                        "[worker.py] Array conversion failed on RHF/ROKS: %s", _e
                    )

            # Attempt to extract optimized XYZ if present (or just current geometry)
            coords = mol.atom_coords(unit="Ang")
            symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

            xyz_lines = [f"{len(symbols)}", "Loaded from Checkpoint"]
            for s, c in zip(symbols, coords):
                xyz_lines.append(f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")

            optimized_xyz = "\n".join(xyz_lines)

            results = {
                "mo_energy": mo_energy,
                "mo_occ": mo_occ,
                "scf_type": scf_type,
                "loaded_xyz": optimized_xyz,
                "chkfile": self.chkfile,
                "out_dir": os.path.dirname(self.chkfile),
            }

            # --- Load Post-Process Data (Freq/Thermo) ---
            base_dir = os.path.dirname(self.chkfile)
            freq_file = os.path.join(base_dir, "freq_analysis.json")

            if os.path.exists(freq_file):
                try:
                    with open(freq_file, "r") as f:
                        data = json.load(f)
                        # Extract and merge
                        if "freq_data" in data:
                            results["freq_data"] = data["freq_data"]
                        if "thermo_data" in data:
                            results["thermo_data"] = data["thermo_data"]
                except Exception as e_json:
                    logging.warning(
                        "[worker.py] LoadWorker: failed to load freq json: %s", e_json
                    )

            # --- Load Scan Data ---
            scan_csv = os.path.join(base_dir, "scan_results.csv")
            if os.path.exists(scan_csv):
                try:
                    results["scan_results"] = self._load_scan_csv(scan_csv)
                    results["scan_type"] = self.load_scan_type(base_dir)
                except Exception as e_scan:
                    logging.warning(
                        "[worker.py] LoadWorker: failed to load scan csv: %s", e_scan
                    )

            scan_traj = os.path.join(base_dir, "scan_trajectory.xyz")
            if os.path.exists(scan_traj):
                # Just pass the path, viewer can read it on demand or we read valid frames
                results["scan_trajectory_path"] = scan_traj

            # --- Load TDDFT Data ---
            tddft_file = os.path.join(base_dir, "tddft_results.json")
            if os.path.exists(tddft_file):
                try:
                    with open(tddft_file, "r") as f:
                        tddft_data = json.load(f)
                        if "tddft_data" in tddft_data:
                            results["tddft_data"] = tddft_data["tddft_data"]
                except Exception as e_tddft:
                    logging.warning(
                        "[worker.py] LoadWorker: failed to load TDDFT json: %s", e_tddft
                    )

            # --- Dipole / Mulliken charges ---
            props_file = os.path.join(base_dir, "properties.json")
            if os.path.exists(props_file):
                try:
                    with open(props_file, "r", encoding="utf-8") as f:
                        results.update(json.load(f))
                except Exception as e_props:
                    logging.warning(
                        "[worker.py] LoadWorker: failed to load properties: %s",
                        e_props,
                    )

            if self._stop_requested:
                return
            self.finished_signal.emit(results)

        except Exception as e:
            if self._stop_requested:
                return
            self.error_signal.emit(str(e) + "\n" + traceback.format_exc())
