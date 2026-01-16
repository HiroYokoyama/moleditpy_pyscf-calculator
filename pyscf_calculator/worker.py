import sys
import os
import io
import tempfile
import traceback
import tempfile
import traceback
import numpy as np
import math
import re
from PyQt6.QtCore import QThread, pyqtSignal

# We import pyscf inside the thread or check availability
try:
    import pyscf
    from pyscf import gto, scf, dft, tools, lo
except ImportError:
    pyscf = None


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
        self.log_file = open(self.filename, 'a', buffering=1, encoding='utf-8')
        
        # Get FDs
        try:
            # Use __stdout__ to ensure we get the real OS FD, even if sys.stdout was MonkeyPatched/Wrapper
            self.original_stdout_fd = sys.__stdout__.fileno()
        except:
            self.original_stdout_fd = 1 # Fallback to standard FD 1

        try:
             self.original_stderr_fd = sys.__stderr__.fileno()
        except:
             self.original_stderr_fd = 2 # Fallback

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
        # Flush
        sys.stdout.flush()
        sys.stderr.flush()
        if hasattr(self, 'log_file'): self.log_file.flush()

        # Restore
        if self.saved_stdout_fd is not None:
            os.dup2(self.saved_stdout_fd, self.original_stdout_fd)
            os.close(self.saved_stdout_fd)
            
        if self.saved_stderr_fd is not None:
            os.dup2(self.saved_stderr_fd, self.original_stderr_fd)
            os.close(self.saved_stderr_fd)
            
        if hasattr(self, 'log_file'): self.log_file.close()

class StreamToSignal(io.TextIOBase):
    def __init__(self, signal, target_stream=None):
        self.signal = signal
        self.target_stream = target_stream
        
    def write(self, text):
        self.signal.emit(text)
        if self.target_stream:
            try:
                self.target_stream.write(text)
                self.target_stream.flush()
            except: pass
            
    def flush(self):
        if self.target_stream:
            try: self.target_stream.flush()
            except: pass
            
    def close(self):
        # Do not close system stdout here
        pass

    @property
    def encoding(self):
        if self.target_stream and hasattr(self.target_stream, 'encoding'):
            return self.target_stream.encoding
        return "utf-8"

    def isatty(self):
        if self.target_stream and hasattr(self.target_stream, 'isatty'):
            return self.target_stream.isatty()
        return False

class PySCFWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict) # Pass back data like XYZ, Cube paths

    def __init__(self, xyz_str, config):
        super().__init__()
        self.xyz_str = xyz_str
        self.config = config

    def run(self):
        if pyscf is None:
            self.error_signal.emit("PySCF is not installed in the python environment.")
            return

        try:
            # Prepare Output Root Directory
            root_dir = self.config.get("out_dir")
            if not root_dir:
                root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
                
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
            sys.stdout = stream
            sys.stderr = stream
            
            # Configure Threads
            n_threads = self.config.get("threads", 0)
            if n_threads > 0:
                pyscf.lib.num_threads(n_threads)
            
            # Prepare Charge and Spin (Convert M -> 2S)
            charge = self.config.get("charge", 0)
            try:
                # User inputs Multiplicity (M)
                # GUI now sends "1 (Singlet)", "2 (Doublet)", etc.
                spin_str = str(self.config.get("spin", "1"))
                if " " in spin_str:
                    spin_mult = int(spin_str.split(" ")[0])
                else:
                    spin_mult = int(spin_str)
                    
                # PySCF expects 2S = M - 1
                spin_2s = spin_mult - 1
                if spin_2s < 0: spin_2s = 0
            except:
                spin_2s = 0
            
            # Setup Molecule
            try:
                mol = gto.M(
                    atom=self.xyz_str,
                    basis=self.config.get("basis", "sto-3g"),
                    charge=charge,
                    spin=spin_2s,
                    verbose=4,
                    output=None,
                    max_memory=self.config.get("memory", 4000)
                )
                mol.stdout = stream
                mol.verbose = 4
                mol.build()
            except (RuntimeError, ValueError) as e_mol:
                # Catch specific PySCF errors (e.g. Spin/Charge mismatch)
                msg = str(e_mol)
                self.error_signal.emit(f"Molecule Build Failed: {msg}\nCheck Charge and Multiplicity settings.")
                return
            
            # Log Parallelism Info
            try:
                n_threads = pyscf.lib.num_threads()
                self.log_signal.emit(f"PySCF running with {n_threads} OpenMP threads.\n")
            except: pass
            
            # Save Input Log
            inp_file = os.path.join(out_dir, "pyscf_input.py")
            with open(inp_file, "w") as f:
                f.write("from pyscf import gto, scf, dft\n")
                f.write(f"mol = gto.M(atom='''{self.xyz_str}''', \n")
                f.write(f"    basis='{self.config.get('basis')}', \n")
                f.write(f"    charge={charge}, \n")
                f.write(f"    spin={spin_2s}, \n")
                f.write(f"    verbose=4)\n")
                f.write("mol.build()\n")
                f.write(f"# Job Type: {self.config.get('job_type')}\n")
                f.write(f"# Method: {self.config.get('method')}\n")
                if "KS" in self.config.get("method"):
                     f.write(f"# Functional: {self.config.get('functional')}\n")
                     f.write(f"mf = dft.{self.config.get('method')}(mol)\n")
                     f.write(f"mf.xc = '{self.config.get('functional')}'\n")
                else:
                     f.write(f"mf = scf.{self.config.get('method')}(mol)\n")
                f.write("mf.kernel()\n")
                
                if "TDDFT" in self.config.get('job_type'):
                    f.write("\n# TDDFT Calculation\n")
                    f.write("from pyscf import tdscf\n")
                    # Simplified selection for generated script
                    f.write(f"td = tdscf.TDDFT(mf) if 'KS' in '{self.config.get('method')}' else tdscf.TDHF(mf)\n")
                    f.write(f"td.nstates = {self.config.get('nstates', 10)}\n")
                    f.write("td.verbose = 4\n")
                    f.write("td.kernel()\n")

            # Select Method
            method_name = self.config.get("method", "RHF")
            functional = self.config.get("functional", "b3lyp")
            
            # Auto-adjust method for Open Shell if needed?
            # If user selected RHF/RKS but spin != 0, PySCF crashes or warns.
            # We should probably force U if spin != 0.
            # But relying on user selection (UHF/UKS) is safer behavior for now.
            # Ideally the GUI updates the combo, but backend safety is good.
            
            if spin_2s != 0:
                if method_name == "RHF": 
                    method_name = "UHF"
                    self.log_signal.emit("Switching to UHF due to spin != 0.\n")
                elif method_name == "RKS": 
                    method_name = "UKS"
                    self.log_signal.emit("Switching to UKS due to spin != 0.\n")

            mf = None
            if method_name == "RHF":
                mf = scf.RHF(mol)
            elif method_name == "UHF":
                mf = scf.UHF(mol)
            elif method_name == "RKS":
                mf = dft.RKS(mol)
                mf.xc = functional
            elif method_name == "UKS":
                mf = dft.UKS(mol)
                mf.xc = functional
            # --- Added: RO Support ---
            elif method_name == "ROHF":
                mf = scf.ROHF(mol)
            elif method_name == "ROKS":
                mf = dft.ROKS(mol)
                mf.xc = functional
            # ------------------------
            else:
                # Default fallback or error
                # If spin != 0 and RHF selected, we might crash if not handled, 
                # but explicit RO selection handles it.
                if spin_2s != 0 and "R" in method_name and "RO" not in method_name:
                     # e.g. RHF with spin. PySCF might complain.
                     # But we follow user instruction strictly.
                     pass
                     
                if method_name not in ["RHF", "UHF", "RKS", "UKS", "ROHF", "ROKS"]:
                     raise ValueError(f"Unknown method {method_name}")
            
            # Ensure Checkpoint is in the new job folder
            chk_path = os.path.join(self.out_dir, "pyscf.chk")
            mf.chkfile = chk_path
            
            # Apply Advanced Settings
            mf.max_cycle = self.config.get("max_cycle", 100)
            try:
                tol_str = self.config.get("conv_tol", "1e-9")
                mf.conv_tol = float(tol_str)
            except: pass

            # Redirect stdout/stderr to capture ALL output (including geometric)
            sys.stdout.flush()
            original_stdout = sys.stdout
            sys.stdout = stream
            
            # Explicitly set pyscf logger stream too for global usage
            from pyscf import lib
            lib.logger.TIMER_LEVEL = 0  # Reduces some noise, or keeps it standard
            
            try:
                # Prepare job type
                job_type = self.config.get("job_type", "Energy")
                results = {}
    
                if "Optimization" in job_type:
                    # Note: PySCF geometric optimization requires 'geometric' or 'berny'
                    self.log_signal.emit(f"Starting Geometry Optimization using {method_name}...\n")
                    try:
                        from pyscf.geomopt.geometric_solver import optimize
                        mol_eq = optimize(mf)
                        
                        # Convert optimized geometry to XYZ string (in Angstroms)
                        coords = mol_eq.atom_coords(unit='Ang')
                        symbols = [mol_eq.atom_symbol(i) for i in range(mol_eq.natm)]
                        
                        xyz_lines = [f"{len(symbols)}", "Generated by PySCF Optimization"]
                        for s, c in zip(symbols, coords):
                            xyz_lines.append(f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
                        
                        results["optimized_xyz"] = "\n".join(xyz_lines)
                        
                        # Re-run single point on optimized geometry for properties
                        # We need to recreate the MF object on new mol
                        if method_name == "RHF": mf = scf.RHF(mol_eq)
                        elif method_name == "UHF": mf = scf.UHF(mol_eq)
                        elif method_name == "RKS": 
                            mf = dft.RKS(mol_eq)
                            mf.xc = functional
                        elif method_name == "UKS": 
                            mf = dft.UKS(mol_eq)
                            mf.xc = functional
                        # --- Added: RO Support for Optimization Re-init ---
                        elif method_name == "ROHF": mf = scf.ROHF(mol_eq)
                        elif method_name == "ROKS":
                            mf = dft.ROKS(mol_eq)
                            mf.xc = functional
                        # -----------------------------------------------
                        
                        # Run Energy on optimized
                        mf.chkfile = chk_path
                        # mf.kernel() # Will be run by next steps or explicitly
                        mol = mol_eq # Update main mol ref for checkfile
                        
                    except ImportError:
                        self.log_signal.emit("\nWARNING: geometric-lib not found. Trying internal optimizer (Berny)...\n")
                        try:
                             from pyscf.geomopt.berny_solver import optimize as optimize_berny
                             mol_eq = optimize_berny(mf)
                             
                             # Convert optimized geometry to XYZ string (in Angstroms)
                             coords = mol_eq.atom_coords(unit='Ang')
                             symbols = [mol_eq.atom_symbol(i) for i in range(mol_eq.natm)]
                             xyz_lines = [f"{len(symbols)}", "Generated by PySCF Optimization (Berny)"]
                             for s, c in zip(symbols, coords):
                                xyz_lines.append(f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
                             results["optimized_xyz"] = "\n".join(xyz_lines)
    
                             # Recalculate energy
                             mf.chkfile = chk_path
                             # mf.kernel()
                             mol = mol_eq
                        except ImportError:
                             self.error_signal.emit("Neither 'geometric' nor 'berny' optimizer found. Please install 'geometric' (pip install geometric).")
                             return

                # Ensure Energy is calculated (if not done by Opt or if detached)
                # Optimization updates mf but we need to ensure kernel is run for properties
                if "Optimization" in job_type or "Energy" in job_type or "Frequency" in job_type:
                     if not mf.e_tot: 
                        self.log_signal.emit(f"Running partial energy calculation using {method_name}...\n")
                        
                        # User Request: Apply Symmetry Breaking for UKS/UHF
                        # "mix_estimation" logic to prevent Alpha=Beta trap
                        # Condition: UKS/UHF, Spin > 0, AND Option Enabled (Default True)
                        should_break = self.config.get("break_symmetry", True)
                        
                        if should_break and method_name in ["UHF", "UKS"] and spin_2s > 0:
                            self.log_signal.emit("Applying Symmetry Breaking to Initial Guess (mix_estimation)...\n")
                            try:
                                # 1. Base Guess
                                dm_guess = mf.get_init_guess(key='minao')
                                
                                # 2. Mix Alpha/Beta
                                if "KS" in method_name: # UKS
                                    dm_mix = dft.uks.mulliken_meta(mol, dm_guess, verbose=4)
                                else: # UHF
                                    dm_mix = scf.uhf.mulliken_meta(mol, dm_guess, verbose=4)
                                    
                                # 3. Kernel with broken symmetry guess
                                mf.kernel(dm0=dm_mix)
                            except Exception as e:
                                self.log_signal.emit(f"WARNING: Symmetry breaking failed ({str(e)}). Proceeding with standard initial guess.\n")
                                mf.kernel()
                        else:
                            mf.kernel()
                
                if "Frequency" in job_type:
                    self.log_signal.emit(f"Starting Frequency Analysis using {method_name}...\n")
                    
                    # Ensure we have a converged SCF on the current molecule
                    if not mf.e_tot: 
                        self.log_signal.emit("Running SCF for Frequency Analysis...\n")
                        mf.kernel()
                        
                    if not mf.converged:
                         self.log_signal.emit("WARNING: SCF did not converge before Frequency Analysis. Results may be inaccurate.\n")

                    self.log_signal.emit("Calculating Hessian...\n")
                    try:
                        h_obj = mf.Hessian()
                        hessian = h_obj.kernel()
                        
                        from pyscf.hessian import thermo
                        self.log_signal.emit("Performing Harmonic Analysis...\n")
                        freq_res = thermo.harmonic_analysis(mol, hessian)
                        
                        # Calculate Thermo
                        self.log_signal.emit("Calculating Thermodynamic Properties...\n")
                        # Ensure temp/pressure are floats
                        T = float(self.config.get("temperature", 298.15))
                        P = float(self.config.get("pressure", 101325))
                        t_data = thermo.thermo(mf, freq_res['freq_au'], temperature=T, pressure=P)
                        
                        # Store data for GUI Visualizer
                        # Check for IR intensity (not always available in standard harmonic_analysis)
                        intensities = freq_res.get("infra_red_intensity", None) 
                        
                        # if intensities is None:
                        #     self.log_signal.emit("Calculating IR Intensities numerically (Finite Difference of Dipole)...\n")
                        #     try:
                        #         intensities = self.calculate_ir_intensities(mol, mf, freq_res["norm_mode"])
                        #     except Exception as e_int:
                        #         self.log_signal.emit(f"IR Intensity calculation failed: {e_int}\n")
                        #         intensities = None
                        
                        results["freq_data"] = {
                            "freqs": freq_res["freq_wavenumber"].tolist(),
                            "modes": freq_res["norm_mode"].tolist(),
                            "intensities": intensities.tolist() if hasattr(intensities, 'tolist') else intensities
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
                                if hasattr(obj, 'tolist'):  # numpy array
                                    obj = obj.tolist()
                                if isinstance(obj, (list, tuple)):
                                    return [make_json_safe(item) for item in obj]
                                if isinstance(obj, dict):
                                    return {k: make_json_safe(v) for k, v in obj.items()}
                                # Fallback: convert to string
                                return str(obj)
                            
                            results["thermo_data"] = make_json_safe(t_data)
                        
                        # Save freq_data and thermo_data to JSON file
                        freq_json_path = os.path.join(self.out_dir, "freq_analysis.json")
                        try:
                            import json
                            
                            # Custom JSON encoder to handle all edge cases
                            class SafeEncoder(json.JSONEncoder):
                                def default(self, obj):
                                    if hasattr(obj, 'tolist'):
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
                            
                            with open(freq_json_path, 'w') as f:
                                json.dump(save_data, f, indent=2, cls=SafeEncoder)
                            self.log_signal.emit(f"Frequency data saved to: {freq_json_path}\n")
                        except Exception as e_save:
                            self.log_signal.emit(f"Warning: Failed to save frequency JSON: {e_save}\n")
                            self.log_signal.emit(traceback.format_exc())
                            
                    except Exception as e_freq:
                        self.log_signal.emit(f"Frequency analysis failed: {e_freq}\n{traceback.format_exc()}\n")

                if "TDDFT" in job_type:
                    self.log_signal.emit(f"Starting TDDFT Calculation...\n")
                    if not mf.e_tot:
                        self.log_signal.emit("Running SCF for TDDFT...\n")
                        mf.kernel()
                    
                    if not mf.converged:
                         self.log_signal.emit("WARNING: SCF did not converge before TDDFT. Results may be inaccurate.\n")

                    try:
                        from pyscf import tdscf
                        
                        # Select TDDFT Method
                        # For RHF/UHF -> TDHF
                        # For RKS/UKS -> TDDFT (or TDA)
                        
                        td_obj = None
                        method_base = method_name.replace("R", "").replace("U", "").replace("RO", "") # 'HF', 'KS'
                        
                        # Simple dispatch based on MF class is usually safer if unsure
                        # But explicit class usage allows TDA control if we add it later
                        
                        if "KS" in method_base: # RKS or UKS
                             # Default to full TDDFT
                             # Could use TDA if we add an option later: tdscf.TDA(mf)
                             td_obj = tdscf.TDDFT(mf)
                        else: # HF
                             td_obj = tdscf.TDHF(mf)

                        nstates = int(self.config.get("nstates", 10))
                        td_obj.nstates = nstates
                        td_obj.verbose = 4
                        # Redirect output? td_obj uses lib.logger which respects global stream we set?
                        # Or explicitly set stdout
                        try:
                            td_obj.stdout = stream
                        except: pass
                        
                        self.log_signal.emit(f"Calculating {nstates} Excited States...\n")
                        td_obj.kernel()
                        
                        # Reporting
                        self.log_signal.emit("\n===== TDDFT Results =====\n")
                        self.log_signal.emit(f"{'State':<6} {'Energy (eV)':<12} {'Osc. Str.':<10}\n")
                        self.log_signal.emit("-" * 30 + "\n")
                        
                        # Results Extraction
                        # td_obj.e_tot are total energies of Excited States
                        # Excitation Energy = E_exc_state - E_ground_state
                        
                        energies_exc = td_obj.e_tot
                        # e_tot can be a list or numpy array
                        if hasattr(energies_exc, 'tolist'): energies_exc = energies_exc.tolist()
                        elif isinstance(energies_exc, float): energies_exc = [energies_exc]
                        
                        # Oscillator Strengths
                        try:
                            oscs = td_obj.oscillator_strength()
                            if hasattr(oscs, 'tolist'): oscs = oscs.tolist()
                            if isinstance(oscs, float): oscs = [oscs]
                        except:
                            oscs = [0.0] * len(energies_exc)
                            
                        HARTREE_TO_EV = 27.2114
                        e_ground = mf.e_tot
                        
                        tddft_list = []
                        
                        for i, e_exc_tot in enumerate(energies_exc):
                            exc_energy_au = e_exc_tot - e_ground
                            exc_ev = exc_energy_au * HARTREE_TO_EV
                            osc = oscs[i] if i < len(oscs) else 0.0
                            
                            self.log_signal.emit(f"{i+1:<6} {exc_ev:<12.4f} {osc:<10.4f}\n")
                            
                            tddft_list.append({
                                "state": i+1,
                                "energy_total": e_exc_tot,
                                "excitation_energy_ev": exc_ev,
                                "oscillator_strength": osc
                            })
                            
                        results["tddft_data"] = tddft_list
                        self.log_signal.emit("-" * 30 + "\n")
                        
                    except Exception as e_td:
                        self.log_signal.emit(f"TDDFT calculation failed: {e_td}\n{traceback.format_exc()}\n")
                        
                if "Energy" == job_type: # Only Energy
                    # Already handled by top block but ensuring...
                    if not mf.e_tot: mf.kernel()
    
                # --- SAVE CHECKPOINT (ALWAYS) ---
                # Checkpoint is already set to self.out_dir/pyscf.chk and written by mf.kernel()
                
                chk_path = os.path.join(self.out_dir, "pyscf.chk")
                
                results.update({
                    "chkfile": chk_path,
                    "out_dir": self.out_dir
                })
                
                # Pass energy/occupancy to GUI for Diagram
                # Handle UHF (tuple) vs RHF (array)
                # Note: mo_energy might be list if loaded from chkfile without full object?
                # Check ndim or length.
                
                is_uhf = False
                if isinstance(mf.mo_energy, tuple):
                     is_uhf = True
                elif isinstance(mf.mo_energy, list) and len(mf.mo_energy) == 2:
                     # Check content types?
                     if hasattr(mf.mo_energy[0], '__len__'):
                         is_uhf = True
                elif hasattr(mf.mo_energy, 'ndim') and mf.mo_energy.ndim == 2:
                     is_uhf = True
                     
                if is_uhf:
                     # Convert to list of lists for JSON safety/GUI compatibility
                     # Use safe list conversion
                     def to_l(arr):
                         return arr.tolist() if hasattr(arr, 'tolist') else list(arr)
                     
                     if isinstance(mf.mo_energy, tuple):
                         e_a, e_b = mf.mo_energy
                         o_a, o_b = mf.mo_occ
                     else:
                         e_a, e_b = mf.mo_energy[0], mf.mo_energy[1]
                         o_a, o_b = mf.mo_occ[0], mf.mo_occ[1]
                         
                     results["mo_energy"] = [to_l(e_a), to_l(e_b)]
                     results["mo_occ"] = [to_l(o_a), to_l(o_b)]
                     results["scf_type"] = "UHF"
                else:
                     val = mf.mo_energy
                     occ = mf.mo_occ
                     results["mo_energy"] = val.tolist() if hasattr(val, 'tolist') else list(val)
                     results["mo_occ"] = occ.tolist() if hasattr(occ, 'tolist') else list(occ)
                     results["scf_type"] = "RHF"
                self.log_signal.emit(f"Checkpoint saved to: {chk_path}\n")
    
                # --- Post Analysis (Optional Auto-Vis) ---
                cube_files = []
                # Only if explicitly requested or job type implies it (omitted for now as requested)
                # The user removed the auto-vis checkbox.
                # So we just return the checkpoint.
                
                results["cube_files"] = [] 
                self.result_signal.emit(results)
                self.finished_signal.emit()

            except Exception as e:
                traceback.print_exc()
                self.error_signal.emit(str(e))
            finally:
                # Restore Python streams
                if 'original_stdout' in locals(): sys.stdout = original_stdout
                if 'original_stderr' in locals(): sys.stderr = original_stderr
                
                # Restore C-Level FDs
                if 'capturer' in locals():
                    try: capturer.__exit__(None, None, None)
                    except: pass

        except Exception as e:
            self.error_signal.emit(str(e) + "\n" + traceback.format_exc())


    def calculate_ir_intensities(self, mol, mf, modes, mass_weighted_modes=False):
        """
        Calculate IR intensities numerically using finite difference of dipole moments.
        
        Args:
            mol: PySCF Mole object
            mf: PySCF SCF object (converged)
            modes: List of normal mode vectors (natm, 3). 
                   *Important*: If 'modes' are raw eigenvectors from Hessian (mass-weighted),
                   set mass_weighted_modes=True. If they are already Cartesian displacements 
                   (visual modes), set False.
            mass_weighted_modes: Boolean, set True if input modes are mass-weighted eigenvectors.
            
        Returns:
            np.array: List of intensities in km/mol.
        """
        import numpy as np
        from pyscf import scf, dft
        
        # Check Unrestricted
        is_unrestricted = False
        if isinstance(mf, (scf.uhf.UHF, dft.uks.UKS)):
            is_unrestricted = True

        # 0. Pre-computation setup
        dm0 = mf.make_rdm1()
        
        # Dipole moment getter with optimization
        def get_dipole(mol_instance, dm_guess=None):
            # Create a new SCF object for the displaced geometry
            if hasattr(mf, 'xc'):
                if is_unrestricted: mf_temp = dft.UKS(mol_instance)
                else: mf_temp = dft.RKS(mol_instance)
                mf_temp.xc = mf.xc
                if hasattr(mf, 'grids'): mf_temp.grids = mf.grids
            else:
                if is_unrestricted: mf_temp = scf.UHF(mol_instance)
                else: mf_temp = scf.RHF(mol_instance)
            
            mf_temp.verbose = 0
            mf_temp.max_cycle = 100
            
            # Disable symmetry check
            mol_instance.symmetry = False 
            
            # Run SCF using the previous density matrix as a guess
            try:
                if dm_guess is not None:
                    mf_temp.kernel(dm0=dm_guess)
                else:
                    mf_temp.kernel()
            except:
                # Fallback to fresh start if guess fails
                mf_temp.kernel()
                
            return mf_temp.dip_moment(mol=mol_instance, unit='Debye', verbose=0)

        # 1. Compute Dipole Derivatives (d_mu / d_X)
        delta = 0.005 # Bohr (Standard step size)
        natm = mol.natm
        dip_derivs = np.zeros((natm, 3, 3)) # (Atom, Axis, Dipole_Component)
        
        coords_original = mol.atom_coords(unit='Bohr')
        
        self.log_signal.emit(f" > Computing Dipole Derivatives (Numerical 6N={6*natm} steps)...")
        
        # Logging progress since this can be slow
        steps_total = natm * 6
        step_count = 0
        
        for i in range(natm):
            for j in range(3): # x, y, z
                # Update coords +delta
                coords_plus = coords_original.copy()
                coords_plus[i, j] += delta 
                
                mol_plus = mol.copy()
                mol_plus.set_geom_(coords_plus, unit='Bohr')
                # Use dm0 from the ground state as the best guess
                mu_plus = get_dipole(mol_plus, dm_guess=dm0)
                
                # Update coords -delta
                coords_minus = coords_original.copy()
                coords_minus[i, j] -= delta
                
                mol_minus = mol.copy()
                mol_minus.set_geom_(coords_minus, unit='Bohr')
                mu_minus = get_dipole(mol_minus, dm_guess=dm0)
                
                # Central difference
                deriv = (mu_plus - mu_minus) / (2.0 * delta) 
                dip_derivs[i, j, :] = deriv 
                
        # 2. Project onto Normal Modes
        intensities = []
        
        # Convert derivs to Debye/Angstrom
        BOHR_TO_ANG = 0.529177210903
        factor_bohr_ang = 1.0 / BOHR_TO_ANG
        dip_derivs_ang = dip_derivs * factor_bohr_ang # Units: Debye / Angstrom
        
        # Mass handling: Need sqrt(amu) for conversion if modes are mass-weighted
        masses = mol.atom_masses() # amu
        
        for k, mode_vec in enumerate(modes): # mode_vec shape: (N_atoms, 3)
            d_mu_dQ = np.zeros(3) # Vector (mu_x, mu_y, mu_z) change
            
            for at in range(natm):
                for j in range(3): # x, y, z displacement coord
                    # Gradient: d_mu / d_R_cartesian
                    grad_vec = dip_derivs_ang[at, j, :] 
                    
                    displacement = mode_vec[at][j]
                    
                    # CORRECTION: Coordinate Transformation
                    # IR Intensity formula requires derivative w.r.t Normal Coordinate Q.
                    # Q has units of [Distance * sqrt(Mass)].
                    # If 'modes' are raw eigenvectors of Mass-Weighted Hessian (L_mw),
                    # then Cartesian displacement dR = L_mw * dQ / sqrt(Mass).
                    # Therefore, we must divide the mode coefficient by sqrt(mass) to get dR/dQ representation
                    # IF the input `modes` are not already converted to Cartesian displacements.
                    
                    if mass_weighted_modes:
                        # If modes are raw eigenvectors (L), divide by sqrt(mass)
                        term = grad_vec * displacement / np.sqrt(masses[at])
                    else:
                        # If modes are already Cartesian displacements (visual modes), use as is
                        term = grad_vec * displacement
                        
                    d_mu_dQ += term
            
            sq_val = np.dot(d_mu_dQ, d_mu_dQ)
            
            # Conversion factor: (Debye / (Angstrom * sqrt(amu)))^2 -> km/mol
            # 42.2561 is the correct prefactor for this unit set.
            inten = 42.2561 * sq_val
            intensities.append(inten)
            
        return np.array(intensities)

class PropertyWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict) # { "type": "mo"|"esp", "files": [...] }

    def __init__(self, chkfile, tasks, out_dir):
        """
        tasks: list of dicts, e.g. [{"type": "mo", "indices": [HOMO, LUMO]}, {"type": "esp"}]
        indices can be relative strings "HOMO", "HOMO-1" or integers.
        """
        super().__init__()
        self.chkfile = chkfile
        self.tasks = tasks # expecting list of orbital names like "HOMO", "LUMO+1" etc. or "ESP"
        self.out_dir = out_dir

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
            sys.stdout = stream
            sys.stderr = stream

            # We need to reload the molecule and SCF object
            mol = lib.chkfile.load_mol(self.chkfile)
            mol.output = None # Ensure no StreamToSignal assignment causing stat errors
            mol.stdout = stream # Capture PySCF specifics
            mol.verbose = 4
            
            # Read SCF data
            scf_data = scf.chkfile.load(self.chkfile, 'scf')
            mo_coeff = scf_data['mo_coeff']
            mo_occ = scf_data['mo_occ']
            
            # Determine HOMO/LUMO indices
            homo_idx = -1
            lumo_idx = -1
            
            # Robust HOMO/LUMO initialization for RHF/UHF/ROHF
            # Use threshold 0.1 to avoid numerical precision issues (e.g., 1e-12)
            occ_threshold = 0.1
            
            try:
                # Handle different mo_occ formats: tuple (UHF), 2D array (ROKS), 1D array (RHF)
                if isinstance(mo_occ, tuple):
                    # UHF: (alpha_occ, beta_occ) - use Alpha for HOMO/LUMO
                    occs = mo_occ[0]
                    for i, occ_val in enumerate(occs):
                        if occ_val > occ_threshold:
                            homo_idx = i
                        else:
                            lumo_idx = i
                            break
                elif hasattr(mo_occ, 'ndim') and mo_occ.ndim == 2:
                    # 2D array (ROKS): use first row (Alpha)
                    occs = mo_occ[0]
                    for i, occ_val in enumerate(occs):
                        if occ_val > occ_threshold:
                            homo_idx = i
                        else:
                            lumo_idx = i
                            break
                elif hasattr(mo_occ, 'shape'):
                    # 1D array (RHF/RKS)
                    for i, occ_val in enumerate(mo_occ):
                        if occ_val > occ_threshold:
                            homo_idx = i
                        else:
                            lumo_idx = i
                            break
                elif isinstance(mo_occ, (list, np.ndarray)):
                    # Fallback for lists or other iterables
                    for i, occ_val in enumerate(mo_occ):
                        if occ_val > occ_threshold:
                            homo_idx = i
                        else:
                            lumo_idx = i
                            break
            except Exception as e:
                self.log_signal.emit(f"Warning: Failed to auto-detect HOMO/LUMO: {e}\n")
            
            if lumo_idx == -1: lumo_idx = homo_idx + 1 # if full
            
            results = {"files": []}
            
            for task in self.tasks:
                if task == "ESP":
                    # Generate Unique Paths
                    from .utils import get_unique_path
                    
                    f_esp_base = os.path.join(self.out_dir, "esp.cube")
                    f_esp = get_unique_path(f_esp_base)

                    f_dens_base = os.path.join(self.out_dir, "density.cube")
                    f_dens = get_unique_path(f_dens_base)
                    
                    # For ESP we need density matrix
                    # Handle both RHF (array) and UHF (tuple)
                    
                    if isinstance(mo_coeff, (tuple, list)) or (isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 3):
                         # UHF case
                         from pyscf.scf import uhf
                         # Safe unpack
                         if isinstance(mo_coeff, tuple):
                             c_a, c_b = mo_coeff
                             o_a, o_b = mo_occ
                         else:
                             c_a, c_b = mo_coeff[0], mo_coeff[1]
                             o_a, o_b = mo_occ[0], mo_occ[1]
                             
                         dm_ab = uhf.make_rdm1((c_a, c_b), (o_a, o_b))
                         dm = dm_ab[0] + dm_ab[1] # Total density for MEP
                    else:
                         # RHF case
                         dm = scf.hf.make_rdm1(mo_coeff, mo_occ)
                    
                    self.log_signal.emit(f"Generating ESP ({os.path.basename(f_esp)})...\n")
                    tools.cubegen.mep(mol, f_esp, dm)
                    
                    self.log_signal.emit(f"Generating Density ({os.path.basename(f_dens)})...\n")
                    tools.cubegen.density(mol, f_dens, dm)
                    
                    results["files"].append(f_esp)
                    results["files"].append(f_dens)
                
                elif task == "SpinDensity":
                    # Check unrestricted
                    if isinstance(mo_coeff, (tuple, list)) or (isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 3):
                         from pyscf.scf import uhf
                         if isinstance(mo_coeff, tuple):
                             c_a, c_b = mo_coeff
                             o_a, o_b = mo_occ
                         else:
                             c_a, c_b = mo_coeff[0], mo_coeff[1]
                             o_a, o_b = mo_occ[0], mo_occ[1]
                             
                         dm_ab = uhf.make_rdm1((c_a, c_b), (o_a, o_b))
                         # Spin Density = Alpha - Beta
                         spin_dens = dm_ab[0] - dm_ab[1]
                         
                         f_spin_base = os.path.join(self.out_dir, "spin_density.cube")
                         from .utils import get_unique_path
                         f_spin = get_unique_path(f_spin_base)
                         
                         self.log_signal.emit(f"Generating Spin Density ({os.path.basename(f_spin)})...\n")
                         tools.cubegen.density(mol, f_spin, spin_dens)
                         results["files"].append(f_spin)

                    # --- Added: ROKS/ROHF Support ---
                    # ROKS mo_occ is often (2, N) (Alpha/Beta occupancy)
                    elif isinstance(mo_occ, np.ndarray) and mo_occ.ndim == 2:
                        # ROKS case: mo_coeff is (N,N), but mo_occ is (2, N)
                        from pyscf import scf
                        # Generate density from orbital coeffs and Alpha/Beta occupancies
                        dm_a = scf.hf.make_rdm1(mo_coeff, mo_occ[0])
                        dm_b = scf.hf.make_rdm1(mo_coeff, mo_occ[1])
                        
                        spin_dens = dm_a - dm_b
                        
                        f_spin_base = os.path.join(self.out_dir, "spin_density.cube")
                        from .utils import get_unique_path
                        f_spin = get_unique_path(f_spin_base)
                        
                        self.log_signal.emit(f"Generating Spin Density (ROKS) ({os.path.basename(f_spin)})...\n")
                        tools.cubegen.density(mol, f_spin, spin_dens)
                        results["files"].append(f_spin)
                    # ---------------------------
                    else:
                         self.log_signal.emit("Skipping Spin Density (Not an open-shell calculation or format unknown).\n")
                    
                elif isinstance(task, str):
                    # Parse offset or absolute index
                    idx = -1
                    label = task
                    spin_suffix = ""
                    target_coeff = mo_coeff
                    
                    # Detect Spin Request in label (internal convention)
                    # "15_HOMO_A" ? NO, task string is likely "HOMO" or "#5"
                    # But if we want to differentiate Alpha/Beta, the GUI must pass it.
                    # Currently strict GUI implementation doesn't pass suffix yet.
                    # But we can try to guess or handle it if we add it to the call.
                    
                    # Assume task might be "HOMO_A" or "HOMO_B" logic?
                    # Or we just assume Alpha for now unless specified?
                    
                    is_uhf = isinstance(mo_coeff, (tuple, list)) or (isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 3)
                    
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
                        if isinstance(mo_coeff, tuple):
                             c_a, c_b = mo_coeff
                             o_a, o_b = mo_occ
                        else:
                             c_a, c_b = mo_coeff[0], mo_coeff[1]
                             o_a, o_b = mo_occ[0], mo_occ[1]
                        
                        # Set target arrays
                        if use_beta:
                            target_coeff = c_b
                            target_occ = o_b
                            # We also need to map HOMO/LUMO indices FOR BETA
                            # Recalculate H/L for Beta
                            h_idx = -1
                            for i, o in enumerate(target_occ):
                                if o > occ_threshold: h_idx = i
                            homo_idx = h_idx
                            lumo_idx = h_idx + 1
                        else:
                            target_coeff = c_a
                            target_occ = o_a
                            # Recalculate H/L for Alpha
                            h_idx = -1
                            for i, o in enumerate(target_occ):
                                if o > occ_threshold: h_idx = i
                            homo_idx = h_idx
                            lumo_idx = h_idx + 1
                    else:
                        target_coeff = mo_coeff
                        # homo_idx already calc for RHF
                    
                    try:
                        # Improved Task Parsing for "MO <idx>_<Label>" format
                        # Explicit regex for "MO <index>_<Label>" (e.g. MO 15_HOMO)
                        mo_lbl_match = re.search(r"MO\s+(\d+)_([A-Za-z0-9+-]+)", task)
                        clean_lbl = "MO" # Default

                        if mo_lbl_match:
                            # e.g. "MO 15_HOMO" -> idx=14, lbl="HOMO"
                            idx = int(mo_lbl_match.group(1)) - 1 # Convert 1-based to 0-based
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
                            clean_task = re.sub(r"\D", "", task) # \D matches non-digits
                            if clean_task:
                                val = int(clean_task)
                                if task.startswith("#"):
                                    idx = val # Internal 0-based index
                                else:
                                    idx = val - 1 # User 1-based index (MO 1 = Index 0)
                            else:
                                raise ValueError(f"Unknown task format: {task}")
                                
                            label = f"MO_{idx+1}" # Normalized label (1-based for display)
                        else:
                             pass # ... same error handling ...

                    except Exception as e:
                        self.log_signal.emit(f"Error parsing orbital: {task} ({e})\n")
                        continue
                        
                    if idx < 0 or idx >= target_coeff.shape[1]:
                        self.log_signal.emit(f"Orbital index {idx} out of bounds for {task}\n")
                        continue
                        
                    
                    # Determine Relative Label (HOMO-X / LUMO+X)
                    # Determine Relative Label (HOMO-X / LUMO+X)
                    clean_lbl = f"MO_{idx}" # Default fallback
                    if idx <= homo_idx:
                        diff = homo_idx - idx
                        clean_lbl = "HOMO" if diff == 0 else f"HOMO-{diff}"
                    elif idx >= lumo_idx:
                        diff = idx - lumo_idx
                        clean_lbl = "LUMO" if diff == 0 else f"LUMO+{diff}"
                    
                    rel_label = clean_lbl
                    
                    
                    # Filename
                    # User Request: Use "10a" prefix style
                    # This ensures sorting separates Alpha (10a) and Beta (10b) clearly.
                    
                    prefix_idx = idx + 1 # Use 1-based index for user-facing filename
                    
                    if is_uhf:
                        # e.g. "015a", "015b"
                        # We use 'a' and 'b' from spin_suffix
                        # spin_suffix is "_A" or "_B".
                        # Convert to 'a' or 'b'
                        s_char = "a" if "_A" in spin_suffix else "b"
                        # User Request: "use 010a not 10a" -> Pad to 3 digits
                        file_prefix = f"{prefix_idx:03d}{s_char}"
                        # If we use this prefix, we might not need suffix at end, but keeping it is safe.
                        # Proposed: "015a_HOMO.cube"
                        fname = f"{file_prefix}_{rel_label}.cube"
                    else:
                        # RHF: Use 1-based index for consistency "016_HOMO"
                        # User Request: "Make them consistent"
                        fname = f"{prefix_idx:03d}_{rel_label}.cube"
                    
                    # Sanitization: Ensure safe filenames but keep readable
                    # fname = fname.replace(" ", "") # User requested spaces in name
                    f_path_base = os.path.join(self.out_dir, fname)
                    
                    from .utils import get_unique_path
                    f_path = get_unique_path(f_path_base)
                    
                    self.log_signal.emit(f"Generating {os.path.basename(f_path)} (Index {idx}{spin_suffix})...\n")
                    tools.cubegen.orbital(mol, f_path, target_coeff[:, idx])
                    results["files"].append(f_path)
            
            self.result_signal.emit(results)
            self.finished_signal.emit()
            
        except Exception as e:
            self.error_signal.emit(str(e) + "\n" + traceback.format_exc())
            
        finally:
            if 'original_stdout' in locals():
                sys.stdout = original_stdout
            
            # Restore C-Level FDs
            if 'capturer' in locals():
                try: capturer.__exit__(None, None, None)
                except: pass


class LoadWorker(QThread):
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, chkfile):
        super().__init__()
        self.chkfile = chkfile

    def run(self):
        if pyscf is None:
            self.error_signal.emit("PySCF not found.")
            return

        try:
            from pyscf import lib, scf
            
            # Load Molecule
            mol = lib.chkfile.load_mol(self.chkfile)
            
            # Load SCF Data
            scf_data = scf.chkfile.load(self.chkfile, 'scf')
            mo_energy = scf_data.get('mo_energy')
            mo_occ = scf_data.get('mo_occ')
            
            # Identify Type (Enhanced: UHF, RHF, ROKS, ROHF)
            scf_type = "RHF"
            
            import numpy as np
            
            # Step 1: Check for Unrestricted (UHF/UKS)
            is_uhf = False
            if isinstance(mo_energy, tuple):
                is_uhf = True
            elif isinstance(mo_energy, list) and len(mo_energy) == 2 and isinstance(mo_energy[0], (list, np.ndarray)):
                 pass # Could be list of lists checking
            elif isinstance(mo_energy, np.ndarray) and mo_energy.ndim == 2:
                 is_uhf = True
            
            # Step 2: Check for Restricted Open-shell (ROKS/ROHF)
            # ROKS has 2D mo_occ: shape (2, N) for Alpha/Beta occupancies
            # and contains partial occupancy (values near 1.0)
            is_roks = False
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
                            is_roks = True
                            scf_type = "ROKS"
                    elif isinstance(mo_occ, list):
                        # Handle list of lists case
                        if len(mo_occ) == 2 and all(isinstance(x, (list, np.ndarray)) for x in mo_occ):
                            has_partial_occ = False
                            for sublist in mo_occ:
                                for occ_val in (sublist if isinstance(sublist, list) else sublist.tolist()):
                                    if 0.5 < occ_val < 1.5:
                                        has_partial_occ = True
                                        break
                                if has_partial_occ:
                                    break
                            if has_partial_occ:
                                is_roks = True
                                scf_type = "ROKS"
                except Exception as e_roks:
                    # If ROKS detection fails, default to RHF (safe fallback)
                    pass
                    
            if is_uhf:
                scf_type = "UHF"
                # Convert to lists for JSON/Qt safety
                try:
                    if isinstance(mo_energy, tuple):
                        mo_energy = [e.tolist() if hasattr(e, 'tolist') else list(e) for e in mo_energy]
                        mo_occ = [o.tolist() if hasattr(o, 'tolist') else list(o) for o in mo_occ]
                    else: 
                     # Numpy 2D case
                        if hasattr(mo_energy, 'tolist'): mo_energy = mo_energy.tolist()
                        if hasattr(mo_occ, 'tolist'): mo_occ = mo_occ.tolist()
                except Exception as e_conv:
                    pass  # Keep as-is if conversion fails
            else:
                 # RHF/ROKS Case
                 try:
                     if hasattr(mo_energy, 'tolist'): mo_energy = mo_energy.tolist()
                     if hasattr(mo_occ, 'tolist'): mo_occ = mo_occ.tolist()
                 except Exception as e_conv:
                     pass  # Keep as-is if conversion fails
            
            # Attempt to extract optimized XYZ if present (or just current geometry)
            coords = mol.atom_coords(unit='Ang')
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
                "out_dir": os.path.dirname(self.chkfile)
            }
            
            # --- Load Post-Process Data (Freq/Thermo) ---
            import json
            base_dir = os.path.dirname(self.chkfile)
            freq_file = os.path.join(base_dir, "freq_analysis.json")
            
            if os.path.exists(freq_file):
                try:
                    with open(freq_file, 'r') as f:
                        data = json.load(f)
                        # Extract and merge
                        if "freq_data" in data: results["freq_data"] = data["freq_data"]
                        if "thermo_data" in data: results["thermo_data"] = data["thermo_data"]
                except Exception as e_json:
                    print(f"Failed to load freq json: {e_json}")
            
            self.finished_signal.emit(results)

        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))

