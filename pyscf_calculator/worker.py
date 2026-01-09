import sys
import os
import io
import tempfile
import traceback
import numpy as np
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
            
            # Setup Molecule
            mol = gto.M(
                atom=self.xyz_str,
                basis=self.config.get("basis", "sto-3g"),
                charge=self.config.get("charge", 0),
                spin=self.config.get("spin", 0),
                verbose=4,
                output=None,
                max_memory=self.config.get("memory", 4000)
            )
            mol.stdout = stream
            mol.verbose = 4
            mol.build()
            
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
                f.write(f"    charge={self.config.get('charge')}, \n")
                f.write(f"    spin={self.config.get('spin')}, \n")
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

            # Select Method
            method_name = self.config.get("method", "RHF")
            functional = self.config.get("functional", "b3lyp")
            
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
            else:
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
    
                if "Geometry Optimization" in job_type:
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
                        
                        # Run Energy on optimized
                        mf.chkfile = chk_path
                        mf.kernel()
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
                             mf.kernel()
                             mol = mol_eq
                        except ImportError:
                             self.error_signal.emit("Neither 'geometric' nor 'berny' optimizer found. Please install 'geometric' (pip install geometric).")
                             return
                elif "Frequency" in job_type:
                    self.log_signal.emit(f"Starting SCF & Frequency Analysis using {method_name}...\n")
                    mf.kernel()
                    
                    self.log_signal.emit("Calculating Hessian...\n")
                    try:
                        h_obj = mf.Hessian()
                        hessian = h_obj.kernel()
                        
                        from pyscf.hessian import thermo
                        self.log_signal.emit("Performing Harmonic Analysis...\n")
                        freq_res = thermo.harmonic_analysis(mol, hessian)
                        
                        # Calculate Thermo
                        self.log_signal.emit("Calculating Thermodynamic Properties...\n")
                        t_data = thermo.thermo(mf, freq_res['freq_au'], freq_res['norm_mode'], mol)
                        
                        # Store data for GUI Visualizer
                        results["freq_data"] = {
                            "freqs": freq_res["freq_wavenumber"].tolist(),
                            "modes": freq_res["norm_mode"].tolist()
                        }
                        
                        # Store Thermo
                        if t_data:
                            # Convert any numpy types to python native
                            safe_t = {}
                            for k, v in t_data.items():
                                if hasattr(v, 'tolist'): safe_t[k] = v.tolist()
                                else: safe_t[k] = v
                            results["thermo_data"] = safe_t
                            
                    except Exception as e_freq:
                        self.log_signal.emit(f"Frequency analysis failed: {e_freq}\n{traceback.format_exc()}\n")
                        
                else:
                    # Energy calculation
                    mf.kernel()
    
                # --- SAVE CHECKPOINT (ALWAYS) ---
                # Checkpoint is already set to self.out_dir/pyscf.chk and written by mf.kernel()
                
                chk_path = os.path.join(self.out_dir, "pyscf.chk")
                
                results.update({
                    "chkfile": chk_path,
                    "out_dir": self.out_dir
                })
                
                # Pass energy/occupancy to GUI for Diagram
                # Handle UHF (tuple) vs RHF (array)
                if isinstance(mf.mo_energy, tuple):
                     results["mo_energy"] = [e.tolist() for e in mf.mo_energy]
                     results["mo_occ"] = [o.tolist() for o in mf.mo_occ]
                     results["scf_type"] = "UHF"
                else:
                     results["mo_energy"] = mf.mo_energy.tolist()
                     results["mo_occ"] = mf.mo_occ.tolist()
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
            
            # Assuming RHF/RKS for now (single set of occ)
            # mo_occ shape (N,)
            if len(mo_occ.shape) == 1:
                for i, occ in enumerate(mo_occ):
                    if occ > 0:
                        homo_idx = i
                    else:
                        lumo_idx = i
                        break
            else:
                 # UHF - not handled in this simple snippet yet
                 pass

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
                         dm_ab = uhf.make_rdm1(mo_coeff, mo_occ)
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
                    # We might want to pass the pair info
                    
                elif isinstance(task, str):
                    # Parse offset or absolute index
                    idx = -1
                    label = task
                    
                    try:
                        # Case 1: Relative to HOMO/LUMO
                        if "HOMO" in task:
                            base = homo_idx
                            if "+" in task:
                                offset = int(task.split("+")[1])
                                idx = base + offset
                            elif "-" in task:
                                offset = int(task.split("-")[1])
                                idx = base - offset
                            else:
                                idx = base
                        elif "LUMO" in task:
                            base = lumo_idx
                            if "+" in task:
                                offset = int(task.split("+")[1])
                                idx = base + offset
                            elif "-" in task:
                                offset = int(task.split("-")[1])
                                idx = base - offset
                            else:
                                idx = base
                        # Case 2: Explicit "MO <n>" or just numbers
                        elif "MO" in task or task.isdigit() or task.startswith("#"):
                            # "MO 15", "15", "#15"
                            clean_task = task.replace("MO", "").replace("#", "").strip()
                            idx = int(clean_task)
                            label = f"MO_{idx}" # Normalized label
                        else:
                            if self.job_type == "Optimization":
                                self.log_signal.emit("Starting Geometry Optimization (geometric)...\n")
                                
                                # We need to allow geometric to write to chkfile during opt?
                                # PySCF gradients read from it.
                                
                                try:
                                    import geometric
                                except ImportError:
                                    self.log_signal.emit("Error: geometric-mol not installed. Skipping optimization.\n")
                                    self.job_type = "Energy" # Fallback                      
                            # Unknown string pattern
                            self.log_signal.emit(f"Unknown orbital syntax: {task}\n")
                            continue

                    except Exception as e:
                        self.log_signal.emit(f"Error parsing orbital: {task} ({e})\n")
                        continue
                        
                    if idx < 0 or idx >= mo_coeff.shape[1]:
                        self.log_signal.emit(f"Orbital index {idx} out of bounds for {task}\n")
                        continue
                        
                    # Determine Relative Label (HOMO-X / LUMO+X)
                    rel_label = f"MO_{idx}"
                    if idx <= homo_idx:
                        diff = homo_idx - idx
                        rel_label = "HOMO" if diff == 0 else f"HOMO-{diff}"
                    elif lumo_idx != -1 and idx >= lumo_idx:
                        diff = idx - lumo_idx
                        rel_label = "LUMO" if diff == 0 else f"LUMO+{diff}"
                        
                    # Filename
                    fname = f"{idx}_{rel_label}.cube"
                        
                    # Sanitization: Ensure safe filenames but keep readable
                    # fname = fname.replace(" ", "") # User requested spaces in name
                    f_path_base = os.path.join(self.out_dir, fname)
                    
                    from .utils import get_unique_path
                    f_path = get_unique_path(f_path_base)
                    
                    self.log_signal.emit(f"Generating {os.path.basename(f_path)} (Index {idx})...\n")
                    tools.cubegen.orbital(mol, f_path, mo_coeff[:, idx])
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
            
            # Identify Type
            scf_type = "RHF"
            if isinstance(mo_energy, tuple):
                scf_type = "UHF"
                # Convert to lists for JSON/Qt safety
                mo_energy = [e.tolist() for e in mo_energy]
                mo_occ = [o.tolist() for o in mo_occ]
            else:
                 mo_energy = mo_energy.tolist()
                 mo_occ = mo_occ.tolist()
            
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
                "optimized_xyz": optimized_xyz,
                "chkfile": self.chkfile
            }
            
            self.finished_signal.emit(results)

        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))

