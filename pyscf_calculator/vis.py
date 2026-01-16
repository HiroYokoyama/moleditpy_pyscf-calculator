import numpy as np
import pyvista as pv
import os
from PyQt6.QtGui import QColor

def parse_cube_data(filename):
    """
    Parses a Gaussian Cube file and returns raw data structures.
    Robust version with strict checks for file format.
    """
    if not filename or not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
        
    with open(filename, 'r') as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError("File too short to be a Cube file.")

    # --- Header Parsing ---
    try:
        # Line 3: Natoms, Origin
        tokens = lines[2].split()
        if len(tokens) < 4: raise ValueError("Invalid Origin line")
        n_atoms_raw = int(tokens[0])
        n_atoms = abs(n_atoms_raw)
        origin_raw = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])

        def parse_vec(line):
            t = line.split()
            if len(t) < 4: raise ValueError("Invalid Axis line")
            return int(t[0]), np.array([float(t[1]), float(t[2]), float(t[3])])

        nx, x_vec_raw = parse_vec(lines[3])
        ny, y_vec_raw = parse_vec(lines[4])
        nz, z_vec_raw = parse_vec(lines[5])
        
        is_angstrom_header = (nx < 0 or ny < 0 or nz < 0)
        nx, ny, nz = abs(nx), abs(ny), abs(nz)
        
    except Exception as e:
        raise ValueError(f"Header parsing failed: {e}")

    # --- Atoms Parsing ---
    atoms = []
    current_line = 6
    
    # Skip extra header line if n_atoms_raw < 0 (MO info line usually)
    if n_atoms_raw < 0:
        if current_line < len(lines):
             parts = lines[current_line].split()
             # MO info line usually has 2 integers, but strict check isn't needed, just skip it
             try: 
                 # Heuristic: if it looks like an atom line (5 chars), don't skip?
                 # Standard: if N < 0, next line is MO info.
                 # Let's verify if the *next* line looks like an atom.
                 _ = int(parts[0]) 
                 # Actually, standard behavior is unconditional skip
                 current_line += 1
             except:
                 current_line += 1

    for _ in range(n_atoms):
        if current_line >= len(lines): break
        line = lines[current_line].split()
        current_line += 1
        
        try:
            if len(line) < 5: 
                # Potentially empty line or malformed
                continue
            atomic_num = int(line[0])
            x, y, z = float(line[2]), float(line[3]), float(line[4])
            atoms.append((atomic_num, np.array([x, y, z])))
        except:
             # Skip malformed atom line
             continue

    # --- Volumetric Data Parsing ---
    # Find start of data
    while current_line < len(lines):
        line_content = lines[current_line].strip()
        parts = line_content.split()
        if not parts:
            current_line += 1
            continue
        
        # Check if this line looks like data (float)
        try:
            float(parts[0])
            break # Start of data found
        except ValueError:
            current_line += 1
            continue

    if current_line >= len(lines):
        # Allow header-only validation if explicitly requested? 
        # But for 'data', we need data.
        # Fallback for empty data
        data_values = np.zeros(nx * ny * nz)
    else:
        full_str = " ".join(lines[current_line:])
        try:
            data_values = np.fromstring(full_str, sep=' ')
        except Exception:
            data_values = np.array([])
    
    expected_size = nx * ny * nz
    actual_size = len(data_values)
    
    # Correct size mismatches defensively
    if actual_size > expected_size:
        # Truncate
        data_values = data_values[:expected_size]
    elif actual_size < expected_size:
        # Pad with zeros
        pad_size = expected_size - actual_size
        if pad_size > 0:
             pad = np.zeros(pad_size)
             data_values = np.concatenate((data_values, pad))
    
    return {
        "atoms": atoms,
        "origin": origin_raw,
        "x_vec": x_vec_raw,
        "y_vec": y_vec_raw,
        "z_vec": z_vec_raw,
        "dims": (nx, ny, nz),
        "data_flat": data_values,
        "is_angstrom_header": is_angstrom_header
    }

def build_grid_from_meta(meta):
    """
    Reconstructs the PyVista grid.
    """
    nx, ny, nz = meta['dims']
    origin = meta['origin'].copy()
    x_vec = meta['x_vec'].copy()
    y_vec = meta['y_vec'].copy()
    z_vec = meta['z_vec'].copy()
    
    # Units Handling
    BOHR_TO_ANGSTROM = 0.529177210903
    convert_to_ang = True
    if meta['is_angstrom_header']:
        convert_to_ang = False
            
    if convert_to_ang:
        origin *= BOHR_TO_ANGSTROM
        x_vec *= BOHR_TO_ANGSTROM
        y_vec *= BOHR_TO_ANGSTROM
        z_vec *= BOHR_TO_ANGSTROM
        
    # Grid Points Generation
    x_range = np.arange(nx)
    y_range = np.arange(ny)
    z_range = np.arange(nz)
    
    gx, gy, gz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    
    gx_f = gx.flatten(order='F')
    gy_f = gy.flatten(order='F')
    gz_f = gz.flatten(order='F')
    
    points = (origin + 
              np.outer(gx_f, x_vec) + 
              np.outer(gy_f, y_vec) + 
              np.outer(gz_f, z_vec))
    
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = [nx, ny, nz]
    
    # Data Mapping
    raw_data = meta['data_flat']
    vol_3d = raw_data.reshape((nx, ny, nz), order='C')
    grid.point_data["values"] = vol_3d.flatten(order='F')
        
    return grid

class CubeVisualizer:
    def __init__(self, mw):
        self.mw = mw
        # self.plotter = mw.plotter # Do not cache!
        self.current_grid = None
        self.actors = {} # Store actors by key
        self.data_max = 1.0

    @property
    def plotter(self):
        if hasattr(self.mw, 'plotter') and self.mw.plotter is not None:
             # Strict check: Ensure RenderWindow exists
             try:
                 if self.mw.plotter.ren_win:
                     return self.mw.plotter
             except: pass
        return None

    def load_file(self, filename):
        try:
            meta = parse_cube_data(filename)
            self.current_grid = build_grid_from_meta(meta)
            
            flat_data = self.current_grid.point_data["values"]
            if len(flat_data) > 0:
                self.data_max = float(np.max(np.abs(flat_data)))
            else:
                self.data_max = 1.0
                
            return True
        except Exception as e:
            print(f"Error loading cube: {e}")
            return False

    def update_iso(self, isovalue, color_p, color_n, opacity, use_comp_color=False):
        if not self.current_grid:
            return

        # Clean previous actors safely
        self.clear_actors()

        # Input Validation
        if isovalue is None or not isinstance(isovalue, (int, float)):
             return

        # Calculate colors
        if use_comp_color:
             c = QColor(color_p)
             h = (c.hue() + 180) % 360
             c_n = QColor.fromHsv(h, c.saturation(), c.value())
             color_n = c_n.name()

        try:
            # Positive
            iso_p = self.current_grid.contour(isosurfaces=[isovalue])
            if iso_p.n_points > 0:
                actor = self.plotter.add_mesh(iso_p, color=color_p, opacity=opacity, name="pyscf_iso_p", reset_camera=False)
                self.actors["p"] = actor
            
            # Negative
            iso_n = self.current_grid.contour(isosurfaces=[-isovalue])
            if iso_n.n_points > 0:
                actor = self.plotter.add_mesh(iso_n, color=color_n, opacity=opacity, name="pyscf_iso_n", reset_camera=False)
                self.actors["n"] = actor
            
            if self.plotter:
                self.plotter.render()
        except Exception as e:
            # print(f"Iso update error: {e}")
            pass

    def clear_actors(self):
        # Remove actors if they exist and plotter is valid
        if not hasattr(self, 'plotter') or self.plotter is None:
             return
             
        try:
            self.plotter.remove_actor("pyscf_iso_p")
            self.plotter.remove_actor("pyscf_iso_n")
        except: pass
        
        self.actors.clear()
        # Do NOT render here. Caller handles it. Rendering on close causes crashes.
        # try: self.plotter.render()
        # except: pass

class MappedVisualizer:
    def __init__(self, mw):
        self.mw = mw
        # self.plotter = mw.plotter # Do not cache
        self.grid_surf = None
        self.grid_prop = None
        self.actor = None
        self.data_surf_max = 1.0
        self.data_prop_range = (-0.1, 0.1)
        
    @property
    def plotter(self):
        if hasattr(self.mw, 'plotter') and self.mw.plotter is not None:
             try:
                 if self.mw.plotter.ren_win:
                     return self.mw.plotter
             except: pass
        return None

    def clear_actors(self):
        # Remove actors if they exist and plotter is valid
        if not self.plotter:
             return

        try:
            if self.actor:
                self.plotter.remove_actor(self.actor)
                self.actor = None 
            self.plotter.remove_actor("pyscf_mapped")
        except: pass
        
        # Do NOT render here.
        # self.plotter.render()

    def load_files(self, surf_file, prop_file):
        try:
            # Load Surface
            meta_s = parse_cube_data(surf_file)
            self.grid_surf = build_grid_from_meta(meta_s)
            
            # Load Property
            meta_p = parse_cube_data(prop_file)
            self.grid_prop = build_grid_from_meta(meta_p)
            
            # Update stats
            flat_s = self.grid_surf.point_data["values"]
            if len(flat_s) > 0:
                self.data_surf_max = float(np.max(np.abs(flat_s)))
            
            flat_p = self.grid_prop.point_data["values"]
            if len(flat_p) > 0:
                self.data_prop_range = (float(np.min(flat_p)), float(np.max(flat_p)))
                
            return True
        except Exception as e:
            print(f"Error loading mapped cubes: {e}")
            return False
            
    def get_mapped_range(self, iso_val):
        """
        Calculates the min/max of the property values on the isosurface.
        Returns (min, max) or (-0.1, 0.1) if empty/error.
        """
        if not self.grid_surf or not self.grid_prop:
            return (-0.1, 0.1)
            
        try:
            iso = self.grid_surf.contour([iso_val], scalars="values")
            if iso.n_points == 0: return (-0.1, 0.1)
            
            mapped = iso.sample(self.grid_prop)
            mvals = mapped.point_data.get("values") # sampled data remains 'values' generally
            
            if mvals is not None and len(mvals) > 0:
                return (float(mvals.min()), float(mvals.max()))
            return (-0.1, 0.1)
        except:
             return (-0.1, 0.1)

    def update_mesh(self, iso_val, opacity, cmap="jet", clim=None):
        if not self.grid_surf or not self.grid_prop:
            return

        try:
            # Clean
            self.clear_actors()

            # Contour Surface
            # The surface grid data is "values"
            iso = self.grid_surf.contour([iso_val], scalars="values")
            if iso.n_points == 0:
                return

            # Sample Property
            # The property grid data is ALSO "values" (from standard parser)
            # rename property data to avoid conflict? No, separate grids.
            # Sample: 'resample_to_image' or 'sample'
            # grid_prop is a StructuredGrid. sample function expects DataSet.
            mapped = iso.sample(self.grid_prop)
            
            # Check sampling result
            if mapped is None or mapped.n_points == 0:
                 return
            
            # The sampled data will be in mapped.point_data["values"] (from prop grid)
            
            if clim is None:
                clim = self.data_prop_range

            self.actor = self.plotter.add_mesh(
                mapped,
                scalars="values",
                cmap=cmap,
                clim=clim,
                smooth_shading=True,
                opacity=opacity,
                name="pyscf_mapped",
                reset_camera=False
            )
            
            if self.plotter:
                self.plotter.render()
            
        except Exception as e:
            print(f"Mapped update error: {e}")
            import traceback
            traceback.print_exc()

    def clear_actors(self):
        if not hasattr(self, 'plotter') or self.plotter is None: return

        try:
            if self.actor:
                self.plotter.remove_actor(self.actor)
                self.actor = None 
            self.plotter.remove_actor("pyscf_mapped")
            # Do NOT render here.
            # self.plotter.render()
        except: pass
