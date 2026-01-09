import numpy as np
import pyvista as pv
from PyQt6.QtGui import QColor

# --- Parsing Logic (Adapted from Cube File Viewer Plugin) ---
def parse_cube_data(filename):
    """
    Parses a Gaussian Cube file and returns raw data structures.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError("File too short to be a Cube file.")

    # --- Header Parsing ---
    tokens = lines[2].split()
    n_atoms_raw = int(tokens[0])
    n_atoms = abs(n_atoms_raw)
    origin_raw = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])

    def parse_vec(line):
        t = line.split()
        return int(t[0]), np.array([float(t[1]), float(t[2]), float(t[3])])

    nx, x_vec_raw = parse_vec(lines[3])
    ny, y_vec_raw = parse_vec(lines[4])
    nz, z_vec_raw = parse_vec(lines[5])
    
    is_angstrom_header = (nx < 0 or ny < 0 or nz < 0)
    nx, ny, nz = abs(nx), abs(ny), abs(nz)

    # --- Atoms Parsing ---
    atoms = []
    current_line = 6
    if n_atoms_raw < 0:
        try:
            parts = lines[current_line].split()
            if len(parts) != 5: 
                 current_line += 1
        except:
             current_line += 1

    for _ in range(n_atoms):
        line = lines[current_line].split()
        current_line += 1
        atomic_num = int(line[0])
        try:
            x, y, z = float(line[2]), float(line[3]), float(line[4])
        except:
            x, y, z = 0.0, 0.0, 0.0
        atoms.append((atomic_num, np.array([x, y, z])))

    # --- Volumetric Data Parsing ---
    while current_line < len(lines):
        line_content = lines[current_line].strip()
        parts = line_content.split()
        if not parts:
            current_line += 1
            continue
        if len(parts) < 6:
            current_line += 1
            continue
        try:
            float(parts[0])
        except ValueError:
            current_line += 1
            continue
        break

    full_str = " ".join(lines[current_line:])
    try:
        data_values = np.fromstring(full_str, sep=' ')
    except:
        data_values = np.array([])
    
    expected_size = nx * ny * nz
    actual_size = len(data_values)
    
    if actual_size > expected_size:
        excess = actual_size - expected_size
        data_values = data_values[excess:]
    elif actual_size < expected_size:
        pad = np.zeros(expected_size - actual_size)
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
        self.plotter = mw.plotter
        self.current_grid = None
        self.actors = {} # Store actors by key
        self.data_max = 1.0

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

        # Clean previous actors
        self.clear_actors()

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
            
            self.plotter.render()
        except Exception as e:
            print(f"Iso update error: {e}")

    def clear_actors(self):
        # Remove actors if they exist
        # We can remove by name if we used name argument
        self.plotter.remove_actor("pyscf_iso_p")
        self.plotter.remove_actor("pyscf_iso_n")
        self.actors.clear()
        self.plotter.render()

class MappedVisualizer:
    def __init__(self, mw):
        self.mw = mw
        self.plotter = mw.plotter
        self.grid_surf = None
        self.grid_prop = None
        self.actor = None
        self.data_surf_max = 1.0
        self.data_prop_range = (-0.1, 0.1)

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

    def update_mesh(self, iso_val, opacity, cmap="jet", clim=None):
        if not self.grid_surf or not self.grid_prop:
            return

        try:
            # Clean
            if self.actor:
                self.plotter.remove_actor(self.actor)
                self.actor = None 
            self.plotter.remove_actor("pyscf_mapped")

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
            
            self.plotter.render()
            
        except Exception as e:
            print(f"Mapped update error: {e}")
            import traceback
            traceback.print_exc()

    def clear_actors(self):
        if self.actor:
            self.plotter.remove_actor(self.actor)
        self.plotter.remove_actor("pyscf_mapped")
        self.plotter.render()
