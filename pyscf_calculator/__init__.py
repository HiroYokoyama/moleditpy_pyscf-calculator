PLUGIN_NAME = "PySCF Calculator"
PLUGIN_VERSION = "1.1.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Perform PySCF quantum chemistry calculations directly in MoleditPy. "
    "Features: Single Point Energy (RHF/UHF/DFT), Geometry Optimization (Geometric/Berny), "
    "Frequency Analysis, and interactive 3D visualization of Molecular Orbitals (HOMO/LUMO) "
    "and Electrostatic Potential (ESP) mapped on Density surfaces."
)
PLUGIN_DEPENDENCIES = ["pyscf", "geometric", "numpy"]

from .gui import PySCFDialog

# Global settings state (persisted in project file)
PLUGIN_SETTINGS = {}

def initialize(context):
    """
    Initialize the PySCF plugin.
    """
    
    # --- Persistence Logic ---
    def save_project_state():
        return PLUGIN_SETTINGS

    def load_project_state(data):
        global PLUGIN_SETTINGS
        PLUGIN_SETTINGS.clear()
        PLUGIN_SETTINGS.update(data)
        
    context.register_save_handler(save_project_state)
    context.register_load_handler(load_project_state)
    
    # --- UI ---
    def show_dialog():
        # Get the main window to parent the dialog
        mw = context.get_main_window()
        
        # Check if we have a molecule loaded
        if not context.current_molecule:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(mw, "No Molecule", "Please load a molecule first.")
            return

        # Create and show the dialog, passing the shared settings dict
        dialog = PySCFDialog(mw, context, settings=PLUGIN_SETTINGS)
        dialog.show()

    # Register the menu action
    context.add_menu_action("Extensions/PySCF Calculator...", show_dialog)
