PLUGIN_NAME = "PySCF Calculator"
PLUGIN_VERSION = "1.3.1"
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
    dialog_instance = None
    
    def show_dialog():
        nonlocal dialog_instance
        # Get the main window to parent the dialog
        mw = context.get_main_window()
        
        if dialog_instance is not None:
            try:
                if dialog_instance.isVisible():
                    dialog_instance.raise_()
                    dialog_instance.activateWindow()
                    return
                else:
                    # Cleanup old instance if hidden/closed
                    dialog_instance.close()
                    dialog_instance.deleteLater()
                    dialog_instance = None
            except RuntimeError:
                # C++ object deleted
                dialog_instance = None

        # Create and show the dialog, passing the shared settings dict
        dialog_instance = PySCFDialog(mw, context, settings=PLUGIN_SETTINGS, version=PLUGIN_VERSION)
        dialog_instance.show()

    # Register the menu action
    context.add_menu_action("Extensions/PySCF Calculator...", show_dialog)
