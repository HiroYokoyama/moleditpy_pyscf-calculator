PLUGIN_NAME = "PySCF Calculator"
PLUGIN_VERSION = "1.6.1"
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
    
    def handle_reset():
        nonlocal dialog_instance
        # 1. Clear Global Settings (Model)
        # Remove association with previous file
        if "associated_filename" in PLUGIN_SETTINGS:
            del PLUGIN_SETTINGS["associated_filename"]
        if "calc_history" in PLUGIN_SETTINGS:
            del PLUGIN_SETTINGS["calc_history"]
        if "struct_source" in PLUGIN_SETTINGS:
            del PLUGIN_SETTINGS["struct_source"]
            
        # 2. Reset Dialog UI (View)
        if dialog_instance is not None:
             if hasattr(dialog_instance, 'on_document_reset'):
                 dialog_instance.on_document_reset()
                 
    context.register_document_reset_handler(handle_reset)
    
    
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
