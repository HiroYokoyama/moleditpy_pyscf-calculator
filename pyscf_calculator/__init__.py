PLUGIN_NAME = "PySCF Calculator"
PLUGIN_VERSION = "4.0.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Perform PySCF quantum chemistry calculations directly in MoleditPy. "
    "Features: Single Point Energy, Geometry and Transition State Optimization, "
    "Frequency Analysis with thermochemistry, TDDFT, Rigid/Relaxed Surface Scans, "
    "implicit solvent and dispersion corrections, and interactive 3D visualization "
    "of Molecular Orbitals, Spin Density and Electrostatic Potential (ESP)."
)
PLUGIN_DEPENDENCIES = ["pyscf", "geometric", "numpy"]
# Extra features only. pyscf-dispersion (D3/D4) is deliberately not listed:
# it has no Apple Silicon wheel, and the one pip falls back to there (1.0.0)
# breaks `import pyscf` -- see README.
PLUGIN_OPTIONAL_DEPENDENCIES = ["pyberny"]
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=4.0.0, <5.0.0"
PLUGIN_SUPPORTED_OS = ["macOS", "Linux", "WSL"]  # pyscf has no native Windows support

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
        PLUGIN_SETTINGS.clear()
        PLUGIN_SETTINGS.update(data)

    context.register_save_handler(save_project_state)
    context.register_load_handler(load_project_state)

    # --- UI ---
    def handle_reset():
        # 1. Clear Global Settings (Model)
        for key in ("associated_filename", "calc_history", "struct_source"):
            PLUGIN_SETTINGS.pop(key, None)

        # 2. Reset Dialog UI (View)
        dlg = context.get_window("dialog")
        if dlg is not None and hasattr(dlg, "on_document_reset"):
            dlg.on_document_reset()

    context.register_document_reset_handler(handle_reset)

    def show_dialog():
        dlg = context.get_window("dialog")
        if dlg is not None:
            try:
                if dlg.isVisible():
                    dlg.raise_()
                    dlg.activateWindow()
                    return
                dlg.close()
                dlg.deleteLater()
            except RuntimeError:
                pass
            # Unregister the stale window so it doesn't block creation of a new one
            context.register_window("dialog", None)

        mw = context.get_main_window()
        new_dlg = PySCFDialog(
            mw, context, settings=PLUGIN_SETTINGS, version=PLUGIN_VERSION
        )
        context.register_window("dialog", new_dlg)
        new_dlg.show()

    # Register the menu action
    context.add_menu_action("Extensions/PySCF Calculator...", show_dialog)
