import re

files_to_process = [
    'pyscf_calculator/vis_tab.py',
    'pyscf_calculator/vis.py',
    'pyscf_calculator/scan_results.py',
    'pyscf_calculator/freq_vis.py'
]

for filepath in files_to_process:
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    # Replace hasattr(mw, 'plotter') -> hasattr(mw, 'view_3d_manager') and hasattr(mw.view_3d_manager, 'plotter')
    content = re.sub(
        r"hasattr\((self\.)?mw,\s*['\"]plotter['\"]\)",
        r"hasattr(\1mw, 'view_3d_manager') and hasattr(\1mw.view_3d_manager, 'plotter')",
        content
    )

    # Replace mw.plotter -> mw.view_3d_manager.plotter
    content = re.sub(r"(self\.)?mw\.plotter", r"\1mw.view_3d_manager.plotter", content)

    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content)

    print(f"  Patched: {filepath}")

print('All replacements completed.')

