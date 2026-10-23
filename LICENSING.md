# Plugin license and separately installed dependencies

PySCF Calculator's first-party code is licensed under the GNU General Public
License, version 3 (GPLv3), with the additional
[geomeTRIC combination permission](GEOMETRIC_EXCEPTION).
The unmodified GPL text is in [LICENSE](LICENSE). Earlier releases retain their
original grants; no third-party code is relicensed by this notice.

The release ZIP contains the Python plugin and these notices. It does not bundle
PySCF, geomeTRIC, Qt, NumPy, or their binaries. Users install dependencies
separately. Separate installation does not by itself establish license
compatibility: the optimizer runs in the same Python process through
`pyscf.geomopt.geometric_solver`. The explicit permission covers that intended
combination without changing the dependency's own terms.

## geomeTRIC

The published geomeTRIC **1.1.1** source archive contains a **BSD 3-clause Non-AI
License**, not the unmodified BSD-3-Clause license. Its fourth condition restricts
including the software's source/binaries and their modifications in AI-training
datasets. It is not a blanket prohibition on molecular geometry optimization,
and the plugin's permission does not remove that condition.

The unmodified geomeTRIC 1.1.1 license is included in
[`licenses/geometric-1.1.1.LICENSE`](licenses/geometric-1.1.1.LICENSE), preserving
the upstream copyright and conditions. The license of the installed version
applies to that dependency and any distribution that includes it.

- [Published 1.1.1 source archive](https://files.pythonhosted.org/packages/4d/57/f53eb30c38aa4eeabddc3bddf283f4c10616de65d95fcd1f43900721f425/geometric-1.1.1.tar.gz)
- [Upstream license](https://github.com/leeping/geomeTRIC/blob/master/LICENSE)
- [GNU GPL FAQ on explicit library exceptions](https://www.gnu.org/licenses/gpl-faq.en.html#GPLIncompatibleLibs)

## Other dependencies

PySCF (Apache-2.0), NumPy (BSD-family), Matplotlib (PSF-based), PySide6/Qt
(LGPLv3 option), and optional pyberny (MPL-2.0) retain their own licenses.
Optional dispersion libraries and their native components also retain their
respective terms. Preserve the notices and corresponding-source/replacement
rights required by the versions actually redistributed. A dependency's license
is not replaced by this plugin's GPL or the geomeTRIC permission.

This plugin retains its GPL license; MoleditPy's permission for independently
developed plugins does not require all plugins to adopt LGPL.
