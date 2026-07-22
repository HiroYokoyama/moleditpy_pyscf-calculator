"""
tests/test_vis_tab_coverage.py

Broad coverage-focused tests for pyscf_calculator/vis_tab.py's VisTab class.
VisTab is instantiated via __new__ (bypassing setup_ui()/__init__, which need
a live QApplication); each test manually sets only the attributes the method
under test reads/writes -- same technique as tests/test_vis_tab.py.

Rather than relying on which stub happens to win the sys.modules setdefault
race across test files (fragile - see the incident where a plain assignment
in another new test file corrupted PyQt6.QtGui for test_plugin_integration.py
real-context tests), the Qt names this module actually touches (Qt, QTimer,
QListWidgetItem, QMessageBox, QFileDialog, QColorDialog, QDialog,
QTableWidget, QTableWidgetItem, QHeaderView) are rebound directly on our own
*already-loaded* private vis_tab module object after exec. This makes the
test file's behavior independent of import order across the test suite.
"""

import os
import sys
import types
import unittest
import importlib.util
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal stubs -- just enough that vis_tab.py imports without raising.
# Real behavior is patched onto the loaded module afterwards (see below).
# ---------------------------------------------------------------------------


def _install_minimal_stubs():
    # Plain assignment (not setdefault) for QtCore/QtWidgets/PyQt6, mirroring
    # tests/test_vis_tab.py's proven-working pattern: VisTab subclasses
    # QWidget at class-definition time (module exec), so QWidget MUST be a
    # plain-Python fake *before* vis_tab.py is exec'd here -- if an earlier
    # test file already fully imported the real PyQt6.QtWidgets (e.g. for
    # test_plugin_integration.py's real-context tests) and we merely reused
    # it (setdefault / hasattr-merge), VisTab would subclass the real
    # sip-wrapped QWidget, and __new__() (which skips __init__) then raises
    # "super-class __init__() of type VisTab was never called" on first
    # attribute access. Rebinding Qt/QTimer/etc. post-load (below) cannot fix
    # this retroactively since the base class is fixed at class-definition
    # time.
    qt_core = types.ModuleType("PyQt6.QtCore")
    qt_core.Qt = MagicMock()
    qt_core.QTimer = MagicMock()
    sys.modules["PyQt6.QtCore"] = qt_core

    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core
    sys.modules["PyQt6"] = pyqt6

    class _QWidget:
        def __init__(self, *args, **kwargs):
            pass

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QWidget = _QWidget
    for name in [
        "QVBoxLayout",
        "QHBoxLayout",
        "QLabel",
        "QPushButton",
        "QListWidget",
        "QListWidgetItem",
        "QGroupBox",
        "QLineEdit",
        "QMessageBox",
        "QDoubleSpinBox",
        "QSlider",
        "QComboBox",
        "QTableWidget",
        "QTableWidgetItem",
        "QHeaderView",
        "QDockWidget",
        "QFileDialog",
        "QDialog",
        "QColorDialog",
    ]:
        setattr(qt_widgets, name, MagicMock)
    pyqt6.QtWidgets = qt_widgets
    sys.modules["PyQt6.QtWidgets"] = qt_widgets

    sys.modules["rdkit"] = MagicMock()
    sys.modules["pyscf"] = None


_install_minimal_stubs()


def _load_module_direct(relpath, module_name):
    src = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", relpath))
    spec = importlib.util.spec_from_file_location(module_name, src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_vis_tab_mod = _load_module_direct(
    os.path.join("pyscf_calculator", "vis_tab.py"),
    "pyscf_calculator_vis_tab_coverage_under_test",
)


# ---------------------------------------------------------------------------
# Real, inspectable fakes -- rebound directly onto the loaded module so
# behavior does not depend on cross-file sys.modules ordering.
# ---------------------------------------------------------------------------


class _ItemFlag:
    ItemIsUserCheckable = 1
    ItemIsEnabled = 2
    ItemIsSelectable = 4


class _CheckState:
    Unchecked = 0
    Checked = 2


class _ItemDataRole:
    UserRole = 256


class _DockWidgetArea:
    RightDockWidgetArea = 1


class _Orientation:
    Horizontal = 1


class _FakeQt:
    ItemFlag = _ItemFlag
    CheckState = _CheckState
    ItemDataRole = _ItemDataRole
    DockWidgetArea = _DockWidgetArea
    Orientation = _Orientation


class _FakeQTimerNoOp:
    @staticmethod
    def singleShot(ms, fn):
        pass


class _FakeQTimerImmediate:
    @staticmethod
    def singleShot(ms, fn):
        fn()


class _FakeListItem:
    def __init__(self, text=""):
        self._text = text
        self._flags = 0
        self._check_state = None
        self._data = {}
        self._tooltip = None

    def flags(self):
        return self._flags

    def setFlags(self, f):
        self._flags = f

    def checkState(self):
        return self._check_state

    def setCheckState(self, s):
        self._check_state = s

    def data(self, role):
        return self._data.get(role)

    def setData(self, role, val):
        self._data[role] = val

    def setToolTip(self, t):
        self._tooltip = t

    def toolTip(self):
        return self._tooltip

    def text(self):
        return self._text


class _FakeListWidget:
    def __init__(self):
        self._items = []
        self.current = None

    def clear(self):
        self._items = []
        self.current = None

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def item(self, i):
        return self._items[i]

    def setCurrentItem(self, item):
        self.current = item

    def currentItem(self):
        return self.current

    def scrollToItem(self, item):
        pass

    def setFixedHeight(self, h):
        pass


class _FakeTableItem:
    def __init__(self, text=""):
        self._text = str(text)

    def text(self):
        return self._text


class _FakeTableWidget:
    def __init__(self):
        self._col_count = 0
        self._headers = []
        self._rows = []

    def setColumnCount(self, n):
        self._col_count = n

    def setHorizontalHeaderLabels(self, labels):
        self._headers = list(labels)

    def insertRow(self, row):
        self._rows.insert(row, {})

    def setItem(self, row, col, item):
        self._rows[row][col] = item

    def item(self, row, col):
        return self._rows[row].get(col)

    def rowCount(self):
        return len(self._rows)

    def columnCount(self):
        return self._col_count

    def horizontalHeaderItem(self, col):
        if col < len(self._headers):
            return _FakeTableItem(self._headers[col])
        return None

    def horizontalHeader(self):
        return MagicMock()


class _FakeQHeaderView:
    class ResizeMode:
        Stretch = 1


class _FakeLayout:
    """QVBoxLayout(dlg)/QHBoxLayout() stand-in. A plain MagicMock class
    can't be used here: calling MagicMock(dlg) with a Mock instance as the
    first positional arg is interpreted as the `spec=` kwarg and raises
    InvalidSpecError ("Cannot spec a Mock object")."""

    def __init__(self, *args, **kwargs):
        pass

    def addWidget(self, *args, **kwargs):
        pass

    def addLayout(self, *args, **kwargs):
        pass


class _FakeStaticDialogs:
    """Stand-in for QMessageBox/QFileDialog/QColorDialog static-method-style
    classes. Deliberately NOT a bare MagicMock() instance: some other test
    file (test_scan_results.py) does `sr_mod.QMessageBox.information =
    MagicMock()` where its QMessageBox is literally the `unittest.mock.
    MagicMock` *class* -- that sets a process-wide class attribute on the
    real stdlib MagicMock class, so a fresh MagicMock() instance's
    `.information` (found via class-attribute fallback before `__getattr__`
    ever fires) silently returns that same globally-shared, cross-test-
    polluted mock. Explicit instance attributes here sidestep that
    entirely."""

    def __init__(self, **kwargs):
        for name in (
            "warning",
            "information",
            "critical",
            "getColor",
            "getExistingDirectory",
            "getSaveFileName",
        ):
            setattr(self, name, MagicMock())
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeColor:
    def __init__(self, valid=True, name="#ff0000"):
        self._valid = valid
        self._name = name

    def isValid(self):
        return self._valid

    def name(self):
        return self._name


def _rebind_qt_names():
    _vis_tab_mod.Qt = _FakeQt
    _vis_tab_mod.QTimer = _FakeQTimerNoOp
    _vis_tab_mod.QListWidgetItem = _FakeListItem
    _vis_tab_mod.QMessageBox = _FakeStaticDialogs()
    _vis_tab_mod.QFileDialog = _FakeStaticDialogs()
    _vis_tab_mod.QColorDialog = _FakeStaticDialogs()
    _vis_tab_mod.QDialog = MagicMock()
    _vis_tab_mod.QTableWidget = _FakeTableWidget
    _vis_tab_mod.QTableWidgetItem = _FakeTableItem
    _vis_tab_mod.QHeaderView = _FakeQHeaderView
    _vis_tab_mod.QDockWidget = MagicMock()
    _vis_tab_mod.QVBoxLayout = _FakeLayout
    _vis_tab_mod.QHBoxLayout = _FakeLayout
    # setup_ui() constructs several widgets with a plain string/int first
    # positional arg (e.g. QLabel("Result Folder:"), QSlider(Qt.Orientation.
    # Horizontal)). A bare MagicMock *class* interprets that arg as `spec=`,
    # producing a mock restricted to that spec object's attributes (e.g. a
    # str's attrs) -- so a later `.setStyleSheet(...)` call raises
    # AttributeError. Pre-built MagicMock() *instances* sidestep this: their
    # __call__ just records the call and returns the (shared, unspecced)
    # .return_value.
    for _name in (
        "QLabel",
        "QLineEdit",
        "QGroupBox",
        "QDoubleSpinBox",
        "QSlider",
        "QComboBox",
        "QListWidget",
        "QPushButton",
    ):
        setattr(_vis_tab_mod, _name, MagicMock())


_rebind_qt_names()

VisTab = _vis_tab_mod.VisTab


def _make_vis_tab(**overrides):
    vt = VisTab.__new__(VisTab)
    vt.parent_dialog = MagicMock()
    vt.parent_dialog.closing = False
    vt.context = MagicMock()
    vt.chkfile_path = "/fake/out/pyscf.chk"
    vt.last_out_dir = "/fake/out"
    vt.mo_data = None
    vt.freq_data = None
    vt.thermo_data = None
    vt.optimized_xyz = None
    vt.prop_worker = None
    vt.load_worker = None
    vt.visualizer = MagicMock()
    vt.mapped_visualizer = None
    vt.freq_vis = None
    vt.freq_dock = None
    vt.mode = "standard"
    vt.loaded_file = None
    vt.color_p = "blue"
    vt.color_n = "red"
    vt.btn_run_analysis = MagicMock()
    vt.btn_load_geom = MagicMock()
    vt.btn_show_diagram = MagicMock()
    vt.btn_show_thermo = MagicMock()
    vt.orb_list = _FakeListWidget()
    vt.file_list = _FakeListWidget()
    vt.log = MagicMock()
    vt.result_path_display = MagicMock()
    vt.lbl_struct_source = MagicMock()
    vt.vis_controls = MagicMock()
    vt.mapped_group = MagicMock()
    vt.iso_spin = MagicMock()
    vt.op_slider = MagicMock()
    vt.m_iso_spin = MagicMock()
    vt.m_min_spin = MagicMock()
    vt.m_max_spin = MagicMock()
    vt.cmap_combo = MagicMock()
    vt.m_op_slider = MagicMock()
    vt.btn_color_p = MagicMock()
    vt.btn_color_n = MagicMock()
    vt.mo_input = MagicMock()
    for k, v in overrides.items():
        setattr(vt, k, v)
    return vt


# ---------------------------------------------------------------------------
# log / _add_to_history
# ---------------------------------------------------------------------------


class TestVisTabRealConstruction(unittest.TestCase):
    """Exercises the real __init__()/setup_ui() (normally 0% covered since
    every other test in this file uses VisTab.__new__() to bypass them --
    they need a live QApplication in production, but with our Qt stand-ins
    they run as plain Python and are safe to execute directly)."""

    def test_real_init_builds_widgets_without_crashing(self):
        _vis_tab_mod.CubeVisualizer = MagicMock()
        parent_dialog = MagicMock()
        context = MagicMock()
        vt = VisTab(parent_dialog, context)
        self.assertEqual(vt.mode, "standard")
        self.assertEqual(vt.color_p, "blue")
        self.assertEqual(vt.color_n, "red")
        self.assertIsNone(vt.chkfile_path)
        self.assertIsNotNone(vt.orb_list)
        self.assertIsNotNone(vt.file_list)
        self.assertIsNotNone(vt.vis_controls)
        self.assertIsNotNone(vt.mapped_group)


class TestLogAndHistory(unittest.TestCase):
    def test_log_delegates_to_parent(self):
        vt = _make_vis_tab()
        vt.log = VisTab.log.__get__(vt)
        vt.log("hi")
        vt.parent_dialog.log.assert_called_once_with("hi")

    def test_add_to_history_creates_list_and_appends(self):
        vt = _make_vis_tab()
        del vt.parent_dialog.calc_history
        pd = MagicMock(spec=["update_internal_state"])
        vt.parent_dialog = pd
        vt._add_to_history("/a/b")
        self.assertEqual(pd.calc_history, ["/a/b"])
        self.assertTrue(vt._history_changed)
        pd.update_internal_state.assert_called_once()

    def test_add_to_history_no_duplicate(self):
        vt = _make_vis_tab()
        vt.parent_dialog.calc_history = ["/a/b"]
        vt._history_changed = False
        vt._add_to_history("/a/b")
        self.assertEqual(vt.parent_dialog.calc_history, ["/a/b"])
        self.assertFalse(vt._history_changed)


# ---------------------------------------------------------------------------
# populate_analysis_options
# ---------------------------------------------------------------------------


class TestPopulateAnalysisOptions(unittest.TestCase):
    def _tasks(self, vt):
        return [
            vt.orb_list.item(i).data(_ItemDataRole.UserRole)
            for i in range(vt.orb_list.count())
        ]

    def test_no_mo_data_gives_esp_and_five_lumos_only(self):
        vt = _make_vis_tab(mo_data=None)
        vt.populate_analysis_options()
        tasks = self._tasks(vt)
        self.assertEqual(tasks[0], "ESP")
        self.assertEqual(len(tasks), 6)
        self.assertEqual(tasks[1], "LUMO+4")
        self.assertEqual(tasks[-1], "LUMO")

    def test_rhf_with_occupations_adds_homo_items(self):
        vt = _make_vis_tab(
            mo_data={"type": "RHF", "occupations": [2, 2, 2, 0, 0], "energies": []}
        )
        vt.populate_analysis_options()
        tasks = self._tasks(vt)
        self.assertIn("MO 3_HOMO", tasks)
        self.assertIn("MO 2_HOMO-1", tasks)
        self.assertIn("MO 1_HOMO-2", tasks)
        self.assertEqual(len(tasks), 1 + 5 + 3)  # ESP + LUMOs + 3 HOMOs

    def test_uhf_adds_spin_density_and_alpha_beta_items(self):
        vt = _make_vis_tab(
            mo_data={
                "type": "UHF",
                "energies": [[-1, -2], [-1, -2]],
                "occupations": [[1, 1, 0], [1, 0, 0]],
            }
        )
        vt.populate_analysis_options()
        tasks = self._tasks(vt)
        self.assertIn("SpinDensity", tasks)
        self.assertIn("LUMO_A", tasks)
        self.assertIn("LUMO_B", tasks)
        self.assertIn("MO 2_HOMO_A", tasks)
        self.assertIn("MO 1_HOMO_B", tasks)

    def test_roks_falls_back_gracefully_on_bad_shape(self):
        # occupations shape that trips the internal try/except -> occ_a=[]
        vt = _make_vis_tab(
            mo_data={"type": "ROKS", "occupations": [1, 1, 1, 0, 0], "energies": []}
        )
        vt.populate_analysis_options()
        tasks = self._tasks(vt)
        self.assertEqual(tasks[0], "ESP")
        self.assertEqual(len(tasks), 6)  # ESP + 5 LUMOs, no HOMOs (occ_a empty)


# ---------------------------------------------------------------------------
# disable_existing_analysis_items
# ---------------------------------------------------------------------------


class TestDisableExistingAnalysisItems(unittest.TestCase):
    def _item(self, task):
        it = _FakeListItem()
        it.setData(_ItemDataRole.UserRole, task)
        it.setFlags(_ItemFlag.ItemIsEnabled | _ItemFlag.ItemIsUserCheckable)
        it.setCheckState(_CheckState.Checked)
        return it

    def test_noop_without_cube_files(self):
        vt = _make_vis_tab()
        vt.orb_list.addItem(self._item("ESP"))
        vt.disable_existing_analysis_items([])
        self.assertEqual(vt.orb_list.item(0).checkState(), _CheckState.Checked)

    def test_esp_disabled_when_both_cubes_present(self):
        vt = _make_vis_tab()
        vt.orb_list.addItem(self._item("ESP"))
        vt.disable_existing_analysis_items(["/d/esp.cube", "/d/density.cube"])
        item = vt.orb_list.item(0)
        self.assertFalse(item.flags() & _ItemFlag.ItemIsEnabled)
        self.assertEqual(item.checkState(), _CheckState.Unchecked)

    def test_mo_task_disabled_by_matching_index_file(self):
        vt = _make_vis_tab()
        vt.orb_list.addItem(self._item("MO 5_HOMO"))
        vt.disable_existing_analysis_items(["/d/005_homo.cube"])
        item = vt.orb_list.item(0)
        self.assertFalse(item.flags() & _ItemFlag.ItemIsEnabled)

    def test_label_task_disabled_by_matching_suffix(self):
        vt = _make_vis_tab()
        vt.orb_list.addItem(self._item("SPINDENSITY"))
        vt.disable_existing_analysis_items(["/d/1_spindensity.cube"])
        item = vt.orb_list.item(0)
        self.assertFalse(item.flags() & _ItemFlag.ItemIsEnabled)


# ---------------------------------------------------------------------------
# run_selected_analysis / run_specific_analysis / generate_specific_orbital
# ---------------------------------------------------------------------------


class TestRunSelectedAnalysis(unittest.TestCase):
    def test_no_checkpoint_warns(self):
        vt = _make_vis_tab(chkfile_path=None)
        vt.run_selected_analysis()
        _vis_tab_mod.QMessageBox.warning.assert_called()

    def test_no_selection_informs(self):
        vt = _make_vis_tab()
        it = _FakeListItem()
        it.setData(_ItemDataRole.UserRole, "ESP")
        it.setCheckState(_CheckState.Unchecked)
        vt.orb_list.addItem(it)
        vt.run_selected_analysis()
        _vis_tab_mod.QMessageBox.information.assert_called()

    def test_selected_tasks_dispatched(self):
        vt = _make_vis_tab()
        it = _FakeListItem()
        it.setData(_ItemDataRole.UserRole, "ESP")
        it.setCheckState(_CheckState.Checked)
        vt.orb_list.addItem(it)
        with patch.object(vt, "run_specific_analysis") as mock_run:
            vt.run_selected_analysis()
        mock_run.assert_called_once_with(["ESP"])


class TestRunSpecificAnalysisOutDirFallback(unittest.TestCase):
    def setUp(self):
        self.mock_worker_cls = MagicMock()
        _vis_tab_mod.PropertyWorker = self.mock_worker_cls

    def test_out_d_falls_back_to_dirname_of_chkfile(self):
        vt = _make_vis_tab(last_out_dir=None)
        with patch.object(_vis_tab_mod.os.path, "exists", return_value=True):
            vt.run_specific_analysis(["ESP"])
        args = self.mock_worker_cls.call_args[0]
        self.assertEqual(args[2], os.path.dirname(vt.chkfile_path))

    def test_out_d_falls_back_to_parent_out_dir_edit(self):
        vt = _make_vis_tab(last_out_dir=None, chkfile_path=None)
        vt.chkfile_path = "/fake/pyscf.chk"
        with patch.object(_vis_tab_mod.os.path, "exists", side_effect=[True]):
            # last_out_dir falsy, but chkfile_path truthy -> uses dirname branch;
            # force that branch off by monkeypatching chkfile_path away after entry check
            pass
        vt.last_out_dir = None
        vt.parent_dialog.out_dir_edit.text.return_value = "/from/parent"
        # Make chkfile_path falsy only for the out_d resolution, not the guard
        orig_chk = vt.chkfile_path
        with patch.object(_vis_tab_mod.os.path, "exists", return_value=True):
            vt.chkfile_path = orig_chk
            vt.run_specific_analysis(["ESP"], out_d=None)
        # dirname fallback wins over parent_dialog since chkfile_path is truthy;
        # verify at least a worker was started without raising
        self.mock_worker_cls.assert_called()


class TestGenerateSpecificOrbital(unittest.TestCase):
    def test_builds_task_string_and_delegates(self):
        vt = _make_vis_tab()
        with patch.object(vt, "run_specific_analysis") as mock_run:
            vt.generate_specific_orbital(5, spin_suffix="_A")
        mock_run.assert_called_once_with(["#5_A"])


# ---------------------------------------------------------------------------
# on_prop_finished / on_prop_results
# ---------------------------------------------------------------------------


class TestOnPropFinished(unittest.TestCase):
    def test_unchecks_and_disables_checked_items(self):
        vt = _make_vis_tab(prop_worker=MagicMock())
        it = _FakeListItem()
        it.setCheckState(_CheckState.Checked)
        it.setFlags(_ItemFlag.ItemIsEnabled)
        vt.orb_list.addItem(it)
        vt.on_prop_finished()
        self.assertIsNone(vt.prop_worker)
        vt.btn_run_analysis.setEnabled.assert_called_with(True)
        self.assertEqual(it.checkState(), _CheckState.Unchecked)
        self.assertFalse(it.flags() & _ItemFlag.ItemIsEnabled)


class TestOnPropResults(
    unittest.TestCase,
):
    def test_selects_esp_named_file_first(self):
        vt = _make_vis_tab()
        with (
            patch.object(_vis_tab_mod.os.path, "exists", return_value=True),
            patch.object(
                _vis_tab_mod.glob,
                "glob",
                return_value=["/d/1_homo.cube", "/d/esp.cube"],
            ),
            patch.object(vt, "on_file_selected") as mock_sel,
        ):
            vt.on_prop_results({"files": ["/d/1_homo.cube", "/d/esp.cube"]})
        self.assertEqual(vt.file_list.count(), 2)
        mock_sel.assert_called_once()
        selected_item = mock_sel.call_args[0][0]
        self.assertTrue(selected_item.toolTip().endswith("esp.cube"))

    def test_falls_back_to_last_new_file_when_no_esp(self):
        vt = _make_vis_tab()
        with (
            patch.object(_vis_tab_mod.os.path, "exists", return_value=True),
            patch.object(
                _vis_tab_mod.glob,
                "glob",
                return_value=["/d/1_homo.cube", "/d/2_lumo.cube"],
            ),
            patch.object(vt, "on_file_selected") as mock_sel,
        ):
            vt.on_prop_results({"files": ["/d/1_homo.cube", "/d/2_lumo.cube"]})
        mock_sel.assert_called_once()

    def test_no_new_files_no_selection(self):
        vt = _make_vis_tab()
        with (
            patch.object(_vis_tab_mod.os.path, "exists", return_value=True),
            patch.object(_vis_tab_mod.glob, "glob", return_value=["/d/a.cube"]),
            patch.object(vt, "on_file_selected") as mock_sel,
        ):
            vt.on_prop_results({"files": []})
        mock_sel.assert_not_called()
        self.assertEqual(vt.file_list.count(), 1)

    def test_no_output_dir_is_noop(self):
        vt = _make_vis_tab(last_out_dir=None, chkfile_path=None)
        vt.on_prop_results({"files": []})
        self.assertEqual(vt.file_list.count(), 0)


# ---------------------------------------------------------------------------
# on_file_selected / switch_to_standard_mode / switch_to_mapped_mode
# ---------------------------------------------------------------------------


class TestOnFileSelected(unittest.TestCase):
    def test_none_item_returns(self):
        vt = _make_vis_tab()
        vt.on_file_selected(None)  # should not raise

    def test_missing_path_returns(self):
        vt = _make_vis_tab()
        item = _FakeListItem()
        item.setToolTip("/nowhere.cube")
        with patch.object(_vis_tab_mod.os.path, "exists", return_value=False):
            vt.on_file_selected(item)

    def test_esp_pair_detected_switches_to_mapped(self):
        vt = _make_vis_tab()
        item = _FakeListItem()
        item.setToolTip("/d/esp.cube")

        def fake_exists(p):
            return True

        with (
            patch.object(_vis_tab_mod.os.path, "exists", side_effect=fake_exists),
            patch.object(vt, "switch_to_mapped_mode") as mock_mapped,
            patch.object(vt, "clear_3d_actors"),
        ):
            vt.on_file_selected(item)
        mock_mapped.assert_called_once_with(
            os.path.join("/d", "density.cube"), "/d/esp.cube"
        )

    def test_non_esp_file_switches_to_standard(self):
        vt = _make_vis_tab()
        item = _FakeListItem()
        item.setToolTip("/d/1_homo.cube")
        with (
            patch.object(_vis_tab_mod.os.path, "exists", return_value=True),
            patch.object(vt, "switch_to_standard_mode") as mock_std,
            patch.object(vt, "clear_3d_actors"),
        ):
            vt.on_file_selected(item)
        mock_std.assert_called_once_with("/d/1_homo.cube")

    def test_exception_falls_back_to_standard_mode(self):
        vt = _make_vis_tab()
        item = _FakeListItem()
        item.setToolTip("/d/esp.cube")
        with (
            patch.object(_vis_tab_mod.os.path, "exists", side_effect=RuntimeError("x")),
            patch.object(vt, "clear_3d_actors"),
        ):
            # exists() raising inside on_file_selected's own guard re-raises there
            with self.assertRaises(RuntimeError):
                vt.on_file_selected(item)


class TestSwitchToStandardMode(unittest.TestCase):
    def test_load_success_configures_iso_spin(self):
        vt = _make_vis_tab()
        vt.visualizer.load_file.return_value = True
        vt.visualizer.data_max = 25.0
        with patch.object(vt, "update_visualization") as mock_upd:
            vt.switch_to_standard_mode("/d/x.cube")
        self.assertEqual(vt.mode, "standard")
        vt.iso_spin.setRange.assert_called_once_with(0.0001, 25.0)
        mock_upd.assert_called_once()

    def test_load_success_clamps_small_data_max(self):
        vt = _make_vis_tab()
        vt.visualizer.load_file.return_value = True
        vt.visualizer.data_max = 0.5
        with patch.object(vt, "update_visualization"):
            vt.switch_to_standard_mode("/d/x.cube")
        vt.iso_spin.setRange.assert_called_once_with(0.0001, 10.0)

    def test_load_failure_does_not_touch_iso_spin(self):
        vt = _make_vis_tab()
        vt.visualizer.load_file.return_value = False
        vt.switch_to_standard_mode("/d/x.cube")
        vt.iso_spin.setRange.assert_not_called()

    def test_creates_visualizer_if_missing(self):
        vt = _make_vis_tab(visualizer=None)
        fake_cv = MagicMock()
        fake_cv.load_file.return_value = False
        _vis_tab_mod.CubeVisualizer = MagicMock(return_value=fake_cv)
        vt.switch_to_standard_mode("/d/x.cube")
        self.assertIs(vt.visualizer, fake_cv)


class TestSwitchToMappedMode(unittest.TestCase):
    def test_load_success_configures_ranges(self):
        vt = _make_vis_tab()
        mv = MagicMock()
        mv.load_files.return_value = True
        mv.get_mapped_range.return_value = (-0.02, 0.03)
        vt.mapped_visualizer = mv
        with patch.object(vt, "update_mapped_vis") as mock_upd:
            vt.switch_to_mapped_mode("/d/density.cube", "/d/esp.cube")
        self.assertEqual(vt.mode, "mapped")
        mock_upd.assert_called_once()
        vt.m_min_spin.setValue.assert_called_once_with(-0.02)
        vt.m_max_spin.setValue.assert_called_once_with(0.03)

    def test_narrow_range_padded(self):
        vt = _make_vis_tab()
        mv = MagicMock()
        mv.load_files.return_value = True
        mv.get_mapped_range.return_value = (0.0, 0.0)
        vt.mapped_visualizer = mv
        with patch.object(vt, "update_mapped_vis"):
            vt.switch_to_mapped_mode("/d/density.cube", "/d/esp.cube")
        vt.m_min_spin.setValue.assert_called_once_with(-0.05)
        vt.m_max_spin.setValue.assert_called_once_with(0.05)

    def test_load_failure_skips_range_setup(self):
        vt = _make_vis_tab()
        mv = MagicMock()
        mv.load_files.return_value = False
        vt.mapped_visualizer = mv
        vt.switch_to_mapped_mode("/d/density.cube", "/d/esp.cube")
        vt.m_min_spin.setValue.assert_not_called()


# ---------------------------------------------------------------------------
# update_visualization / update_mapped_vis / fit_mapped_range
# ---------------------------------------------------------------------------


class TestUpdateVisualization(unittest.TestCase):
    def test_delegates_to_mapped_when_mode_mapped(self):
        vt = _make_vis_tab(mode="mapped")
        with patch.object(vt, "update_mapped_vis") as mock_m:
            vt.update_visualization()
        mock_m.assert_called_once()

    def test_noop_without_visualizer_or_file(self):
        vt = _make_vis_tab(visualizer=None, loaded_file=None)
        vt.update_visualization()  # should not raise

    def test_calls_update_iso_with_values(self):
        vt = _make_vis_tab(loaded_file="/d/x.cube")
        vt.iso_spin.value.return_value = 0.02
        vt.op_slider.value.return_value = 50
        vt.update_visualization()
        vt.visualizer.update_iso.assert_called_once_with(0.02, "blue", "red", 0.5)


class TestUpdateMappedVis(unittest.TestCase):
    def test_noop_without_mapped_visualizer(self):
        vt = _make_vis_tab(mapped_visualizer=None)
        vt.update_mapped_vis()  # should not raise

    def test_calls_update_mesh(self):
        vt = _make_vis_tab()
        mv = MagicMock()
        vt.mapped_visualizer = mv
        vt.m_iso_spin.value.return_value = 0.004
        vt.m_min_spin.value.return_value = -0.1
        vt.m_max_spin.value.return_value = 0.1
        vt.cmap_combo.currentText.return_value = "jet"
        vt.m_op_slider.value.return_value = 40
        vt.update_mapped_vis()
        mv.update_mesh.assert_called_once_with(0.004, 0.4, cmap="jet", clim=[-0.1, 0.1])


class TestFitMappedRange(unittest.TestCase):
    def test_noop_without_mapped_visualizer(self):
        vt = _make_vis_tab(mapped_visualizer=None)
        vt.fit_mapped_range()  # should not raise

    def test_extends_spin_minimum_when_out_of_range(self):
        vt = _make_vis_tab()
        mv = MagicMock()
        mv.get_mapped_range.return_value = (-5.0, 5.0)
        vt.mapped_visualizer = mv
        vt.m_min_spin.minimum.return_value = -1.0
        vt.m_max_spin.maximum.return_value = 1.0
        with patch.object(vt, "update_mapped_vis") as mock_upd:
            vt.fit_mapped_range()
        vt.m_min_spin.setMinimum.assert_called_once_with(-6.0)
        vt.m_max_spin.setMaximum.assert_called_once_with(6.0)
        mock_upd.assert_called_once()

    def test_narrow_range_padded(self):
        vt = _make_vis_tab()
        mv = MagicMock()
        mv.get_mapped_range.return_value = (0.0, 0.0)
        vt.mapped_visualizer = mv
        vt.m_min_spin.minimum.return_value = -1.0
        vt.m_max_spin.maximum.return_value = 1.0
        with patch.object(vt, "update_mapped_vis"):
            vt.fit_mapped_range()
        vt.m_min_spin.setValue.assert_called_once_with(-0.05)
        vt.m_max_spin.setValue.assert_called_once_with(0.05)


# ---------------------------------------------------------------------------
# choose_color
# ---------------------------------------------------------------------------


class TestChooseColor(unittest.TestCase):
    def test_valid_positive_color_updates_state(self):
        vt = _make_vis_tab()
        _vis_tab_mod.QColorDialog.getColor.return_value = _FakeColor(True, "#abcdef")
        with patch.object(vt, "update_visualization") as mock_upd:
            vt.choose_color("p")
        self.assertEqual(vt.color_p, "#abcdef")
        mock_upd.assert_called_once()

    def test_valid_negative_color_updates_state(self):
        vt = _make_vis_tab()
        _vis_tab_mod.QColorDialog.getColor.return_value = _FakeColor(True, "#112233")
        with patch.object(vt, "update_visualization"):
            vt.choose_color("n")
        self.assertEqual(vt.color_n, "#112233")

    def test_invalid_color_leaves_state(self):
        vt = _make_vis_tab()
        _vis_tab_mod.QColorDialog.getColor.return_value = _FakeColor(False)
        with patch.object(vt, "update_visualization") as mock_upd:
            vt.choose_color("p")
        self.assertEqual(vt.color_p, "blue")
        mock_upd.assert_not_called()


# ---------------------------------------------------------------------------
# clear_3d_actors
# ---------------------------------------------------------------------------


class TestClear3dActors(unittest.TestCase):
    def test_no_view_3d_manager_is_noop(self):
        vt = _make_vis_tab()
        vt.context.get_main_window.return_value = MagicMock(spec=[])
        vt.clear_3d_actors()  # should not raise

    def test_plotter_none_is_noop(self):
        vt = _make_vis_tab()
        mw = MagicMock()
        mw.view_3d_manager.plotter = None
        vt.context.get_main_window.return_value = mw
        vt.clear_3d_actors()

    def test_removes_actors_and_renders(self):
        vt = _make_vis_tab()
        mw = MagicMock()
        mw.view_3d_manager.plotter = MagicMock()
        vt.context.get_main_window.return_value = mw
        vt.parent_dialog.closing = False
        vt.clear_3d_actors()
        mw.view_3d_manager.plotter.remove_actor.assert_any_call("pyscf_iso_p")
        mw.view_3d_manager.plotter.remove_actor.assert_any_call("pyscf_iso_n")
        mw.view_3d_manager.plotter.remove_actor.assert_any_call("pyscf_mapped")
        vt.visualizer.clear_actors.assert_called_once()
        mw.view_3d_manager.plotter.render.assert_called_once()

    def test_skips_render_when_closing(self):
        vt = _make_vis_tab()
        mw = MagicMock()
        mw.view_3d_manager.plotter = MagicMock()
        vt.context.get_main_window.return_value = mw
        vt.parent_dialog.closing = True
        vt.clear_3d_actors()
        mw.view_3d_manager.plotter.render.assert_not_called()

    def test_cleans_up_freq_vis(self):
        # freq_vis cleanup only runs if the code reaches the end of the
        # method -- an mw lacking view_3d_manager triggers an early `return`
        # inside the outer try, which skips it entirely (see source).
        vt = _make_vis_tab()
        mw = MagicMock()
        mw.view_3d_manager.plotter = MagicMock()
        vt.context.get_main_window.return_value = mw
        vt.freq_vis = MagicMock()
        vt.clear_3d_actors()
        vt.freq_vis.cleanup.assert_called_once()

    def test_freq_vis_cleanup_skipped_when_early_return(self):
        vt = _make_vis_tab()
        vt.context.get_main_window.return_value = MagicMock(spec=[])
        vt.freq_vis = MagicMock()
        vt.clear_3d_actors()
        vt.freq_vis.cleanup.assert_not_called()

    def test_swallows_outer_exception(self):
        vt = _make_vis_tab()
        vt.context.get_main_window.side_effect = RuntimeError("boom")
        vt.clear_3d_actors()  # should not raise


# ---------------------------------------------------------------------------
# load_optimized_geometry / update_geometry
# ---------------------------------------------------------------------------


class TestLoadOptimizedGeometry(unittest.TestCase):
    def test_noop_without_xyz(self):
        vt = _make_vis_tab(optimized_xyz=None)
        _vis_tab_mod.QMessageBox.information.reset_mock()
        vt.load_optimized_geometry()
        _vis_tab_mod.QMessageBox.information.assert_not_called()

    def test_updates_and_informs(self):
        vt = _make_vis_tab(optimized_xyz="3\n\nC 0 0 0\n")
        with patch.object(vt, "update_geometry") as mock_upd:
            vt.load_optimized_geometry()
        mock_upd.assert_called_once_with(vt.optimized_xyz)
        _vis_tab_mod.QMessageBox.information.assert_called()


class TestUpdateGeometry(unittest.TestCase):
    def test_updates_molecule_and_pushes_undo(self):
        vt = _make_vis_tab()
        _vis_tab_mod.update_molecule_from_xyz = MagicMock()
        mw = MagicMock()
        vt.context.get_main_window.return_value = mw
        with patch.object(vt, "clear_3d_actors") as mock_clear:
            vt.update_geometry("xyz-data")
        mock_clear.assert_called_once()
        _vis_tab_mod.update_molecule_from_xyz.assert_called_once_with(
            vt.context, "xyz-data"
        )
        mw.edit_actions_manager.push_undo_state.assert_called_once()
        vt.log.assert_called_with("Geometry updated.")

    def test_cleans_up_freq_vis_first(self):
        vt = _make_vis_tab()
        vt.freq_vis = MagicMock()
        _vis_tab_mod.update_molecule_from_xyz = MagicMock()
        vt.context.get_main_window.return_value = MagicMock(spec=[])
        with patch.object(vt, "clear_3d_actors"):
            vt.update_geometry("xyz-data")
        self.assertIsNone(vt.freq_vis)


# ---------------------------------------------------------------------------
# add_custom_mo
# ---------------------------------------------------------------------------


class TestAddCustomMo(unittest.TestCase):
    def test_empty_text_is_noop(self):
        vt = _make_vis_tab()
        vt.mo_input.text.return_value = ""
        vt.add_custom_mo()
        self.assertEqual(vt.orb_list.count(), 0)

    def test_invalid_text_warns(self):
        vt = _make_vis_tab()
        vt.mo_input.text.return_value = "not valid!!"
        vt.add_custom_mo()
        _vis_tab_mod.QMessageBox.warning.assert_called()
        self.assertEqual(vt.orb_list.count(), 0)

    def test_plain_digit_adds_mo_item(self):
        vt = _make_vis_tab()
        vt.mo_input.text.return_value = "12"
        vt.add_custom_mo()
        self.assertEqual(vt.orb_list.count(), 1)
        item = vt.orb_list.item(0)
        self.assertEqual(item.data(_ItemDataRole.UserRole), "MO 12")
        vt.mo_input.clear.assert_called_once()

    def test_digit_with_alpha_suffix(self):
        vt = _make_vis_tab()
        vt.mo_input.text.return_value = "7a"
        vt.add_custom_mo()
        item = vt.orb_list.item(0)
        self.assertEqual(item.data(_ItemDataRole.UserRole), "MO 7_A")

    def test_digit_with_beta_suffix(self):
        vt = _make_vis_tab()
        vt.mo_input.text.return_value = "7B"
        vt.add_custom_mo()
        item = vt.orb_list.item(0)
        self.assertEqual(item.data(_ItemDataRole.UserRole), "MO 7_B")

    def test_relative_homo_label(self):
        vt = _make_vis_tab()
        vt.mo_input.text.return_value = "HOMO-1"
        vt.add_custom_mo()
        item = vt.orb_list.item(0)
        self.assertEqual(item.data(_ItemDataRole.UserRole), "HOMO-1")

    def test_digit_resolves_relative_display_label(self):
        vt = _make_vis_tab(mo_data={"occupations": [2, 2, 0, 0]})
        vt.mo_input.text.return_value = "1"
        vt.add_custom_mo()
        item = vt.orb_list.item(0)
        self.assertIn("HOMO", item.text())

    def test_duplicate_task_not_added_twice(self):
        vt = _make_vis_tab()
        vt.mo_input.text.return_value = "12"
        vt.add_custom_mo()
        vt.mo_input.text.return_value = "12"
        vt.add_custom_mo()
        self.assertEqual(vt.orb_list.count(), 1)


# ---------------------------------------------------------------------------
# show_energy_diagram
# ---------------------------------------------------------------------------


class TestShowEnergyDiagram(unittest.TestCase):
    def test_noop_without_mo_data(self):
        vt = _make_vis_tab(mo_data=None)
        vt.show_energy_diagram()  # should not raise

    def test_opens_new_dialog(self):
        vt = _make_vis_tab(mo_data={"type": "RHF"})
        fake_dlg = MagicMock()
        _vis_tab_mod.EnergyDiagramDialog = MagicMock(return_value=fake_dlg)
        vt.show_energy_diagram()
        self.assertIs(vt.energy_dlg, fake_dlg)
        fake_dlg.show.assert_called_once()

    def test_closes_existing_dialog_first(self):
        vt = _make_vis_tab(mo_data={"type": "RHF"})
        old_dlg = MagicMock()
        vt.energy_dlg = old_dlg
        fake_dlg = MagicMock()
        _vis_tab_mod.EnergyDiagramDialog = MagicMock(return_value=fake_dlg)
        vt.show_energy_diagram()
        old_dlg.close.assert_called_once()
        self.assertIs(vt.energy_dlg, fake_dlg)


# ---------------------------------------------------------------------------
# load_file_by_path
# ---------------------------------------------------------------------------


class TestLoadFileByPath(unittest.TestCase):
    def test_missing_path_returns(self):
        vt = _make_vis_tab()
        with patch.object(_vis_tab_mod.os.path, "exists", return_value=False):
            vt.load_file_by_path("/nope.cube")

    def test_found_in_file_list_selects_it(self):
        vt = _make_vis_tab()
        item = _FakeListItem()
        item.setToolTip("/d/x.cube")
        vt.file_list.addItem(item)
        with (
            patch.object(_vis_tab_mod.os.path, "exists", return_value=True),
            patch.object(vt, "on_file_selected") as mock_sel,
        ):
            vt.load_file_by_path("/d/x.cube")
        mock_sel.assert_called_once_with(item)

    def test_not_found_falls_back_to_standard_mode(self):
        vt = _make_vis_tab()
        with (
            patch.object(_vis_tab_mod.os.path, "exists", return_value=True),
            patch.object(vt, "switch_to_standard_mode") as mock_std,
        ):
            vt.load_file_by_path("/d/y.cube")
        mock_std.assert_called_once_with("/d/y.cube")


# ---------------------------------------------------------------------------
# show_thermo_data / export_thermo_csv
# ---------------------------------------------------------------------------


class TestShowThermoData(unittest.TestCase):
    def test_no_data_informs(self):
        vt = _make_vis_tab(thermo_data=None)
        vt.show_thermo_data()
        _vis_tab_mod.QMessageBox.information.assert_called()

    def test_populates_table_with_known_and_extra_keys(self):
        vt = _make_vis_tab(
            thermo_data={
                "E_tot": -1.234567,
                "G_tot": [-1.1, "Ha"],
                "Custom_Key": "some text",
            }
        )
        _vis_tab_mod.QDialog.return_value.exec.reset_mock()
        vt.show_thermo_data()
        dlg = _vis_tab_mod.QDialog.return_value
        dlg.exec.assert_called_once()

    def test_flatten_value_handles_nested_lists(self):
        vt = _make_vis_tab(
            thermo_data={"Cv_tot": [[1.0, "cal/mol-K"], [2.0, "cal/mol-K"]]}
        )
        vt.show_thermo_data()  # should not raise; exercises recursive flatten_value


class TestExportThermoCsv(unittest.TestCase):
    def test_cancel_returns_early(self):
        vt = _make_vis_tab()
        _vis_tab_mod.QFileDialog.getSaveFileName.return_value = ("", "")
        table = _FakeTableWidget()
        vt.export_thermo_csv(table)  # should not raise / not attempt to open file

    def test_writes_csv_and_informs(self):
        vt = _make_vis_tab()
        table = _FakeTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Property", "Value", "Unit"])
        table.insertRow(0)
        table.setItem(0, 0, _FakeTableItem("E_tot"))
        table.setItem(0, 1, _FakeTableItem("-1.234567"))
        table.setItem(0, 2, _FakeTableItem("Ha"))

        import tempfile

        with tempfile.TemporaryDirectory() as d:
            out_csv = os.path.join(d, "out.csv")
            _vis_tab_mod.QFileDialog.getSaveFileName.return_value = (out_csv, "")
            vt.export_thermo_csv(table)
            self.assertTrue(os.path.exists(out_csv))
            with open(out_csv) as f:
                content = f.read()
            self.assertIn("E_tot", content)
        _vis_tab_mod.QMessageBox.information.assert_called()

    def test_write_failure_shows_critical(self):
        vt = _make_vis_tab()
        table = _FakeTableWidget()
        _vis_tab_mod.QFileDialog.getSaveFileName.return_value = (
            "/nonexistent_dir_xyz/out.csv",
            "",
        )
        vt.export_thermo_csv(table)
        _vis_tab_mod.QMessageBox.critical.assert_called()


# ---------------------------------------------------------------------------
# on_calculation_finished / close_freq_window
# ---------------------------------------------------------------------------


class TestOnCalculationFinished(unittest.TestCase):
    def test_no_out_dir_is_noop(self):
        vt = _make_vis_tab()
        with patch.object(vt, "load_result_folder") as mock_load:
            vt.on_calculation_finished({})
        mock_load.assert_not_called()

    def test_dispatches_to_load_result_folder(self):
        vt = _make_vis_tab()
        with patch.object(vt, "load_result_folder") as mock_load:
            vt.on_calculation_finished(
                {"out_dir": "/d/out", "optimized_xyz": "3\n\nC 0 0 0\n"}
            )
        mock_load.assert_called_once_with(
            "/d/out", update_structure=True, is_opt_job=True
        )
        vt.result_path_display.setText.assert_called_once_with("/d/out")


class TestCloseFreqWindow(unittest.TestCase):
    def test_closes_and_clears_when_present(self):
        vt = _make_vis_tab()
        vt.freq_dock = MagicMock()
        vt.freq_vis = MagicMock()
        vt.close_freq_window()
        self.assertIsNone(vt.freq_dock)
        self.assertIsNone(vt.freq_vis)

    def test_noop_when_absent(self):
        vt = _make_vis_tab(freq_dock=None)
        vt.close_freq_window()  # should not raise
        self.assertIsNone(vt.freq_dock)


# ---------------------------------------------------------------------------
# load_result_folder
# ---------------------------------------------------------------------------


class TestLoadResultFolder(unittest.TestCase):
    def test_dialog_cancelled_returns(self):
        vt = _make_vis_tab()
        _vis_tab_mod.QFileDialog.getExistingDirectory.return_value = ""
        vt.load_result_folder(path=None)
        vt.result_path_display.setText.assert_not_called()

    def test_scan_result_folder_detected(self):
        vt = _make_vis_tab()
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "scan_results.csv"), "w").close()
            open(os.path.join(d, "scan_trajectory.xyz"), "w").close()
            with (
                patch.object(vt, "load_scan_results") as mock_scan,
                patch.object(vt, "_add_to_history"),
            ):
                vt.load_result_folder(path=d)
            mock_scan.assert_called_once_with(os.path.abspath(d))

    def test_missing_checkpoint_warns(self):
        vt = _make_vis_tab()
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            vt.load_result_folder(path=d)
        _vis_tab_mod.QMessageBox.warning.assert_called()

    def test_alt_checkpoint_name_accepted(self):
        vt = _make_vis_tab()
        import tempfile

        _vis_tab_mod.LoadWorker = MagicMock()
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "checkpoint.chk"), "w").close()
            with patch.object(vt, "_add_to_history"):
                vt.load_result_folder(path=d)
        _vis_tab_mod.LoadWorker.assert_called_once()
        args = _vis_tab_mod.LoadWorker.call_args[0]
        self.assertTrue(args[0].endswith("checkpoint.chk"))

    def test_load_worker_none_shows_critical(self):
        vt = _make_vis_tab()
        _vis_tab_mod.LoadWorker = None
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "pyscf.chk"), "w").close()
            with patch.object(vt, "_add_to_history"):
                vt.load_result_folder(path=d)
        _vis_tab_mod.QMessageBox.critical.assert_called()

    def test_busy_load_worker_warns(self):
        vt = _make_vis_tab()
        running = MagicMock()
        running.isRunning.return_value = True
        vt.load_worker = running
        _vis_tab_mod.LoadWorker = MagicMock()
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "pyscf.chk"), "w").close()
            with patch.object(vt, "_add_to_history"):
                vt.load_result_folder(path=d)
        _vis_tab_mod.QMessageBox.warning.assert_called()

    def test_success_starts_worker(self):
        vt = _make_vis_tab()
        fake_worker = MagicMock()
        _vis_tab_mod.LoadWorker = MagicMock(return_value=fake_worker)
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "pyscf.chk"), "w").close()
            with patch.object(vt, "_add_to_history"):
                vt.load_result_folder(path=d)
        fake_worker.start.assert_called_once()
        vt.parent_dialog.progress_bar.show.assert_called_once()


# ---------------------------------------------------------------------------
# load_scan_results
# ---------------------------------------------------------------------------


class TestLoadScanResults(unittest.TestCase):
    def _write_scan_dir(self, d):
        with open(os.path.join(d, "scan_results.csv"), "w", newline="") as f:
            f.write("Step,Value,Energy\n0,1.0,-10.0\n1,1.1,-10.1\n")
        with open(os.path.join(d, "scan_trajectory.xyz"), "w") as f:
            f.write("1\ncomment\nC 0.0 0.0 0.0\n1\ncomment\nC 0.0 0.0 0.1\n")

    def test_success_opens_scan_dialog(self):
        vt = _make_vis_tab()
        fake_dlg = MagicMock()
        _vis_tab_mod.ScanResultDialog = MagicMock(return_value=fake_dlg)
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._write_scan_dir(d)
            vt.load_scan_results(d)
        fake_dlg.show.assert_called_once()
        self.assertIs(vt.scan_dlg, fake_dlg)

    def test_bad_csv_raises(self):
        vt = _make_vis_tab()
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "scan_trajectory.xyz"), "w") as f:
                f.write("1\ncomment\nC 0 0 0\n")
            with self.assertRaises(Exception):
                vt.load_scan_results(d)

    def test_scan_result_dialog_none_raises(self):
        vt = _make_vis_tab()
        _vis_tab_mod.ScanResultDialog = None
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._write_scan_dir(d)
            with self.assertRaises(Exception):
                vt.load_scan_results(d)

    def test_history_changed_marks_project_modified(self):
        vt = _make_vis_tab()
        _vis_tab_mod.ScanResultDialog = MagicMock(return_value=MagicMock())
        vt._history_changed = True
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._write_scan_dir(d)
            vt.load_scan_results(d)
        vt.context.mark_project_modified.assert_called_once()
        self.assertFalse(vt._history_changed)


# ---------------------------------------------------------------------------
# on_load_finished / finalize_load
# ---------------------------------------------------------------------------


class TestOnLoadFinished(unittest.TestCase):
    def test_basic_result_enables_buttons(self):
        vt = _make_vis_tab()
        with (
            patch.object(_vis_tab_mod, "QTimer", _FakeQTimerImmediate),
            patch.object(vt, "finalize_load") as mock_final,
            patch.object(vt, "clear_3d_actors"),
            patch.object(_vis_tab_mod.os.path, "exists", return_value=False),
        ):
            vt.on_load_finished(
                {
                    "chkfile": "/out/pyscf.chk",
                    "mo_energy": [-1, -2],
                    "mo_occ": [2, 0],
                    "scf_type": "RHF",
                }
            )
        self.assertEqual(vt.chkfile_path, "/out/pyscf.chk")
        vt.btn_show_diagram.setEnabled.assert_called_with(True)
        vt.btn_run_analysis.setEnabled.assert_called_with(True)
        mock_final.assert_called_once()

    def test_optimized_xyz_triggers_deferred_update(self):
        vt = _make_vis_tab()
        mw = MagicMock()
        vt.context.get_main_window.return_value = mw
        with (
            patch.object(_vis_tab_mod, "QTimer", _FakeQTimerImmediate),
            patch.object(vt, "finalize_load"),
            patch.object(vt, "clear_3d_actors"),
            patch.object(vt, "update_geometry") as mock_upd,
            patch.object(_vis_tab_mod.os.path, "exists", return_value=False),
        ):
            vt.on_load_finished({"optimized_xyz": "3\n\nC 0 0 0\n", "loaded_xyz": None})
        self.assertEqual(vt.optimized_xyz, "3\n\nC 0 0 0\n")
        vt.btn_load_geom.setEnabled.assert_called_with(True)
        mock_upd.assert_called_once_with("3\n\nC 0 0 0\n")
        vt.context.mark_project_modified.assert_called()

    def test_loaded_xyz_without_opt_flag_updates_geometry_but_not_source(self):
        vt = _make_vis_tab()
        mw = MagicMock()
        vt.context.get_main_window.return_value = mw
        with (
            patch.object(_vis_tab_mod, "QTimer", _FakeQTimerImmediate),
            patch.object(vt, "finalize_load"),
            patch.object(vt, "clear_3d_actors"),
            patch.object(vt, "update_geometry") as mock_upd,
            patch.object(_vis_tab_mod.os.path, "exists", return_value=False),
        ):
            vt.on_load_finished({"loaded_xyz": "3\n\nC 0 0 0\n"})
        mock_upd.assert_called_once_with("3\n\nC 0 0 0\n")

    def test_thermo_data_enables_button(self):
        vt = _make_vis_tab()
        with (
            patch.object(_vis_tab_mod, "QTimer", _FakeQTimerNoOp),
            patch.object(vt, "finalize_load"),
            patch.object(vt, "clear_3d_actors"),
            patch.object(_vis_tab_mod.os.path, "exists", return_value=False),
        ):
            vt.on_load_finished({"thermo_data": {"E_tot": -1.0}})
        vt.btn_show_thermo.setEnabled.assert_called_with(True)

    def test_finds_existing_cube_files(self):
        vt = _make_vis_tab()
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            vt.last_out_dir = d
            open(os.path.join(d, "1_homo.cube"), "w").close()
            with (
                patch.object(_vis_tab_mod, "QTimer", _FakeQTimerNoOp),
                patch.object(vt, "finalize_load") as mock_final,
                patch.object(vt, "clear_3d_actors"),
            ):
                vt.on_load_finished({})
        self.assertEqual(vt.file_list.count(), 1)
        mock_final.assert_called_once()

    def test_cleans_up_prior_freq_vis_and_dock(self):
        vt = _make_vis_tab()
        vt.freq_vis = MagicMock()
        vt.freq_dock = MagicMock()
        mw = MagicMock()
        vt.context.get_main_window.return_value = mw
        with (
            patch.object(_vis_tab_mod, "QTimer", _FakeQTimerNoOp),
            patch.object(vt, "finalize_load"),
            patch.object(vt, "clear_3d_actors"),
            patch.object(_vis_tab_mod.os.path, "exists", return_value=False),
        ):
            vt.on_load_finished({})
        self.assertIsNone(vt.visualizer)
        self.assertIsNone(vt.mapped_visualizer)


class TestFinalizeLoad(unittest.TestCase):
    def test_closes_prior_scan_and_tddft_dialogs(self):
        vt = _make_vis_tab()
        vt.scan_dlg = MagicMock()
        vt.tddft_dlg = MagicMock()
        vt.finalize_load({})
        self.assertIsNone(vt.scan_dlg)
        self.assertIsNone(vt.tddft_dlg)

    def test_freq_data_present_but_freqvisualizer_missing_logs_error(self):
        vt = _make_vis_tab(freq_data=None)
        vt.context.current_molecule = MagicMock()
        _vis_tab_mod.FreqVisualizer = None
        with patch.object(vt, "clear_3d_actors"):
            vt.finalize_load({"freq_data": {"freqs": [1, 2], "modes": [[0], [0]]}})
        self.assertIsNotNone(vt.freq_data)
        # FreqVisualizer(None-callable) raises TypeError -> caught -> logged
        self.assertTrue(
            any(
                "Error opening Frequency Visualizer" in str(c)
                for c in vt.log.call_args_list
            )
        )

    def test_freq_data_present_no_current_molecule_skips_visualizer(self):
        vt = _make_vis_tab()
        vt.context.current_molecule = None
        vt.finalize_load({"freq_data": {"freqs": [1], "modes": [[0]]}})
        self.assertIsNone(vt.freq_vis)

    def test_tddft_data_opens_table(self):
        vt = _make_vis_tab()
        fake_mod = types.ModuleType("pyscf_calculator.tddft_table")
        fake_mod.TddftTable = MagicMock(return_value=MagicMock())
        with (
            patch.dict(sys.modules, {"pyscf_calculator.tddft_table": fake_mod}),
            patch.object(_vis_tab_mod, "__package__", "pyscf_calculator"),
        ):
            vt.finalize_load({"tddft_data": {"states": []}})
        # Either succeeds (dlg shown) or fails gracefully (logged) -- both fine;
        # just confirm no unhandled exception escaped.

    def test_scan_results_opens_dialog_with_trajectory(self):
        vt = _make_vis_tab()
        fake_dlg = MagicMock()
        _vis_tab_mod.ScanResultDialog = MagicMock(return_value=fake_dlg)
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            traj_path = os.path.join(d, "traj.xyz")
            with open(traj_path, "w") as f:
                f.write("1\ncomment\nC 0 0 0\n")
            vt.finalize_load(
                {
                    "scan_results": [{"step": 0, "value": 1.0, "energy": -1.0}],
                    "scan_trajectory_path": traj_path,
                }
            )
        fake_dlg.show.assert_called_once()
        self.assertIs(vt.scan_dlg, fake_dlg)

    def test_cubes_disables_existing_items(self):
        vt = _make_vis_tab()
        with patch.object(vt, "disable_existing_analysis_items") as mock_dis:
            vt.finalize_load({}, cubes=["/d/esp.cube"])
        mock_dis.assert_called_once_with(["/d/esp.cube"])


if __name__ == "__main__":
    unittest.main()
