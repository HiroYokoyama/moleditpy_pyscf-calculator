"""
tests/test_update_zenodo.py

plan_file_uploads(): a re-run against a new-version draft that an earlier
run left unpublished (v4.0.0: Zenodo answered the publish with a 500) must
not re-register files the draft already holds -- that fails with "File
with key ... already exists" and made the archive unrecoverable by re-run.
"""

import hashlib
import importlib.util
import os
import tempfile
import unittest

_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "scripts", "update_zenodo.py")
)
_spec = importlib.util.spec_from_file_location("update_zenodo_under_test", _SRC)
zen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(zen)


class TestPlanFileUploads(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.zip = self._write("plugin_4.0.0.zip", b"zip-bytes")
        self.src = self._write("source.tar.gz", b"tar-bytes")

    def _write(self, name, data):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    @staticmethod
    def _entry(key, data, status="completed"):
        return {
            "key": key,
            "checksum": "md5:" + hashlib.md5(data).hexdigest(),
            "status": status,
        }

    def test_fresh_draft_uploads_everything(self):
        up, stale = zen.plan_file_uploads([], [self.zip, self.src])
        self.assertEqual(up, [self.zip, self.src])
        self.assertEqual(stale, [])

    def test_files_already_committed_are_skipped(self):
        entries = [
            self._entry("plugin_4.0.0.zip", b"zip-bytes"),
            self._entry("source.tar.gz", b"tar-bytes"),
        ]
        up, stale = zen.plan_file_uploads(entries, [self.zip, self.src])
        self.assertEqual((up, stale), ([], []))

    def test_changed_file_is_replaced(self):
        entries = [self._entry("plugin_4.0.0.zip", b"older-bytes")]
        up, stale = zen.plan_file_uploads(entries, [self.zip, self.src])
        self.assertEqual(up, [self.zip, self.src])
        self.assertEqual(stale, ["plugin_4.0.0.zip"])

    def test_half_uploaded_file_is_replaced(self):
        entries = [self._entry("source.tar.gz", b"tar-bytes", status="pending")]
        up, stale = zen.plan_file_uploads(entries, [self.src])
        self.assertEqual((up, stale), ([self.src], ["source.tar.gz"]))

    def test_md5_matches_hashlib(self):
        self.assertEqual(zen.file_md5(self.zip), hashlib.md5(b"zip-bytes").hexdigest())


if __name__ == "__main__":
    unittest.main()
