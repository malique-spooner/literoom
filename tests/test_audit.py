from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from photo_unifier.audit import build_audit_report, format_audit_report
from photo_unifier.metadata import manifest


class AuditTests(unittest.TestCase):
    def test_audit_report_summarizes_sources_and_flags_weak_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            manifest.upsert_raw(
                [
                    {
                        "source": "local",
                        "abs_zip": str(root / "local.jpg"),
                        "zip_path": "local.jpg",
                        "source_kind": "file",
                        "source_locator": str(root / "local.jpg"),
                        "source_path": "local.jpg",
                        "media_type": "image",
                        "orig_filename": "local.jpg",
                        "orig_ext": ".jpg",
                        "orig_size": 10,
                        "dt_original": "2024-01-01T12:00:00",
                        "src_mtime": "2024-01-01T12:00:00",
                        "source_dt": "exif_datetime",
                    },
                    {
                        "source": "google",
                        "abs_zip": str(root / "google.zip"),
                        "zip_path": "google.jpg",
                        "source_kind": "zip",
                        "source_locator": str(root / "google.zip"),
                        "source_path": "google.jpg",
                        "media_type": "image",
                        "orig_filename": "google.jpg",
                        "orig_ext": ".jpg",
                        "orig_size": 10,
                        "dt_original": None,
                        "src_mtime": "2024-01-01T12:00:00",
                        "source_dt": None,
                    },
                ],
                db_path,
            )

            asset = manifest.list_assets(db_path, limit=10, include_hidden=True)[0]
            manifest.replace_faces_for_asset(
                db_path,
                asset["id"],
                [{"id": "face-1", "bbox": {"x": 1, "y": 2, "w": 3, "h": 4}}],
                source_name="test",
            )

            report = build_audit_report(db_path, sample_size=10)
            text = format_audit_report(report)

            self.assertIn("Literoom audit", text)
            self.assertIn("local (file)", text)
            self.assertIn("google (zip)", text)
            self.assertIn("potential inaccuracies", text)
            self.assertEqual(report["samples"]["local"]["sample_size"], 1)
            self.assertEqual(report["samples"]["google"]["sample_size"], 1)
            self.assertTrue(report["samples"]["google"]["issues"])


if __name__ == "__main__":
    unittest.main()
