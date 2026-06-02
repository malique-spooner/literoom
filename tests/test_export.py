from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from literoom.config import AppConfig, save_config
from literoom.metadata import manifest
from literoom.pipeline import run_export_library
from literoom.utils.hashing import sha256_file


class ExportTests(unittest.TestCase):
    def test_run_export_library_copies_built_assets_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "literoom.local.yaml"
            save_config(AppConfig(), config_path)

            managed_dir = root / "library"
            source_file = managed_dir / "2024" / "01" / "example.jpg"
            source_file.parent.mkdir(parents=True, exist_ok=True)
            source_file.write_bytes(b"example-image-bytes")

            db_path = root / ".literoom" / "manifest.sqlite"
            manifest.init_db(db_path)

            asset_hash = sha256_file(source_file)
            con = sqlite3.connect(db_path)
            try:
                con.execute(
                    """
                    INSERT INTO assets (
                      id, source, abs_zip, zip_path, source_kind, source_locator, source_path,
                      media_type, orig_filename, orig_ext, orig_size, dt_original,
                      status, sha256, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "asset-1",
                        "local",
                        str(source_file),
                        "2024/01/example.jpg",
                        "file",
                        str(source_file),
                        "2024/01/example.jpg",
                        "image",
                        "example.jpg",
                        ".jpg",
                        source_file.stat().st_size,
                        "2024-01-02T03:04:05",
                        "EMBEDDED",
                        asset_hash,
                        "2024-01-02T03:04:05",
                    ),
                )
                con.execute(
                    """
                    INSERT INTO managed_assets (id, asset_id, relpath, filename, managed_path, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "managed-1",
                        "asset-1",
                        "2024/01",
                        "example.jpg",
                        "2024/01/example.jpg",
                        "BUILT",
                    ),
                )
                con.commit()
            finally:
                con.close()

            export_dir = root / "exports"
            result = run_export_library(config_path, destination=export_dir)

            exported_file = export_dir / "2024" / "01" / "example.jpg"
            manifest_file = export_dir / "literoom-export.jsonl"
            summary_file = export_dir / "literoom-export-summary.json"

            self.assertEqual(result["exported"], 1)
            self.assertEqual(result["skipped"], 0)
            self.assertEqual(result["missing"], 0)
            self.assertTrue(exported_file.exists())
            self.assertTrue(manifest_file.exists())
            self.assertTrue(summary_file.exists())
            self.assertEqual(exported_file.read_bytes(), b"example-image-bytes")
            self.assertIn('"status": "exported"', manifest_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
