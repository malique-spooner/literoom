from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from photo_unifier.metadata import manifest, master


class MasterBuildTests(unittest.TestCase):
    def test_build_writes_provider_agnostic_metadata_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            source_file = root / "IMG_0001.JPG"
            managed_dir = root / "managed"
            source_file.write_bytes(b"fake-image-bytes")

            manifest.init_db(db_path)
            manifest.upsert_raw(
                [
                    {
                        "source": "local",
                        "abs_zip": str(source_file),
                        "zip_path": source_file.name,
                        "source_kind": "file",
                        "source_root": str(root),
                        "source_locator": str(source_file),
                        "source_path": str(source_file),
                        "media_type": "image",
                        "orig_filename": source_file.name,
                        "orig_ext": ".jpg",
                        "orig_size": source_file.stat().st_size,
                        "dt_original": "2024-02-03T04:05:06",
                        "title": "Test Title",
                        "description": "Test Description",
                        "keywords": ["test"],
                        "people": ["Alex"],
                        "src_mtime": "2024-02-03T04:05:06",
                        "source_dt": "filename",
                        "source_gps": None,
                    }
                ],
                db_path,
            )
            manifest.plan_targets(db_path)

            master.build(managed_dir, db_path)

            asset = manifest.list_assets(db_path, limit=1)[0]
            artifacts = manifest.list_asset_artifacts(db_path, asset["id"])
            sidecars = [item for item in artifacts if item["artifact_type"] == "metadata_sidecar"]
            self.assertEqual(len(sidecars), 1)

            sidecar_path = Path(sidecars[0]["path"])
            self.assertTrue(sidecar_path.exists())
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["normalized_metadata"]["title"], "Test Title")
            self.assertEqual(payload["normalized_metadata"]["people"], ["Alex"])

    def test_build_recovers_legacy_zip_locator_from_current_source_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed_dir = root / "managed"
            imports_dir = root / "Imports" / "Apple takeout"
            imports_dir.mkdir(parents=True)
            zip_path = imports_dir / "iCloud Photos Part 6 of 11.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("iCloud Photos Part 6 of 11/Photos/IMG_0001.HEIC", b"fake-heic")

            manifest.init_db(db_path)
            manifest.upsert_raw(
                [
                    {
                        "source": "apple",
                        "abs_zip": r"F:\MSp\Camera\imports\Apple takeout\iCloud Photos Part 6 of 11.zip",
                        "zip_path": "iCloud Photos Part 6 of 11/Photos/IMG_0001.HEIC",
                        "source_kind": "zip",
                        "source_root": r"F:\MSp\Camera\imports",
                        "source_locator": r"F:\MSp\Camera\imports\Apple takeout\iCloud Photos Part 6 of 11.zip",
                        "source_path": "iCloud Photos Part 6 of 11/Photos/IMG_0001.HEIC",
                        "media_type": "image",
                        "orig_filename": "IMG_0001.HEIC",
                        "orig_ext": ".heic",
                        "orig_size": 8,
                        "dt_original": "2024-02-03T04:05:06",
                        "src_mtime": "2024-02-03T04:05:06",
                        "source_dt": "apple_csv",
                    }
                ],
                db_path,
            )
            manifest.plan_targets(db_path)

            processed = master.build(managed_dir, db_path, source_roots=[root / "Imports"])

            self.assertEqual(processed, 1)
            asset = manifest.get_asset(db_path, manifest.list_assets(db_path, limit=1)[0]["id"])
            self.assertEqual(asset["status"], "COPIED")
            self.assertEqual(asset["source_locator"], str(zip_path))


if __name__ == "__main__":
    unittest.main()
