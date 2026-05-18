from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from photo_unifier.metadata_repair import repair_metadata
from photo_unifier.metadata import manifest


class MetadataRepairTests(unittest.TestCase):
    def test_repair_metadata_updates_weak_time_and_missing_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed_dir = root / "managed"
            managed_dir.mkdir()
            manifest.init_db(db_path)

            manifest.upsert_raw(
                [
                    {
                        "source": "local",
                        "abs_zip": str(root / "IMG_0001.JPG"),
                        "zip_path": "IMG_0001.JPG",
                        "source_kind": "file",
                        "source_locator": str(root / "IMG_0001.JPG"),
                        "source_path": str(root / "IMG_0001.JPG"),
                        "media_type": "image",
                        "orig_filename": "IMG_0001.JPG",
                        "orig_ext": ".jpg",
                        "orig_size": 100,
                        "dt_original": "2024-01-01T00:00:00",
                        "src_mtime": "2024-01-01T00:00:00",
                        "source_dt": "file_mtime",
                    }
                ],
                db_path,
            )
            manifest.plan_targets(db_path)
            asset = manifest.list_assets(db_path, limit=1)[0]
            managed_path = managed_dir / "2024" / "2024-01" / f"20240101_000000_{asset['id'][:8]}.jpg"
            managed_path.parent.mkdir(parents=True, exist_ok=True)
            managed_path.write_bytes(b"image")
            manifest.mark_copied(db_path, asset["id"], "abc123")

            with patch(
                "photo_unifier.metadata_repair.read_core_metadata_batch",
                return_value={
                    str(managed_path.resolve()): {
                        "dt_original": "2024-01-01T12:34:56",
                        "dt_source": "DateTimeOriginal",
                        "gps_lat": 51.5,
                        "gps_lon": -0.12,
                        "gps_alt": 35.0,
                        "gps_source": "GPS",
                    }
                },
            ):
                result = repair_metadata(db_path, managed_dir)

            self.assertGreaterEqual(result["repaired"], 1)
            payload = manifest.export_asset_metadata_payload(db_path, asset["id"])
            self.assertEqual(payload["normalized_metadata"]["captured_at"], "2024-01-01T12:34:56")
            self.assertEqual(payload["normalized_metadata"]["location"]["lat"], 51.5)

    def test_repair_metadata_uses_apple_photo_details_csv_and_detects_screenshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "imports" / "iphone take out"
            source_root.mkdir(parents=True)
            db_path = root / "manifest.sqlite"
            managed_dir = root / "managed"
            managed_dir.mkdir()
            manifest.init_db(db_path)

            source_file = source_root / "IMG_5831.PNG"
            source_file.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR"
                b"\x00\x00\x04\xb0\x00\x00\t`\x08\x02\x00\x00\x00"
                b"\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            csv_path = source_root / "Photo Details-1.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "imgName",
                        "fileChecksum",
                        "favorite",
                        "hidden",
                        "deleted",
                        "originalCreationDate",
                        "viewCount",
                        "importDate",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "imgName": "IMG_5831.PNG",
                        "fileChecksum": "abc",
                        "favorite": "no",
                        "hidden": "no",
                        "deleted": "no",
                        "originalCreationDate": "Tuesday May 13,2025 10:46 PM GMT",
                        "viewCount": "0",
                        "importDate": "Wednesday May 14,2025 1:00 AM GMT",
                    }
                )

            manifest.upsert_raw(
                [
                    {
                        "source": "local",
                        "abs_zip": str(source_file),
                        "zip_path": "iphone take out/IMG_5831.PNG",
                        "source_kind": "file",
                        "source_root": str(root / "imports"),
                        "source_locator": str(source_file),
                        "source_path": "iphone take out/IMG_5831.PNG",
                        "media_type": "image",
                        "orig_filename": "IMG_5831.PNG",
                        "orig_ext": ".png",
                        "orig_size": source_file.stat().st_size,
                        "dt_original": None,
                        "src_mtime": "2024-01-01T00:00:00",
                        "source_dt": "file_mtime",
                    }
                ],
                db_path,
            )
            manifest.plan_targets(db_path)
            asset = manifest.list_assets(db_path, limit=1)[0]
            managed_path = managed_dir / manifest.get_asset(db_path, asset["id"])["managed_path"]
            managed_path.parent.mkdir(parents=True, exist_ok=True)
            managed_path.write_bytes(source_file.read_bytes())
            manifest.mark_copied(db_path, asset["id"], "abc123")

            result = repair_metadata(db_path, managed_dir, source_roots=[root / "imports"])

            self.assertGreaterEqual(result["repaired"], 1)
            payload = manifest.export_asset_metadata_payload(db_path, asset["id"])
            self.assertEqual(payload["normalized_metadata"]["captured_at"], "2025-05-13T22:46:00")
            self.assertEqual(payload["normalized_metadata"]["captured_at_utc"], "2025-05-13T22:46:00+00:00")
            self.assertTrue(payload["normalized_metadata"]["is_screenshot"])

    def test_repair_metadata_recovers_sidecar_text_keywords_device_and_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "imports"
            source_root.mkdir(parents=True)
            db_path = root / "manifest.sqlite"
            managed_dir = root / "managed"
            managed_dir.mkdir()
            manifest.init_db(db_path)

            source_file = source_root / "IMG_2001.HEIC"
            source_file.write_bytes(b"heic")
            sidecar_path = source_root / "IMG_2001.HEIC.json"
            sidecar_path.write_text(
                json.dumps(
                    {
                        "title": "Beach sunset",
                        "description": "Evening walk by the sea",
                        "keywords": ["beach", "sunset"],
                        "deviceMake": "Apple",
                        "deviceModel": "iPhone 15 Pro",
                        "photoTakenTime": {"timestamp": "1712445696"},
                        "timezone": "-0400",
                        "geoData": {"latitude": 25.7617, "longitude": -80.1918, "altitude": 3.0},
                    }
                ),
                encoding="utf-8",
            )

            manifest.upsert_raw(
                [
                    {
                        "source": "local",
                        "abs_zip": str(source_file),
                        "zip_path": "IMG_2001.HEIC",
                        "source_kind": "file",
                        "source_root": str(source_root),
                        "source_locator": str(source_file),
                        "source_path": "IMG_2001.HEIC",
                        "media_type": "image",
                        "orig_filename": "IMG_2001.HEIC",
                        "orig_ext": ".heic",
                        "orig_size": source_file.stat().st_size,
                        "dt_original": None,
                        "src_mtime": "2024-01-01T00:00:00",
                        "source_dt": "file_mtime",
                    }
                ],
                db_path,
            )
            manifest.plan_targets(db_path)
            asset = manifest.list_assets(db_path, limit=1)[0]
            managed_path = managed_dir / manifest.get_asset(db_path, asset["id"])["managed_path"]
            managed_path.parent.mkdir(parents=True, exist_ok=True)
            managed_path.write_bytes(source_file.read_bytes())
            manifest.mark_copied(db_path, asset["id"], "abc123")

            with patch("photo_unifier.metadata_repair.read_core_metadata_batch", return_value={}):
                result = repair_metadata(db_path, managed_dir, source_roots=[source_root])

            self.assertGreaterEqual(result["repaired"], 1)
            payload = manifest.export_asset_metadata_payload(db_path, asset["id"])
            normalized = payload["normalized_metadata"]
            self.assertEqual(normalized["title"], "Beach sunset")
            self.assertEqual(normalized["description"], "Evening walk by the sea")
            self.assertEqual(normalized["keywords"], ["beach", "sunset"])
            self.assertEqual(normalized["camera_make"], "Apple")
            self.assertEqual(normalized["camera_model"], "iPhone 15 Pro")
            self.assertEqual(normalized["captured_at"], "2024-04-06T23:21:36-04:00")
            self.assertEqual(normalized["captured_at_utc"], "2024-04-07T03:21:36+00:00")
            self.assertEqual(normalized["timezone_name"], "UTC-04:00")
            self.assertEqual(normalized["location"]["lat"], 25.7617)

    def test_repair_metadata_recovers_nested_google_style_sidecar_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "imports"
            source_root.mkdir(parents=True)
            db_path = root / "manifest.sqlite"
            managed_dir = root / "managed"
            managed_dir.mkdir()
            manifest.init_db(db_path)

            source_file = source_root / "IMG_3001.JPG"
            source_file.write_bytes(b"jpg")
            sidecar_path = source_root / "IMG_3001.JPG.json"
            sidecar_path.write_text(
                json.dumps(
                    {
                        "mediaMetadata": {
                            "photoTakenTime": {"timestamp": {"seconds": "1712445696"}},
                            "geoDataExif": {
                                "location": {
                                    "latitudeE7": 515000000,
                                    "longitudeE7": -1200000,
                                    "altitude": 41,
                                }
                            },
                        },
                        "title": {"text": "Mountain hike"},
                        "description": {"caption": "summit view"},
                        "recognizedFaces": [{"name": "Alice"}, {"label": "Bob"}],
                    }
                ),
                encoding="utf-8",
            )

            manifest.upsert_raw(
                [
                    {
                        "source": "local",
                        "abs_zip": str(source_file),
                        "zip_path": "IMG_3001.JPG",
                        "source_kind": "file",
                        "source_root": str(source_root),
                        "source_locator": str(source_file),
                        "source_path": "IMG_3001.JPG",
                        "media_type": "image",
                        "orig_filename": "IMG_3001.JPG",
                        "orig_ext": ".jpg",
                        "orig_size": source_file.stat().st_size,
                        "dt_original": None,
                        "src_mtime": "2024-01-01T00:00:00",
                        "source_dt": "file_mtime",
                    }
                ],
                db_path,
            )
            manifest.plan_targets(db_path)
            asset = manifest.list_assets(db_path, limit=1)[0]
            managed_path = managed_dir / manifest.get_asset(db_path, asset["id"])["managed_path"]
            managed_path.parent.mkdir(parents=True, exist_ok=True)
            managed_path.write_bytes(source_file.read_bytes())
            manifest.mark_copied(db_path, asset["id"], "abc123")

            with patch("photo_unifier.metadata_repair.read_core_metadata_batch", return_value={}):
                result = repair_metadata(db_path, managed_dir, source_roots=[source_root])

            self.assertGreaterEqual(result["repaired"], 1)
            payload = manifest.export_asset_metadata_payload(db_path, asset["id"])
            normalized = payload["normalized_metadata"]
            self.assertEqual(normalized["captured_at"], "2024-04-06T23:21:36")
            self.assertEqual(normalized["location"]["lat"], 51.5)
            self.assertEqual(normalized["title"], "Mountain hike")
            self.assertEqual(normalized["description"], "summit view")
            self.assertEqual(normalized["people"], ["Alice", "Bob"])


if __name__ == "__main__":
    unittest.main()
