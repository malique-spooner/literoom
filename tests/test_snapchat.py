from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from photo_unifier.metadata import manifest
from photo_unifier.metadata.gather import scan_sources, scan_zip
from photo_unifier.metadata_repair import repair_metadata
from photo_unifier.utils.location import parse_location_candidate


class SnapchatTests(unittest.TestCase):
    def test_snapchat_location_string_parses(self):
        payload = parse_location_candidate("Latitude, Longitude: 51.445824, -0.1195754")
        self.assertIsNotNone(payload)
        self.assertAlmostEqual(payload["lat"], 51.445824)
        self.assertAlmostEqual(payload["lon"], -0.1195754)

    def test_scan_zip_recovers_snapchat_bundle_rows_and_skips_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history_zip = root / "mydata~000.zip"
            media_zip = root / "mydata~001.zip"
            with zipfile.ZipFile(history_zip, "w") as zf:
                zf.writestr(
                    "json/memories_history.json",
                    json.dumps(
                        {
                            "Saved Media": [
                                {
                                    "Date": "2026-05-09 10:24:55 UTC",
                                    "Media Type": "Video",
                                    "Location": "Latitude, Longitude: 51.445824, -0.1195754",
                                    "Download Link": "",
                                    "Media Download Url": "",
                                },
                                {
                                    "Date": "2026-05-09 02:19:31 UTC",
                                    "Media Type": "Video",
                                    "Location": "Latitude, Longitude: 51.449646, -0.12195039",
                                    "Download Link": "",
                                    "Media Download Url": "",
                                },
                            ]
                        }
                    ),
                )
                zf.writestr("html/memories_history.html", "<html></html>")
            with zipfile.ZipFile(media_zip, "w") as zf:
                zf.writestr("memories/2026-05-09_ABC-main.mp4", b"video-bytes-1")
                zf.writestr("memories/2026-05-09_DEF-main.mp4", b"video-bytes-2")
                zf.writestr("memories/2026-05-09_ABC-overlay.png", b"overlay-bytes")

            rows = list(scan_zip(media_zip, source_hint="snapchat"))
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["orig_filename"] for row in rows], ["2026-05-09_ABC-main.mp4", "2026-05-09_DEF-main.mp4"])
            self.assertEqual(rows[0]["dt_original"], "2026-05-09T10:24:55")
            self.assertEqual(rows[1]["dt_original"], "2026-05-09T02:19:31")
            self.assertAlmostEqual(rows[0]["gps_lat"], 51.445824)
            self.assertAlmostEqual(rows[1]["gps_lat"], 51.449646)

    def test_scan_directory_recurses_into_snapchat_zip_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "snapchat"
            bundle.mkdir()
            history_zip = bundle / "mydata~000.zip"
            media_zip = bundle / "mydata~001.zip"
            with zipfile.ZipFile(history_zip, "w") as zf:
                zf.writestr(
                    "json/memories_history.json",
                    json.dumps(
                        {
                            "Saved Media": [
                                {
                                    "Date": "2026-05-09 10:24:55 UTC",
                                    "Media Type": "Video",
                                    "Location": "Latitude, Longitude: 51.445824, -0.1195754",
                                },
                                {
                                    "Date": "2026-05-09 02:19:31 UTC",
                                    "Media Type": "Video",
                                    "Location": "Latitude, Longitude: 51.449646, -0.12195039",
                                },
                            ]
                        }
                    ),
                )
            with zipfile.ZipFile(media_zip, "w") as zf:
                zf.writestr("memories/2026-05-09_ABC-main.mp4", b"video-bytes-1")
                zf.writestr("memories/2026-05-09_DEF-main.mp4", b"video-bytes-2")

            rows = list(scan_sources([bundle], source_hint="snapchat"))
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["orig_filename"] for row in rows], ["2026-05-09_ABC-main.mp4", "2026-05-09_DEF-main.mp4"])
            self.assertEqual(rows[0]["dt_original"], "2026-05-09T10:24:55")
            self.assertEqual(rows[1]["dt_original"], "2026-05-09T02:19:31")
            self.assertAlmostEqual(rows[0]["gps_lat"], 51.445824)
            self.assertAlmostEqual(rows[1]["gps_lat"], 51.449646)

    def test_repair_metadata_matches_snapchat_exact_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "mydata~000.zip"
            managed_dir = root / "managed"
            managed_dir.mkdir()
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "json/memories_history.json",
                    json.dumps(
                        {
                            "Saved Media": [
                                {
                                    "Date": "2026-05-09 10:24:55 UTC",
                                    "Media Type": "Video",
                                    "Location": "Latitude, Longitude: 51.445824, -0.1195754",
                                    "Download Link": "",
                                    "Media Download Url": "",
                                }
                            ]
                        }
                    ),
                )
                zf.writestr("memories/2026-05-09_ABC-main.mp4", b"video-bytes")

            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)
            manifest.upsert_raw(
                [
                    {
                        "source": "snapchat",
                        "abs_zip": str(zip_path),
                        "zip_path": "memories/2026-05-09_ABC-main.mp4",
                        "source_kind": "zip",
                        "source_root": str(root),
                        "source_locator": str(zip_path),
                        "source_path": "memories/2026-05-09_ABC-main.mp4",
                        "media_type": "video",
                        "orig_filename": "2026-05-09_ABC-main.mp4",
                        "orig_ext": ".mp4",
                        "orig_size": 11,
                        "dt_original": None,
                        "src_mtime": "2026-05-09T10:24:55",
                        "source_dt": "snapchat_history",
                    }
                ],
                db_path,
            )
            manifest.plan_targets(db_path)
            asset = manifest.list_assets(db_path, limit=1)[0]
            managed_path = managed_dir / manifest.get_asset(db_path, asset["id"])["managed_path"]
            managed_path.parent.mkdir(parents=True, exist_ok=True)
            managed_path.write_bytes(b"video-bytes")
            manifest.mark_copied(db_path, asset["id"], "abc123")

            with patch(
                "photo_unifier.metadata_repair.read_core_metadata_batch",
                return_value={
                    str(managed_path.resolve()): {
                        "dt_original": "2026-05-09T10:24:55",
                        "dt_source": "CreateDate",
                    }
                },
            ):
                repair_metadata(db_path, managed_dir)

            payload = manifest.export_asset_metadata_payload(db_path, asset["id"])
            normalized = payload["normalized_metadata"]
            self.assertEqual(normalized["captured_at"], "2026-05-09T10:24:55")
            self.assertAlmostEqual(normalized["location"]["lat"], 51.445824)
            self.assertAlmostEqual(normalized["location"]["lon"], -0.1195754)


if __name__ == "__main__":
    unittest.main()
