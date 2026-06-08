from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from literoom.metadata import gather, manifest
from literoom.pipeline import run_ingest


class GatherTests(unittest.TestCase):
    def test_ingest_local_directory_indexes_media_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_dir = root / "media"
            media_dir.mkdir()
            (media_dir / "IMG_20240102_030405.jpg").write_bytes(b"fake-jpeg")
            (media_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
            db_path = root / "manifest.sqlite"

            rows = gather.run([media_dir], db_path=db_path, batch_size=10)
            overview = manifest.get_overview(db_path)

            self.assertEqual(rows, 1)
            self.assertEqual(overview["assets_total"], 1)

    def test_smoke_ingest_samples_recent_media_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_dir = root / "media"
            media_dir.mkdir()
            older = media_dir / "older.jpg"
            middle = media_dir / "middle.mp4"
            newest = media_dir / "newest.jpg"
            older.write_bytes(b"older")
            middle.write_bytes(b"middle")
            newest.write_bytes(b"newest")
            os.utime(older, (1700000000, 1700000000))
            os.utime(middle, (1800000000, 1800000000))
            os.utime(newest, (1900000000, 1900000000))
            db_path = root / "manifest.sqlite"

            rows = gather.run([media_dir], db_path=db_path, batch_size=2, limit=2, sample_recent=True)
            assets = manifest.list_assets(db_path, limit=10, include_hidden=True)

            self.assertEqual(rows, 2)
            self.assertEqual([row["orig_filename"] for row in assets], ["newest.jpg", "middle.mp4"])

    def test_ingest_reports_missing_and_malformed_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_dir = root / "media"
            media_dir.mkdir()
            (media_dir / "IMG_20240102_030405.jpg").write_bytes(b"fake-jpeg")
            (media_dir / "broken.zip").write_bytes(b"not-a-real-zip")
            missing_dir = root / "missing"
            db_path = root / "manifest.sqlite"
            job_id = manifest.create_job(db_path, "ingest", {"sources": [str(media_dir), str(missing_dir)]})
            manifest.start_job(db_path, job_id)

            rows = gather.run([media_dir, missing_dir], db_path=db_path, batch_size=1, job_id=job_id)
            jobs = manifest.list_jobs(db_path, limit=10)
            job = next(row for row in jobs if row["id"] == job_id)

            self.assertEqual(rows, 1)
            self.assertEqual(job["status"], "RUNNING")
            metrics = job["metrics_json"]
            self.assertIsNotNone(metrics)
            self.assertIn("missing_sources", metrics)
            self.assertIn("malformed_sources", metrics)
            self.assertIn("source_issues", metrics)
            self.assertIn("broken.zip", metrics)

    def test_run_ingest_surfaces_source_reporting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_dir = root / "media"
            media_dir.mkdir()
            (media_dir / "IMG_20240102_030405.jpg").write_bytes(b"fake-jpeg")
            (media_dir / "broken.zip").write_bytes(b"not-a-real-zip")
            missing_dir = root / "missing"
            config_path = root / "literoom.local.yaml"

            result = run_ingest(config_path, sources=[media_dir, missing_dir])

            self.assertEqual(result["rows_upserted"], 1)
            self.assertEqual(result["missing_sources"], 1)
            self.assertGreaterEqual(result["malformed_sources"], 1)
            self.assertTrue(result["source_issues"])
            self.assertTrue(any("broken.zip" in issue["path"] for issue in result["source_issues"]))

    def test_ingest_insta360_directory_indexes_insv_and_ignores_lrv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_dir = root / "imports"
            clip_dir = media_dir / "random" / "nested" / "path" / "DCIM" / "Camera01"
            clip_dir.mkdir(parents=True)
            (clip_dir / "VID_20250701_145851_00_001.insv").write_bytes(b"fake-insv")
            (clip_dir / "LRV_20250701_145851_01_001.lrv").write_bytes(b"fake-lrv")
            (clip_dir / "IMG_20250701_145851_00_001.JPG").write_bytes(b"fake-jpg")
            (media_dir / "random" / "nested" / "path" / "DCIM" / "fileinfo_list.list").parent.mkdir(parents=True, exist_ok=True)
            (media_dir / "random" / "nested" / "path" / "DCIM" / "fileinfo_list.list").write_text("ignored", encoding="utf-8")
            db_path = root / "manifest.sqlite"

            rows = gather.run([media_dir], db_path=db_path, batch_size=10)
            assets = manifest.list_assets(db_path, limit=10, include_hidden=True)
            asset_by_name = {row["orig_filename"]: row for row in assets}
            insv_asset = manifest.get_asset(db_path, asset_by_name["VID_20250701_145851_00_001.insv"]["id"])

            self.assertEqual(rows, 2)
            self.assertEqual(len(assets), 2)
            self.assertEqual(asset_by_name["VID_20250701_145851_00_001.insv"]["source"], "insta360")
            self.assertEqual(asset_by_name["VID_20250701_145851_00_001.insv"]["media_type"], "video")
            self.assertEqual(insv_asset["orig_ext"], ".insv")
            self.assertEqual(asset_by_name["IMG_20250701_145851_00_001.JPG"]["source"], "insta360")
            self.assertEqual(asset_by_name["IMG_20250701_145851_00_001.JPG"]["media_type"], "image")

    def test_ingest_insta360_sidecar_metadata_applies_to_media_anywhere_under_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            imports_dir = root / "imports"
            clip_dir = imports_dir / "mixed" / "nest" / "DCIM" / "Camera01"
            clip_dir.mkdir(parents=True)
            (clip_dir / "clip.insv").write_bytes(b"fake-insv")
            (clip_dir / "still.jpg").write_bytes(b"fake-jpg")
            sidecar = imports_dir / "mixed" / "nest" / "DCIM" / "fileinfo_list.list"
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_bytes(b"fake-proto")
            db_path = root / "manifest.sqlite"
            sidecar_root = imports_dir / "mixed" / "nest"
            sidecar_index = {
                sidecar_root: {
                    "dcim/camera01/clip.insv": {
                        "dt_original": datetime(2025, 7, 1, 14, 58, 51),
                        "source_dt": "insta360_proto",
                    },
                    "dcim/camera01/still.jpg": {
                        "dt_original": datetime(2025, 7, 1, 14, 58, 51),
                        "source_dt": "insta360_proto",
                    },
                }
            }

            with patch.object(gather, "_build_insta360_index_from_root", return_value=sidecar_index):
                rows = gather.run([imports_dir], db_path=db_path, batch_size=10)

            assets = manifest.list_assets(db_path, limit=10, include_hidden=True)
            asset_by_name = {row["orig_filename"]: row for row in assets}
            clip_asset = manifest.get_asset(db_path, asset_by_name["clip.insv"]["id"])
            still_asset = manifest.get_asset(db_path, asset_by_name["still.jpg"]["id"])

            self.assertEqual(rows, 2)
            self.assertEqual(len(assets), 2)
            self.assertEqual(asset_by_name["clip.insv"]["source"], "insta360")
            self.assertEqual(clip_asset["source_dt"], "insta360_proto")
            self.assertEqual(clip_asset["dt_original"], "2025-07-01T14:58:51")
            self.assertEqual(asset_by_name["still.jpg"]["source"], "insta360")
            self.assertEqual(still_asset["source_dt"], "insta360_proto")
            self.assertEqual(still_asset["dt_original"], "2025-07-01T14:58:51")


if __name__ == "__main__":
    unittest.main()
