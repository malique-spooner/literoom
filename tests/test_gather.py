from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from photo_unifier.metadata import gather, manifest


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
