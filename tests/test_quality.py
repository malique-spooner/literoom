from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from photo_unifier.metadata_repair import repair_metadata
from photo_unifier.metadata.gather import _parse_google
from photo_unifier import dedupe
from photo_unifier.metadata import manifest
from photo_unifier.utils.location import parse_location_candidate


class QualityTests(unittest.TestCase):
    def test_blur_flag_is_recorded_for_soft_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            managed.mkdir()
            manifest.init_db(db_path)

            source = root / "soft.jpg"
            Image.new("RGB", (1200, 1200), color="white").save(source)
            manifest.upsert_raw(
                [
                    {
                        "source": "local",
                        "abs_zip": str(source),
                        "zip_path": source.name,
                        "source_kind": "file",
                        "source_locator": str(source),
                        "source_path": source.name,
                        "media_type": "image",
                        "orig_filename": source.name,
                        "orig_ext": ".jpg",
                        "orig_size": source.stat().st_size,
                        "dt_original": "2024-01-01T12:00:00",
                        "src_mtime": "2024-01-01T12:00:00",
                        "source_dt": "image_exif",
                    }
                ],
                db_path,
            )
            manifest.plan_targets(db_path)
            asset = manifest.list_assets(db_path, limit=1)[0]
            managed_path = managed / manifest.get_asset(db_path, asset["id"])["managed_path"]
            managed_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (1200, 1200), color="white").save(managed_path)
            manifest.mark_copied(db_path, asset["id"], "abc123")

            result = repair_metadata(db_path, managed)

            self.assertGreaterEqual(result["examined"], 1)
            payload = manifest.export_asset_metadata_payload(db_path, asset["id"])
            self.assertTrue(payload["normalized_metadata"]["is_blurry"])
            self.assertIn("blur_score", payload["normalized_metadata"])

    def test_near_duplicate_detection_groups_similar_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            managed.mkdir()
            manifest.init_db(db_path)

            rows = []
            for idx in range(2):
                source = root / f"image-{idx}.jpg"
                img = Image.new("RGB", (800, 800), color="white")
                draw = ImageDraw.Draw(img)
                draw.rectangle([40 + idx * 2, 40, 240 + idx * 2, 240], fill="black")
                img.save(source)
                rows.append(
                    {
                        "source": "local",
                        "abs_zip": str(source),
                        "zip_path": source.name,
                        "source_kind": "file",
                        "source_locator": str(source),
                        "source_path": source.name,
                        "media_type": "image",
                        "orig_filename": source.name,
                        "orig_ext": ".jpg",
                        "orig_size": source.stat().st_size,
                        "dt_original": "2024-01-01T12:00:00",
                        "src_mtime": "2024-01-01T12:00:00",
                    }
                )
            manifest.upsert_raw(rows, db_path)
            manifest.plan_targets(db_path)
            assets = list(manifest.iter_for_master(db_path))
            for row in assets:
                path = managed / row["managed_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                source = root / row["orig_filename"]
                path.write_bytes(source.read_bytes())
                manifest.mark_copied(db_path, row["id"], "abc123")

            result = dedupe.run_near(db_path, managed, max_distance=10)
            groups = manifest.list_duplicate_groups(db_path)

            self.assertEqual(result["groups"], 1)
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]["group_type"], "NEAR_VISUAL")

    def test_iso6709_location_strings_are_parsed(self):
        payload = parse_location_candidate("+51.5000-0.1200+35.5/")
        self.assertIsNotNone(payload)
        self.assertAlmostEqual(payload["lat"], 51.5)
        self.assertAlmostEqual(payload["lon"], -0.12)
        self.assertAlmostEqual(payload["alt"], 35.5)

    def test_google_e7_location_payload_is_parsed(self):
        payload = parse_location_candidate(
            {
                "latitudeE7": 515000000,
                "longitudeE7": -1200000,
                "altitude": 35,
            }
        )
        self.assertIsNotNone(payload)
        self.assertAlmostEqual(payload["lat"], 51.5)
        self.assertAlmostEqual(payload["lon"], -0.12)
        self.assertAlmostEqual(payload["alt"], 35.0)

    def test_nested_google_metadata_payload_is_parsed(self):
        payload = _parse_google(
            {
                "mediaMetadata": {
                    "photoTakenTime": {"timestamp": {"seconds": "1712445696"}},
                    "geoDataExif": {
                        "latitudeE7": 515000000,
                        "longitudeE7": -1200000,
                        "altitude": 41,
                    },
                },
                "title": {"text": "Mountain hike"},
                "description": {"caption": "summit view"},
                "recognizedFaces": [{"name": "Alice"}, {"label": "Bob"}],
            }
        )
        self.assertEqual(payload.ts_utc, 1712445696)
        self.assertEqual(payload.title, "Mountain hike")
        self.assertEqual(payload.description, "summit view")
        self.assertEqual(payload.people, ["Alice", "Bob"])
        self.assertAlmostEqual(payload.gps_lat, 51.5)
        self.assertAlmostEqual(payload.gps_lon, -0.12)
        self.assertAlmostEqual(payload.gps_alt, 41.0)


if __name__ == "__main__":
    unittest.main()
