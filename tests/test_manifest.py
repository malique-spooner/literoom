from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from photo_unifier.metadata import manifest


class ManifestTests(unittest.TestCase):
    def test_init_db_creates_foundation_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "manifest.sqlite"
            manifest.init_db(db_path)

            con = sqlite3.connect(db_path)
            try:
                tables = {
                    row[0]
                    for row in con.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
            finally:
                con.close()

            expected = {
                "assets",
                "jobs",
                "managed_assets",
                "hashes",
                "metadata_fields",
                "artifacts",
                "thumbnails",
                "duplicate_groups",
                "duplicate_items",
                "faces",
                "face_identities",
                "embeddings",
                "extraction_results",
                "face_clarifications",
                "review_decisions",
                "mutation_history",
            }
            self.assertTrue(expected.issubset(tables))

    def test_job_lifecycle_and_overview(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "manifest.sqlite"
            manifest.init_db(db_path)

            job_id = manifest.create_job(db_path, "ingest", {"source_count": 1})
            manifest.start_job(db_path, job_id)
            manifest.complete_job(db_path, job_id, {"rows_upserted": 4})

            jobs = manifest.list_jobs(db_path)
            self.assertEqual(jobs[0]["status"], "COMPLETED")
            overview = manifest.get_overview(db_path)
            self.assertEqual(overview["jobs"], 1)

    def test_upsert_raw_creates_normalized_metadata_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "manifest.sqlite"
            manifest.init_db(db_path)

            manifest.upsert_raw(
                [
                    {
                        "source": "google",
                        "abs_zip": "/tmp/takeout.zip",
                        "zip_path": "Takeout/Photos/IMG_0001.JPG",
                        "source_kind": "zip",
                        "source_locator": "/tmp/takeout.zip",
                        "source_path": "Takeout/Photos/IMG_0001.JPG",
                        "media_type": "image",
                        "orig_filename": "IMG_0001.JPG",
                        "orig_ext": ".jpg",
                        "orig_size": 1234,
                        "dt_original": "2024-02-03T04:05:06",
                        "gps_lat": 51.5,
                        "gps_lon": -0.12,
                        "gps_alt": 35.0,
                        "title": "Night out",
                        "description": "Rooftop drinks",
                        "keywords": ["london", "friends"],
                        "people": ["Ethan"],
                        "src_mtime": "2024-02-04T05:06:07",
                        "source_dt": "google_epoch",
                        "source_gps": "google_json",
                    }
                ],
                db_path,
            )

            asset = manifest.list_assets(db_path, limit=1)[0]
            payload = manifest.export_asset_metadata_payload(db_path, asset["id"])

            self.assertEqual(payload["normalized_metadata"]["captured_at"], "2024-02-03T04:05:06")
            self.assertEqual(payload["normalized_metadata"]["title"], "Night out")
            self.assertEqual(payload["normalized_metadata"]["people"], ["Ethan"])
            self.assertEqual(payload["normalized_metadata"]["location"]["lat"], 51.5)
            self.assertTrue(
                any(
                    item["field"] == "captured_at" and item["source_field"] == "google_epoch"
                    for item in payload["metadata_provenance"]
                )
            )

    def test_backfill_missing_metadata_populates_existing_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "manifest.sqlite"
            manifest.init_db(db_path)

            con = sqlite3.connect(db_path)
            try:
                con.execute(
                    """
                    INSERT INTO assets (
                      id, source, abs_zip, zip_path, source_kind, source_locator, source_path,
                      media_type, orig_filename, orig_ext, orig_size, dt_original,
                      title, description, keywords_json, people_json, status, src_mtime,
                      source_dt, source_gps, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "asset123",
                        "apple",
                        "/tmp/archive.zip",
                        "Photos/IMG_0001.HEIC",
                        "zip",
                        "/tmp/archive.zip",
                        "Photos/IMG_0001.HEIC",
                        "image",
                        "IMG_0001.HEIC",
                        ".heic",
                        555,
                        "2024-01-01T12:00:00",
                        "Lunch",
                        "Family lunch",
                        '["family"]',
                        '["Mum"]',
                        "NEW",
                        "2024-01-01T12:00:00",
                        "apple_csv",
                        None,
                        "2024-01-01T12:00:00",
                    ),
                )
                con.commit()
            finally:
                con.close()

            count = manifest.backfill_missing_metadata(db_path)
            self.assertEqual(count, 1)

            payload = manifest.export_asset_metadata_payload(db_path, "asset123")
            self.assertEqual(payload["normalized_metadata"]["title"], "Lunch")
            self.assertEqual(payload["normalized_metadata"]["keywords"], ["family"])

    def test_metadata_audit_scores_and_missing_lists_respect_media_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "manifest.sqlite"
            manifest.init_db(db_path)

            manifest.upsert_raw(
                [
                    {
                        "source": "local",
                        "abs_zip": "/tmp/video.mov",
                        "zip_path": "video.mov",
                        "source_kind": "file",
                        "source_locator": "/tmp/video.mov",
                        "source_path": "video.mov",
                        "media_type": "video",
                        "orig_filename": "video.mov",
                        "orig_ext": ".mov",
                        "orig_size": 100,
                        "dt_original": "2024-02-03T04:05:06",
                        "src_mtime": "2024-02-03T04:05:06",
                        "source_dt": "createdate",
                    },
                    {
                        "source": "local",
                        "abs_zip": "/tmp/image.heic",
                        "zip_path": "image.heic",
                        "source_kind": "file",
                        "source_locator": "/tmp/image.heic",
                        "source_path": "image.heic",
                        "media_type": "image",
                        "orig_filename": "image.heic",
                        "orig_ext": ".heic",
                        "orig_size": 100,
                        "dt_original": "2024-02-03T04:05:06",
                        "src_mtime": "2024-02-03T04:05:06",
                        "source_dt": "createdate",
                    },
                ],
                db_path,
            )
            assets = manifest.list_assets(db_path, limit=10)
            ids = {asset["orig_filename"]: asset["id"] for asset in assets}
            manifest.set_metadata_field(
                db_path,
                ids["image.heic"],
                field_name="lens_model",
                value="iPhone 15 Pro back triple camera 6.86mm f/1.78",
                source_name="embedded_metadata",
                source_field="LensModel",
                is_canonical=True,
            )

            image_audit = manifest.asset_metadata_audit(db_path, ids["image.heic"])
            self.assertIn("lens_model", image_audit["present_fields"])

            overview = manifest.get_metadata_audit_overview(db_path)
            lens_summary = next(item for item in overview["field_summaries"] if item["field_name"] == "lens_model")
            self.assertEqual(lens_summary["applicable_assets"], 1)
            self.assertEqual(lens_summary["missing_assets"], 0)

            missing_lens = manifest.list_assets_missing_metadata(db_path, "lens_model", limit=10)
            self.assertEqual(missing_lens, [])

    def test_weaker_metadata_updates_do_not_replace_stronger_canonical_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "manifest.sqlite"
            manifest.init_db(db_path)

            manifest.upsert_raw(
                [
                    {
                        "source": "local",
                        "abs_zip": "/tmp/image.jpg",
                        "zip_path": "image.jpg",
                        "source_kind": "file",
                        "source_locator": "/tmp/image.jpg",
                        "source_path": "image.jpg",
                        "media_type": "image",
                        "orig_filename": "image.jpg",
                        "orig_ext": ".jpg",
                        "orig_size": 100,
                        "dt_original": "2024-01-01T10:00:00",
                        "src_mtime": "2024-01-01T10:00:00",
                        "source_dt": "image_exif",
                    }
                ],
                db_path,
            )
            asset = manifest.list_assets(db_path, limit=1)[0]
            manifest.apply_metadata_updates(
                db_path,
                asset["id"],
                dt_original="2024-01-01T10:00:00",
                dt_source="image_exif",
                source_name="metadata_repair",
                confidence=0.96,
            )
            manifest.apply_metadata_updates(
                db_path,
                asset["id"],
                dt_original="2024-01-01T09:00:00",
                dt_source="filename",
                source_name="metadata_repair",
                confidence=0.40,
            )

            refreshed = manifest.get_asset(db_path, asset["id"])
            self.assertEqual(refreshed["dt_original"], "2024-01-01T10:00:00")
            payload = manifest.export_asset_metadata_payload(db_path, asset["id"])
            self.assertEqual(payload["normalized_metadata"]["captured_at"], "2024-01-01T10:00:00")

    def test_set_asset_review_persists_rating_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "manifest.sqlite"
            manifest.init_db(db_path)

            manifest.upsert_raw(
                [
                    {
                        "source": "local",
                        "abs_zip": "/tmp/image.jpg",
                        "zip_path": "image.jpg",
                        "source_kind": "file",
                        "source_locator": "/tmp/image.jpg",
                        "source_path": "image.jpg",
                        "media_type": "image",
                        "orig_filename": "image.jpg",
                        "orig_ext": ".jpg",
                        "orig_size": 100,
                        "dt_original": "2024-01-01T10:00:00",
                        "src_mtime": "2024-01-01T10:00:00",
                    }
                ],
                db_path,
            )
            asset = manifest.list_assets(db_path, limit=1)[0]

            manifest.set_asset_review(
                db_path,
                asset["id"],
                user_rating=5,
                review_state="reviewed",
                review_score=0.88,
                favorite=True,
            )

            updated = manifest.get_asset(db_path, asset["id"])
            provenance = manifest.list_asset_metadata(db_path, asset["id"])

            self.assertEqual(updated["user_rating"], 5)
            self.assertEqual(updated["review_state"], "reviewed")
            self.assertEqual(updated["favorite"], 1)
            self.assertAlmostEqual(float(updated["review_score"]), 0.88, places=2)
            self.assertTrue(any(row["field_name"] == "user_rating" for row in provenance))
            self.assertTrue(any(row["field_name"] == "favorite" for row in provenance))

            history = manifest.get_mutation_history(db_path, limit=4)
            self.assertEqual(history[0]["action_type"], "SET_ASSET_REVIEW")
            status = manifest.get_mutation_status(db_path)
            self.assertTrue(status["can_undo"])
            self.assertFalse(status["can_redo"])

            manifest.undo_last_mutation(db_path)
            reverted = manifest.get_asset(db_path, asset["id"])
            self.assertIsNone(reverted["user_rating"])
            self.assertIsNone(reverted["review_state"])
            self.assertEqual(int(reverted["favorite"] or 0), 0)
            self.assertFalse(manifest.get_mutation_status(db_path)["can_undo"])
            self.assertTrue(manifest.get_mutation_status(db_path)["can_redo"])

            manifest.redo_last_mutation(db_path)
            redone = manifest.get_asset(db_path, asset["id"])
            self.assertEqual(redone["user_rating"], 5)
            self.assertEqual(redone["review_state"], "reviewed")
            self.assertEqual(redone["favorite"], 1)

    def test_undo_redo_restores_person_tag_and_duplicate_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "manifest.sqlite"
            manifest.init_db(db_path)

            manifest.upsert_raw(
                [
                    {
                        "source": "local",
                        "abs_zip": "/tmp/a.jpg",
                        "zip_path": "a.jpg",
                        "source_kind": "file",
                        "source_locator": "/tmp/a.jpg",
                        "source_path": "a.jpg",
                        "media_type": "image",
                        "orig_filename": "a.jpg",
                        "orig_ext": ".jpg",
                        "orig_size": 100,
                        "dt_original": "2024-01-01T10:00:00",
                        "src_mtime": "2024-01-01T10:00:00",
                    },
                    {
                        "source": "local",
                        "abs_zip": "/tmp/b.jpg",
                        "zip_path": "b.jpg",
                        "source_kind": "file",
                        "source_locator": "/tmp/b.jpg",
                        "source_path": "b.jpg",
                        "media_type": "image",
                        "orig_filename": "b.jpg",
                        "orig_ext": ".jpg",
                        "orig_size": 100,
                        "dt_original": "2024-01-02T10:00:00",
                        "src_mtime": "2024-01-02T10:00:00",
                    },
                ],
                db_path,
            )
            assets = {row["orig_filename"]: row for row in manifest.list_assets(db_path, limit=10)}
            a_id = assets["a.jpg"]["id"]
            b_id = assets["b.jpg"]["id"]

            manifest.add_asset_person(db_path, a_id, "Avery")
            tagged = manifest.get_asset(db_path, a_id)
            self.assertIn("Avery", tagged["people_json"])
            manifest.remove_asset_person(db_path, a_id, "Avery")
            removed = manifest.get_asset(db_path, a_id)
            self.assertNotIn("Avery", removed["people_json"])
            manifest.undo_last_mutation(db_path)
            untagged = manifest.get_asset(db_path, a_id)
            self.assertIn("Avery", untagged["people_json"])
            manifest.redo_last_mutation(db_path)
            retagged = manifest.get_asset(db_path, a_id)
            self.assertNotIn("Avery", retagged["people_json"])

            manifest.replace_duplicate_groups(
                db_path,
                group_type="EXACT_SHA256",
                groups=[
                    {
                        "canonical_asset_id": a_id,
                        "items": [
                            {"asset_id": a_id, "score": 1.0, "rationale": "seed"},
                            {"asset_id": b_id, "score": 1.0, "rationale": "seed"},
                        ],
                    }
                ],
            )
            group_id = manifest.list_duplicate_groups(db_path, group_type="EXACT_SHA256")[0]["id"]
            manifest.resolve_duplicate_group(db_path, group_id, a_id)
            group = manifest.list_duplicate_groups(db_path, group_type="EXACT_SHA256")[0]
            self.assertEqual(group["status"], "RESOLVED")
            manifest.undo_last_mutation(db_path)
            group = manifest.list_duplicate_groups(db_path, group_type="EXACT_SHA256")[0]
            self.assertEqual(group["status"], "OPEN")
            manifest.redo_last_mutation(db_path)
            group = manifest.list_duplicate_groups(db_path, group_type="EXACT_SHA256")[0]
            self.assertEqual(group["status"], "RESOLVED")

    def test_upsert_raw_is_idempotent_for_same_source_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "manifest.sqlite"
            manifest.init_db(db_path)

            row = {
                "source": "local",
                "abs_zip": "/tmp/image.jpg",
                "zip_path": "image.jpg",
                "source_kind": "file",
                "source_locator": "/tmp/image.jpg",
                "source_path": "image.jpg",
                "media_type": "image",
                "orig_filename": "image.jpg",
                "orig_ext": ".jpg",
                "orig_size": 100,
                "dt_original": "2024-01-01T10:00:00",
                "src_mtime": "2024-01-01T10:00:00",
            }

            first = manifest.upsert_raw([row], db_path)
            second = manifest.upsert_raw([row], db_path)
            assets = manifest.list_assets(db_path, limit=10, include_hidden=True)

            self.assertEqual(first, 1)
            self.assertEqual(second, 1)
            self.assertEqual(len(assets), 1)

    def test_pipeline_health_reports_missing_managed_and_preview_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "manifest.sqlite"
            manifest.init_db(db_path)

            manifest.upsert_raw(
                [
                    {
                        "source": "local",
                        "abs_zip": "/tmp/image.jpg",
                        "zip_path": "image.jpg",
                        "source_kind": "file",
                        "source_locator": "/tmp/image.jpg",
                        "source_path": "image.jpg",
                        "media_type": "image",
                        "orig_filename": "image.jpg",
                        "orig_ext": ".jpg",
                        "orig_size": 100,
                        "dt_original": "2024-01-01T10:00:00",
                        "src_mtime": "2024-01-01T10:00:00",
                    }
                ],
                db_path,
            )
            manifest.plan_targets(db_path)

            health = manifest.get_pipeline_health(db_path)
            self.assertFalse(health["healthy"])
            self.assertEqual(health["missing_managed_assets"], 1)
            self.assertEqual(health["missing_previews"], 1)

    def test_people_albums_duplicate_threshold_and_import_progress_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "managed"
            managed.mkdir()
            manifest.init_db(db_path)

            manifest.upsert_raw(
                [
                    {
                        "source": "local",
                        "abs_zip": "/tmp/a.jpg",
                        "zip_path": "a.jpg",
                        "source_kind": "file",
                        "source_locator": "/tmp/a.jpg",
                        "source_path": "a.jpg",
                        "media_type": "image",
                        "orig_filename": "a.jpg",
                        "orig_ext": ".jpg",
                        "orig_size": 100,
                        "dt_original": "2024-01-01T10:00:00",
                        "src_mtime": "2024-01-01T10:00:00",
                    }
                ],
                db_path,
            )
            manifest.plan_targets(db_path)
            asset = manifest.list_assets(db_path, limit=1)[0]
            managed_path = managed / asset["managed_path"]
            managed_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (120, 120), color="white").save(managed_path)
            manifest.mark_copied(db_path, asset["id"], "abc123")
            manifest.record_thumbnail(db_path, asset["id"], "primary", str(managed_path), "READY", width=120, height=120)

            identity_id = manifest.create_face_identity(db_path, "Avery")
            manifest.replace_faces_for_asset(
                db_path,
                asset["id"],
                [{"id": "face-a", "bbox": {"x": 1, "y": 2, "w": 20, "h": 20}, "frame_time_ms": 0}],
            )
            manifest.assign_face_identity(db_path, "face-a", identity_id)
            manifest.replace_duplicate_groups(
                db_path,
                group_type="NEAR_AHASH",
                groups=[
                    {
                        "canonical_asset_id": asset["id"],
                        "items": [
                            {"asset_id": asset["id"], "score": 1.0, "rationale": "reference"},
                        ],
                    }
                ],
            )

            albums = manifest.list_person_albums(db_path)
            assets_for_person = manifest.list_assets_for_person(db_path, "Avery")
            groups = manifest.list_duplicate_groups(db_path, min_score=0.9, group_type="NEAR_AHASH")
            import_progress = manifest.get_import_progress(db_path)

            self.assertEqual(albums[0]["label"], "Avery")
            self.assertEqual(assets_for_person[0]["id"], asset["id"])
            self.assertEqual(len(groups), 1)
            self.assertGreater(import_progress["completion_pct"], 0)


if __name__ == "__main__":
    unittest.main()
