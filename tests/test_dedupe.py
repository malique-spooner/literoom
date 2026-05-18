from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from photo_unifier.phase_dedupe import dedupe
from photo_unifier.phase_metadata import manifest

try:
    from fastapi.testclient import TestClient
    from photo_unifier.api import create_app

    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False
    TestClient = None  # type: ignore[assignment]
    create_app = None  # type: ignore[assignment]


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class DedupeTests(unittest.TestCase):
    @staticmethod
    def _write_config(config_path: Path, db_path: Path, managed: Path, derived: Path, root: Path):
        config_path.write_text(
            "\n".join(
                [
                    "workspace_root: .",
                    "sources: []",
                    "paths:",
                    f"  db_path: {db_path}",
                    f"  managed_library_dir: {managed}",
                    f"  derivatives_dir: {derived}",
                    f"  logs_dir: {root / 'logs'}",
                    f"  temp_dir: {root / 'tmp'}",
                ]
            ),
            encoding="utf-8",
        )

    def test_exact_dedupe_groups_matching_managed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            managed.mkdir()
            manifest.init_db(db_path)

            rows = []
            for idx in range(2):
                locator = root / f"source-{idx}.jpg"
                locator.write_bytes(b"same-bytes")
                rows.append(
                    {
                        "source": "local",
                        "abs_zip": str(locator),
                        "zip_path": locator.name,
                        "source_kind": "file",
                        "source_root": str(root),
                        "source_locator": str(locator),
                        "source_path": locator.name,
                        "media_type": "image",
                        "orig_filename": locator.name,
                        "orig_ext": ".jpg",
                        "orig_size": 10,
                        "dt_original": "2024-01-02T03:04:05",
                        "src_mtime": "2024-01-02T03:04:05",
                    }
                )

            manifest.upsert_raw(rows, db_path)
            manifest.plan_targets(db_path)

            assets = list(manifest.iter_for_master(db_path))
            for row in assets:
                path = managed / row["managed_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"same-bytes")
                manifest.mark_copied(db_path, row["id"], "abc123", warning="embed skipped")

            result = dedupe.run_exact(db_path, managed)
            groups = manifest.list_duplicate_groups(db_path)
            items = manifest.list_duplicate_group_items(db_path, groups[0]["id"])

            self.assertEqual(result["groups"], 1)
            self.assertEqual(len(groups), 1)
            self.assertEqual(len(items), 2)

    def test_duplicate_group_can_be_resolved_via_app(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            derived = root / "derived"
            managed.mkdir()
            derived.mkdir()
            manifest.init_db(db_path)

            rows = []
            for idx in range(2):
                locator = root / f"source-{idx}.jpg"
                locator.write_bytes(b"same-bytes")
                rows.append(
                    {
                        "source": "local",
                        "abs_zip": str(locator),
                        "zip_path": locator.name,
                        "source_kind": "file",
                        "source_root": str(root),
                        "source_locator": str(locator),
                        "source_path": locator.name,
                        "media_type": "image",
                        "orig_filename": locator.name,
                        "orig_ext": ".jpg",
                        "orig_size": 10 + idx,
                        "dt_original": "2024-01-02T03:04:05",
                        "src_mtime": "2024-01-02T03:04:05",
                    }
                )

            manifest.upsert_raw(rows, db_path)
            manifest.plan_targets(db_path)
            assets = list(manifest.iter_for_master(db_path))
            for row in assets:
                path = managed / row["managed_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"same-bytes")
                manifest.mark_copied(db_path, row["id"], "abc123", warning="embed skipped")
                thumb = derived / f"{row['id']}.jpg"
                thumb.write_bytes(b"thumb")
                manifest.record_thumbnail(db_path, row["id"], "primary", str(thumb), "READY")

            dedupe.run_exact(db_path, managed)
            groups = manifest.list_duplicate_groups(db_path)
            group_id = groups[0]["id"]
            items = manifest.list_duplicate_group_items(db_path, group_id)
            canonical_asset_id = items[0]["asset_id"]

            config_path = root / "config.yaml"
            self._write_config(config_path, db_path, managed, derived, root)
            client = TestClient(create_app(config_path))
            response = client.get(
                f"/app/duplicates/{group_id}/resolve?canonical_asset_id={canonical_asset_id}",
                follow_redirects=False,
            )

            self.assertEqual(response.status_code, 303)
            updated_groups = manifest.list_duplicate_groups(db_path)
            self.assertEqual(updated_groups[0]["status"], "RESOLVED")

    def test_hidden_duplicates_are_excluded_from_default_asset_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            derived = root / "derived"
            managed.mkdir()
            derived.mkdir()
            manifest.init_db(db_path)

            rows = []
            for idx in range(2):
                locator = root / f"source-{idx}.jpg"
                locator.write_bytes(b"same-bytes")
                rows.append(
                    {
                        "source": "local",
                        "abs_zip": str(locator),
                        "zip_path": locator.name,
                        "source_kind": "file",
                        "source_root": str(root),
                        "source_locator": str(locator),
                        "source_path": locator.name,
                        "media_type": "image",
                        "orig_filename": locator.name,
                        "orig_ext": ".jpg",
                        "orig_size": 10 + idx,
                        "dt_original": "2024-01-02T03:04:05",
                        "src_mtime": "2024-01-02T03:04:05",
                    }
                )
            manifest.upsert_raw(rows, db_path)
            manifest.plan_targets(db_path)
            assets = list(manifest.iter_for_master(db_path))
            for row in assets:
                path = managed / row["managed_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"same-bytes")
                manifest.mark_copied(db_path, row["id"], "abc123", warning="embed skipped")
                thumb = derived / f"{row['id']}.jpg"
                thumb.write_bytes(b"thumb")
                manifest.record_thumbnail(db_path, row["id"], "primary", str(thumb), "READY")

            dedupe.run_exact(db_path, managed)
            group = manifest.list_duplicate_groups(db_path)[0]
            items = manifest.list_duplicate_group_items(db_path, group["id"])
            manifest.resolve_duplicate_group(db_path, group["id"], items[0]["asset_id"])

            visible = manifest.list_assets(db_path)
            all_items = manifest.list_assets(db_path, include_hidden=True)
            self.assertEqual(len(visible), 1)
            self.assertEqual(len(all_items), 2)

    def test_asset_detail_page_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            derived = root / "derived"
            managed.mkdir()
            derived.mkdir()
            manifest.init_db(db_path)

            locator = root / "source.jpg"
            locator.write_bytes(b"same-bytes")
            manifest.upsert_raw(
                [
                    {
                        "source": "local",
                        "abs_zip": str(locator),
                        "zip_path": locator.name,
                        "source_kind": "file",
                        "source_root": str(root),
                        "source_locator": str(locator),
                        "source_path": locator.name,
                        "media_type": "image",
                        "orig_filename": locator.name,
                        "orig_ext": ".jpg",
                        "orig_size": 10,
                        "dt_original": "2024-01-02T03:04:05",
                        "src_mtime": "2024-01-02T03:04:05",
                    }
                ],
                db_path,
            )
            manifest.plan_targets(db_path)
            asset = list(manifest.iter_for_master(db_path))[0]
            path = managed / asset["managed_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"same-bytes")
            manifest.mark_copied(db_path, asset["id"], "abc123", warning="embed skipped")
            thumb = derived / f"{asset['id']}.jpg"
            thumb.write_bytes(b"thumb")
            manifest.record_thumbnail(db_path, asset["id"], "primary", str(thumb), "READY")

            config_path = root / "config.yaml"
            self._write_config(config_path, db_path, managed, derived, root)
            client = TestClient(create_app(config_path))
            response = client.get(f"/app/assets/{asset['id']}")

            self.assertEqual(response.status_code, 200)
            self.assertIn("Normalized Metadata", response.text)
            self.assertIn("Open File", response.text)

    def test_video_asset_detail_renders_inline_viewer_and_inline_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            derived = root / "derived"
            managed.mkdir()
            derived.mkdir()
            manifest.init_db(db_path)

            locator = root / "clip.mp4"
            locator.write_bytes(b"video-bytes")
            manifest.upsert_raw(
                [
                    {
                        "source": "local",
                        "abs_zip": str(locator),
                        "zip_path": locator.name,
                        "source_kind": "file",
                        "source_root": str(root),
                        "source_locator": str(locator),
                        "source_path": locator.name,
                        "media_type": "video",
                        "orig_filename": locator.name,
                        "orig_ext": ".mp4",
                        "orig_size": 11,
                        "dt_original": "2024-01-02T03:04:05",
                        "src_mtime": "2024-01-02T03:04:05",
                    }
                ],
                db_path,
            )
            manifest.plan_targets(db_path)
            asset = list(manifest.iter_for_master(db_path))[0]
            path = managed / asset["managed_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"video-bytes")
            manifest.mark_copied(db_path, asset["id"], "abc123", warning="embed skipped")
            thumb = derived / f"{asset['id']}.jpg"
            thumb.write_bytes(b"thumb")
            manifest.record_thumbnail(db_path, asset["id"], "video_preview", str(thumb), "READY")

            config_path = root / "config.yaml"
            self._write_config(config_path, db_path, managed, derived, root)
            client = TestClient(create_app(config_path))

            page = client.get(f"/app/assets/{asset['id']}")
            inline = client.get(f"/inline/{asset['id']}")

            self.assertEqual(page.status_code, 200)
            self.assertIn("<video controls", page.text)
            self.assertEqual(inline.status_code, 200)

    def test_settings_page_and_save_route_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            derived = root / "derived"
            managed.mkdir()
            derived.mkdir()
            manifest.init_db(db_path)

            config_path = root / "config.yaml"
            self._write_config(config_path, db_path, managed, derived, root)
            client = TestClient(create_app(config_path))

            page = client.get("/app/settings")
            self.assertEqual(page.status_code, 200)
            self.assertIn("Settings", page.text)

            response = client.get(
                "/app/settings/save",
                params={
                    "workspace_root": str(root),
                    "sources_text": "/Volumes/Extreme SSD/MSp/Camera\n/Volumes/Extreme SSD/MSp/Camera/Master",
                    "db_path_value": str(db_path),
                    "managed_library_dir_value": str(managed),
                    "derivatives_dir_value": str(derived),
                    "logs_dir_value": str(root / "logs"),
                    "temp_dir_value": str(root / "tmp"),
                    "batch_size": 123,
                    "image_thumbnail_size": 256,
                    "video_preview_offset_seconds": 2,
                    "auto_sync_interval_seconds": 120,
                    "managed_naming": "{YYYY}/{shortid}",
                    "exiftool": "/usr/local/bin/exiftool",
                    "ffmpeg": "/usr/local/bin/ffmpeg",
                },
                follow_redirects=False,
            )
            self.assertEqual(response.status_code, 303)

            from photo_unifier.config import load_config

            loaded, _ = load_config(config_path)
            self.assertEqual(loaded.sources[0], "/Volumes/Extreme SSD/MSp/Camera")
            self.assertEqual(loaded.pipeline.batch_size, 123)
            self.assertEqual(loaded.pipeline.auto_sync_interval_seconds, 120)

    def test_keep_all_route_marks_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            derived = root / "derived"
            managed.mkdir()
            derived.mkdir()
            manifest.init_db(db_path)

            rows = []
            for idx in range(2):
                locator = root / f"source-{idx}.jpg"
                locator.write_bytes(b"same-bytes")
                rows.append(
                    {
                        "source": "local",
                        "abs_zip": str(locator),
                        "zip_path": locator.name,
                        "source_kind": "file",
                        "source_root": str(root),
                        "source_locator": str(locator),
                        "source_path": locator.name,
                        "media_type": "image",
                        "orig_filename": locator.name,
                        "orig_ext": ".jpg",
                        "orig_size": 10 + idx,
                        "dt_original": "2024-01-02T03:04:05",
                        "src_mtime": "2024-01-02T03:04:05",
                    }
                )
            manifest.upsert_raw(rows, db_path)
            manifest.plan_targets(db_path)
            assets = list(manifest.iter_for_master(db_path))
            for row in assets:
                path = managed / row["managed_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"same-bytes")
                manifest.mark_copied(db_path, row["id"], "abc123", warning="embed skipped")
                thumb = derived / f"{row['id']}.jpg"
                thumb.write_bytes(b"thumb")
                manifest.record_thumbnail(db_path, row["id"], "primary", str(thumb), "READY")

            dedupe.run_exact(db_path, managed)
            group_id = manifest.list_duplicate_groups(db_path)[0]["id"]
            config_path = root / "config.yaml"
            self._write_config(config_path, db_path, managed, derived, root)
            client = TestClient(create_app(config_path))

            response = client.get(f"/app/duplicates/{group_id}/keep-all", follow_redirects=False)
            self.assertEqual(response.status_code, 303)
            updated_group = manifest.list_duplicate_groups(db_path)[0]
            self.assertEqual(updated_group["status"], "KEPT_ALL")


if __name__ == "__main__":
    unittest.main()
