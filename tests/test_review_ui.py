from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from photo_unifier.metadata import manifest

try:
    from fastapi.testclient import TestClient
    from photo_unifier.api import create_app

    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False
    TestClient = None  # type: ignore[assignment]
    create_app = None  # type: ignore[assignment]


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class ReviewUiTests(unittest.TestCase):
    @staticmethod
    def _write_config(config_path: Path, db_path: Path, managed: Path, derived: Path, root: Path):
        config_path.write_text(
            "\n".join(
                [
                    "workspace_root: .",
                    "sources: []",
                    "paths:",
                    f"  db_path: {db_path}",
                    f"  library_dir: {managed}",
                    f"  previews_dir: {derived}",
                    f"  logs_dir: {root / 'logs'}",
                    f"  temp_dir: {root / 'tmp'}",
                ]
            ),
            encoding="utf-8",
        )

    def _seed_assets(self, root: Path, db_path: Path, managed: Path):
        rows = []
        for idx, name in enumerate(["alpha.jpg", "beta.jpg"]):
            source = root / name
            Image.new("RGB", (180, 180), color=("white" if idx == 0 else "lightgray")).save(source)
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
                    "dt_original": f"2024-01-0{idx + 1}T12:00:00",
                    "src_mtime": f"2024-01-0{idx + 1}T12:00:00",
                }
            )
        manifest.upsert_raw(rows, db_path)
        manifest.plan_targets(db_path)
        assets = {row["orig_filename"]: row for row in manifest.list_assets(db_path, limit=10)}
        for asset in assets.values():
            managed_path = managed / asset["managed_path"]
            managed_path.parent.mkdir(parents=True, exist_ok=True)
            managed_path.write_bytes((root / asset["orig_filename"]).read_bytes())
            manifest.mark_copied(db_path, asset["id"], "abc123", warning="embed skipped")
            manifest.record_thumbnail(db_path, asset["id"], "primary", str(managed_path), "READY", width=180, height=180)
        return assets

    def test_search_compare_and_review_pages_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            derived = root / "derived"
            managed.mkdir()
            derived.mkdir()
            manifest.init_db(db_path)

            assets = self._seed_assets(root, db_path, managed)
            favorite_id = assets["alpha.jpg"]["id"]
            other_id = assets["beta.jpg"]["id"]
            manifest.set_asset_review(
                db_path,
                favorite_id,
                user_rating=5,
                review_state="reviewed",
                favorite=True,
                review_score=0.92,
            )
            face_rows = manifest.replace_faces_for_asset(
                db_path,
                favorite_id,
                [
                    {
                        "id": "face_alpha",
                        "bbox": {"x": 10, "y": 10, "w": 60, "h": 60},
                        "frame_time_ms": 0,
                        "embedding_ref": "seed",
                    }
                ],
            )
            self.assertEqual(face_rows, 1)
            identity_id = manifest.create_face_identity(db_path, "Avery")
            manifest.assign_face_identity(db_path, "face_alpha", identity_id)
            manifest.replace_duplicate_groups(
                db_path,
                group_type="EXACT_SHA256",
                groups=[
                    {
                        "canonical_asset_id": favorite_id,
                        "items": [
                            {"asset_id": favorite_id, "score": 1.0, "rationale": "seed exact"},
                            {"asset_id": other_id, "score": 1.0, "rationale": "seed exact"},
                        ],
                    }
                ],
            )
            manifest.replace_duplicate_groups(
                db_path,
                group_type="NEAR_AHASH",
                groups=[
                    {
                        "canonical_asset_id": favorite_id,
                        "items": [
                            {"asset_id": favorite_id, "score": 1.0, "rationale": "reference"},
                            {"asset_id": other_id, "score": 0.87, "rationale": "close match"},
                        ],
                    }
                ],
            )
            manifest.set_metadata_field(
                db_path,
                other_id,
                field_name="is_blurry",
                value=True,
                source_name="heuristic",
                is_canonical=True,
                confidence=0.8,
            )

            config_path = root / "config.yaml"
            self._write_config(config_path, db_path, managed, derived, root)
            client = TestClient(create_app(config_path))

            home_page = client.get("/")
            library_page = client.get("/app/assets")
            people_page = client.get("/app/people")
            system_page = client.get("/app/system")
            search_page = client.get("/app/search")
            similar_page = client.get(f"/app/search?similar_to={favorite_id}")
            review_page = client.get("/app/review")
            compare_page = client.get(f"/app/compare?left={favorite_id}&right={other_id}")
            asset_page = client.get(f"/app/assets/{favorite_id}")
            save_search = client.get("/app/search/save", params={"q": "alpha", "sort": "rated"}, follow_redirects=False)
            saved_search_page = client.get("/app/search?q=alpha&sort=rated")
            review_action = client.get(
                f"/app/assets/{other_id}/review?rating=4&state=reviewed",
                follow_redirects=False,
            )
            tag_action = client.get(
                f"/app/assets/{other_id}/people?person=Nova",
                follow_redirects=False,
            )
            remove_action = client.get(
                f"/app/assets/{other_id}/people/remove?person=Nova",
                follow_redirects=False,
            )
            history = manifest.get_mutation_history(db_path, limit=8)
            review_entry = next(row for row in history if row["action_type"] == "SET_ASSET_REVIEW")
            restore_history = client.get(
                f"/app/history/{review_entry['seq']}/restore?return_to=/app/assets/{other_id}",
                follow_redirects=False,
            )
            health = client.get("/healthz")
            status = client.get("/status")
            jobs = client.get("/jobs")

            self.assertEqual(home_page.status_code, 200)
            self.assertIn("Everything, in one calm place", home_page.text)
            self.assertIn("Open Library", home_page.text)
            self.assertIn("data-home-carousel", home_page.text)
            self.assertIn("featured-people", home_page.text)
            self.assertIn("carousel-dot", home_page.text)
            self.assertEqual(library_page.status_code, 200)
            self.assertIn("Showing 1-2 of 2", library_page.text)
            self.assertIn("Page 1 of 1", library_page.text)
            self.assertEqual(people_page.status_code, 200)
            self.assertIn("People albums", people_page.text)
            self.assertIn("Avery", people_page.text)
            self.assertEqual(system_page.status_code, 200)
            self.assertIn("System", system_page.text)
            self.assertIn("Imports", system_page.text)
            self.assertNotIn("Media tools", system_page.text)
            self.assertIn("Stack", system_page.text)
            self.assertIn("Choose import folder", system_page.text)
            self.assertEqual(search_page.status_code, 200)
            self.assertIn("Search", search_page.text)
            self.assertEqual(similar_page.status_code, 200)
            self.assertIn("Similar photos", similar_page.text)
            self.assertIn("Open photo", similar_page.text)
            self.assertIn("More Like This", search_page.text)
            self.assertEqual(review_page.status_code, 200)
            self.assertIn("Review", review_page.text)
            self.assertIn("Blurry", review_page.text)
            self.assertIn("/app/duplicates/", review_page.text)
            exact_group_id = manifest.list_duplicate_groups(db_path, limit=10, group_type="EXACT_SHA256")[0]["id"]
            exact_group_items = manifest.list_duplicate_group_items(db_path, exact_group_id)
            compare_group_page = client.get(f"/app/duplicates/{exact_group_id}/review")
            self.assertEqual(compare_group_page.status_code, 200)
            self.assertIn("Compare duplicates", compare_group_page.text)
            self.assertIn("duplicate-action delete", compare_group_page.text)
            hide_action = client.get(
                f"/app/duplicates/{exact_group_id}/hide",
                params={"asset_id": exact_group_items[0]["asset_id"]},
                follow_redirects=False,
            )
            self.assertEqual(hide_action.status_code, 303)
            self.assertEqual(compare_page.status_code, 200)
            self.assertIn("Compare", compare_page.text)
            self.assertIn("Fast side-by-side review for two assets", compare_page.text)
            self.assertEqual(asset_page.status_code, 200)
            self.assertIn("Quick Rank", asset_page.text)
            self.assertIn("Quick person tagger", asset_page.text)
            self.assertIn("ArrowLeft", asset_page.text)
            self.assertEqual(save_search.status_code, 303)
            self.assertEqual(saved_search_page.status_code, 200)
            self.assertIn("Saved Searches", saved_search_page.text)
            self.assertEqual(review_action.status_code, 303)
            self.assertEqual(tag_action.status_code, 303)
            self.assertEqual(remove_action.status_code, 303)
            self.assertEqual(restore_history.status_code, 303)
            self.assertEqual(health.status_code, 200)
            self.assertIn("pipeline", health.json())
            self.assertIn("tools", health.json())
            self.assertIn("exiftool", health.json()["tools"])
            self.assertIn("locked_stack", health.json()["tools"])
            self.assertEqual(status.status_code, 200)
            self.assertIn("pipeline", status.json())
            self.assertIn("tools", status.json())
            self.assertIn("locked_stack", status.json()["tools"])
            self.assertEqual(jobs.status_code, 200)
            self.assertIn("items", jobs.json())

            refreshed = manifest.get_asset(db_path, other_id)
            self.assertEqual(refreshed["user_rating"], 4)
            self.assertEqual(refreshed["review_state"], "reviewed")
            self.assertNotIn("Nova", refreshed["people_json"])

            import_progress = manifest.get_import_progress(db_path)
            self.assertGreaterEqual(import_progress["completion_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
