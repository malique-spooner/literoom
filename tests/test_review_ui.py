from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from literoom.metadata import manifest

try:
    from fastapi.testclient import TestClient
    from literoom.api import create_app

    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False
    TestClient = None  # type: ignore[assignment]
    create_app = None  # type: ignore[assignment]


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class ReviewUiTests(unittest.TestCase):
    @staticmethod
    def _write_config(config_path: Path, db_path: Path, managed: Path, derived: Path, root: Path):
        import_dir = root / "imports"
        import_dir.mkdir(exist_ok=True)
        config_path.write_text(
            "\n".join(
                [
                    "workspace_root: .",
                    "sources:",
                    f"  - {import_dir}",
                    "onboarding_complete: true",
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
                group_type="NEAR_VISUAL",
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
            compare_page = client.get(f"/app/compare?left={favorite_id}&right={other_id}", follow_redirects=False)
            asset_page = client.get(f"/app/assets/{favorite_id}")
            save_search = client.get("/app/search/save", params={"q": "alpha", "sort": "rated"}, follow_redirects=False)
            saved_search_page = client.get("/app/search?q=alpha&sort=rated")
            review_action = client.get(
                f"/app/assets/{other_id}/review?rating=4&state=reviewed",
                follow_redirects=False,
            )
            edit_action = client.get(
                f"/app/assets/{other_id}/edit",
                params={
                    "title": "Sunlit desk",
                    "description": "A quick browser smoke test",
                    "address": "12 Example Street",
                },
                follow_redirects=False,
            )
            tag_action = client.get(
                f"/app/assets/{other_id}/people?person=Nova",
                follow_redirects=False,
            )
            bulk_tag_action = client.get(
                "/app/library/tag-people",
                params={
                    "asset_ids": f"{favorite_id},{other_id}",
                    "person": "Malik",
                    "return_to": "/app/assets",
                },
                follow_redirects=False,
            )
            remove_action = client.get(
                f"/app/assets/{other_id}/people/remove?person=Nova",
                follow_redirects=False,
            )
            rename_action = client.get(
                "/app/people/rename",
                params={
                    "identity_id": identity_id,
                    "label": "Avery Lane",
                    "return_to": f"/app/people?identity_id={identity_id}",
                },
                follow_redirects=False,
            )
            renamed_people_page = client.get("/app/people")
            people_detail_page = client.get(f"/app/people?identity_id={identity_id}")
            malik_assets_before_restore = manifest.list_assets_tagged_with_person(db_path, "Malik", limit=10)
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
            self.assertIn("We keep this love in a photograph", home_page.text)
            self.assertIn("Open Library", home_page.text)
            self.assertIn("data-home-carousel", home_page.text)
            self.assertIn("featured-people", home_page.text)
            self.assertIn("carousel-dot", home_page.text)
            self.assertEqual(library_page.status_code, 200)
            self.assertIn("library-select-mode-toggle", library_page.text)
            self.assertIn("/app/library/tag-people", library_page.text)
            self.assertIn("library-load-more", library_page.text)
            self.assertTrue(
                "Scroll to load more" in library_page.text or "All assets loaded" in library_page.text
            )
            self.assertNotIn("Show Hidden Duplicates", library_page.text)
            self.assertEqual(people_page.status_code, 200)
            self.assertEqual(renamed_people_page.status_code, 200)
            self.assertEqual(people_detail_page.status_code, 200)
            self.assertNotIn("Review suggestions", renamed_people_page.text)
            self.assertNotIn("Unreviewed Clusters", renamed_people_page.text)
            self.assertIn("Avery Lane", people_detail_page.text)
            self.assertIn("Press Enter to save the name.", people_detail_page.text)
            self.assertEqual(len(malik_assets_before_restore), 2)
            self.assertEqual(system_page.status_code, 200)
            self.assertIn("System", system_page.text)
            self.assertIn("Imports", system_page.text)
            self.assertNotIn("Media tools", system_page.text)
            self.assertIn("v1.0.0", system_page.text)
            self.assertIn("Ingest check", system_page.text)
            self.assertIn("Check imports", system_page.text)
            self.assertIn("Smoke ingest", system_page.text)
            self.assertIn("Run full ingest", system_page.text)
            self.assertIn("Startup", system_page.text)
            self.assertIn("First startup", system_page.text)
            self.assertIn("Select import folder", system_page.text)
            self.assertIn("Stack", system_page.text)
            self.assertIn("Select import folder", system_page.text)
            self.assertIn("Recent jobs", system_page.text)
            self.assertIn("history-list", system_page.text)
            self.assertIn("data-history-seq", system_page.text)
            self.assertIn('option value="local"', library_page.text)
            self.assertEqual(search_page.status_code, 200)
            self.assertIn("Search", search_page.text)
            self.assertEqual(similar_page.status_code, 200)
            self.assertIn("Similar photos", similar_page.text)
            self.assertIn("Open photo", similar_page.text)
            self.assertIn("More Like This", search_page.text)
            self.assertEqual(review_page.status_code, 200)
            self.assertIn("Review", review_page.text)
            self.assertIn("Exact duplicates", review_page.text)
            self.assertIn("Near duplicates", review_page.text)
            self.assertNotIn("Sequence candidates", review_page.text)
            self.assertIn("/poster/", review_page.text)
            self.assertIn("/app/duplicates/", review_page.text)
            exact_group_id = manifest.list_duplicate_groups(db_path, limit=10, group_type="EXACT_SHA256")[0]["id"]
            exact_group_items = manifest.list_duplicate_group_items(db_path, exact_group_id)
            compare_group_page = client.get(f"/app/duplicates/{exact_group_id}/review", follow_redirects=False)
            self.assertEqual(compare_group_page.status_code, 303)
            self.assertEqual(compare_group_page.headers["location"], "/app/review")
            hide_action = client.get(
                f"/app/duplicates/{exact_group_id}/hide",
                params={"asset_id": exact_group_items[0]["asset_id"]},
                follow_redirects=False,
            )
            self.assertEqual(hide_action.status_code, 303)
            review_after_hide = client.get("/app/review")
            self.assertEqual(review_after_hide.status_code, 200)
            self.assertIn("No exact duplicates right now.", review_after_hide.text)
            self.assertNotIn("100% duplicate", review_after_hide.text)
            self.assertEqual(compare_page.status_code, 303)
            self.assertEqual(compare_page.headers["location"], "/app/review")
            self.assertEqual(asset_page.status_code, 200)
            self.assertIn("Quick Rank", asset_page.text)
            self.assertIn("Press Enter to tag someone.", asset_page.text)
            self.assertIn("ArrowLeft", asset_page.text)
            self.assertEqual(save_search.status_code, 303)
            self.assertEqual(saved_search_page.status_code, 200)
            self.assertIn("Saved Searches", saved_search_page.text)
            self.assertEqual(review_action.status_code, 303)
            self.assertEqual(edit_action.status_code, 303)
            self.assertEqual(tag_action.status_code, 303)
            self.assertEqual(bulk_tag_action.status_code, 303)
            self.assertEqual(remove_action.status_code, 303)
            self.assertEqual(rename_action.status_code, 303)
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
            self.assertEqual(refreshed["title"], "Sunlit desk")
            self.assertEqual(refreshed["description"], "A quick browser smoke test")
            refreshed_payload = manifest.export_asset_metadata_payload(db_path, other_id)
            self.assertEqual(refreshed_payload["normalized_metadata"]["title"], "Sunlit desk")
            self.assertEqual(refreshed_payload["normalized_metadata"]["description"], "A quick browser smoke test")
            self.assertEqual(refreshed_payload["normalized_metadata"]["location"], "12 Example Street")
            refreshed_identity = manifest.list_face_identities(db_path, limit=10, status="CONFIRMED")[0]
            self.assertEqual(refreshed_identity["label"], "Avery Lane")
            assets_for_person = manifest.list_assets_for_person(db_path, "Avery Lane")
            self.assertEqual(assets_for_person[0]["id"], favorite_id)

            import_progress = manifest.get_import_progress(db_path)
            self.assertGreaterEqual(import_progress["completion_pct"], 100.0)

    def test_library_click_and_poster_load_is_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            derived = root / "derived"
            managed.mkdir()
            derived.mkdir()
            manifest.init_db(db_path)

            source = root / "large.jpg"
            Image.new("RGB", (4200, 3200), color="slategray").save(source, quality=92)
            with source.open("ab") as fh:
                fh.write(b"\0" * (3 * 1024 * 1024))
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
                        "dt_original": "2024-02-01T12:00:00",
                        "src_mtime": "2024-02-01T12:00:00",
                    }
                ],
                db_path,
            )
            manifest.plan_targets(db_path)
            asset = manifest.list_assets(db_path, limit=5)[0]
            managed_path = managed / asset["managed_path"]
            managed_path.parent.mkdir(parents=True, exist_ok=True)
            managed_path.write_bytes(source.read_bytes())
            manifest.mark_copied(db_path, asset["id"], "abc123", warning=None)

            config_path = root / "config.yaml"
            self._write_config(config_path, db_path, managed, derived, root)
            client = TestClient(create_app(config_path))

            library_page = client.get("/app/assets")
            self.assertEqual(library_page.status_code, 200)
            self.assertIn(f"/app/assets/{asset['id']}", library_page.text)
            self.assertNotIn('class="asset-name"', library_page.text)

            start = time.perf_counter()
            asset_page = client.get(f"/app/assets/{asset['id']}")
            poster_page = client.get(f"/poster/{asset['id']}")
            elapsed = time.perf_counter() - start

            self.assertEqual(asset_page.status_code, 200)
            self.assertEqual(poster_page.status_code, 200)
            self.assertLess(elapsed, 5.0, f"Asset click and poster load took too long: {elapsed:.2f}s")
            self.assertIn("/poster/", asset_page.text)

    def test_people_page_shows_clustered_identities_on_landing_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            derived = root / "derived"
            managed.mkdir()
            derived.mkdir()
            manifest.init_db(db_path)

            assets = self._seed_assets(root, db_path, managed)
            cluster_id = manifest.create_face_identity(db_path, "Cluster 001", status="CLUSTERED")
            manifest.replace_faces_for_asset(
                db_path,
                assets["alpha.jpg"]["id"],
                [{"id": "face_cluster", "bbox": {"x": 10, "y": 10, "w": 60, "h": 60}, "frame_time_ms": 0, "embedding_ref": "seed"}],
            )
            manifest.update_face_cluster(db_path, "face_cluster", cluster_id, labeled=False)

            config_path = root / "config.yaml"
            self._write_config(config_path, db_path, managed, derived, root)
            client = TestClient(create_app(config_path))

            people_page = client.get("/app/people")
            cluster_detail = client.get(f"/app/people?identity_id={cluster_id}")

            self.assertEqual(people_page.status_code, 200)
            self.assertIn("Cluster 001", people_page.text)
            self.assertIn("Cluster", people_page.text)
            self.assertEqual(cluster_detail.status_code, 200)
            self.assertIn("Press Enter to save the name.", cluster_detail.text)

    def test_library_source_filter_shows_insta360_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            derived = root / "derived"
            managed.mkdir()
            derived.mkdir()
            manifest.init_db(db_path)

            rows = [
                {
                    "source": "insta360",
                    "abs_zip": str(root / "clip.insv"),
                    "zip_path": "clip.insv",
                    "source_kind": "file",
                    "source_locator": str(root / "clip.insv"),
                    "source_path": "clip.insv",
                    "media_type": "video",
                    "orig_filename": "clip.insv",
                    "orig_ext": ".insv",
                    "orig_size": 10,
                    "dt_original": "2024-01-01T12:00:00",
                    "src_mtime": "2024-01-01T12:00:00",
                },
                {
                    "source": "local",
                    "abs_zip": str(root / "alpha.jpg"),
                    "zip_path": "alpha.jpg",
                    "source_kind": "file",
                    "source_locator": str(root / "alpha.jpg"),
                    "source_path": "alpha.jpg",
                    "media_type": "image",
                    "orig_filename": "alpha.jpg",
                    "orig_ext": ".jpg",
                    "orig_size": 10,
                    "dt_original": "2024-01-02T12:00:00",
                    "src_mtime": "2024-01-02T12:00:00",
                },
            ]
            manifest.upsert_raw(rows, db_path)
            config_path = root / "config.yaml"
            self._write_config(config_path, db_path, managed, derived, root)
            client = TestClient(create_app(config_path))

            page = client.get("/app/assets?source=insta360")
            self.assertEqual(page.status_code, 200)
            self.assertIn('option value="insta360" selected', page.text)
            self.assertIn('insta360 (1)', page.text)
            self.assertIn('option value="local"', page.text)
            self.assertIn('local (1)', page.text)
            self.assertIn("clip.insv", page.text)
            self.assertNotIn("alpha.jpg", page.text)



if __name__ == "__main__":
    unittest.main()
