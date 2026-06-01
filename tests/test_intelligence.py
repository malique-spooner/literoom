from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from photo_unifier import intelligence
from photo_unifier.metadata import manifest


class IntelligenceTests(unittest.TestCase):
    def test_extract_content_records_ocr_and_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            managed.mkdir()
            manifest.init_db(db_path)

            source = root / "scan.jpg"
            Image.new("RGB", (120, 120), color="white").save(source)
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
                    }
                ],
                db_path,
            )
            manifest.plan_targets(db_path)
            asset = manifest.list_assets(db_path, limit=1)[0]
            managed_path = managed / manifest.get_asset(db_path, asset["id"])["managed_path"]
            managed_path.parent.mkdir(parents=True, exist_ok=True)
            managed_path.write_bytes(source.read_bytes())
            manifest.mark_copied(db_path, asset["id"], "abc123")

            with patch("photo_unifier.intelligence._ocr_text_from_image", return_value=("boarding pass seat 14A", "mock-ocr")):
                result = intelligence.extract_asset_content(db_path, managed, limit=1)

            extractions = manifest.list_extraction_results(db_path, asset["id"])
            embeddings = manifest.list_asset_embeddings(db_path, asset["id"])

            self.assertEqual(result["ocr_saved"], 1)
            self.assertTrue(any(row["result_type"] == "ocr" and row["text_content"] for row in extractions))
            self.assertTrue(any(row["embedding_type"] == "asset_text" for row in embeddings))
            self.assertTrue(any(row["embedding_type"] == "asset_visual" for row in embeddings))

    def test_semantic_search_ranks_matching_text_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            managed.mkdir()
            manifest.init_db(db_path)

            rows = []
            for name, title in [("scan.jpg", "Airport gate"), ("other.jpg", "Office desk")]:
                source = root / name
                Image.new("RGB", (120, 120), color="white").save(source)
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
                        "title": title,
                    }
                )
            manifest.upsert_raw(rows, db_path)
            manifest.plan_targets(db_path)
            assets = {row["orig_filename"]: row for row in manifest.list_assets(db_path, limit=10)}
            for row in assets.values():
                path = managed / manifest.get_asset(db_path, row["id"])["managed_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((root / row["orig_filename"]).read_bytes())
                manifest.mark_copied(db_path, row["id"], "abc123")

            def _ocr_side_effect(path: Path):
                if path.name == "scan.jpg":
                    return "boarding pass gate 12", "mock-ocr"
                return None, "mock-ocr"

            with patch("photo_unifier.intelligence._ocr_text_from_image", side_effect=_ocr_side_effect):
                intelligence.extract_asset_content(db_path, managed, limit=10)

            report = intelligence.search_assets_semantic(db_path, "boarding pass gate")
            self.assertGreaterEqual(report["count"], 1)
            self.assertEqual(report["items"][0]["orig_filename"], "scan.jpg")

    def test_review_metadata_boosts_semantic_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            rows = []
            for name in ["plain.jpg", "favorite.jpg"]:
                source = root / name
                Image.new("RGB", (120, 120), color="white").save(source)
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
            assets = {row["orig_filename"]: row for row in manifest.list_assets(db_path, limit=10)}
            manifest.set_asset_review(
                db_path,
                assets["favorite.jpg"]["id"],
                user_rating=5,
                review_state="reviewed",
                review_score=0.95,
                favorite=True,
            )

            report = intelligence.search_assets_semantic(db_path, None)
            self.assertGreaterEqual(report["count"], 2)
            self.assertEqual(report["items"][0]["orig_filename"], "favorite.jpg")

    def test_near_confident_face_match_stays_unassigned_for_small_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            rows = []
            for idx in range(2):
                source = root / f"face-{idx}.jpg"
                Image.new("RGB", (120, 120), color="white").save(source)
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
            assets = manifest.list_assets(db_path, limit=10, include_hidden=True)
            vector_a = [1.0] + [0.0] * 167
            vector_b = [0.88, (1 - 0.88**2) ** 0.5] + [0.0] * 166
            manifest.replace_faces_for_asset(
                db_path,
                assets[0]["id"],
                [{"id": "face-a", "bbox": {}, "embedding_vector": vector_a}],
                source_name="test",
            )
            manifest.save_face_embedding(db_path, asset_id=assets[0]["id"], face_id="face-a", vector=vector_a)
            identity_id = manifest.create_face_identity(db_path, "Ethan", status="CONFIRMED")
            manifest.assign_face_identity(db_path, "face-a", identity_id)

            manifest.replace_faces_for_asset(
                db_path,
                assets[1]["id"],
                [{"id": "face-b", "bbox": {}, "embedding_vector": vector_b}],
                source_name="test",
            )
            manifest.save_face_embedding(db_path, asset_id=assets[1]["id"], face_id="face-b", vector=vector_b)

            from photo_unifier import faces

            cluster_result = faces.cluster_faces(db_path, similarity_threshold=0.94)
            clarifications = manifest.list_face_clarifications(db_path, limit=10)
            updated_faces = manifest.list_faces(db_path, limit=10)

            self.assertEqual(cluster_result["assigned"], 0)
            self.assertEqual(cluster_result["clarified"], 0)
            self.assertEqual(clarifications, [])
            self.assertTrue(any(row["id"] == "face-b" and row["identity_id"] is None for row in updated_faces))


if __name__ == "__main__":
    unittest.main()
