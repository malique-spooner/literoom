from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from literoom import faces
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
class FacesTests(unittest.TestCase):


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

    @staticmethod
    def _unit_vector(degrees: float) -> list[float]:
        radians = math.radians(degrees)
        return [math.cos(radians), math.sin(radians)]

    @staticmethod
    def _create_face_assets(root: Path, db_path: Path, prefix: str, count: int) -> list[dict]:
        rows = []
        for idx in range(count):
            locator = root / f"{prefix}-{idx}.jpg"
            locator.write_bytes(b"same-bytes")
            rows.append(
                {
                    "source": "local",
                    "abs_zip": str(locator),
                    "zip_path": locator.name,
                    "source_kind": "file",
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
        assets_by_name = {
            row["orig_filename"]: row
            for row in manifest.list_assets(db_path, limit=count + 10, include_hidden=True)
        }
        return [assets_by_name[f"{prefix}-{idx}.jpg"] for idx in range(count)]

    @staticmethod
    def _cluster_face(db_path: Path, asset: dict, face_id: str, vector: list[float], identity_id: str) -> None:
        manifest.replace_faces_for_asset(
            db_path,
            asset["id"],
            [{"id": face_id, "bbox": {}, "embedding_vector": vector}],
            source_name="test",
        )
        manifest.save_face_embedding(
            db_path,
            asset_id=asset["id"],
            face_id=face_id,
            vector=vector,
        )
        manifest.update_face_cluster(db_path, face_id, identity_id, labeled=False)

    def test_face_detection_and_label_assignment(self):
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
                        "orig_size": 10,
                        "dt_original": "2024-01-02T03:04:05",
                        "src_mtime": "2024-01-02T03:04:05",
                        "source_dt": "exif_datetime",
                    }
                )
            manifest.upsert_raw(rows, db_path)
            manifest.plan_targets(db_path)
            assets = list(manifest.iter_for_master(db_path))
            for asset in assets:
                path = managed / asset["managed_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (100, 100), color="white").save(path)
                manifest.mark_copied(db_path, asset["id"], "abc123", warning="embed skipped")
                thumb = derived / f"{asset['id']}.jpg"
                Image.new("RGB", (100, 100), color="gray").save(thumb)
                manifest.record_thumbnail(db_path, asset["id"], "primary", str(thumb), "READY")

            fake_image = __import__("numpy").random.randint(0, 255, size=(100, 100, 3), dtype="uint8")
            with patch("literoom.faces._ensure_insightface_ready") as fake_ready, patch(
                "literoom.faces._insightface_detections"
            ) as fake_insightface, patch("literoom.faces._cv2") as fake_cv2:
                fake_insightface.return_value = [
                    {
                        "bbox": {
                            "x": 10,
                            "y": 20,
                            "w": 40,
                            "h": 40,
                            "image_width": 100,
                            "image_height": 100,
                            "quality": 99.0,
                        },
                        "embedding_vector": [0.1] * 512,
                    }
                ]
                fake_cv2.return_value.imread.return_value = fake_image
                result = faces.detect_faces(db_path, managed, face_model="antelopev2")

            self.assertEqual(result["detected"], 2)
            self.assertTrue(fake_ready.called)
            detected = manifest.list_faces(db_path, limit=10)
            self.assertEqual(len(detected), 2)
            self.assertTrue(all(row["identity_label"] in (None, "") for row in detected))

            config_path = root / "config.yaml"
            self._write_config(config_path, db_path, managed, derived, root)
            client = TestClient(create_app(config_path))
            response = client.get(
                f"/app/faces/assign?face_id={detected[0]['id']}&identity_label=Ethan",
                follow_redirects=False,
            )
            self.assertEqual(response.status_code, 303)
            updated = next(row for row in manifest.list_faces(db_path, limit=10) if row["id"] == detected[0]["id"])
            self.assertEqual(updated["identity_label"], "Ethan")
            crop = client.get(f"/face-crop/{updated['id']}")
            self.assertEqual(crop.status_code, 200)
            self.assertEqual(crop.headers["content-type"], "image/jpeg")

            reject = client.get(
                f"/app/faces/reject?face_id={updated['id']}",
                follow_redirects=False,
            )
            self.assertEqual(reject.status_code, 303)
            visible = manifest.list_faces(db_path, limit=10)
            self.assertEqual(len(visible), 1)
            self.assertNotEqual(visible[0]["id"], updated["id"])

    def test_detect_faces_requires_insightface_for_quality_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            managed.mkdir()
            manifest.init_db(db_path)

            with patch(
                "literoom.faces._ensure_insightface_ready",
                side_effect=faces.FaceRecognitionQualityError("InsightFace required"),
            ):
                with self.assertRaises(faces.FaceRecognitionQualityError):
                    faces.detect_faces(db_path, managed, face_model="antelopev2")

    def test_cluster_faces_requires_multiple_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            manifest.upsert_raw(
                [
                    {
                        "source": "local",
                        "abs_zip": str(root / "single.jpg"),
                        "zip_path": "single.jpg",
                        "source_kind": "file",
                        "source_locator": str(root / "single.jpg"),
                        "source_path": "single.jpg",
                        "media_type": "image",
                        "orig_filename": "single.jpg",
                        "orig_ext": ".jpg",
                        "orig_size": 10,
                        "dt_original": "2024-01-02T03:04:05",
                        "src_mtime": "2024-01-02T03:04:05",
                    }
                ],
                db_path,
            )
            manifest.plan_targets(db_path)
            asset = manifest.list_assets(db_path, limit=1)[0]
            detections = [
                {"id": "face-a", "bbox": {}, "embedding_vector": [0.1] * 168},
                {"id": "face-b", "bbox": {}, "embedding_vector": [0.1] * 168},
            ]
            manifest.replace_faces_for_asset(db_path, asset["id"], detections, source_name="test")
            for detection in detections:
                manifest.save_face_embedding(
                    db_path,
                    asset_id=asset["id"],
                    face_id=detection["id"],
                    vector=detection["embedding_vector"],
                )

            result = faces.cluster_faces(db_path, similarity_threshold=0.95)
            identities = manifest.list_face_identities(db_path, limit=10)
            self.assertEqual(result["clusters_created"], 0)
            self.assertEqual(len(identities), 0)

    def test_flatten_insightface_pack_copies_nested_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_name = "antelopev2"
            nested = root / "models" / model_name / model_name
            nested.mkdir(parents=True)
            for filename in ["scrfd_10g_bnkps.onnx", "glintr100.onnx", "._ignored.onnx"]:
                (nested / filename).write_bytes(b"model-bytes" if not filename.startswith("._") else b"ignored")

            with patch.dict("os.environ", {"INSIGHTFACE_HOME": str(root)}):
                result_dir = faces._flatten_insightface_pack(model_name)

            self.assertEqual(result_dir, (root / "models" / model_name).resolve())
            self.assertTrue((root / "models" / model_name / "scrfd_10g_bnkps.onnx").exists())
            self.assertTrue((root / "models" / model_name / "glintr100.onnx").exists())
            self.assertFalse((root / "models" / model_name / "._ignored.onnx").exists())

    def test_cluster_faces_does_not_create_machine_cluster_until_enough_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            rows = []
            for idx in range(2):
                locator = root / f"asset-{idx}.jpg"
                locator.write_bytes(b"same-bytes")
                rows.append(
                    {
                        "source": "local",
                        "abs_zip": str(locator),
                        "zip_path": locator.name,
                        "source_kind": "file",
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
            assets = manifest.list_assets(db_path, limit=10)
            vectors = [[0.1] * 168, [0.1] * 168]
            for asset, vector in zip(assets, vectors):
                face_id = f"face-{asset['id']}"
                manifest.replace_faces_for_asset(
                    db_path,
                    asset["id"],
                    [{"id": face_id, "bbox": {}, "embedding_vector": vector}],
                    source_name="test",
                )
                manifest.save_face_embedding(
                    db_path,
                    asset_id=asset["id"],
                    face_id=face_id,
                    vector=vector,
                )

            result = faces.cluster_faces(db_path, similarity_threshold=0.95)
            identities = manifest.list_face_identities(db_path, limit=10)
            labeled_faces = manifest.list_faces(db_path, limit=10, include_rejected=True)

            self.assertEqual(result["clusters_created"], 0)
            self.assertEqual(len(identities), 0)
            self.assertEqual({row["identity_id"] for row in labeled_faces}, {None})

    def test_cluster_faces_creates_machine_cluster_after_enough_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            rows = []
            for idx in range(10):
                locator = root / f"asset-{idx}.jpg"
                locator.write_bytes(b"same-bytes")
                rows.append(
                    {
                        "source": "local",
                        "abs_zip": str(locator),
                        "zip_path": locator.name,
                        "source_kind": "file",
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
            assets = manifest.list_assets(db_path, limit=20)
            vector = [0.1] * 168
            for asset in assets:
                face_id = f"face-{asset['id']}"
                manifest.replace_faces_for_asset(
                    db_path,
                    asset["id"],
                    [{"id": face_id, "bbox": {}, "embedding_vector": vector}],
                    source_name="test",
                )
                manifest.save_face_embedding(
                    db_path,
                    asset_id=asset["id"],
                    face_id=face_id,
                    vector=vector,
                )

            result = faces.cluster_faces(db_path, similarity_threshold=0.95)
            identities = manifest.list_face_identities(db_path, limit=10)
            labeled_faces = manifest.list_faces(db_path, limit=20, include_rejected=True)

            self.assertEqual(result["clusters_created"], 1)
            self.assertEqual(len(identities), 1)
            self.assertEqual({row["identity_id"] for row in labeled_faces}, {identities[0]["id"]})

    def test_cluster_faces_snowballs_strong_cluster_matches_without_naming(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            rows = []
            for idx in range(11):
                locator = root / f"snowball-{idx}.jpg"
                locator.write_bytes(b"same-bytes")
                rows.append(
                    {
                        "source": "local",
                        "abs_zip": str(locator),
                        "zip_path": locator.name,
                        "source_kind": "file",
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
            assets = manifest.list_assets(db_path, limit=20)
            seed_vector = [1.0, 0.0]
            candidate_vector = [0.91, 0.4146082488325576]
            for asset in assets[:10]:
                face_id = f"face-{asset['id']}"
                manifest.replace_faces_for_asset(
                    db_path,
                    asset["id"],
                    [{"id": face_id, "bbox": {}, "embedding_vector": seed_vector}],
                    source_name="test",
                )
                manifest.save_face_embedding(
                    db_path,
                    asset_id=asset["id"],
                    face_id=face_id,
                    vector=seed_vector,
                )

            candidate_asset = assets[10]
            candidate_face_id = f"face-{candidate_asset['id']}"
            manifest.replace_faces_for_asset(
                db_path,
                candidate_asset["id"],
                [{"id": candidate_face_id, "bbox": {}, "embedding_vector": candidate_vector}],
                source_name="test",
            )
            manifest.save_face_embedding(
                db_path,
                asset_id=candidate_asset["id"],
                face_id=candidate_face_id,
                vector=candidate_vector,
            )

            result = faces.cluster_faces(db_path, similarity_threshold=0.9)
            identities = manifest.list_face_identities(db_path, limit=10)
            labeled_faces = {row["id"]: row for row in manifest.list_faces(db_path, limit=20, include_rejected=True)}

            self.assertEqual(result["clusters_created"], 1)
            self.assertGreaterEqual(result["assigned"], 1)
            self.assertEqual(len(identities), 1)
            self.assertEqual(labeled_faces[candidate_face_id]["identity_id"], identities[0]["id"])

    def test_cluster_faces_snowballs_more_aggressively_for_larger_strong_clusters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            rows = []
            for idx in range(21):
                locator = root / f"snowball-large-{idx}.jpg"
                locator.write_bytes(b"same-bytes")
                rows.append(
                    {
                        "source": "local",
                        "abs_zip": str(locator),
                        "zip_path": locator.name,
                        "source_kind": "file",
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
            assets = manifest.list_assets(db_path, limit=30)
            seed_vector = [1.0, 0.0]
            candidate_vector = [0.8, 0.6]
            for asset in assets[:20]:
                face_id = f"face-{asset['id']}"
                manifest.replace_faces_for_asset(
                    db_path,
                    asset["id"],
                    [{"id": face_id, "bbox": {}, "embedding_vector": seed_vector}],
                    source_name="test",
                )
                manifest.save_face_embedding(
                    db_path,
                    asset_id=asset["id"],
                    face_id=face_id,
                    vector=seed_vector,
                )

            candidate_asset = assets[20]
            candidate_face_id = f"face-{candidate_asset['id']}"
            manifest.replace_faces_for_asset(
                db_path,
                candidate_asset["id"],
                [{"id": candidate_face_id, "bbox": {}, "embedding_vector": candidate_vector}],
                source_name="test",
            )
            manifest.save_face_embedding(
                db_path,
                asset_id=candidate_asset["id"],
                face_id=candidate_face_id,
                vector=candidate_vector,
            )

            result = faces.cluster_faces(db_path, similarity_threshold=0.9)
            identities = manifest.list_face_identities(db_path, limit=10)
            labeled_faces = {row["id"]: row for row in manifest.list_faces(db_path, limit=30, include_rejected=True)}

            self.assertEqual(result["clusters_created"], 1)
            self.assertGreaterEqual(result["assigned"], 1)
            self.assertEqual(len(identities), 1)
            self.assertEqual(labeled_faces[candidate_face_id]["identity_id"], identities[0]["id"])

    def test_cluster_faces_large_tight_clusters_auto_assign_lower_score_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            rows = []
            for idx in range(31):
                locator = root / f"snowball-tight-{idx}.jpg"
                locator.write_bytes(b"same-bytes")
                rows.append(
                    {
                        "source": "local",
                        "abs_zip": str(locator),
                        "zip_path": locator.name,
                        "source_kind": "file",
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
            assets = manifest.list_assets(db_path, limit=40)
            seed_vector = [1.0, 0.0]
            candidate_vector = [0.72, 0.6939740629158989]
            for asset in assets[:30]:
                face_id = f"face-{asset['id']}"
                manifest.replace_faces_for_asset(
                    db_path,
                    asset["id"],
                    [{"id": face_id, "bbox": {}, "embedding_vector": seed_vector}],
                    source_name="test",
                )
                manifest.save_face_embedding(
                    db_path,
                    asset_id=asset["id"],
                    face_id=face_id,
                    vector=seed_vector,
                )

            candidate_asset = assets[30]
            candidate_face_id = f"face-{candidate_asset['id']}"
            manifest.replace_faces_for_asset(
                db_path,
                candidate_asset["id"],
                [{"id": candidate_face_id, "bbox": {}, "embedding_vector": candidate_vector}],
                source_name="test",
            )
            manifest.save_face_embedding(
                db_path,
                asset_id=candidate_asset["id"],
                face_id=candidate_face_id,
                vector=candidate_vector,
            )

            result = faces.cluster_faces(db_path, similarity_threshold=0.9)
            identities = manifest.list_face_identities(db_path, limit=10)
            labeled_faces = {row["id"]: row for row in manifest.list_faces(db_path, limit=40, include_rejected=True)}

            self.assertEqual(result["clusters_created"], 1)
            self.assertGreaterEqual(result["assigned"], 1)
            self.assertEqual(len(identities), 1)
            self.assertEqual(labeled_faces[candidate_face_id]["identity_id"], identities[0]["id"])

    def test_cluster_faces_propagates_confirmed_identity_to_asset_people(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            rows = []
            for idx in range(2):
                locator = root / f"person-{idx}.jpg"
                locator.write_bytes(b"same-bytes")
                rows.append(
                    {
                        "source": "local",
                        "abs_zip": str(locator),
                        "zip_path": locator.name,
                        "source_kind": "file",
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
            assets = {row["orig_filename"]: row for row in manifest.list_assets(db_path, limit=10, include_hidden=True)}
            vector = [0.2] * 168
            face_a = "face-a"
            face_b = "face-b"
            manifest.replace_faces_for_asset(
                db_path,
                assets["person-0.jpg"]["id"],
                [{"id": face_a, "bbox": {}, "embedding_vector": vector}],
                source_name="test",
            )
            manifest.save_face_embedding(
                db_path,
                asset_id=assets["person-0.jpg"]["id"],
                face_id=face_a,
                vector=vector,
            )
            identity_id = manifest.create_face_identity(db_path, "Ethan", status="CONFIRMED")
            manifest.assign_face_identity(db_path, face_a, identity_id)

            manifest.replace_faces_for_asset(
                db_path,
                assets["person-1.jpg"]["id"],
                [{"id": face_b, "bbox": {}, "embedding_vector": vector}],
                source_name="test",
            )
            manifest.save_face_embedding(
                db_path,
                asset_id=assets["person-1.jpg"]["id"],
                face_id=face_b,
                vector=vector,
            )

            result = faces.cluster_faces(db_path, similarity_threshold=0.95)
            updated_asset = manifest.get_asset(db_path, assets["person-1.jpg"]["id"])
            labeled_faces = {row["id"]: row for row in manifest.list_faces(db_path, limit=10, include_rejected=True)}

            self.assertEqual(result["assigned"], 1)
            self.assertEqual(labeled_faces[face_b]["identity_label"], "Ethan")
            self.assertIn("Ethan", updated_asset["people_json"])

    def test_cluster_faces_auto_assigns_strong_confirmed_matches_with_aggressive_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            rows = []
            for idx in range(2):
                locator = root / f"person-{idx}.jpg"
                locator.write_bytes(b"same-bytes")
                rows.append(
                    {
                        "source": "local",
                        "abs_zip": str(locator),
                        "zip_path": locator.name,
                        "source_kind": "file",
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
            assets = {row["orig_filename"]: row for row in manifest.list_assets(db_path, limit=10, include_hidden=True)}
            confirmed_vector = [1.0, 0.0]
            candidate_vector = [0.955, 0.2966479394838265]
            face_a = "face-a"
            face_b = "face-b"

            manifest.replace_faces_for_asset(
                db_path,
                assets["person-0.jpg"]["id"],
                [{"id": face_a, "bbox": {}, "embedding_vector": confirmed_vector}],
                source_name="test",
            )
            manifest.save_face_embedding(
                db_path,
                asset_id=assets["person-0.jpg"]["id"],
                face_id=face_a,
                vector=confirmed_vector,
            )
            identity_id = manifest.create_face_identity(db_path, "Malik", status="CONFIRMED")
            manifest.assign_face_identity(db_path, face_a, identity_id)

            manifest.replace_faces_for_asset(
                db_path,
                assets["person-1.jpg"]["id"],
                [{"id": face_b, "bbox": {}, "embedding_vector": candidate_vector}],
                source_name="test",
            )
            manifest.save_face_embedding(
                db_path,
                asset_id=assets["person-1.jpg"]["id"],
                face_id=face_b,
                vector=candidate_vector,
            )

            result = faces.cluster_faces(db_path, similarity_threshold=0.9)
            labeled_faces = {row["id"]: row for row in manifest.list_faces(db_path, limit=10, include_rejected=True)}
            clarifications = manifest.list_face_clarifications(db_path, suggested_identity_id=identity_id)

            self.assertEqual(result["assigned"], 1)
            self.assertEqual(result["clarified"], 0)
            self.assertEqual(labeled_faces[face_b]["identity_id"], identity_id)
            self.assertEqual(clarifications, [])

    def test_cluster_faces_lowers_confirmed_auto_assign_bar_after_five_face_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            assets = self._create_face_assets(root, db_path, "confirmed-strong", 11)
            identity_id = manifest.create_face_identity(db_path, "Malique", status="CONFIRMED")
            seed_vector = [1.0, 0.0]
            for idx, asset in enumerate(assets[:10]):
                face_id = f"face-confirmed-{idx}"
                manifest.replace_faces_for_asset(
                    db_path,
                    asset["id"],
                    [{"id": face_id, "bbox": {}, "embedding_vector": seed_vector}],
                    source_name="test",
                )
                manifest.save_face_embedding(
                    db_path,
                    asset_id=asset["id"],
                    face_id=face_id,
                    vector=seed_vector,
                )
                manifest.update_face_cluster(db_path, face_id, identity_id, labeled=True)

            candidate_face_id = "face-candidate"
            candidate_vector = [0.91, math.sqrt(1 - 0.91 * 0.91)]
            manifest.replace_faces_for_asset(
                db_path,
                assets[10]["id"],
                [{"id": candidate_face_id, "bbox": {}, "embedding_vector": candidate_vector}],
                source_name="test",
            )
            manifest.save_face_embedding(
                db_path,
                asset_id=assets[10]["id"],
                face_id=candidate_face_id,
                vector=candidate_vector,
            )

            result = faces.cluster_faces(db_path, similarity_threshold=0.9)
            labeled_faces = {row["id"]: row for row in manifest.list_faces(db_path, limit=20, include_rejected=True)}

            self.assertEqual(result["assigned"], 1)
            self.assertEqual(labeled_faces[candidate_face_id]["identity_id"], identity_id)
            self.assertEqual(labeled_faces[candidate_face_id]["status"], "LABELED")

    def test_refresh_person_candidate_clarifications_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            rows = []
            for idx in range(2):
                locator = root / f"candidate-{idx}.jpg"
                locator.write_bytes(b"same-bytes")
                rows.append(
                    {
                        "source": "local",
                        "abs_zip": str(locator),
                        "zip_path": locator.name,
                        "source_kind": "file",
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
            assets = {row["orig_filename"]: row for row in manifest.list_assets(db_path, limit=10, include_hidden=True)}
            confirmed_vector = [1.0, 0.0]
            suggested_vector = [0.75, 0.6614378277661477]

            manifest.replace_faces_for_asset(
                db_path,
                assets["candidate-0.jpg"]["id"],
                [{"id": "face-confirmed", "bbox": {}, "embedding_vector": confirmed_vector}],
                source_name="test",
            )
            manifest.save_face_embedding(
                db_path,
                asset_id=assets["candidate-0.jpg"]["id"],
                face_id="face-confirmed",
                vector=confirmed_vector,
            )
            identity_id = manifest.create_face_identity(db_path, "Ethan", status="CONFIRMED")
            manifest.assign_face_identity(db_path, "face-confirmed", identity_id)

            manifest.replace_faces_for_asset(
                db_path,
                assets["candidate-1.jpg"]["id"],
                [{"id": "face-suggested", "bbox": {}, "embedding_vector": suggested_vector}],
                source_name="test",
            )
            manifest.save_face_embedding(
                db_path,
                asset_id=assets["candidate-1.jpg"]["id"],
                face_id="face-suggested",
                vector=suggested_vector,
            )

            saved = manifest.refresh_person_candidate_clarifications(db_path, identity_id)
            clarifications = manifest.list_face_clarifications(db_path, suggested_identity_id=identity_id)

            self.assertEqual(saved, 0)
            self.assertEqual(clarifications, [])

    def test_refresh_person_candidate_clarifications_stays_disabled_as_profile_strengthens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            rows = []
            for idx in range(4):
                locator = root / f"profile-{idx}.jpg"
                locator.write_bytes(b"same-bytes")
                rows.append(
                    {
                        "source": "local",
                        "abs_zip": str(locator),
                        "zip_path": locator.name,
                        "source_kind": "file",
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
            assets = {row["orig_filename"]: row for row in manifest.list_assets(db_path, limit=10, include_hidden=True)}
            identity_id = manifest.create_face_identity(db_path, "Malik", status="CONFIRMED")

            base_vector = [1.0, 0.0]
            for idx in range(3):
                face_id = f"face-confirmed-{idx}"
                asset = assets[f"profile-{idx}.jpg"]
                manifest.replace_faces_for_asset(
                    db_path,
                    asset["id"],
                    [{"id": face_id, "bbox": {}, "embedding_vector": base_vector}],
                    source_name="test",
                )
                manifest.save_face_embedding(
                    db_path,
                    asset_id=asset["id"],
                    face_id=face_id,
                    vector=base_vector,
                )
                manifest.assign_face_identity(db_path, face_id, identity_id)

            candidate_vector = [0.69, 0.7238093673182802]
            manifest.replace_faces_for_asset(
                db_path,
                assets["profile-3.jpg"]["id"],
                [{"id": "face-candidate", "bbox": {}, "embedding_vector": candidate_vector}],
                source_name="test",
            )
            manifest.save_face_embedding(
                db_path,
                asset_id=assets["profile-3.jpg"]["id"],
                face_id="face-candidate",
                vector=candidate_vector,
            )

            saved = manifest.refresh_person_candidate_clarifications(db_path, identity_id)
            clarifications = manifest.list_face_clarifications(db_path, suggested_identity_id=identity_id)

            self.assertEqual(saved, 0)
            self.assertEqual(clarifications, [])

    def test_refresh_person_candidate_clarifications_dismisses_stale_suggestions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            rows = []
            for idx in range(2):
                locator = root / f"stale-{idx}.jpg"
                locator.write_bytes(b"same-bytes")
                rows.append(
                    {
                        "source": "local",
                        "abs_zip": str(locator),
                        "zip_path": locator.name,
                        "source_kind": "file",
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
            assets = {row["orig_filename"]: row for row in manifest.list_assets(db_path, limit=10, include_hidden=True)}
            identity_id = manifest.create_face_identity(db_path, "Ethan", status="CONFIRMED")

            manifest.replace_faces_for_asset(
                db_path,
                assets["stale-0.jpg"]["id"],
                [{"id": "face-confirmed", "bbox": {}, "embedding_vector": [1.0, 0.0]}],
                source_name="test",
            )
            manifest.save_face_embedding(
                db_path,
                asset_id=assets["stale-0.jpg"]["id"],
                face_id="face-confirmed",
                vector=[1.0, 0.0],
            )
            manifest.assign_face_identity(db_path, "face-confirmed", identity_id)

            manifest.replace_faces_for_asset(
                db_path,
                assets["stale-1.jpg"]["id"],
                [{"id": "face-loose", "bbox": {}, "embedding_vector": [0.95, 0.31224989991991997]}],
                source_name="test",
            )
            manifest.save_face_embedding(
                db_path,
                asset_id=assets["stale-1.jpg"]["id"],
                face_id="face-loose",
                vector=[0.95, 0.31224989991991997],
            )

            manifest.save_face_clarification(
                db_path,
                "face-loose",
                suggested_identity_id=identity_id,
                suggested_label="Ethan",
                score=0.9,
                rationale="stale suggestion",
            )

            manifest.replace_faces_for_asset(
                db_path,
                assets["stale-1.jpg"]["id"],
                [{"id": "face-loose", "bbox": {}, "embedding_vector": [0.2, 0.9797958971132712]}],
                source_name="test",
            )
            manifest.save_face_embedding(
                db_path,
                asset_id=assets["stale-1.jpg"]["id"],
                face_id="face-loose",
                vector=[0.2, 0.9797958971132712],
            )

            saved = manifest.refresh_person_candidate_clarifications(db_path, identity_id)
            open_clarifications = manifest.list_face_clarifications(db_path, suggested_identity_id=identity_id)

            self.assertEqual(saved, 0)
            self.assertEqual(open_clarifications, [])

    def test_replace_faces_for_asset_clears_stale_face_people_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            source = root / "portrait.jpg"
            source.write_bytes(b"same-bytes")
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
                        "dt_original": "2024-01-02T03:04:05",
                        "src_mtime": "2024-01-02T03:04:05",
                    }
                ],
                db_path,
            )
            asset = manifest.list_assets(db_path, limit=1, include_hidden=True)[0]
            identity_id = manifest.create_face_identity(db_path, "Malik", status="CONFIRMED")

            manifest.replace_faces_for_asset(
                db_path,
                asset["id"],
                [{"id": "face-old", "bbox": {}, "embedding_vector": [0.3] * 168}],
                source_name="test",
            )
            manifest.assign_face_identity(db_path, "face-old", identity_id)
            manifest.add_asset_person(db_path, asset["id"], "Manual Tag")

            labeled_asset = manifest.get_asset(db_path, asset["id"])
            self.assertIn("Malik", labeled_asset["people_json"])
            self.assertIn("Manual Tag", labeled_asset["people_json"])

            manifest.replace_faces_for_asset(
                db_path,
                asset["id"],
                [{"id": "face-new", "bbox": {}, "embedding_vector": [0.2] * 168}],
                source_name="test",
            )

            refreshed = manifest.get_asset(db_path, asset["id"])
            self.assertNotIn("Malik", refreshed["people_json"])
            self.assertIn("Manual Tag", refreshed["people_json"])

    def test_related_clusters_are_separate_from_loose_face_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            rows = []
            for idx in range(2):
                locator = root / f"person-{idx}.jpg"
                locator.write_bytes(b"same-bytes")
                rows.append(
                    {
                        "source": "local",
                        "abs_zip": str(locator),
                        "zip_path": locator.name,
                        "source_kind": "file",
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
            assets = {row["orig_filename"]: row for row in manifest.list_assets(db_path, limit=10, include_hidden=True)}

            confirmed_id = manifest.create_face_identity(db_path, "Malik", status="CONFIRMED")
            cluster_id = manifest.create_face_identity(db_path, "Cluster 1", status="CLUSTERED")
            vector = [0.4] * 168

            manifest.replace_faces_for_asset(
                db_path,
                assets["person-0.jpg"]["id"],
                [{"id": "face-confirmed", "bbox": {}, "embedding_vector": vector}],
                source_name="test",
            )
            manifest.save_face_embedding(
                db_path,
                asset_id=assets["person-0.jpg"]["id"],
                face_id="face-confirmed",
                vector=vector,
            )
            manifest.assign_face_identity(db_path, "face-confirmed", confirmed_id)

            manifest.replace_faces_for_asset(
                db_path,
                assets["person-1.jpg"]["id"],
                [{"id": "face-cluster", "bbox": {}, "embedding_vector": vector}],
                source_name="test",
            )
            manifest.save_face_embedding(
                db_path,
                asset_id=assets["person-1.jpg"]["id"],
                face_id="face-cluster",
                vector=vector,
            )
            manifest.update_face_cluster(db_path, "face-cluster", cluster_id, labeled=False)

            loose_faces = manifest.list_person_candidate_faces(db_path, confirmed_id, limit=10)
            related_clusters = manifest.list_related_identity_candidates(db_path, confirmed_id, limit=10, threshold=0.7)

            self.assertEqual(loose_faces, [])
            self.assertEqual(len(related_clusters), 1)
            self.assertEqual(related_clusters[0]["id"], cluster_id)

    def test_merge_similar_cluster_identities_merges_sibling_clusters_with_cross_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            assets = self._create_face_assets(root, db_path, "sibling", 20)
            cluster_a = manifest.create_face_identity(db_path, "Cluster A", status="CLUSTERED")
            cluster_b = manifest.create_face_identity(db_path, "Cluster B", status="CLUSTERED")

            angles_a = [0.0] * 8 + [12.0] * 2
            angles_b = [42.0] * 8 + [30.0] * 2
            for idx, angle in enumerate(angles_a):
                self._cluster_face(
                    db_path,
                    assets[idx],
                    f"face-a-{idx}",
                    self._unit_vector(angle),
                    cluster_a,
                )
            for idx, angle in enumerate(angles_b):
                self._cluster_face(
                    db_path,
                    assets[idx + 10],
                    f"face-b-{idx}",
                    self._unit_vector(angle),
                    cluster_b,
                )

            result = faces.merge_similar_cluster_identities(db_path, similarity_threshold=0.7)
            identities = manifest.list_face_identities(db_path, limit=10, status="CLUSTERED")
            labeled_faces = manifest.list_faces(db_path, limit=25, include_rejected=True)

            self.assertEqual(result["merged"], 1)
            self.assertEqual(len(identities), 1)
            self.assertEqual({row["identity_id"] for row in labeled_faces}, {identities[0]["id"]})

    def test_merge_similar_cluster_identities_does_not_chain_bridge_clusters_into_blob(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            manifest.init_db(db_path)

            assets = self._create_face_assets(root, db_path, "bridge", 30)
            cluster_a = manifest.create_face_identity(db_path, "Cluster A", status="CLUSTERED")
            cluster_b = manifest.create_face_identity(db_path, "Cluster B", status="CLUSTERED")
            cluster_c = manifest.create_face_identity(db_path, "Cluster C", status="CLUSTERED")

            cluster_specs = [
                (cluster_a, 0.0, 0),
                (cluster_b, 35.0, 10),
                (cluster_c, 70.0, 20),
            ]
            for identity_id, angle, offset in cluster_specs:
                for idx in range(10):
                    self._cluster_face(
                        db_path,
                        assets[offset + idx],
                        f"face-{offset + idx}",
                        self._unit_vector(angle),
                        identity_id,
                    )

            result = faces.merge_similar_cluster_identities(db_path, similarity_threshold=0.7)
            identities = manifest.list_face_identities(db_path, limit=10, status="CLUSTERED")
            labeled_faces = manifest.list_faces(db_path, limit=35, include_rejected=True)
            remaining_cluster_sizes = sorted(
                sum(1 for face in labeled_faces if face["identity_id"] == identity["id"])
                for identity in identities
            )

            self.assertEqual(result["merged"], 1)
            self.assertEqual(len(identities), 2)
            self.assertEqual(remaining_cluster_sizes, [10, 20])

    def test_detect_faces_prefers_insightface_model_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            managed.mkdir()
            manifest.init_db(db_path)

            source = root / "portrait.jpg"
            Image.new("RGB", (200, 200), color="white").save(source)
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
                        "dt_original": "2024-01-02T03:04:05",
                        "src_mtime": "2024-01-02T03:04:05",
                    }
                ],
                db_path,
            )
            manifest.plan_targets(db_path)
            asset = manifest.list_assets(db_path, limit=1)[0]
            managed_path = managed / asset["managed_path"]
            managed_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (200, 200), color="white").save(managed_path)
            manifest.mark_copied(db_path, asset["id"], "abc123")

            with patch("literoom.faces._ensure_insightface_ready") as fake_ready, patch(
                "literoom.faces._insightface_detections",
                return_value=[
                    {
                        "bbox": {
                            "x": 10,
                            "y": 20,
                            "w": 50,
                            "h": 60,
                            "image_width": 200,
                            "image_height": 200,
                            "quality": 88.0,
                        },
                        "embedding_vector": [0.5] * 168,
                    }
                ],
            ) as mock_insightface, patch("literoom.faces._cv2") as fake_cv2:
                fake_cv2.return_value.imread.return_value = __import__("numpy").zeros((200, 200, 3), dtype="uint8")
                result = faces.detect_faces(db_path, managed, face_model="antelopev2")

            self.assertEqual(result["detected"], 1)
            self.assertTrue(mock_insightface.called)
            self.assertTrue(fake_ready.called)


if __name__ == "__main__":
    unittest.main()
