from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from photo_unifier import faces
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
            with patch("photo_unifier.faces._detect_boxes", return_value=[(10, 20, 40, 40)]), patch(
                "photo_unifier.faces._cv2"
            ) as fake_cv2, patch(
                "photo_unifier.faces._face_quality", return_value=99.0
            ), patch(
                "photo_unifier.faces._compute_embedding", return_value=[0.1] * 272
            ):
                fake_cv2.return_value.imread.return_value = fake_image
                result = faces.detect_faces(db_path, managed)

            self.assertEqual(result["detected"], 2)
            detected = manifest.list_faces(db_path, limit=10)
            self.assertEqual(len(detected), 2)
            self.assertTrue(any(row["identity_label"].startswith("Cluster ") for row in detected))

            config_path = root / "config.yaml"
            self._write_config(config_path, db_path, managed, derived, root)
            client = TestClient(create_app(config_path))
            response = client.get(
                f"/app/faces/assign?face_id={detected[0]['id']}&identity_label=Ethan",
                follow_redirects=False,
            )
            self.assertEqual(response.status_code, 303)
            updated = manifest.list_faces(db_path, limit=10)[0]
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
            self.assertEqual(visible, [])

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

    def test_cluster_faces_groups_matching_faces_across_assets(self):
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

            self.assertEqual(result["clusters_created"], 1)
            self.assertEqual(len(identities), 1)
            self.assertEqual({row["identity_id"] for row in labeled_faces}, {identities[0]["id"]})


if __name__ == "__main__":
    unittest.main()
