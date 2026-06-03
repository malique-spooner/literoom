from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from fastapi.testclient import TestClient
    from literoom.api import create_app
    from literoom.updates import UpdateStatus, check_for_updates

    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False
    TestClient = None  # type: ignore[assignment]
    create_app = None  # type: ignore[assignment]
    check_for_updates = None  # type: ignore[assignment]
    UpdateStatus = None  # type: ignore[assignment]


class _DummyResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class UpdateTests(unittest.TestCase):
    def test_check_for_updates_detects_new_release(self):
        payload = {
            "tag_name": "v1.0.1",
            "html_url": "https://github.com/malique-spooner/literoom/releases/tag/v1.0.1",
            "assets": [
                {
                    "name": "Literoom.zip",
                    "browser_download_url": "https://github.com/malique-spooner/literoom/releases/download/v1.0.1/Literoom.zip",
                }
            ],
        }
        with mock.patch("literoom.updates.urlopen", return_value=_DummyResponse(json.dumps(payload).encode("utf-8"))):
            status = check_for_updates(current_version="1.0.0")

        self.assertTrue(status.available)
        self.assertEqual(status.current_version, "v1.0.0")
        self.assertEqual(status.latest_version, "v1.0.1")
        self.assertIn("Update available", status.message)
        self.assertIn("Literoom.zip", status.download_url)

    def test_check_for_updates_handles_up_to_date_release(self):
        payload = {
            "tag_name": "v1.0.0",
            "html_url": "https://github.com/malique-spooner/literoom/releases/tag/v1.0.0",
            "assets": [],
        }
        with mock.patch("literoom.updates.urlopen", return_value=_DummyResponse(json.dumps(payload).encode("utf-8"))):
            status = check_for_updates(current_version="1.0.0")

        self.assertFalse(status.available)
        self.assertEqual(status.latest_version, "v1.0.0")
        self.assertIn("up to date", status.message)

    def test_update_page_renders_release_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            config_path = root / "config.yaml"
            import_dir = root / "imports"
            managed_dir = root / "library"
            derived_dir = root / "previews"
            import_dir.mkdir()
            managed_dir.mkdir()
            derived_dir.mkdir()
            config_path.write_text(
                "\n".join(
                    [
                        "workspace_root: .",
                        "sources:",
                        f"  - {import_dir}",
                        "paths:",
                        f"  db_path: {db_path}",
                        f"  managed_library_dir: {managed_dir}",
                        f"  derivatives_dir: {derived_dir}",
                        f"  logs_dir: {root / 'logs'}",
                        f"  temp_dir: {root / 'tmp'}",
                    ]
                ),
                encoding="utf-8",
            )
            with mock.patch("literoom.api.check_for_updates") as mocked_check:
                mocked_check.return_value = UpdateStatus(
                    current_version="v1.0.0",
                    latest_version="v1.0.1",
                    release_url="https://github.com/malique-spooner/literoom/releases/tag/v1.0.1",
                    download_url="https://github.com/malique-spooner/literoom/releases/download/v1.0.1/Literoom.zip",
                    available=True,
                    checked=True,
                    message="Update available: v1.0.1. Open the release page to download the latest ZIP.",
                )
                client = TestClient(create_app(config_path))
                response = client.get("/app/update")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Updates", response.text)
        self.assertIn("Download latest ZIP", response.text)
        self.assertIn("v1.0.1", response.text)


if __name__ == "__main__":
    unittest.main()
