from __future__ import annotations

import io
import json
import unittest
from unittest import mock

try:
    from literoom.updates import UpdateStatus, check_for_updates

    UPDATES_AVAILABLE = True
except Exception:
    UPDATES_AVAILABLE = False
    check_for_updates = None  # type: ignore[assignment]
    UpdateStatus = None  # type: ignore[assignment]


class _DummyResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@unittest.skipUnless(UPDATES_AVAILABLE, "update helpers are not installed")
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

if __name__ == "__main__":
    unittest.main()
