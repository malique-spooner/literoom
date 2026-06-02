from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from literoom.desktop import launch_desktop_app


class DesktopLauncherTests(TestCase):
    def test_launches_browser_on_friendlier_local_url(self) -> None:
        fake_uvicorn = SimpleNamespace(run=Mock())

        with patch("literoom.api.create_app", return_value="app"), \
            patch("literoom.desktop.webbrowser.open") as open_mock, \
            patch.dict(sys.modules, {"uvicorn": fake_uvicorn}):
            launch_desktop_app(Path("literoom.local.yaml"), host="127.0.0.1", port=8000, open_browser=True)

        open_mock.assert_called_once_with("http://literoom.localhost:8000/")
        fake_uvicorn.run.assert_called_once_with("app", host="127.0.0.1", port=8000)

    def test_preserves_custom_host_when_not_loopback(self) -> None:
        fake_uvicorn = SimpleNamespace(run=Mock())

        with patch("literoom.api.create_app", return_value="app"), \
            patch("literoom.desktop.webbrowser.open") as open_mock, \
            patch.dict(sys.modules, {"uvicorn": fake_uvicorn}):
            launch_desktop_app(Path("literoom.local.yaml"), host="10.0.0.5", port=8123, open_browser=True)

        open_mock.assert_called_once_with("http://10.0.0.5:8123/")
        fake_uvicorn.run.assert_called_once_with("app", host="10.0.0.5", port=8123)
