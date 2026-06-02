from __future__ import annotations

from pathlib import Path

from literoom.desktop import launch_desktop_app


def _bundle_config_path() -> Path:
    return Path.home() / "Library" / "Application Support" / "Literoom" / "literoom.local.yaml"


def main() -> None:
    config_path = _bundle_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    launch_desktop_app(config_path, host="127.0.0.1", port=8000, open_browser=True)


if __name__ == "__main__":
    main()
