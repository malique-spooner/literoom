from __future__ import annotations

import webbrowser
from pathlib import Path

import uvicorn

from literoom.api import create_app


def _bundle_config_path() -> Path:
    return Path.home() / "Library" / "Application Support" / "Literoom" / "literoom.local.yaml"


def main() -> None:
    config_path = _bundle_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    webbrowser.open("http://127.0.0.1:8000/")
    uvicorn.run(create_app(config_path), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
