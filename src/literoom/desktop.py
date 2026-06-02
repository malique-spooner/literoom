from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

_LOCAL_BROWSER_HOST = "literoom.localhost"


def _browser_url(host: str, port: int) -> str:
    if host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}:
        return f"http://{_LOCAL_BROWSER_HOST}:{port}/"
    return f"http://{host}:{port}/"


def launch_desktop_app(config_path: Path, *, host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("uvicorn is required for the desktop launcher") from exc

    from .api import create_app

    if open_browser:
        webbrowser.open(_browser_url(host, port))
    uvicorn.run(create_app(config_path), host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="literoom-desktop")
    parser.add_argument("--config", default="literoom.local.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--no-open-browser", action="store_true")
    args = parser.parse_args()
    launch_desktop_app(
        Path(args.config),
        host=args.host,
        port=args.port,
        open_browser=not args.no_open_browser,
    )
