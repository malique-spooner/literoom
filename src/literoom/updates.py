from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .version import APP_VERSION

_DEFAULT_REPO = "malique-spooner/literoom"
_GITHUB_RELEASES_API = "https://api.github.com/repos/{repo}/releases/latest"
_VERSION_RE = re.compile(r"^v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:[-+].*)?$")


@dataclass(frozen=True)
class UpdateStatus:
    current_version: str
    latest_version: Optional[str]
    release_url: Optional[str]
    download_url: Optional[str]
    available: bool
    checked: bool
    message: str
    error: Optional[str] = None


def _normalize_tag(value: str) -> str:
    cleaned = value.strip()
    return cleaned if cleaned.startswith("v") else f"v{cleaned}"


def _version_tuple(value: str) -> tuple[int, int, int] | None:
    match = _VERSION_RE.match(value.strip())
    if not match:
        return None
    return tuple(int(match.group(part)) for part in ("major", "minor", "patch"))


def _select_download_url(payload: dict[str, Any]) -> Optional[str]:
    assets = payload.get("assets") or []
    for asset in assets:
        name = str(asset.get("name") or "").lower()
        if name == "literoom.zip":
            return str(asset.get("browser_download_url") or asset.get("url") or "")
    if assets:
        asset = assets[0]
        return str(asset.get("browser_download_url") or asset.get("url") or "")
    return None


def fetch_latest_release(repo_full_name: str = _DEFAULT_REPO, timeout: float = 5.0) -> dict[str, Any]:
    request = Request(
        _GITHUB_RELEASES_API.format(repo=repo_full_name),
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"Literoom/{APP_VERSION}",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Unexpected response from GitHub releases API.")
    return payload


def check_for_updates(
    current_version: str = APP_VERSION,
    repo_full_name: str = _DEFAULT_REPO,
    timeout: float = 5.0,
) -> UpdateStatus:
    current_version = current_version.strip()
    current_tag = _normalize_tag(current_version)
    try:
        payload = fetch_latest_release(repo_full_name=repo_full_name, timeout=timeout)
        latest_tag = str(payload.get("tag_name") or payload.get("name") or "").strip()
        latest_version = latest_tag.lstrip("v") if latest_tag else None
        release_url = str(payload.get("html_url") or "").strip() or None
        download_url = _select_download_url(payload)
        current_tuple = _version_tuple(current_version)
        latest_tuple = _version_tuple(latest_version or "") if latest_version else None
        available = bool(current_tuple and latest_tuple and latest_tuple > current_tuple)
        if available:
            message = f"Update available: {_normalize_tag(latest_version or latest_tag)}. Open the release page to download the latest ZIP."
        else:
            message = f"You are up to date on {current_tag}."
        return UpdateStatus(
            current_version=current_tag,
            latest_version=_normalize_tag(latest_version) if latest_version else None,
            release_url=release_url,
            download_url=download_url,
            available=available,
            checked=True,
            message=message,
        )
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return UpdateStatus(
            current_version=current_tag,
            latest_version=None,
            release_url=None,
            download_url=None,
            available=False,
            checked=False,
            message="Unable to check for updates right now.",
            error=str(exc),
        )
