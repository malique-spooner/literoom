from __future__ import annotations

import json
import re
import zipfile
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from ..utils.location import parse_location_candidate


SNAPCHAT_HISTORY_JSON = "json/memories_history.json"
SNAPCHAT_HISTORY_HTML = "html/memories_history.html"
SNAPCHAT_MEDIA_ROOT = "memories/"

_SNAPCHAT_MAIN_RE = re.compile(r"^memories/(?P<day>\d{4}-\d{2}-\d{2})_[^/]+-main\.(?P<ext>jpg|jpeg|mp4|mov)$", re.IGNORECASE)
_SNAPCHAT_OVERLAY_RE = re.compile(r"^memories/(?P<day>\d{4}-\d{2}-\d{2})_[^/]+-overlay\.png$", re.IGNORECASE)


def looks_like_snapchat_export(path: Path, inner_names_sample: Iterable[str] = ()) -> bool:
    name = path.name.lower()
    if "snapchat" in name or name.startswith("mydata~"):
        return True
    inner = " ".join(n.lower() for n in list(inner_names_sample)[:20])
    return any(token in inner for token in ("memories_history.json", "memories_history.html", "memories/"))


def is_snapchat_main_media(inner_path: str) -> bool:
    return bool(_SNAPCHAT_MAIN_RE.match(inner_path))


def is_snapchat_overlay(inner_path: str) -> bool:
    return bool(_SNAPCHAT_OVERLAY_RE.match(inner_path))


def _parse_snapchat_datetime(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        return datetime.fromisoformat(text)
    except Exception:
        return None


def load_history_rows_from_zip(zp: zipfile.ZipFile) -> list[dict[str, Any]]:
    try:
        raw = zp.read(SNAPCHAT_HISTORY_JSON).decode("utf-8", errors="replace")
    except Exception:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    rows = data.get("Saved Media")
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        dt = _parse_snapchat_datetime(row.get("Date"))
        location = parse_location_candidate(row.get("Location"))
        media_type = str(row.get("Media Type") or "").strip().lower()
        normalized.append(
            {
                "captured_at_utc": dt.isoformat(timespec="seconds") if dt else None,
                "captured_date": dt.date().isoformat() if dt else None,
                "media_type": media_type,
                "location": location,
                "raw": row,
            }
        )
    return normalized


@lru_cache(maxsize=32)
def load_history_rows_for_export(zip_path: str) -> list[dict[str, Any]]:
    path = Path(zip_path)
    if not path.exists():
        return []
    candidates = [path]
    candidates.extend(sorted(path.parent.glob("*.zip")))
    for candidate in candidates:
        try:
            with zipfile.ZipFile(candidate, "r") as zp:
                rows = load_history_rows_from_zip(zp)
                if rows:
                    return rows
        except Exception:
            continue
    return []


def group_rows_by_day_type(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("captured_date") or "", row.get("media_type") or "")
        grouped.setdefault(key, []).append(row)
    return grouped


def order_rows_for_snapchat_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get("captured_at_utc") or "",
            json.dumps(row.get("raw") or {}, sort_keys=True, ensure_ascii=False),
        ),
        reverse=True,
    )


def row_for_exact_timestamp(rows: list[dict[str, Any]], dt_text: str) -> Optional[dict[str, Any]]:
    if not dt_text:
        return None
    needle = str(dt_text).replace("+00:00", "Z")[:19]
    for row in rows:
        captured = row.get("captured_at_utc") or ""
        if captured[:19] == needle:
            return row
    return None
