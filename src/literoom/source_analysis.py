from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

from .metadata import gather


def _record_issue(issues: list[dict[str, str]], path: Path, exc: Exception) -> None:
    if len(issues) >= 20:
        return
    issues.append({"path": str(path), "error": str(exc)})


def analyze_sources(
    paths: Iterable[Path | str],
    *,
    source_hint: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    source_paths = [Path(item).expanduser().resolve() for item in paths]
    media_counts = Counter()
    basename_counts = Counter()
    issues: list[dict[str, str]] = []
    total_media = 0
    selected_sources = 0

    for raw_path in source_paths:
        if limit is not None and total_media >= limit:
            break
        if not raw_path.exists():
            _record_issue(issues, raw_path, FileNotFoundError("missing source"))
            continue
        selected_sources += 1
        if raw_path.is_dir():
            iterator = gather.scan_directory(raw_path, source_hint=source_hint, on_error=lambda p, e: _record_issue(issues, p, e))
        else:
            iterator = gather.scan_file(raw_path, source_hint=source_hint, on_error=lambda p, e: _record_issue(issues, p, e))
        for row in iterator:
            if limit is not None and total_media >= limit:
                break
            media_type = str(row.get("media_type") or "other").lower()
            media_counts[media_type] += 1
            basename = str(row.get("orig_filename") or "").strip().lower()
            if basename:
                basename_counts[basename] += 1
            total_media += 1

    duplicate_basename_groups = sum(1 for count in basename_counts.values() if count > 1)
    duplicate_basename_items = sum(count - 1 for count in basename_counts.values() if count > 1)
    repeated_basenames = [
        {"name": name, "count": count}
        for name, count in basename_counts.most_common()
        if count > 1
    ][:20]

    return {
        "stage": "source_analysis",
        "selected_sources": selected_sources,
        "media_files": total_media,
        "image_files": media_counts.get("image", 0),
        "video_files": media_counts.get("video", 0),
        "raw_files": media_counts.get("raw", 0),
        "other_files": media_counts.get("other", 0),
        "duplicate_basename_groups": duplicate_basename_groups,
        "duplicate_basename_items": duplicate_basename_items,
        "repeated_basenames": repeated_basenames,
        "source_issues": issues,
    }
