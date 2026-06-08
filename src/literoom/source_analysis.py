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
    db_path: Path | None = None,
    source_hint: Optional[str] = None,
    limit: Optional[int] = None,
    job_id: Optional[str] = None,
) -> dict[str, Any]:
    source_paths = [Path(item).expanduser().resolve() for item in paths]
    media_counts = Counter()
    basename_counts = Counter()
    issues: list[dict[str, str]] = []
    total_media = 0
    selected_sources = 0
    stats: dict[str, Any] = {
        "stage": "source_analysis",
        "selected_sources": 0,
        "total_sources": len(source_paths),
        "processed_assets": 0,
        "media_files": 0,
        "image_files": 0,
        "video_files": 0,
        "raw_files": 0,
        "other_files": 0,
        "source_issues": issues,
        "detail": "Scanning import folders",
        "target_limit": limit,
    }

    def record_progress() -> None:
        if job_id and db_path is not None:
            gather.manifest.attach_job_metrics(Path(db_path), job_id, stats)

    for raw_path in source_paths:
        if limit is not None and total_media >= limit:
            break
        if not raw_path.exists():
            _record_issue(issues, raw_path, FileNotFoundError("missing source"))
            stats["source_issues"] = issues
            stats["detail"] = f"Missing: {raw_path.name}"
            record_progress()
            continue
        selected_sources += 1
        stats["selected_sources"] = selected_sources
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
            stats["processed_assets"] = total_media
            stats["media_files"] = total_media
            stats["image_files"] = media_counts.get("image", 0)
            stats["video_files"] = media_counts.get("video", 0)
            stats["raw_files"] = media_counts.get("raw", 0)
            stats["other_files"] = media_counts.get("other", 0)
            stats["detail"] = f"{total_media:,} files seen"
            if total_media % 100 == 0:
                record_progress()
        record_progress()

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
