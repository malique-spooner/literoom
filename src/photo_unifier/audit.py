from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


_WEAK_TIMESTAMP_SOURCES = {
    "local": {"file_mtime", "filename"},
    "apple": {"file_mtime", "zip_mtime", "filename"},
    "google": {"filename", "zip_mtime", "file_mtime"},
    "snapchat": {"filename", "file_mtime", None},
}


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _evenly_sample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    if limit == 1:
        return [rows[len(rows) // 2]]
    picks: list[dict[str, Any]] = []
    seen: set[int] = set()
    for i in range(limit):
        idx = round(i * (len(rows) - 1) / (limit - 1))
        if idx not in seen:
            picks.append(rows[idx])
            seen.add(idx)
    if len(picks) < limit:
        for idx, row in enumerate(rows):
            if idx in seen:
                continue
            picks.append(row)
            seen.add(idx)
            if len(picks) >= limit:
                break
    return picks[:limit]


def _compact_counter(values: Iterable[Any]) -> dict[str, int]:
    counter = Counter("None" if value in (None, "") else str(value) for value in values)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _clean_people_json(raw: Any) -> bool:
    if raw in (None, "", "[]", "null", "{}"):
        return False
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return bool(raw)
    return bool(payload)


def _row_issues(row: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    source = row.get("source") or "unknown"
    source_dt = row.get("source_dt")
    dt_original = row.get("dt_original")
    gps_present = row.get("gps_lat") is not None and row.get("gps_lon") is not None
    title = row.get("title")
    description = row.get("description")
    people_json = row.get("people_json")

    if not dt_original:
        issues.append("missing timestamp")
    elif source_dt in _WEAK_TIMESTAMP_SOURCES.get(source, set()):
        issues.append(f"weak timestamp source ({source_dt or 'missing'})")

    if source == "snapchat":
        if source_dt != "snapchat_history":
            issues.append(f"unexpected Snapchat timestamp source ({source_dt or 'missing'})")
        if not gps_present:
            issues.append("missing Snapchat GPS")
    elif source == "google":
        if source_dt not in {
            "google_epoch",
            "google_formatted",
            "createdate",
            "datetimeoriginal",
            "exif_datetime",
            "filename",
            "zip_mtime",
            "file_mtime",
        }:
            issues.append(f"unexpected Google timestamp source ({source_dt or 'missing'})")
    elif source == "apple":
        if source_dt not in {
            "apple_csv",
            "createdate",
            "datetimeoriginal",
            "exif_datetime",
            "filename",
            "zip_mtime",
            "file_mtime",
        }:
            issues.append(f"unexpected Apple timestamp source ({source_dt or 'missing'})")
    elif source == "local":
        if source_dt not in {
            "exif_datetime",
            "datetimeoriginal",
            "createdate",
            "apple_csv_gmt",
            "file_mtime",
            "filename",
        }:
            issues.append(f"unexpected local timestamp source ({source_dt or 'missing'})")
    return issues


def build_audit_report(db_path: Path, *, sample_size: int = 100, sources: Optional[Iterable[str]] = None) -> dict[str, Any]:
    con = _connect(db_path)
    try:
        source_filter = [str(item) for item in sources] if sources else None
        if source_filter:
            placeholders = ",".join("?" for _ in source_filter)
            asset_rows = [dict(r) for r in con.execute(
                f"""
                SELECT source, source_kind, COUNT(*) AS assets,
                       SUM(CASE WHEN dt_original IS NOT NULL AND dt_original <> '' THEN 1 ELSE 0 END) AS dt_recovered,
                       SUM(CASE WHEN gps_lat IS NOT NULL AND gps_lon IS NOT NULL THEN 1 ELSE 0 END) AS gps_recovered,
                       SUM(CASE WHEN title IS NOT NULL AND title <> '' THEN 1 ELSE 0 END) AS title_recovered,
                       SUM(CASE WHEN description IS NOT NULL AND description <> '' THEN 1 ELSE 0 END) AS description_recovered,
                       SUM(CASE WHEN people_json IS NOT NULL AND people_json NOT IN ('', '[]', 'null', '{{}}') THEN 1 ELSE 0 END) AS people_recovered,
                       SUM(CASE WHEN source_dt IS NOT NULL AND source_dt <> '' THEN 1 ELSE 0 END) AS source_dt_set
                FROM assets
                WHERE source IN ({placeholders})
                GROUP BY source, source_kind
                ORDER BY source, source_kind
                """,
                source_filter,
            ).fetchall()]
        else:
            asset_rows = [dict(r) for r in con.execute(
                """
                SELECT source, source_kind, COUNT(*) AS assets,
                       SUM(CASE WHEN dt_original IS NOT NULL AND dt_original <> '' THEN 1 ELSE 0 END) AS dt_recovered,
                       SUM(CASE WHEN gps_lat IS NOT NULL AND gps_lon IS NOT NULL THEN 1 ELSE 0 END) AS gps_recovered,
                       SUM(CASE WHEN title IS NOT NULL AND title <> '' THEN 1 ELSE 0 END) AS title_recovered,
                       SUM(CASE WHEN description IS NOT NULL AND description <> '' THEN 1 ELSE 0 END) AS description_recovered,
                       SUM(CASE WHEN people_json IS NOT NULL AND people_json NOT IN ('', '[]', 'null', '{}') THEN 1 ELSE 0 END) AS people_recovered,
                       SUM(CASE WHEN source_dt IS NOT NULL AND source_dt <> '' THEN 1 ELSE 0 END) AS source_dt_set
                FROM assets
                GROUP BY source, source_kind
                ORDER BY source, source_kind
                """
            ).fetchall()]

        face_rows = [dict(r) for r in con.execute(
            """
            SELECT a.source,
                   COUNT(*) AS faces,
                   SUM(CASE WHEN f.status = 'LABELED' THEN 1 ELSE 0 END) AS labeled,
                   SUM(CASE WHEN f.status = 'REJECTED' THEN 1 ELSE 0 END) AS rejected,
                   SUM(CASE WHEN f.status = 'DETECTED' THEN 1 ELSE 0 END) AS detected,
                   COUNT(DISTINCT f.asset_id) AS assets_with_faces,
                   COUNT(DISTINCT f.identity_id) AS identity_ids
            FROM faces f
            JOIN assets a ON a.id = f.asset_id
            GROUP BY a.source
            ORDER BY a.source
            """
        ).fetchall()]

        samples: dict[str, dict[str, Any]] = {}
        for row in asset_rows:
            source = row["source"]
            if source_filter and source not in source_filter:
                continue
            rows = [dict(r) for r in con.execute(
                """
                SELECT id, source, source_kind, media_type, orig_filename, source_path,
                       source_locator, dt_original, source_dt, gps_lat, gps_lon, gps_alt,
                       title, description, people_json, status
                FROM assets
                WHERE source = ?
                ORDER BY COALESCE(dt_original, ''), COALESCE(orig_filename, ''), id
                """,
                (source,),
            ).fetchall()]
            sample = _evenly_sample(rows, sample_size)
            sample_issues: list[dict[str, Any]] = []
            for item in sample:
                issues = _row_issues(item)
                if issues:
                    sample_issues.append(
                        {
                            "orig_filename": item.get("orig_filename"),
                            "source_path": item.get("source_path"),
                            "source_dt": item.get("source_dt"),
                            "issues": issues,
                        }
                    )
            samples[source] = {
                "sample_size": len(sample),
                "timestamp_recovered": sum(1 for item in sample if item.get("dt_original")),
                "gps_recovered": sum(1 for item in sample if item.get("gps_lat") is not None and item.get("gps_lon") is not None),
                "title_recovered": sum(1 for item in sample if item.get("title")),
                "description_recovered": sum(1 for item in sample if item.get("description")),
                "people_recovered": sum(1 for item in sample if item.get("people_json") not in (None, "", "[]", "null", "{}")),
                "media_type_breakdown": _compact_counter(item.get("media_type") for item in sample),
                "source_dt_breakdown": _compact_counter(item.get("source_dt") for item in sample),
                "issues": sample_issues[:10],
            }

        totals = {
            "assets": sum(int(row["assets"] or 0) for row in asset_rows),
            "faces": sum(int(row["faces"] or 0) for row in face_rows),
            "face_identities": con.execute("SELECT COUNT(*) FROM face_identities").fetchone()[0],
            "duplicate_groups": con.execute("SELECT COUNT(*) FROM duplicate_groups").fetchone()[0],
        }
        return {
            "db_path": str(db_path),
            "totals": totals,
            "assets_by_source": asset_rows,
            "faces_by_source": face_rows,
            "samples": samples,
        }
    finally:
        con.close()


def format_audit_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Photo Unifier audit")
    lines.append(f"Database: {report['db_path']}")
    totals = report["totals"]
    lines.append(
        f"Totals: assets={totals['assets']} faces={totals['faces']} face_identities={totals['face_identities']} duplicate_groups={totals['duplicate_groups']}"
    )
    lines.append("")

    lines.append("Coverage by source")
    for row in report["assets_by_source"]:
        source = row["source"]
        source_kind = row["source_kind"]
        lines.append(
            f"- {source} ({source_kind}): assets={row['assets']}, timestamps={row['dt_recovered']}/{row['assets']}, "
            f"gps={row['gps_recovered']}/{row['assets']}, titles={row['title_recovered']}/{row['assets']}, "
            f"descriptions={row['description_recovered']}/{row['assets']}, people={row['people_recovered']}/{row['assets']}"
        )
        sample = report["samples"].get(source, {})
        if sample:
            lines.append(
                f"  sample={sample['sample_size']} timestamps={sample['timestamp_recovered']}/{sample['sample_size']} gps={sample['gps_recovered']}/{sample['sample_size']} "
                f"titles={sample['title_recovered']}/{sample['sample_size']} descriptions={sample['description_recovered']}/{sample['sample_size']} people={sample['people_recovered']}/{sample['sample_size']}"
            )
            lines.append(
                f"  sample source_dt: {', '.join(f'{k}={v}' for k, v in sample['source_dt_breakdown'].items()) or 'none'}"
            )
            if sample["issues"]:
                lines.append("  potential inaccuracies:")
                for issue in sample["issues"]:
                    lines.append(
                        f"    - {issue['orig_filename']}: {', '.join(issue['issues'])}"
                    )
            else:
                lines.append("  potential inaccuracies: none in sample")
    lines.append("")

    lines.append("Faces by source")
    for row in report["faces_by_source"]:
        lines.append(
            f"- {row['source']}: faces={row['faces']}, labeled={row['labeled']}, rejected={row['rejected']}, detected={row['detected']}, assets_with_faces={row['assets_with_faces']}, identity_ids={row['identity_ids']}"
        )
    return "\n".join(lines)
