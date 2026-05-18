from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


JOB_STATES = ("QUEUED", "RUNNING", "COMPLETED", "FAILED", "RETRYABLE")
MANAGED_STATES = ("PLANNED", "BUILT", "MISSING", "FAILED")

CANONICAL_METADATA_STANDARD: List[Dict[str, Any]] = [
    {"field_name": "captured_at", "label": "Captured Time", "weight": 24, "media_types": ["image", "raw", "video"]},
    {"field_name": "captured_at_utc", "label": "Captured Time (UTC)", "weight": 8, "media_types": ["image", "raw", "video"]},
    {"field_name": "timezone_name", "label": "Timezone", "weight": 6, "media_types": ["image", "raw", "video"]},
    {"field_name": "location", "label": "Location", "weight": 18, "media_types": ["image", "raw", "video"]},
    {"field_name": "title", "label": "Title", "weight": 6, "media_types": ["image", "raw", "video"]},
    {"field_name": "description", "label": "Description", "weight": 6, "media_types": ["image", "raw", "video"]},
    {"field_name": "keywords", "label": "Keywords", "weight": 6, "media_types": ["image", "raw", "video"]},
    {"field_name": "people", "label": "People", "weight": 12, "media_types": ["image", "raw", "video"]},
    {"field_name": "camera_make", "label": "Camera Make", "weight": 5, "media_types": ["image", "raw", "video"]},
    {"field_name": "camera_model", "label": "Camera Model", "weight": 5, "media_types": ["image", "raw", "video"]},
    {"field_name": "software", "label": "Software", "weight": 3, "media_types": ["image", "raw", "video"]},
    {"field_name": "pixel_width", "label": "Width", "weight": 3, "media_types": ["image", "raw", "video"]},
    {"field_name": "pixel_height", "label": "Height", "weight": 3, "media_types": ["image", "raw", "video"]},
    {"field_name": "format", "label": "Format", "weight": 3, "media_types": ["image", "raw", "video"]},
    {"field_name": "duration_seconds", "label": "Duration", "weight": 7, "media_types": ["video"]},
    {"field_name": "frame_rate", "label": "Frame Rate", "weight": 4, "media_types": ["video"]},
    {"field_name": "bitrate", "label": "Bitrate", "weight": 3, "media_types": ["video"]},
    {"field_name": "video_codec", "label": "Video Codec", "weight": 4, "media_types": ["video"]},
    {"field_name": "audio_codec", "label": "Audio Codec", "weight": 3, "media_types": ["video"]},
    {"field_name": "audio_channels", "label": "Audio Channels", "weight": 2, "media_types": ["video"]},
    {"field_name": "container_format", "label": "Container", "weight": 2, "media_types": ["video"]},
    {"field_name": "encoder", "label": "Encoder", "weight": 2, "media_types": ["video"]},
    {"field_name": "mime_type", "label": "MIME Type", "weight": 2, "media_types": ["image", "raw", "video"]},
    {"field_name": "lens_model", "label": "Lens Model", "weight": 4, "media_types": ["image", "raw"]},
]


def utcnow_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def init_db(db_path: Path) -> None:
    con = connect(db_path)
    try:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS assets (
              id TEXT PRIMARY KEY,
              source TEXT NOT NULL,
              abs_zip TEXT NOT NULL,
              zip_path TEXT NOT NULL,
              source_kind TEXT DEFAULT 'zip',
              source_root TEXT,
              source_locator TEXT,
              source_path TEXT,
              media_type TEXT,
              orig_filename TEXT,
              orig_ext TEXT,
              orig_size INTEGER,
              dt_original TEXT,
              gps_lat REAL, gps_lon REAL, gps_alt REAL,
              title TEXT, description TEXT,
              keywords_json TEXT, people_json TEXT,
              live_group_id TEXT, burst_id TEXT,
              status TEXT DEFAULT 'NEW',
              target_relpath TEXT,
              target_filename TEXT,
              managed_asset_id TEXT,
              sha256 TEXT,
              last_embed_hash TEXT,
              exiftool_version TEXT,
              last_updated TEXT,
              error_msg TEXT,
              src_mtime TEXT,
              source_dt TEXT,
              source_gps TEXT
            );

            CREATE UNIQUE INDEX IF NOT EXISTS ux_src_zip_path ON assets(source, abs_zip, zip_path);
            CREATE INDEX IF NOT EXISTS ix_assets_status ON assets(status);
            CREATE INDEX IF NOT EXISTS ix_assets_media_type ON assets(media_type);
            CREATE INDEX IF NOT EXISTS ix_assets_dt_original ON assets(dt_original);

            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              job_type TEXT NOT NULL,
              status TEXT NOT NULL,
              config_json TEXT,
              metrics_json TEXT,
              error_msg TEXT,
              started_at TEXT,
              finished_at TEXT,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS managed_assets (
              id TEXT PRIMARY KEY,
              asset_id TEXT NOT NULL UNIQUE REFERENCES assets(id) ON DELETE CASCADE,
              relpath TEXT NOT NULL,
              filename TEXT NOT NULL,
              managed_path TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'PLANNED',
              built_at TEXT,
              validated_at TEXT,
              error_msg TEXT
            );

            CREATE TABLE IF NOT EXISTS hashes (
              asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
              algorithm TEXT NOT NULL,
              scope TEXT NOT NULL,
              value TEXT NOT NULL,
              created_at TEXT NOT NULL,
              PRIMARY KEY (asset_id, algorithm, scope)
            );

            CREATE TABLE IF NOT EXISTS thumbnails (
              asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
              kind TEXT NOT NULL,
              path TEXT NOT NULL,
              width INTEGER,
              height INTEGER,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              error_msg TEXT,
              PRIMARY KEY (asset_id, kind)
            );

            CREATE TABLE IF NOT EXISTS duplicate_groups (
              id TEXT PRIMARY KEY,
              group_type TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'OPEN',
              canonical_asset_id TEXT REFERENCES assets(id),
              created_at TEXT NOT NULL,
              resolved_at TEXT
            );

            CREATE TABLE IF NOT EXISTS duplicate_items (
              group_id TEXT NOT NULL REFERENCES duplicate_groups(id) ON DELETE CASCADE,
              asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
              score REAL,
              rationale TEXT,
              keep_decision TEXT,
              PRIMARY KEY (group_id, asset_id)
            );

            CREATE TABLE IF NOT EXISTS face_identities (
              id TEXT PRIMARY KEY,
              label TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'UNCONFIRMED',
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS faces (
              id TEXT PRIMARY KEY,
              asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
              identity_id TEXT REFERENCES face_identities(id),
              embedding_ref TEXT,
              bbox_json TEXT,
              frame_time_ms INTEGER,
              status TEXT NOT NULL DEFAULT 'DETECTED',
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS embeddings (
              id TEXT PRIMARY KEY,
              asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
              embedding_type TEXT NOT NULL,
              model_name TEXT,
              vector_ref TEXT,
              payload_json TEXT,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS extraction_results (
              id TEXT PRIMARY KEY,
              asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
              result_type TEXT NOT NULL,
              segment_start_ms INTEGER,
              segment_end_ms INTEGER,
              text_content TEXT,
              payload_json TEXT,
              status TEXT NOT NULL DEFAULT 'PENDING',
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS metadata_fields (
              asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
              field_name TEXT NOT NULL,
              value_json TEXT NOT NULL,
              value_text TEXT,
              value_real REAL,
              source_name TEXT NOT NULL,
              source_field TEXT,
              is_canonical INTEGER NOT NULL DEFAULT 0,
              confidence REAL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (asset_id, field_name, source_name, source_field)
            );

            CREATE TABLE IF NOT EXISTS artifacts (
              asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
              artifact_type TEXT NOT NULL,
              path TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              error_msg TEXT,
              PRIMARY KEY (asset_id, artifact_type, path)
            );

            CREATE TABLE IF NOT EXISTS review_decisions (
              id TEXT PRIMARY KEY,
              asset_id TEXT REFERENCES assets(id) ON DELETE CASCADE,
              decision_type TEXT NOT NULL,
              value_json TEXT,
              created_at TEXT NOT NULL
            );
            """
        )
        con.commit()
    finally:
        con.close()
    _ensure_columns(db_path)
    backfill_missing_metadata(db_path)


def _ensure_columns(db_path: Path) -> None:
    con = connect(db_path)
    try:
        cur = con.execute("PRAGMA table_info(assets)")
        cols = {row["name"] for row in cur.fetchall()}
        wanted = (
            ("source_kind", "TEXT DEFAULT 'zip'"),
            ("source_root", "TEXT"),
            ("source_locator", "TEXT"),
            ("source_path", "TEXT"),
            ("managed_asset_id", "TEXT"),
            ("target_relpath", "TEXT"),
            ("target_filename", "TEXT"),
            ("sha256", "TEXT"),
            ("last_embed_hash", "TEXT"),
            ("exiftool_version", "TEXT"),
            ("last_updated", "TEXT"),
            ("error_msg", "TEXT"),
            ("src_mtime", "TEXT"),
            ("source_dt", "TEXT"),
            ("source_gps", "TEXT"),
        )
        for name, typ in wanted:
            if name not in cols:
                con.execute(f"ALTER TABLE assets ADD COLUMN {name} {typ}")

        con.execute(
            """
            UPDATE assets
            SET source_kind = COALESCE(source_kind, 'zip'),
                source_locator = COALESCE(source_locator, abs_zip),
                source_path = COALESCE(source_path, zip_path)
            WHERE source_kind IS NULL OR source_locator IS NULL OR source_path IS NULL
            """
        )
        con.commit()
    finally:
        con.close()


def _row_id(source: str, locator: str, source_path: str) -> str:
    payload = f"{source}|{locator}|{source_path}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def _managed_asset_id(asset_id: str) -> str:
    return hashlib.sha1(f"managed|{asset_id}".encode("utf-8")).hexdigest()[:20]


def _metadata_value_parts(value: Any) -> tuple[str, Optional[str], Optional[float]]:
    value_json = json.dumps(value, ensure_ascii=True, sort_keys=True)
    value_text = value if isinstance(value, str) else None
    value_real = float(value) if isinstance(value, (int, float)) else None
    return value_json, value_text, value_real


def _coerce_list(value: Any) -> List[Any]:
    if value in (None, "", []):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return [value]
    return [value]


def canonical_metadata_standard(media_type: Optional[str] = None) -> List[Dict[str, Any]]:
    if not media_type:
        return [dict(item) for item in CANONICAL_METADATA_STANDARD]
    return [
        dict(item)
        for item in CANONICAL_METADATA_STANDARD
        if media_type in item.get("media_types", [])
    ]


def _canonical_field_definition(field_name: str) -> Optional[Dict[str, Any]]:
    for item in CANONICAL_METADATA_STANDARD:
        if item["field_name"] == field_name:
            return dict(item)
    return None


def _record_metadata_field(
    con: sqlite3.Connection,
    *,
    asset_id: str,
    field_name: str,
    value: Any,
    source_name: str,
    source_field: Optional[str] = None,
    is_canonical: bool = False,
    confidence: Optional[float] = None,
) -> None:
    if value in (None, "", [], {}):
        return
    ts = utcnow_iso()
    value_json, value_text, value_real = _metadata_value_parts(value)
    existing_row = con.execute(
        """
        SELECT is_canonical, confidence
        FROM metadata_fields
        WHERE asset_id = ?
          AND field_name = ?
          AND source_name = ?
          AND COALESCE(source_field, '') = COALESCE(?, '')
        LIMIT 1
        """,
        (asset_id, field_name, source_name, source_field),
    ).fetchone()
    existing_canonical_confidence: Optional[float] = None
    if is_canonical:
        row = con.execute(
            """
            SELECT confidence
            FROM metadata_fields
            WHERE asset_id = ?
              AND field_name = ?
              AND is_canonical = 1
            ORDER BY
              CASE
                WHEN confidence IS NULL THEN -1
                ELSE confidence
              END DESC,
              updated_at DESC
            LIMIT 1
            """,
            (asset_id, field_name),
        ).fetchone()
        if row:
            existing_canonical_confidence = row["confidence"]
        if (
            existing_canonical_confidence is not None
            and confidence is not None
            and float(existing_canonical_confidence) > float(confidence)
        ):
            is_canonical = False
    if (
        existing_row
        and not is_canonical
        and bool(existing_row["is_canonical"])
        and (
            existing_row["confidence"] is None
            or confidence is None
            or float(existing_row["confidence"]) >= float(confidence)
        )
    ):
        return
    if is_canonical:
        con.execute(
            """
            UPDATE metadata_fields
            SET is_canonical = 0,
                updated_at = ?
            WHERE asset_id = ?
              AND field_name = ?
            """,
            (ts, asset_id, field_name),
        )
    con.execute(
        """
        INSERT INTO metadata_fields (
          asset_id, field_name, value_json, value_text, value_real,
          source_name, source_field, is_canonical, confidence, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(asset_id, field_name, source_name, source_field) DO UPDATE SET
          value_json=excluded.value_json,
          value_text=excluded.value_text,
          value_real=excluded.value_real,
          is_canonical=excluded.is_canonical,
          confidence=excluded.confidence,
          updated_at=excluded.updated_at
        """,
        (
            asset_id,
            field_name,
            value_json,
            value_text,
            value_real,
            source_name,
            source_field,
            1 if is_canonical else 0,
            confidence,
            ts,
            ts,
        ),
    )


def _sync_ingest_metadata(con: sqlite3.Connection, asset_id: str, row: Dict[str, Any]) -> None:
    provider = row.get("source") or "unknown"
    source_dt = row.get("source_dt") or "unknown"
    source_gps = row.get("source_gps") or "unknown"
    keywords = _coerce_list(row.get("keywords"))
    if not keywords:
        keywords = _coerce_list(row.get("keywords_json"))
    people = _coerce_list(row.get("people"))
    if not people:
        people = _coerce_list(row.get("people_json"))

    _record_metadata_field(
        con,
        asset_id=asset_id,
        field_name="provider",
        value=provider,
        source_name="ingest",
        source_field="source",
    )
    _record_metadata_field(
        con,
        asset_id=asset_id,
        field_name="media_type",
        value=row.get("media_type"),
        source_name="ingest",
        source_field="media_type",
    )
    _record_metadata_field(
        con,
        asset_id=asset_id,
        field_name="original_filename",
        value=row.get("orig_filename"),
        source_name="ingest",
        source_field="orig_filename",
    )
    _record_metadata_field(
        con,
        asset_id=asset_id,
        field_name="original_extension",
        value=row.get("orig_ext"),
        source_name="ingest",
        source_field="orig_ext",
    )
    _record_metadata_field(
        con,
        asset_id=asset_id,
        field_name="original_size_bytes",
        value=row.get("orig_size"),
        source_name="ingest",
        source_field="orig_size",
    )
    _record_metadata_field(
        con,
        asset_id=asset_id,
        field_name="captured_at",
        value=row.get("dt_original"),
        source_name="canonical",
        source_field=source_dt,
        is_canonical=True,
    )
    _record_metadata_field(
        con,
        asset_id=asset_id,
        field_name="source_modified_at",
        value=row.get("src_mtime"),
        source_name="ingest",
        source_field="src_mtime",
    )
    if row.get("gps_lat") is not None and row.get("gps_lon") is not None:
        _record_metadata_field(
            con,
            asset_id=asset_id,
            field_name="location",
            value={
                "lat": row.get("gps_lat"),
                "lon": row.get("gps_lon"),
                "alt": row.get("gps_alt"),
            },
            source_name="canonical",
            source_field=source_gps,
            is_canonical=True,
        )
    _record_metadata_field(
        con,
        asset_id=asset_id,
        field_name="title",
        value=row.get("title"),
        source_name="canonical",
        source_field=provider,
        is_canonical=True,
    )
    _record_metadata_field(
        con,
        asset_id=asset_id,
        field_name="description",
        value=row.get("description"),
        source_name="canonical",
        source_field=provider,
        is_canonical=True,
    )
    _record_metadata_field(
        con,
        asset_id=asset_id,
        field_name="keywords",
        value=keywords,
        source_name="canonical",
        source_field=provider,
        is_canonical=True,
    )
    _record_metadata_field(
        con,
        asset_id=asset_id,
        field_name="people",
        value=people,
        source_name="canonical",
        source_field=provider,
        is_canonical=True,
    )


def backfill_missing_metadata(db_path: Path) -> int:
    con = connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT a.*
            FROM assets a
            WHERE NOT EXISTS (
              SELECT 1
              FROM metadata_fields mf
              WHERE mf.asset_id = a.id
            )
            ORDER BY a.last_updated ASC, a.id ASC
            """
        ).fetchall()
        count = 0
        for row in rows:
            _sync_ingest_metadata(con, row["id"], dict(row))
            count += 1
        if count:
            con.commit()
        return count
    finally:
        con.close()


def upsert_raw(rows: List[Dict[str, Any]], db_path: Path, job_id: Optional[str] = None) -> int:
    _ensure_columns(db_path)
    con = connect(db_path)
    try:
        n = 0
        for r in rows:
            locator = r.get("source_locator") or r.get("abs_zip") or ""
            source_path = r.get("source_path") or r.get("zip_path") or ""
            rid = _row_id(r.get("source", ""), locator, source_path)
            con.execute(
                """
                INSERT INTO assets (
                  id, source, abs_zip, zip_path, source_kind, source_root, source_locator, source_path,
                  media_type, orig_filename, orig_ext, orig_size, dt_original,
                  gps_lat, gps_lon, gps_alt, title, description,
                  keywords_json, people_json, live_group_id, burst_id, status, src_mtime,
                  source_dt, source_gps, last_updated
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                  source_kind     = COALESCE(excluded.source_kind, assets.source_kind),
                  source_root     = COALESCE(excluded.source_root, assets.source_root),
                  source_locator  = COALESCE(excluded.source_locator, assets.source_locator),
                  source_path     = COALESCE(excluded.source_path, assets.source_path),
                  media_type      = excluded.media_type,
                  orig_filename   = excluded.orig_filename,
                  orig_ext        = excluded.orig_ext,
                  orig_size       = excluded.orig_size,
                  dt_original     = COALESCE(excluded.dt_original, assets.dt_original),
                  gps_lat         = COALESCE(excluded.gps_lat, assets.gps_lat),
                  gps_lon         = COALESCE(excluded.gps_lon, assets.gps_lon),
                  gps_alt         = COALESCE(excluded.gps_alt, assets.gps_alt),
                  title           = COALESCE(excluded.title, assets.title),
                  description     = COALESCE(excluded.description, assets.description),
                  keywords_json   = CASE WHEN excluded.keywords_json IS NOT NULL AND excluded.keywords_json <> '[]' THEN excluded.keywords_json ELSE assets.keywords_json END,
                  people_json     = CASE WHEN excluded.people_json IS NOT NULL AND excluded.people_json <> '[]' THEN excluded.people_json ELSE assets.people_json END,
                  live_group_id   = COALESCE(excluded.live_group_id, assets.live_group_id),
                  burst_id        = COALESCE(excluded.burst_id, assets.burst_id),
                  src_mtime       = excluded.src_mtime,
                  source_dt       = COALESCE(excluded.source_dt, assets.source_dt),
                  source_gps      = COALESCE(excluded.source_gps, assets.source_gps),
                  last_updated    = excluded.last_updated
                """
                ,
                (
                    rid,
                    r.get("source"),
                    r.get("abs_zip") or locator,
                    r.get("zip_path") or source_path,
                    r.get("source_kind"),
                    r.get("source_root"),
                    locator,
                    source_path,
                    r.get("media_type"),
                    r.get("orig_filename"),
                    r.get("orig_ext"),
                    r.get("orig_size"),
                    r.get("dt_original"),
                    r.get("gps_lat"),
                    r.get("gps_lon"),
                    r.get("gps_alt"),
                    r.get("title"),
                    r.get("description"),
                    json.dumps(r.get("keywords") or []),
                    json.dumps(r.get("people") or []),
                    r.get("live_group_id"),
                    r.get("burst_id"),
                    "NEW",
                    r.get("src_mtime"),
                    r.get("source_dt"),
                    r.get("source_gps"),
                    utcnow_iso(),
                ),
            )
            _sync_ingest_metadata(con, rid, r)
            n += 1

        con.commit()
        return n
    finally:
        con.close()


def _parse_filename_dt(name: str) -> Optional[datetime]:
    from datetime import datetime as dt
    import re

    patterns = [
        re.compile(r"(?P<y>20\d{2})(?P<m>\d{2})(?P<d>\d{2})[_-]?(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})"),
        re.compile(r"IMG[_-](?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})[_-](?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})"),
        re.compile(r"PXL[_-](?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})[_-](?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})"),
    ]
    base = Path(name).stem
    for pattern in patterns:
        match = pattern.search(base)
        if not match:
            continue
        try:
            return dt(
                int(match["y"]),
                int(match["m"]),
                int(match["d"]),
                int(match["H"]),
                int(match["M"]),
                int(match["S"]),
            )
        except Exception:
            continue
    return None


def plan_targets(db_path: Path, naming: str = "{YYYY}/{YYYY-MM}/{YYYYMMDD}_{hhmmss}_{shortid}") -> int:
    _ensure_columns(db_path)
    con = connect(db_path)
    try:
        rows = con.execute(
            "SELECT id, orig_filename, orig_ext, dt_original, src_mtime FROM assets"
        ).fetchall()
        planned = 0
        for row in rows:
            d: Optional[datetime] = None
            if row["dt_original"]:
                try:
                    d = datetime.strptime(str(row["dt_original"])[:19], "%Y-%m-%dT%H:%M:%S")
                except Exception:
                    d = None
            if d is None:
                d = _parse_filename_dt(row["orig_filename"] or "")
            if d is None and row["src_mtime"]:
                try:
                    d = datetime.strptime(str(row["src_mtime"])[:19], "%Y-%m-%dT%H:%M:%S")
                except Exception:
                    d = None

            if d is None:
                year, month, ymd, hms = "0000", "00", "00000000", "000000"
            else:
                year = f"{d.year:04d}"
                month = f"{d.month:02d}"
                ymd = f"{d.year:04d}{d.month:02d}{d.day:02d}"
                hms = f"{d.hour:02d}{d.minute:02d}{d.second:02d}"

            base = (
                naming.replace("{YYYY}", year)
                .replace("{MM}", month)
                .replace("{YYYY-MM}", f"{year}-{month}")
                .replace("{YYYYMMDD}", ymd)
                .replace("{hhmmss}", hms)
                .replace("{shortid}", row["id"][:8])
            )
            relpath, fname_noext = base.rsplit("/", 1) if "/" in base else ("", base)
            ext = (row["orig_ext"] or Path(row["orig_filename"] or "").suffix or "").lower()
            target_filename = f"{fname_noext}{ext}"
            managed_id = _managed_asset_id(row["id"])
            managed_path = str((Path(relpath) / target_filename).as_posix())

            con.execute(
                """
                UPDATE assets
                SET target_relpath=?, target_filename=?, managed_asset_id=?, last_updated=?
                WHERE id=?
                """,
                (relpath, target_filename, managed_id, utcnow_iso(), row["id"]),
            )
            con.execute(
                """
                INSERT INTO managed_assets (id, asset_id, relpath, filename, managed_path, status)
                VALUES (?, ?, ?, ?, ?, 'PLANNED')
                ON CONFLICT(asset_id) DO UPDATE SET
                  relpath=excluded.relpath,
                  filename=excluded.filename,
                  managed_path=excluded.managed_path,
                  status=CASE
                    WHEN managed_assets.status='BUILT' THEN managed_assets.status
                    ELSE 'PLANNED'
                  END
                """,
                (managed_id, row["id"], relpath, target_filename, managed_path),
            )
            planned += 1
        con.commit()
        return planned
    finally:
        con.close()


def iter_for_master(db_path: Path, limit: int | None = None) -> Iterator[Dict[str, Any]]:
    _ensure_columns(db_path)
    con = connect(db_path)
    try:
        sql = """
            SELECT a.*, m.id AS managed_id, m.relpath AS managed_relpath, m.filename AS managed_filename,
                   m.managed_path, m.status AS managed_status
            FROM assets a
            JOIN managed_assets m ON m.asset_id = a.id
            WHERE a.target_relpath IS NOT NULL
              AND a.target_filename IS NOT NULL
              AND (m.status = 'PLANNED' OR m.status = 'FAILED' OR a.status = 'NEW')
            ORDER BY a.dt_original, a.id
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = con.execute(sql).fetchall()
        for row in rows:
            yield dict(row)
    finally:
        con.close()


def mark_embedded(
    db_path: Path,
    asset_id: str,
    sha256: str,
    embed_hash: Optional[str],
    tool_version: Optional[str],
) -> None:
    _ensure_columns(db_path)
    con = connect(db_path)
    try:
        ts = utcnow_iso()
        con.execute(
            """
            UPDATE assets
            SET sha256=?, last_embed_hash=?, exiftool_version=?,
                status='EMBEDDED', last_updated=?, error_msg=NULL
            WHERE id=?
            """,
            (sha256, embed_hash, tool_version, ts, asset_id),
        )
        con.execute(
            """
            UPDATE managed_assets
            SET status='BUILT', built_at=?, error_msg=NULL
            WHERE asset_id=?
            """,
            (ts, asset_id),
        )
        con.execute(
            """
            INSERT INTO hashes (asset_id, algorithm, scope, value, created_at)
            VALUES (?, 'sha256', 'managed', ?, ?)
            ON CONFLICT(asset_id, algorithm, scope) DO UPDATE SET
              value=excluded.value,
              created_at=excluded.created_at
            """,
            (asset_id, sha256, ts),
        )
        con.commit()
    finally:
        con.close()


def mark_copied(
    db_path: Path,
    asset_id: str,
    sha256: str,
    *,
    tool_version: Optional[str] = None,
    warning: Optional[str] = None,
) -> None:
    _ensure_columns(db_path)
    con = connect(db_path)
    try:
        ts = utcnow_iso()
        con.execute(
            """
            UPDATE assets
            SET sha256=?, exiftool_version=?, status='COPIED',
                last_updated=?, error_msg=?
            WHERE id=?
            """,
            (sha256, tool_version, ts, warning, asset_id),
        )
        con.execute(
            """
            UPDATE managed_assets
            SET status='BUILT', built_at=?, error_msg=?
            WHERE asset_id=?
            """,
            (ts, warning, asset_id),
        )
        con.execute(
            """
            INSERT INTO hashes (asset_id, algorithm, scope, value, created_at)
            VALUES (?, 'sha256', 'managed', ?, ?)
            ON CONFLICT(asset_id, algorithm, scope) DO UPDATE SET
              value=excluded.value,
              created_at=excluded.created_at
            """,
            (asset_id, sha256, ts),
        )
        con.commit()
    finally:
        con.close()


def update_status(db_path: Path, asset_id: str, status: str, error: Optional[str] = None) -> None:
    _ensure_columns(db_path)
    con = connect(db_path)
    try:
        con.execute(
            "UPDATE assets SET status=?, error_msg=?, last_updated=? WHERE id=?",
            (status, error, utcnow_iso(), asset_id),
        )
        if status == "ERROR":
            con.execute(
                "UPDATE managed_assets SET status='FAILED', error_msg=? WHERE asset_id=?",
                (error, asset_id),
            )
        con.commit()
    finally:
        con.close()


def update_source_locator(
    db_path: Path,
    asset_id: str,
    *,
    source_locator: str,
    source_root: Optional[str] = None,
) -> None:
    con = connect(db_path)
    try:
        con.execute(
            """
            UPDATE assets
            SET source_locator=?,
                abs_zip=?,
                source_root=COALESCE(?, source_root),
                last_updated=?
            WHERE id=?
            """,
            (source_locator, source_locator, source_root, utcnow_iso(), asset_id),
        )
        con.commit()
    finally:
        con.close()


def record_hash(db_path: Path, asset_id: str, algorithm: str, scope: str, value: str) -> None:
    con = connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO hashes (asset_id, algorithm, scope, value, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(asset_id, algorithm, scope) DO UPDATE SET
              value=excluded.value,
              created_at=excluded.created_at
            """,
            (asset_id, algorithm, scope, value, utcnow_iso()),
        )
        con.commit()
    finally:
        con.close()


def record_thumbnail(
    db_path: Path,
    asset_id: str,
    kind: str,
    path: str,
    status: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    con = connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO thumbnails (asset_id, kind, path, width, height, status, created_at, error_msg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asset_id, kind) DO UPDATE SET
              path=excluded.path,
              width=excluded.width,
              height=excluded.height,
              status=excluded.status,
              created_at=excluded.created_at,
              error_msg=excluded.error_msg
            """,
            (asset_id, kind, path, width, height, status, utcnow_iso(), error),
        )
        con.commit()
    finally:
        con.close()


def record_artifact(
    db_path: Path,
    asset_id: str,
    artifact_type: str,
    path: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    con = connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO artifacts (asset_id, artifact_type, path, status, created_at, error_msg)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(asset_id, artifact_type, path) DO UPDATE SET
              status=excluded.status,
              created_at=excluded.created_at,
              error_msg=excluded.error_msg
            """,
            (asset_id, artifact_type, path, status, utcnow_iso(), error),
        )
        con.commit()
    finally:
        con.close()


def iter_assets_for_derivatives(db_path: Path, limit: int | None = None) -> Iterator[Dict[str, Any]]:
    con = connect(db_path)
    try:
        sql = """
            SELECT a.id, a.media_type, m.managed_path, m.relpath, m.filename
            FROM assets a
            JOIN managed_assets m ON m.asset_id = a.id
            WHERE a.status IN ('EMBEDDED', 'COPIED') AND m.status='BUILT'
              AND NOT EXISTS (
                SELECT 1
                FROM thumbnails t
                WHERE t.asset_id = a.id
                  AND (
                    (a.media_type IN ('image', 'raw') AND t.kind = 'primary' AND t.status = 'READY')
                    OR
                    (a.media_type = 'video' AND t.kind = 'video_preview' AND t.status = 'READY')
                  )
              )
            ORDER BY a.dt_original, a.id
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        for row in con.execute(sql):
            yield dict(row)
    finally:
        con.close()


def iter_assets_for_metadata_repair(db_path: Path, limit: int | None = None) -> Iterator[Dict[str, Any]]:
    con = connect(db_path)
    try:
        sql = """
            SELECT a.*, m.managed_path
            FROM assets a
            JOIN managed_assets m ON m.asset_id = a.id
            WHERE m.status='BUILT'
              AND a.status IN ('EMBEDDED', 'COPIED')
            ORDER BY a.last_updated ASC, a.id ASC
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        for row in con.execute(sql):
            yield dict(row)
    finally:
        con.close()


def apply_metadata_updates(
    db_path: Path,
    asset_id: str,
    *,
    dt_original: Optional[str] = None,
    dt_source: Optional[str] = None,
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None,
    gps_alt: Optional[float] = None,
    gps_source: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    people: Optional[List[str]] = None,
    source_name: str = "metadata_repair",
    confidence: Optional[float] = None,
) -> None:
    con = connect(db_path)
    try:
        sets = ["last_updated=?"]
        params: List[Any] = [utcnow_iso()]
        def _can_promote(field_name: str) -> bool:
            if confidence is None:
                return True
            row = con.execute(
                """
                SELECT confidence
                FROM metadata_fields
                WHERE asset_id = ?
                  AND field_name = ?
                  AND is_canonical = 1
                ORDER BY
                  CASE
                    WHEN confidence IS NULL THEN -1
                    ELSE confidence
                  END DESC,
                  updated_at DESC
                LIMIT 1
                """,
                (asset_id, field_name),
            ).fetchone()
            if not row:
                return True
            current_confidence = row["confidence"]
            if current_confidence is None:
                return True
            return float(confidence) >= float(current_confidence)

        if dt_original:
            if _can_promote("captured_at"):
                sets.extend(["dt_original=?", "source_dt=?"])
                params.extend([dt_original, dt_source or source_name])
        if gps_lat is not None and gps_lon is not None:
            if _can_promote("location"):
                sets.extend(["gps_lat=?", "gps_lon=?", "gps_alt=?", "source_gps=?"])
                params.extend([gps_lat, gps_lon, gps_alt, gps_source or source_name])
        if title:
            if _can_promote("title"):
                sets.append("title=?")
                params.append(title)
        if description:
            if _can_promote("description"):
                sets.append("description=?")
                params.append(description)
        if keywords is not None:
            if _can_promote("keywords"):
                sets.append("keywords_json=?")
                params.append(json.dumps(keywords))
        if people is not None:
            if _can_promote("people"):
                sets.append("people_json=?")
                params.append(json.dumps(people))
        params.append(asset_id)
        con.execute(f"UPDATE assets SET {', '.join(sets)} WHERE id=?", params)

        if dt_original:
            _record_metadata_field(
                con,
                asset_id=asset_id,
                field_name="captured_at",
                value=dt_original,
                source_name=source_name,
                source_field=dt_source or source_name,
                is_canonical=_can_promote("captured_at"),
                confidence=confidence,
            )
        if gps_lat is not None and gps_lon is not None:
            _record_metadata_field(
                con,
                asset_id=asset_id,
                field_name="location",
                value={"lat": gps_lat, "lon": gps_lon, "alt": gps_alt},
                source_name=source_name,
                source_field=gps_source or source_name,
                is_canonical=_can_promote("location"),
                confidence=confidence,
            )
        if title:
            _record_metadata_field(
                con,
                asset_id=asset_id,
                field_name="title",
                value=title,
                source_name=source_name,
                source_field="Title",
                is_canonical=_can_promote("title"),
                confidence=confidence,
            )
        if description:
            _record_metadata_field(
                con,
                asset_id=asset_id,
                field_name="description",
                value=description,
                source_name=source_name,
                source_field="Description",
                is_canonical=_can_promote("description"),
                confidence=confidence,
            )
        if keywords is not None:
            _record_metadata_field(
                con,
                asset_id=asset_id,
                field_name="keywords",
                value=keywords,
                source_name=source_name,
                source_field="Subject",
                is_canonical=_can_promote("keywords"),
                confidence=confidence,
            )
        if people is not None:
            _record_metadata_field(
                con,
                asset_id=asset_id,
                field_name="people",
                value=people,
                source_name=source_name,
                source_field="people",
                is_canonical=_can_promote("people"),
                confidence=confidence,
            )
        con.commit()
    finally:
        con.close()


def set_metadata_field(
    db_path: Path,
    asset_id: str,
    *,
    field_name: str,
    value: Any,
    source_name: str,
    source_field: Optional[str] = None,
    is_canonical: bool = False,
    confidence: Optional[float] = None,
) -> None:
    con = connect(db_path)
    try:
        _record_metadata_field(
            con,
            asset_id=asset_id,
            field_name=field_name,
            value=value,
            source_name=source_name,
            source_field=source_field,
            is_canonical=is_canonical,
            confidence=confidence,
        )
        con.commit()
    finally:
        con.close()


def create_job(db_path: Path, job_type: str, config: Optional[Dict[str, Any]] = None) -> str:
    init_db(db_path)
    job_id = uuid.uuid4().hex[:20]
    con = connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO jobs (id, job_type, status, config_json, created_at)
            VALUES (?, ?, 'QUEUED', ?, ?)
            """,
            (job_id, job_type, json.dumps(config or {}), utcnow_iso()),
        )
        con.commit()
        return job_id
    finally:
        con.close()


def start_job(db_path: Path, job_id: str) -> None:
    con = connect(db_path)
    try:
        con.execute(
            "UPDATE jobs SET status='RUNNING', started_at=?, error_msg=NULL WHERE id=?",
            (utcnow_iso(), job_id),
        )
        con.commit()
    finally:
        con.close()


def complete_job(db_path: Path, job_id: str, metrics: Optional[Dict[str, Any]] = None) -> None:
    con = connect(db_path)
    try:
        con.execute(
            "UPDATE jobs SET status='COMPLETED', finished_at=?, metrics_json=? WHERE id=?",
            (utcnow_iso(), json.dumps(metrics or {}), job_id),
        )
        con.commit()
    finally:
        con.close()


def fail_job(
    db_path: Path,
    job_id: str,
    error: str,
    *,
    retryable: bool = True,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    con = connect(db_path)
    try:
        status = "RETRYABLE" if retryable else "FAILED"
        con.execute(
            """
            UPDATE jobs
            SET status=?, finished_at=?, error_msg=?, metrics_json=?
            WHERE id=?
            """,
            (status, utcnow_iso(), error, json.dumps(metrics or {}), job_id),
        )
        con.commit()
    finally:
        con.close()


def attach_job_metrics(db_path: Path, job_id: str, metrics: Dict[str, Any]) -> None:
    con = connect(db_path)
    try:
        row = con.execute("SELECT metrics_json FROM jobs WHERE id=?", (job_id,)).fetchone()
        current = {}
        if row and row["metrics_json"]:
            try:
                current = json.loads(row["metrics_json"])
            except Exception:
                current = {}
        current.update(metrics)
        con.execute("UPDATE jobs SET metrics_json=? WHERE id=?", (json.dumps(current), job_id))
        con.commit()
    finally:
        con.close()


def list_jobs(db_path: Path, limit: int = 20) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        rows = con.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def get_overview(db_path: Path) -> Dict[str, Any]:
    init_db(db_path)
    con = connect(db_path)
    try:
        assets_total = con.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
        embedded = con.execute("SELECT COUNT(*) FROM assets WHERE status='EMBEDDED'").fetchone()[0]
        copied = con.execute("SELECT COUNT(*) FROM assets WHERE status='COPIED'").fetchone()[0]
        errors = con.execute("SELECT COUNT(*) FROM assets WHERE status='ERROR'").fetchone()[0]
        planned = con.execute("SELECT COUNT(*) FROM managed_assets").fetchone()[0]
        thumbs = con.execute("SELECT COUNT(*) FROM thumbnails WHERE status='READY'").fetchone()[0]
        jobs = con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        return {
            "assets_total": assets_total,
            "embedded_assets": embedded,
            "copied_assets": copied,
            "errored_assets": errors,
            "managed_assets": planned,
            "ready_thumbnails": thumbs,
            "jobs": jobs,
        }
    finally:
        con.close()


def get_metadata_overview(db_path: Path) -> Dict[str, Any]:
    con = connect(db_path)
    try:
        total = con.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
        with_time = con.execute("SELECT COUNT(*) FROM assets WHERE dt_original IS NOT NULL").fetchone()[0]
        strong_time = con.execute(
            """
            SELECT COUNT(*)
            FROM assets
            WHERE dt_original IS NOT NULL
              AND COALESCE(source_dt, '') NOT IN ('', 'filename', 'file_mtime', 'zip_mtime', 'unknown')
            """
        ).fetchone()[0]
        with_gps = con.execute(
            "SELECT COUNT(*) FROM assets WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL"
        ).fetchone()[0]
        with_title = con.execute(
            "SELECT COUNT(*) FROM assets WHERE COALESCE(title, '') <> ''"
        ).fetchone()[0]
        with_description = con.execute(
            "SELECT COUNT(*) FROM assets WHERE COALESCE(description, '') <> ''"
        ).fetchone()[0]
        with_people = con.execute(
            """
            SELECT COUNT(*)
            FROM assets
            WHERE COALESCE(people_json, '') <> ''
              AND COALESCE(people_json, '[]') <> '[]'
            """
        ).fetchone()[0]
        screenshots = con.execute(
            """
            SELECT COUNT(DISTINCT asset_id)
            FROM metadata_fields
            WHERE field_name='is_screenshot'
              AND value_json='true'
            """
        ).fetchone()[0]
        blurry = con.execute(
            """
            SELECT COUNT(DISTINCT asset_id)
            FROM metadata_fields
            WHERE field_name='is_blurry'
              AND value_json='true'
            """
        ).fetchone()[0]
        return {
            "total_assets": total,
            "timestamped_assets": with_time,
            "strong_timestamps": strong_time,
            "assets_with_location": with_gps,
            "assets_with_title": with_title,
            "assets_with_description": with_description,
            "assets_with_people": with_people,
            "screenshots_detected": screenshots,
            "blurry_items": blurry,
        }
    finally:
        con.close()


def _metadata_value_present(value: Any) -> bool:
    if value in (None, "", [], {}):
        return False
    if isinstance(value, list):
        return any(item not in (None, "") for item in value)
    if isinstance(value, dict):
        return bool(value)
    return True


def asset_metadata_audit(db_path: Path, asset_id: str) -> Dict[str, Any]:
    asset = get_asset(db_path, asset_id)
    if not asset:
        raise ValueError("asset not found")
    payload = export_asset_metadata_payload(db_path, asset_id)
    metadata = payload["normalized_metadata"]
    standard = canonical_metadata_standard(asset.get("media_type"))
    total_weight = sum(int(item["weight"]) for item in standard) or 1
    filled_weight = 0
    present_fields: List[str] = []
    missing_fields: List[str] = []
    for item in standard:
        field_name = item["field_name"]
        if _metadata_value_present(metadata.get(field_name)):
            filled_weight += int(item["weight"])
            present_fields.append(field_name)
        else:
            missing_fields.append(field_name)
    score = int(round((filled_weight / total_weight) * 100))
    return {
        "asset_id": asset_id,
        "media_type": asset.get("media_type"),
        "score": score,
        "filled_weight": filled_weight,
        "total_weight": total_weight,
        "present_fields": present_fields,
        "missing_fields": missing_fields,
        "standard": standard,
    }


def get_metadata_audit_overview(db_path: Path) -> Dict[str, Any]:
    con = connect(db_path)
    try:
        counts: Dict[str, int] = {}
        field_summaries: List[Dict[str, Any]] = []
        for item in CANONICAL_METADATA_STANDARD:
            field_name = item["field_name"]
            media_types = item.get("media_types", [])
            placeholders = ",".join("?" for _ in media_types)
            applicable_row = con.execute(
                f"""
                SELECT COUNT(*)
                FROM assets a
                WHERE a.media_type IN ({placeholders})
                """,
                media_types,
            ).fetchone()
            applicable_assets = applicable_row[0] if applicable_row else 0
            row = con.execute(
                f"""
                SELECT COUNT(DISTINCT a.id)
                FROM assets a
                LEFT JOIN metadata_fields mf
                  ON mf.asset_id = a.id
                 AND mf.field_name = ?
                 AND mf.is_canonical = 1
                WHERE a.media_type IN ({placeholders})
                  AND (
                    mf.asset_id IS NULL
                    OR mf.value_json IN ('null', '""', '[]', '{{}}')
                  )
                """,
                [field_name, *media_types],
            ).fetchone()
            missing_assets = row[0] if row else 0
            present_assets = max(applicable_assets - missing_assets, 0)
            completeness = round((present_assets / applicable_assets) * 100, 1) if applicable_assets else 0.0
            counts[f"missing_{field_name}_assets"] = missing_assets
            field_summaries.append(
                {
                    "field_name": field_name,
                    "label": item["label"],
                    "weight": item["weight"],
                    "applicable_assets": applicable_assets,
                    "present_assets": present_assets,
                    "missing_assets": missing_assets,
                    "completeness_pct": completeness,
                }
            )

        rows = con.execute(
            """
            SELECT a.id, a.media_type
            FROM assets a
            ORDER BY a.id ASC
            """
        ).fetchall()
        field_rows = con.execute(
            """
            SELECT asset_id, field_name, value_json
            FROM metadata_fields
            WHERE is_canonical = 1
            """
        ).fetchall()
    finally:
        con.close()

    present_by_asset: Dict[str, set[str]] = {}
    for row in field_rows:
        if row["value_json"] in ('null', '""', '[]', '{}'):
            continue
        present_by_asset.setdefault(row["asset_id"], set()).add(row["field_name"])

    scores: List[int] = []
    complete = 0
    for row in rows:
        standard = canonical_metadata_standard(row["media_type"])
        total_weight = sum(int(item["weight"]) for item in standard) or 1
        present = present_by_asset.get(row["id"], set())
        filled_weight = sum(int(item["weight"]) for item in standard if item["field_name"] in present)
        score = int(round((filled_weight / total_weight) * 100))
        scores.append(score)
        if score >= 90:
            complete += 1
    average_score = round(sum(scores) / len(scores), 1) if scores else 0.0
    return {
        "assets_meeting_90_score": complete,
        "average_metadata_score": average_score,
        "field_summaries": field_summaries,
        **counts,
    }


def list_assets(
    db_path: Path,
    limit: int = 100,
    *,
    offset: int = 0,
    query: Optional[str] = None,
    include_hidden: bool = False,
) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        sql = """
            SELECT a.id, a.source, a.source_kind, a.media_type, a.orig_filename, a.dt_original,
                   a.status, a.target_relpath, a.target_filename, m.status AS managed_status,
                   m.managed_path,
                   COALESCE((
                     SELECT di.keep_decision
                     FROM duplicate_items di
                     WHERE di.asset_id = a.id
                     ORDER BY CASE di.keep_decision
                       WHEN 'HIDE' THEN 0
                       WHEN 'CANONICAL' THEN 1
                       WHEN 'KEEP' THEN 2
                       ELSE 3
                     END
                     LIMIT 1
                   ), '') AS duplicate_decision
            FROM assets a
            LEFT JOIN managed_assets m ON m.asset_id = a.id
        """
        params: List[Any] = []
        clauses: List[str] = []
        if not include_hidden:
            clauses.append(
                """
                NOT EXISTS (
                  SELECT 1
                  FROM duplicate_items di_hide
                  WHERE di_hide.asset_id = a.id
                    AND di_hide.keep_decision = 'HIDE'
                )
                """
            )
        if query:
            clauses.append(
                """
                (a.orig_filename LIKE ?
                 OR COALESCE(a.title, '') LIKE ?
                 OR COALESCE(a.description, '') LIKE ?)
                """
            )
            like = f"%{query}%"
            params.extend([like, like, like])
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY a.dt_original DESC, a.id DESC LIMIT ? OFFSET ?"
        params.extend([limit, max(0, offset)])
        rows = con.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def list_assets_missing_metadata(
    db_path: Path,
    field_name: str,
    *,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    field_def = _canonical_field_definition(field_name)
    if not field_def:
        return []
    media_types = field_def.get("media_types", [])
    placeholders = ",".join("?" for _ in media_types)
    con = connect(db_path)
    try:
        rows = con.execute(
            f"""
            SELECT a.id, a.source, a.source_kind, a.media_type, a.orig_filename, a.dt_original,
                   a.status, m.managed_path
            FROM assets a
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            LEFT JOIN metadata_fields mf
              ON mf.asset_id = a.id
             AND mf.field_name = ?
             AND mf.is_canonical = 1
            WHERE a.media_type IN ({placeholders})
              AND (
                  mf.asset_id IS NULL
               OR mf.value_json IN ('null', '""', '[]', '{{}}')
              )
            ORDER BY a.dt_original DESC, a.id DESC
            LIMIT ?
            """,
            (field_name, *media_types, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def list_assets_with_flag(
    db_path: Path,
    field_name: str,
    *,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT a.id, a.source, a.source_kind, a.media_type, a.orig_filename, a.dt_original,
                   a.status, m.managed_path, mf.value_json, mf.confidence
            FROM assets a
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            JOIN metadata_fields mf
              ON mf.asset_id = a.id
             AND mf.field_name = ?
            WHERE mf.value_json = 'true'
            ORDER BY a.dt_original DESC, a.id DESC
            LIMIT ?
            """,
            (field_name, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def get_asset(db_path: Path, asset_id: str) -> Optional[Dict[str, Any]]:
    con = connect(db_path)
    try:
        row = con.execute(
            """
            SELECT a.*, m.status AS managed_status, m.managed_path,
                   t.path AS thumbnail_path
            FROM assets a
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            LEFT JOIN thumbnails t ON t.asset_id = a.id AND t.kind IN ('primary', 'video_preview')
            WHERE a.id = ?
            LIMIT 1
            """,
            (asset_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def list_asset_metadata(db_path: Path, asset_id: str) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT asset_id, field_name, value_json, value_text, value_real,
                   source_name, source_field, is_canonical, confidence,
                   created_at, updated_at
            FROM metadata_fields
            WHERE asset_id = ?
            ORDER BY field_name ASC, is_canonical DESC, updated_at DESC, source_name ASC, source_field ASC
            """,
            (asset_id,),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["value"] = json.loads(item["value_json"])
            except Exception:
                item["value"] = item["value_text"] or item["value_real"]
            out.append(item)
        return out
    finally:
        con.close()


def list_asset_artifacts(db_path: Path, asset_id: str) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT asset_id, artifact_type, path, status, created_at, error_msg
            FROM artifacts
            WHERE asset_id = ?
            ORDER BY artifact_type ASC, path ASC
            """,
            (asset_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def export_asset_metadata_payload(db_path: Path, asset_id: str) -> Dict[str, Any]:
    asset = get_asset(db_path, asset_id)
    if not asset:
        raise ValueError("asset not found")
    fields = list_asset_metadata(db_path, asset_id)
    artifacts = list_asset_artifacts(db_path, asset_id)
    canonical: Dict[str, Any] = {}
    provenance: List[Dict[str, Any]] = []
    for field in fields:
        entry = {
            "field": field["field_name"],
            "value": field.get("value"),
            "source_name": field.get("source_name"),
            "source_field": field.get("source_field"),
            "is_canonical": bool(field.get("is_canonical")),
            "confidence": field.get("confidence"),
            "updated_at": field.get("updated_at"),
        }
        provenance.append(entry)
        if entry["is_canonical"] and entry["field"] not in canonical:
            canonical[entry["field"]] = entry["value"]
    return {
        "asset_id": asset_id,
        "normalized_metadata": canonical,
        "metadata_provenance": provenance,
        "artifacts": artifacts,
        "source": {
            "provider": asset.get("source"),
            "source_kind": asset.get("source_kind"),
            "source_locator": asset.get("source_locator"),
            "source_path": asset.get("source_path"),
        },
    }


def list_duplicate_groups(db_path: Path, limit: int = 50) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT g.*, COUNT(i.asset_id) AS item_count,
                   a.orig_filename AS canonical_filename
            FROM duplicate_groups g
            LEFT JOIN duplicate_items i ON i.group_id = g.id
            LEFT JOIN assets a ON a.id = g.canonical_asset_id
            GROUP BY g.id
            ORDER BY g.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def list_duplicate_group_items(db_path: Path, group_id: str) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT i.*, a.orig_filename, a.media_type, a.dt_original, a.orig_size,
                   m.managed_path, t.path AS thumbnail_path
            FROM duplicate_items i
            JOIN assets a ON a.id = i.asset_id
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            LEFT JOIN thumbnails t ON t.asset_id = a.id AND t.kind IN ('primary', 'video_preview')
            WHERE i.group_id = ?
            ORDER BY a.orig_size DESC, a.dt_original ASC, a.id ASC
            """,
            (group_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def list_faces(db_path: Path, limit: int = 50, *, include_rejected: bool = False) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        sql = """
            SELECT f.*, i.label AS identity_label, a.orig_filename, a.media_type, t.path AS thumbnail_path,
                   m.managed_path
            FROM faces f
            LEFT JOIN face_identities i ON i.id = f.identity_id
            LEFT JOIN assets a ON a.id = f.asset_id
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            LEFT JOIN thumbnails t ON t.asset_id = a.id AND t.kind IN ('primary', 'video_preview')
        """
        params: List[Any] = []
        if not include_rejected:
            sql += " WHERE f.status <> 'REJECTED'"
        sql += """
            ORDER BY CASE WHEN f.identity_id IS NULL THEN 0 ELSE 1 END ASC,
                     f.created_at DESC
            LIMIT ?
        """
        params.append(limit)
        rows = con.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def list_face_identities(db_path: Path, limit: int = 200) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT i.*, COUNT(f.id) AS face_count
            FROM face_identities i
            LEFT JOIN faces f ON f.identity_id = i.id
            GROUP BY i.id
            ORDER BY i.label ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def create_face_identity(db_path: Path, label: str, *, status: str = "CONFIRMED") -> str:
    con = connect(db_path)
    try:
        existing = con.execute(
            "SELECT id FROM face_identities WHERE lower(label)=lower(?) LIMIT 1",
            (label,),
        ).fetchone()
        if existing:
            return existing["id"]
        identity_id = uuid.uuid4().hex[:20]
        con.execute(
            """
            INSERT INTO face_identities (id, label, status, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (identity_id, label, status, utcnow_iso()),
        )
        con.commit()
        return identity_id
    finally:
        con.close()


def save_face_embedding(
    db_path: Path,
    *,
    asset_id: str,
    face_id: str,
    vector: List[float],
    model_name: str = "simple-face-v1",
) -> str:
    embedding_id = uuid.uuid4().hex[:20]
    con = connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO embeddings (id, asset_id, embedding_type, model_name, vector_ref, payload_json, created_at)
            VALUES (?, ?, 'face', ?, ?, ?, ?)
            """,
            (
                embedding_id,
                asset_id,
                model_name,
                face_id,
                json.dumps({"vector": vector}),
                utcnow_iso(),
            ),
        )
        con.execute(
            "UPDATE faces SET embedding_ref=? WHERE id=?",
            (embedding_id, face_id),
        )
        con.commit()
        return embedding_id
    finally:
        con.close()


def list_face_embeddings(db_path: Path) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT f.id AS face_id, f.asset_id, f.identity_id, f.status, e.id AS embedding_id,
                   e.model_name, e.payload_json
            FROM faces f
            JOIN embeddings e ON e.id = f.embedding_ref
            WHERE f.status <> 'REJECTED'
              AND e.embedding_type = 'face'
            ORDER BY f.created_at ASC
            """
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            payload = json.loads(item.get("payload_json") or "{}")
            item["vector"] = payload.get("vector") or []
            out.append(item)
        return out
    finally:
        con.close()


def update_face_cluster(db_path: Path, face_id: str, identity_id: str, *, labeled: bool = False) -> None:
    con = connect(db_path)
    try:
        con.execute(
            "UPDATE faces SET identity_id=?, status=? WHERE id=?",
            (identity_id, "LABELED" if labeled else "CLUSTERED", face_id),
        )
        con.commit()
    finally:
        con.close()


def rename_face_identity(db_path: Path, identity_id: str, label: str, *, status: str = "CONFIRMED") -> None:
    con = connect(db_path)
    try:
        con.execute(
            "UPDATE face_identities SET label=?, status=? WHERE id=?",
            (label, status, identity_id),
        )
        con.commit()
    finally:
        con.close()


def apply_identity_to_assets(db_path: Path, identity_id: str) -> None:
    con = connect(db_path)
    try:
        identity = con.execute("SELECT label FROM face_identities WHERE id=?", (identity_id,)).fetchone()
        if not identity:
            raise ValueError("identity not found")
        label = identity["label"]
        asset_rows = con.execute(
            "SELECT DISTINCT asset_id FROM faces WHERE identity_id=? AND status <> 'REJECTED'",
            (identity_id,),
        ).fetchall()
        for asset_row in asset_rows:
            asset_id = asset_row["asset_id"]
            row = con.execute("SELECT people_json FROM assets WHERE id=?", (asset_id,)).fetchone()
            people = _coerce_list(row["people_json"] if row else [])
            if label not in people:
                people.append(label)
            con.execute(
                "UPDATE assets SET people_json=?, last_updated=? WHERE id=?",
                (json.dumps(sorted(set(people))), utcnow_iso(), asset_id),
            )
            _record_metadata_field(
                con,
                asset_id=asset_id,
                field_name="people",
                value=sorted(set(people)),
                source_name="face_label",
                source_field=label,
                is_canonical=True,
                confidence=0.98,
            )
        con.execute(
            "UPDATE faces SET status='LABELED' WHERE identity_id=? AND status <> 'REJECTED'",
            (identity_id,),
        )
        con.commit()
    finally:
        con.close()


def replace_faces_for_asset(
    db_path: Path,
    asset_id: str,
    detections: List[Dict[str, Any]],
    *,
    source_name: str = "opencv_haar",
) -> int:
    con = connect(db_path)
    try:
        con.execute("DELETE FROM faces WHERE asset_id=?", (asset_id,))
        created = 0
        for detection in detections:
            con.execute(
                """
                INSERT INTO faces (id, asset_id, identity_id, embedding_ref, bbox_json, frame_time_ms, status, created_at)
                VALUES (?, ?, NULL, ?, ?, ?, 'DETECTED', ?)
                """,
                (
                    detection.get("id") or uuid.uuid4().hex[:20],
                    asset_id,
                    detection.get("embedding_ref") or source_name,
                    json.dumps(detection.get("bbox") or {}),
                    detection.get("frame_time_ms"),
                    utcnow_iso(),
                ),
            )
            created += 1
        con.commit()
        return created
    finally:
        con.close()


def assign_face_identity(db_path: Path, face_id: str, identity_id: str) -> None:
    con = connect(db_path)
    try:
        face = con.execute(
            "SELECT asset_id, identity_id FROM faces WHERE id=?",
            (face_id,),
        ).fetchone()
        identity = con.execute(
            "SELECT label FROM face_identities WHERE id=?",
            (identity_id,),
        ).fetchone()
        if not face or not identity:
            raise ValueError("face or identity not found")
        con.commit()
    finally:
        con.close()
    cluster_identity = face["identity_id"] or identity_id
    if face["identity_id"] and face["identity_id"] != identity_id:
        con = connect(db_path)
        try:
            con.execute(
                "UPDATE faces SET identity_id=? WHERE identity_id=?",
                (identity_id, face["identity_id"]),
            )
            con.execute("DELETE FROM face_identities WHERE id=?", (face["identity_id"],))
            con.commit()
        finally:
            con.close()
        cluster_identity = identity_id
    else:
        update_face_cluster(db_path, face_id, identity_id, labeled=True)
    apply_identity_to_assets(db_path, cluster_identity)


def reject_face(db_path: Path, face_id: str) -> None:
    con = connect(db_path)
    try:
        con.execute(
            "UPDATE faces SET status='REJECTED', identity_id=NULL WHERE id=?",
            (face_id,),
        )
        con.commit()
    finally:
        con.close()


def get_face(db_path: Path, face_id: str) -> Optional[Dict[str, Any]]:
    con = connect(db_path)
    try:
        row = con.execute(
            """
            SELECT f.*, i.label AS identity_label, a.orig_filename, a.media_type,
                   m.managed_path, t.path AS thumbnail_path
            FROM faces f
            LEFT JOIN face_identities i ON i.id = f.identity_id
            LEFT JOIN assets a ON a.id = f.asset_id
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            LEFT JOIN thumbnails t ON t.asset_id = a.id AND t.kind IN ('primary', 'video_preview')
            WHERE f.id = ?
            """,
            (face_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def get_face_overview(db_path: Path) -> Dict[str, Any]:
    con = connect(db_path)
    try:
        total = con.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        unlabeled = con.execute("SELECT COUNT(*) FROM faces WHERE identity_id IS NULL AND status <> 'REJECTED'").fetchone()[0]
        clustered = con.execute("SELECT COUNT(*) FROM faces WHERE identity_id IS NOT NULL AND status='CLUSTERED'").fetchone()[0]
        labeled = con.execute("SELECT COUNT(*) FROM faces WHERE identity_id IS NOT NULL AND status='LABELED'").fetchone()[0]
        rejected = con.execute("SELECT COUNT(*) FROM faces WHERE status='REJECTED'").fetchone()[0]
        assets = con.execute("SELECT COUNT(DISTINCT asset_id) FROM faces WHERE status <> 'REJECTED'").fetchone()[0]
        identities = con.execute("SELECT COUNT(*) FROM face_identities").fetchone()[0]
        return {
            "total_faces": total,
            "unlabeled_faces": unlabeled,
            "clustered_faces": clustered,
            "labeled_faces": labeled,
            "rejected_faces": rejected,
            "assets_with_faces": assets,
            "identities": identities,
        }
    finally:
        con.close()


def iter_assets_for_face_detection(
    db_path: Path,
    limit: int | None = None,
    *,
    include_existing: bool = False,
) -> Iterator[Dict[str, Any]]:
    con = connect(db_path)
    try:
        sql = """
            SELECT a.id, a.media_type, a.orig_filename, a.dt_original, m.managed_path,
                   t.path AS thumbnail_path,
                   COALESCE(md.value_real, CAST(md.value_text AS REAL)) AS duration_seconds
            FROM assets a
            JOIN managed_assets m ON m.asset_id = a.id
            LEFT JOIN thumbnails t ON t.asset_id = a.id AND t.kind IN ('primary', 'video_preview')
            LEFT JOIN metadata_fields md
              ON md.asset_id = a.id
             AND md.field_name = 'duration_seconds'
             AND md.is_canonical = 1
            WHERE m.status='BUILT'
              AND a.status IN ('EMBEDDED', 'COPIED')
              AND a.media_type IN ('image', 'raw', 'video')
        """
        if not include_existing:
            sql += """
              AND NOT EXISTS (
                SELECT 1 FROM faces f WHERE f.asset_id = a.id
              )
            """
        sql += """
            ORDER BY a.dt_original ASC, a.id ASC
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        for row in con.execute(sql):
            yield dict(row)
    finally:
        con.close()


def iter_built_assets(db_path: Path, limit: int | None = None) -> Iterator[Dict[str, Any]]:
    con = connect(db_path)
    try:
        sql = """
            SELECT a.id, a.orig_filename, a.orig_size, a.dt_original, a.media_type,
                   a.sha256, m.managed_path
            FROM assets a
            JOIN managed_assets m ON m.asset_id = a.id
            WHERE m.status='BUILT'
            ORDER BY a.dt_original, a.id
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        for row in con.execute(sql):
            yield dict(row)
    finally:
        con.close()


def replace_duplicate_groups(
    db_path: Path,
    *,
    group_type: str,
    groups: List[Dict[str, Any]],
) -> int:
    con = connect(db_path)
    try:
        existing = con.execute(
            "SELECT id FROM duplicate_groups WHERE group_type=?",
            (group_type,),
        ).fetchall()
        for row in existing:
            con.execute("DELETE FROM duplicate_items WHERE group_id=?", (row["id"],))
        con.execute("DELETE FROM duplicate_groups WHERE group_type=?", (group_type,))

        created = 0
        for group in groups:
            group_id = uuid.uuid4().hex[:20]
            created += 1
            con.execute(
                """
                INSERT INTO duplicate_groups (id, group_type, status, canonical_asset_id, created_at)
                VALUES (?, ?, 'OPEN', ?, ?)
                """,
                (group_id, group_type, group["canonical_asset_id"], utcnow_iso()),
            )
            for item in group["items"]:
                con.execute(
                    """
                    INSERT INTO duplicate_items (group_id, asset_id, score, rationale, keep_decision)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        group_id,
                        item["asset_id"],
                        item.get("score"),
                        item.get("rationale"),
                        "CANONICAL" if item["asset_id"] == group["canonical_asset_id"] else None,
                    ),
                )
        con.commit()
        return created
    finally:
        con.close()


def thumbnail_for_asset(db_path: Path, asset_id: str) -> Optional[str]:
    con = connect(db_path)
    try:
        row = con.execute(
            """
            SELECT path
            FROM thumbnails
            WHERE asset_id=? AND status='READY'
            ORDER BY CASE kind
              WHEN 'primary' THEN 0
              WHEN 'video_preview' THEN 1
              ELSE 2
            END
            LIMIT 1
            """,
            (asset_id,),
        ).fetchone()
        return row["path"] if row else None
    finally:
        con.close()


def managed_path_for_asset(db_path: Path, asset_id: str) -> Optional[str]:
    con = connect(db_path)
    try:
        row = con.execute(
            "SELECT managed_path FROM managed_assets WHERE asset_id=?",
            (asset_id,),
        ).fetchone()
        return row["managed_path"] if row else None
    finally:
        con.close()


def resolve_duplicate_group(
    db_path: Path,
    group_id: str,
    canonical_asset_id: str,
    *,
    kept_asset_ids: Optional[List[str]] = None,
) -> None:
    kept = set(kept_asset_ids or [])
    kept.add(canonical_asset_id)
    con = connect(db_path)
    try:
        exists = con.execute(
            "SELECT 1 FROM duplicate_items WHERE group_id=? AND asset_id=?",
            (group_id, canonical_asset_id),
        ).fetchone()
        if not exists:
            raise ValueError("canonical asset is not part of duplicate group")

        con.execute(
            """
            UPDATE duplicate_groups
            SET canonical_asset_id=?, status='RESOLVED', resolved_at=?
            WHERE id=?
            """,
            (canonical_asset_id, utcnow_iso(), group_id),
        )
        rows = con.execute(
            "SELECT asset_id FROM duplicate_items WHERE group_id=?",
            (group_id,),
        ).fetchall()
        for row in rows:
            asset_id = row["asset_id"]
            decision = "CANONICAL" if asset_id == canonical_asset_id else ("KEEP" if asset_id in kept else "HIDE")
            con.execute(
                "UPDATE duplicate_items SET keep_decision=? WHERE group_id=? AND asset_id=?",
                (decision, group_id, asset_id),
            )
        con.execute(
            """
            INSERT INTO review_decisions (id, asset_id, decision_type, value_json, created_at)
            VALUES (?, ?, 'duplicate_resolution', ?, ?)
            """,
            (
                uuid.uuid4().hex[:20],
                canonical_asset_id,
                json.dumps(
                    {
                        "group_id": group_id,
                        "canonical_asset_id": canonical_asset_id,
                        "kept_asset_ids": sorted(kept),
                    }
                ),
                utcnow_iso(),
            ),
        )
        con.commit()
    finally:
        con.close()


def keep_all_duplicate_group(db_path: Path, group_id: str) -> None:
    con = connect(db_path)
    try:
        rows = con.execute(
            "SELECT asset_id FROM duplicate_items WHERE group_id=?",
            (group_id,),
        ).fetchall()
        if not rows:
            raise ValueError("duplicate group not found")
        canonical_asset_id = rows[0]["asset_id"]
        con.execute(
            """
            UPDATE duplicate_groups
            SET canonical_asset_id=?, status='KEPT_ALL', resolved_at=?
            WHERE id=?
            """,
            (canonical_asset_id, utcnow_iso(), group_id),
        )
        for row in rows:
            con.execute(
                "UPDATE duplicate_items SET keep_decision='KEEP' WHERE group_id=? AND asset_id=?",
                (group_id, row["asset_id"]),
            )
        con.execute(
            """
            INSERT INTO review_decisions (id, asset_id, decision_type, value_json, created_at)
            VALUES (?, ?, 'duplicate_keep_all', ?, ?)
            """,
            (
                uuid.uuid4().hex[:20],
                canonical_asset_id,
                json.dumps({"group_id": group_id}),
                utcnow_iso(),
            ),
        )
        con.commit()
    finally:
        con.close()


def skip_duplicate_group(db_path: Path, group_id: str) -> None:
    con = connect(db_path)
    try:
        exists = con.execute(
            "SELECT 1 FROM duplicate_groups WHERE id=?",
            (group_id,),
        ).fetchone()
        if not exists:
            raise ValueError("duplicate group not found")
        con.execute(
            "UPDATE duplicate_groups SET status='SKIPPED', resolved_at=? WHERE id=?",
            (utcnow_iso(), group_id),
        )
        con.execute(
            "UPDATE duplicate_items SET keep_decision='SKIP' WHERE group_id=? AND keep_decision IS NULL",
            (group_id,),
        )
        con.commit()
    finally:
        con.close()
