from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


JOB_STATES = ("QUEUED", "RUNNING", "COMPLETED", "FAILED", "RETRYABLE")
MANAGED_STATES = ("PLANNED", "BUILT", "MISSING", "FAILED")
PERSON_CANDIDATE_SCORE_FLOOR = 0.60
PERSON_CANDIDATE_CLARIFICATION_THRESHOLD = 0.72
PERSON_CANDIDATE_LIMIT = 32
PERSON_CANDIDATE_REFRESH_LIMIT = 120
PERSON_CANDIDATE_REFRESH_MIN_THRESHOLD = 0.66
PERSON_CANDIDATE_REFRESH_STEP = 0.02
PERSON_AUTO_ASSIGN_THRESHOLD = 0.95
PERSON_AUTO_ASSIGN_MIN_THRESHOLD = 0.50
PERSON_AUTO_ASSIGN_STEP = 0.01

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
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    limit = min(len(a), len(b))
    if limit <= 0:
        return 0.0
    dot = sum(float(a[i]) * float(b[i]) for i in range(limit))
    an = sum(float(a[i]) * float(a[i]) for i in range(limit)) ** 0.5 or 1.0
    bn = sum(float(b[i]) * float(b[i]) for i in range(limit)) ** 0.5 or 1.0
    return dot / (an * bn)


def _centroid_from_vectors(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = max(len(vector) for vector in vectors)
    centroid = [0.0] * dim
    for vector in vectors:
        for index, value in enumerate(vector[:dim]):
            centroid[index] += float(value)
    scale = float(len(vectors)) or 1.0
    return [value / scale for value in centroid]


def _average_centroid_similarity(vectors: List[List[float]]) -> float:
    if len(vectors) < 2:
        return 1.0
    centroid = _centroid_from_vectors(vectors)
    if not centroid:
        return 0.0
    scores = [_cosine_similarity(vector, centroid) for vector in vectors]
    return sum(scores) / len(scores) if scores else 0.0


def _confirmed_auto_assign_threshold(
    face_count: int,
    vectors: List[List[float]],
    *,
    base_threshold: float = PERSON_AUTO_ASSIGN_THRESHOLD,
) -> float:
    profile_size = max(face_count, len(vectors))
    # Small confirmed profiles should stay cautious, while stronger profiles
    # can relax enough to absorb obvious matches automatically.
    threshold = base_threshold + 0.03 - max(0, profile_size - 1) * 0.02
    return max(PERSON_AUTO_ASSIGN_MIN_THRESHOLD, min(0.98, threshold))


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
              user_rating INTEGER,
              review_score REAL,
              review_state TEXT,
              favorite INTEGER DEFAULT 0,
              ranked_at TEXT,
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

            CREATE TABLE IF NOT EXISTS face_clarifications (
              face_id TEXT PRIMARY KEY REFERENCES faces(id) ON DELETE CASCADE,
              suggested_identity_id TEXT REFERENCES face_identities(id),
              suggested_label TEXT,
              score REAL,
              rationale TEXT,
              status TEXT NOT NULL DEFAULT 'OPEN',
              created_at TEXT NOT NULL,
              resolved_at TEXT
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

            CREATE TABLE IF NOT EXISTS mutation_history (
              seq INTEGER PRIMARY KEY AUTOINCREMENT,
              action_type TEXT NOT NULL,
              target_id TEXT,
              summary TEXT,
              before_json TEXT NOT NULL,
              after_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              undone_at TEXT,
              redone_at TEXT
            );

            CREATE TABLE IF NOT EXISTS saved_searches (
              id TEXT PRIMARY KEY,
              label TEXT NOT NULL,
              params_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
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
            ("user_rating", "INTEGER"),
            ("review_score", "REAL"),
            ("review_state", "TEXT"),
            ("favorite", "INTEGER DEFAULT 0"),
            ("ranked_at", "TEXT"),
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


def _coerce_str_list(value: Any) -> List[str]:
    return [str(item) for item in _coerce_list(value) if str(item).strip()]


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
                  user_rating, review_score, review_state, favorite, ranked_at,
                  keywords_json, people_json, live_group_id, burst_id, status, src_mtime,
                  source_dt, source_gps, last_updated
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                  user_rating     = COALESCE(excluded.user_rating, assets.user_rating),
                  review_score    = COALESCE(excluded.review_score, assets.review_score),
                  review_state    = COALESCE(excluded.review_state, assets.review_state),
                  favorite        = COALESCE(excluded.favorite, assets.favorite),
                  ranked_at       = COALESCE(excluded.ranked_at, assets.ranked_at),
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
                    r.get("user_rating"),
                    r.get("review_score"),
                    r.get("review_state"),
                    int(bool(r.get("favorite"))) if r.get("favorite") is not None else 0,
                    r.get("ranked_at"),
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


def count_for_master(db_path: Path, limit: int | None = None) -> int:
    _ensure_columns(db_path)
    con = connect(db_path)
    try:
        sql = """
            SELECT COUNT(*)
            FROM assets a
            JOIN managed_assets m ON m.asset_id = a.id
            WHERE a.target_relpath IS NOT NULL
              AND a.target_filename IS NOT NULL
              AND (m.status = 'PLANNED' OR m.status = 'FAILED' OR a.status = 'NEW')
        """
        params: List[Any] = []
        if limit:
            sql = f"SELECT COUNT(*) FROM ({sql} LIMIT ?) AS counted"
            params.append(int(limit))
        return int(con.execute(sql, params).fetchone()[0])
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


def delete_metadata_field(
    db_path: Path,
    asset_id: str,
    *,
    field_name: str,
    source_name: Optional[str] = None,
    source_field: Optional[str] = None,
) -> None:
    con = connect(db_path)
    try:
        sql = "DELETE FROM metadata_fields WHERE asset_id = ? AND field_name = ?"
        params: List[Any] = [asset_id, field_name]
        if source_name is not None:
            sql += " AND source_name = ?"
            params.append(source_name)
        if source_field is not None:
            sql += " AND COALESCE(source_field, '') = COALESCE(?, '')"
            params.append(source_field)
        con.execute(sql, params)
        con.commit()
    finally:
        con.close()


def _capture_asset_mutation_snapshot(
    con: sqlite3.Connection,
    asset_id: str,
    *,
    columns: List[str],
    field_names: List[str],
) -> Dict[str, Any]:
    select_cols = ["id", "last_updated"]
    for column in columns:
        if column not in select_cols:
            select_cols.append(column)
    asset_row = con.execute(
        f"SELECT {', '.join(select_cols)} FROM assets WHERE id=?",
        (asset_id,),
    ).fetchone()
    if not asset_row:
        raise ValueError("asset not found")
    metadata_rows: List[Dict[str, Any]] = []
    if field_names:
        placeholders = ", ".join("?" for _ in field_names)
        rows = con.execute(
            f"""
            SELECT asset_id, field_name, value_json, value_text, value_real,
                   source_name, source_field, is_canonical, confidence, created_at, updated_at
            FROM metadata_fields
            WHERE asset_id=? AND field_name IN ({placeholders})
            """,
            (asset_id, *field_names),
        ).fetchall()
        metadata_rows = [dict(row) for row in rows]
    return {
        "type": "asset_state",
        "asset_id": asset_id,
        "columns": columns,
        "field_names": field_names,
        "asset": dict(asset_row),
        "metadata_rows": metadata_rows,
    }


def _capture_duplicate_group_snapshot(con: sqlite3.Connection, group_id: str) -> Dict[str, Any]:
    group = con.execute(
        "SELECT * FROM duplicate_groups WHERE id=?",
        (group_id,),
    ).fetchone()
    if not group:
        raise ValueError("duplicate group not found")
    items = con.execute(
        "SELECT * FROM duplicate_items WHERE group_id=? ORDER BY asset_id",
        (group_id,),
    ).fetchall()
    return {
        "type": "duplicate_group_state",
        "group_id": group_id,
        "group": dict(group),
        "items": [dict(row) for row in items],
    }


def _restore_asset_snapshot(con: sqlite3.Connection, snapshot: Dict[str, Any]) -> None:
    asset = snapshot.get("asset") or {}
    asset_id = snapshot.get("asset_id") or asset.get("id")
    if not asset_id:
        raise ValueError("asset snapshot missing asset id")
    columns = list(snapshot.get("columns") or [])
    if not columns:
        return
    updates = []
    params: List[Any] = []
    for column in columns:
        updates.append(f"{column}=?")
        params.append(asset.get(column))
    params.append(asset_id)
    con.execute(f"UPDATE assets SET {', '.join(updates)} WHERE id=?", params)
    field_names = list(snapshot.get("field_names") or [])
    if field_names:
        placeholders = ", ".join("?" for _ in field_names)
        con.execute(
            f"DELETE FROM metadata_fields WHERE asset_id=? AND field_name IN ({placeholders})",
            (asset_id, *field_names),
        )
        for row in snapshot.get("metadata_rows") or []:
            con.execute(
                """
                INSERT INTO metadata_fields (
                  asset_id, field_name, value_json, value_text, value_real,
                  source_name, source_field, is_canonical, confidence, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["asset_id"],
                    row["field_name"],
                    row["value_json"],
                    row["value_text"],
                    row["value_real"],
                    row["source_name"],
                    row["source_field"],
                    row["is_canonical"],
                    row["confidence"],
                    row["created_at"],
                    row["updated_at"],
                ),
            )


def _restore_duplicate_group_snapshot(con: sqlite3.Connection, snapshot: Dict[str, Any]) -> None:
    group = snapshot.get("group") or {}
    group_id = snapshot.get("group_id") or group.get("id")
    if not group_id:
        raise ValueError("duplicate group snapshot missing group id")
    con.execute("DELETE FROM duplicate_items WHERE group_id=?", (group_id,))
    con.execute("DELETE FROM duplicate_groups WHERE id=?", (group_id,))
    con.execute(
        """
        INSERT INTO duplicate_groups (id, group_type, status, canonical_asset_id, created_at, resolved_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            group["id"],
            group["group_type"],
            group["status"],
            group["canonical_asset_id"],
            group["created_at"],
            group.get("resolved_at"),
        ),
    )
    for row in snapshot.get("items") or []:
        con.execute(
            """
            INSERT INTO duplicate_items (group_id, asset_id, score, rationale, keep_decision)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                row["group_id"],
                row["asset_id"],
                row["score"],
                row["rationale"],
                row["keep_decision"],
            ),
        )


def _mutation_snapshot_for_review(con: sqlite3.Connection, asset_id: str) -> Dict[str, Any]:
    return _capture_asset_mutation_snapshot(
        con,
        asset_id,
        columns=["user_rating", "review_state", "review_score", "favorite", "ranked_at", "last_updated"],
        field_names=["user_rating", "review_state", "review_score", "favorite"],
    )


def _mutation_snapshot_for_people(con: sqlite3.Connection, asset_id: str) -> Dict[str, Any]:
    return _capture_asset_mutation_snapshot(
        con,
        asset_id,
        columns=["people_json", "last_updated"],
        field_names=["people"],
    )


def _sync_people_json_from_metadata(con: sqlite3.Connection, asset_id: str) -> List[str]:
    rows = con.execute(
        """
        SELECT value_json, value_text
        FROM metadata_fields
        WHERE asset_id = ?
          AND field_name = 'people'
        ORDER BY created_at ASC, source_name ASC, source_field ASC
        """,
        (asset_id,),
    ).fetchall()
    labels: List[str] = []
    seen: set[str] = set()
    for row in rows:
        value: Any
        raw_json = row["value_json"]
        raw_text = row["value_text"]
        if raw_json:
            try:
                value = json.loads(raw_json)
            except Exception:
                value = raw_json
        else:
            value = raw_text
        for label in _coerce_str_list(value):
            cleaned = label.strip()
            key = cleaned.casefold()
            if cleaned and key not in seen:
                seen.add(key)
                labels.append(cleaned)
    labels = sorted(labels, key=lambda item: item.casefold())
    con.execute(
        "UPDATE assets SET people_json=?, last_updated=? WHERE id=?",
        (json.dumps(labels), utcnow_iso(), asset_id),
    )
    return labels


def _sync_people_label_for_asset(
    con: sqlite3.Connection,
    asset_id: str,
    identity_id: str,
    label: str,
) -> None:
    _record_metadata_field(
        con,
        asset_id=asset_id,
        field_name="people",
        value=label,
        source_name="face_label",
        source_field=identity_id,
        is_canonical=False,
        confidence=0.98,
    )
    _sync_people_json_from_metadata(con, asset_id)


def _write_mutation_history(
    con: sqlite3.Connection,
    *,
    action_type: str,
    target_id: Optional[str],
    summary: str,
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> None:
    con.execute("DELETE FROM mutation_history WHERE undone_at IS NOT NULL")
    con.execute(
        """
        INSERT INTO mutation_history (
          action_type, target_id, summary, before_json, after_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            action_type,
            target_id,
            summary,
            json.dumps(before),
            json.dumps(after),
            utcnow_iso(),
        ),
    )


def get_mutation_history(db_path: Path, *, limit: int = 8) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT *
            FROM mutation_history
            ORDER BY seq DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def get_mutation_status(db_path: Path) -> Dict[str, Any]:
    con = connect(db_path)
    try:
        applied = con.execute(
            """
            SELECT seq, action_type, target_id, summary, created_at
            FROM mutation_history
            WHERE undone_at IS NULL
            ORDER BY seq DESC
            LIMIT 1
            """
        ).fetchone()
        undone = con.execute(
            """
            SELECT seq, action_type, target_id, summary, created_at, undone_at
            FROM mutation_history
            WHERE undone_at IS NOT NULL
            ORDER BY undone_at DESC, seq DESC
            LIMIT 1
            """
        ).fetchone()
        return {
            "can_undo": bool(applied),
            "can_redo": bool(undone),
            "last_action": dict(applied) if applied else None,
            "next_redo": dict(undone) if undone else None,
        }
    finally:
        con.close()


def undo_last_mutation(db_path: Path) -> Dict[str, Any]:
    con = connect(db_path)
    try:
        row = con.execute(
            """
            SELECT *
            FROM mutation_history
            WHERE undone_at IS NULL
            ORDER BY seq DESC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            raise ValueError("nothing to undo")
        payload = json.loads(row["before_json"])
        if payload.get("type") == "asset_state":
            _restore_asset_snapshot(con, payload)
        elif payload.get("type") == "duplicate_group_state":
            _restore_duplicate_group_snapshot(con, payload)
        else:
            raise ValueError(f"unsupported undo action: {row['action_type']}")
        con.execute(
            "UPDATE mutation_history SET undone_at=? WHERE seq=?",
            (utcnow_iso(), row["seq"]),
        )
        con.commit()
        return dict(row)
    finally:
        con.close()


def redo_last_mutation(db_path: Path) -> Dict[str, Any]:
    con = connect(db_path)
    try:
        row = con.execute(
            """
            SELECT *
            FROM mutation_history
            WHERE undone_at IS NOT NULL
            ORDER BY undone_at DESC, seq DESC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            raise ValueError("nothing to redo")
        payload = json.loads(row["after_json"])
        if payload.get("type") == "asset_state":
            _restore_asset_snapshot(con, payload)
        elif payload.get("type") == "duplicate_group_state":
            _restore_duplicate_group_snapshot(con, payload)
        else:
            raise ValueError(f"unsupported redo action: {row['action_type']}")
        con.execute(
            "UPDATE mutation_history SET undone_at=NULL, redone_at=? WHERE seq=?",
            (utcnow_iso(), row["seq"]),
        )
        con.commit()
        return dict(row)
    finally:
        con.close()


def restore_mutation_history_entry(db_path: Path, seq: int) -> Dict[str, Any]:
    con = connect(db_path)
    try:
        target = con.execute(
            """
            SELECT *
            FROM mutation_history
            WHERE seq=?
            """,
            (seq,),
        ).fetchone()
        if not target:
            raise ValueError("history entry not found")
        if target["undone_at"] is not None:
            raise ValueError("history entry is not currently active")
        later_rows = con.execute(
            """
            SELECT *
            FROM mutation_history
            WHERE seq > ? AND undone_at IS NULL
            ORDER BY seq DESC
            """,
            (seq,),
        ).fetchall()
        for row in later_rows:
            payload = json.loads(row["before_json"])
            if payload.get("type") == "asset_state":
                _restore_asset_snapshot(con, payload)
            elif payload.get("type") == "duplicate_group_state":
                _restore_duplicate_group_snapshot(con, payload)
            else:
                raise ValueError(f"unsupported history entry: {row['action_type']}")
            con.execute(
                "UPDATE mutation_history SET undone_at=? WHERE seq=?",
                (utcnow_iso(), row["seq"]),
            )
        con.commit()
        return dict(target)
    finally:
        con.close()


def set_asset_review(
    db_path: Path,
    asset_id: str,
    *,
    user_rating: Optional[int] = None,
    review_state: Optional[str] = None,
    review_score: Optional[float] = None,
    favorite: Optional[bool] = None,
) -> None:
    con = connect(db_path)
    try:
        before = _mutation_snapshot_for_review(con, asset_id)
        ts = utcnow_iso()
        updates = ["last_updated=?", "ranked_at=?"]
        params: list[Any] = [ts, ts]
        if user_rating is not None:
            rating = max(0, min(int(user_rating), 5))
            updates.append("user_rating=?")
            params.append(rating)
            _record_metadata_field(
                con,
                asset_id=asset_id,
                field_name="user_rating",
                value=rating,
                source_name="user",
                source_field="rating",
                is_canonical=True,
                confidence=1.0,
            )
        if review_state is not None:
            updates.append("review_state=?")
            params.append(review_state)
            _record_metadata_field(
                con,
                asset_id=asset_id,
                field_name="review_state",
                value=review_state,
                source_name="user",
                source_field="state",
                is_canonical=True,
                confidence=1.0,
            )
        if review_score is not None:
            updates.append("review_score=?")
            params.append(float(review_score))
            _record_metadata_field(
                con,
                asset_id=asset_id,
                field_name="review_score",
                value=float(review_score),
                source_name="user",
                source_field="score",
                is_canonical=False,
                confidence=1.0,
            )
        if favorite is not None:
            updates.append("favorite=?")
            params.append(1 if favorite else 0)
            _record_metadata_field(
                con,
                asset_id=asset_id,
                field_name="favorite",
                value=bool(favorite),
                source_name="user",
                source_field="favorite",
                is_canonical=True,
                confidence=1.0,
            )
        params.append(asset_id)
        con.execute(f"UPDATE assets SET {', '.join(updates)} WHERE id=?", params)
        after = _mutation_snapshot_for_review(con, asset_id)
        _write_mutation_history(
            con,
            action_type="SET_ASSET_REVIEW",
            target_id=asset_id,
            summary="Changed rating or review state",
            before=before,
            after=after,
        )
        con.commit()
    finally:
        con.close()


def add_asset_person(
    db_path: Path,
    asset_id: str,
    person: str,
    *,
    source_name: str = "user",
    source_field: str = "people",
) -> None:
    label = str(person or "").strip()
    if not label:
        return
    con = connect(db_path)
    try:
        before = _mutation_snapshot_for_people(con, asset_id)
        row = con.execute(
            """
            SELECT value_json, value_text
            FROM metadata_fields
            WHERE asset_id = ?
              AND field_name = 'people'
              AND source_name = ?
              AND source_field = ?
            LIMIT 1
            """,
            (asset_id, source_name, source_field),
        ).fetchone()
        people = _coerce_list(json.loads(row["value_json"]) if row and row["value_json"] else row["value_text"] if row else [])
        if label not in people:
            people.append(label)
        people = sorted(set(people))
        _record_metadata_field(
            con,
            asset_id=asset_id,
            field_name="people",
            value=people,
            source_name=source_name,
            source_field=source_field,
            is_canonical=True,
            confidence=1.0,
        )
        _sync_people_json_from_metadata(con, asset_id)
        after = _mutation_snapshot_for_people(con, asset_id)
        _write_mutation_history(
            con,
            action_type="ADD_ASSET_PERSON",
            target_id=asset_id,
            summary=f"Tagged person {label}",
            before=before,
            after=after,
        )
        con.commit()
    finally:
        con.close()


def remove_asset_person(
    db_path: Path,
    asset_id: str,
    person: str,
    *,
    source_name: str = "user",
    source_field: str = "people",
) -> None:
    label = str(person or "").strip()
    if not label:
        return
    con = connect(db_path)
    try:
        before = _mutation_snapshot_for_people(con, asset_id)
        row = con.execute(
            """
            SELECT value_json, value_text
            FROM metadata_fields
            WHERE asset_id = ?
              AND field_name = 'people'
              AND source_name = ?
              AND source_field = ?
            LIMIT 1
            """,
            (asset_id, source_name, source_field),
        ).fetchone()
        people = _coerce_list(json.loads(row["value_json"]) if row and row["value_json"] else row["value_text"] if row else [])
        people = [name for name in people if name != label]
        if people:
            _record_metadata_field(
                con,
                asset_id=asset_id,
                field_name="people",
                value=sorted(set(people)),
                source_name=source_name,
                source_field=source_field,
                is_canonical=True,
                confidence=1.0,
            )
        else:
            con.execute(
                """
                DELETE FROM metadata_fields
                WHERE asset_id = ?
                  AND field_name = 'people'
                  AND source_name = ?
                  AND COALESCE(source_field, '') = COALESCE(?, '')
                """,
                (asset_id, source_name, source_field),
            )
        _sync_people_json_from_metadata(con, asset_id)
        after = _mutation_snapshot_for_people(con, asset_id)
        _write_mutation_history(
            con,
            action_type="REMOVE_ASSET_PERSON",
            target_id=asset_id,
            summary=f"Removed person {label}",
            before=before,
            after=after,
        )
        con.commit()
    finally:
        con.close()


def list_assets_for_review(
    db_path: Path,
    *,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT a.id, a.source, a.source_kind, a.media_type, a.orig_filename, a.dt_original,
                   a.status, a.review_state, a.user_rating, a.review_score, a.favorite,
                   a.target_relpath, a.target_filename, m.status AS managed_status, m.managed_path
            FROM assets a
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            WHERE COALESCE(a.review_state, 'NEW') IN ('NEW', 'REVIEW', '')
               OR a.user_rating IS NULL
            ORDER BY COALESCE(a.user_rating, 0) ASC, COALESCE(a.review_score, 0) DESC, a.dt_original DESC, a.id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def list_asset_neighbors(db_path: Path, asset_id: str, *, sort: str = "recent") -> Dict[str, Optional[Dict[str, Any]]]:
    con = connect(db_path)
    try:
        order_clause = {
            "rated": "COALESCE(user_rating, 0) DESC, COALESCE(review_score, 0) DESC, dt_original DESC, id DESC",
            "review": "COALESCE(user_rating, 0) ASC, COALESCE(review_score, 0) DESC, dt_original DESC, id DESC",
            "recent": "dt_original DESC, id DESC",
        }.get(sort, "dt_original DESC, id DESC")
        row = con.execute(
            f"""
            WITH ordered AS (
              SELECT id,
                     LAG(id) OVER (ORDER BY {order_clause}) AS prev_id,
                     LEAD(id) OVER (ORDER BY {order_clause}) AS next_id
              FROM assets
            )
            SELECT prev_id, next_id
            FROM ordered
            WHERE id = ?
            LIMIT 1
            """,
            (asset_id,),
        ).fetchone()
        if not row:
            return {"prev": None, "next": None}
        prev_item = con.execute("SELECT id, orig_filename FROM assets WHERE id=?", (row["prev_id"],)).fetchone() if row["prev_id"] else None
        next_item = con.execute("SELECT id, orig_filename FROM assets WHERE id=?", (row["next_id"],)).fetchone() if row["next_id"] else None
        return {"prev": dict(prev_item) if prev_item else None, "next": dict(next_item) if next_item else None}
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


def clear_jobs(
    db_path: Path,
    *,
    job_type: Optional[str] = None,
    statuses: Optional[List[str]] = None,
) -> int:
    con = connect(db_path)
    try:
        clauses: List[str] = []
        params: List[Any] = []
        if job_type:
            clauses.append("job_type = ?")
            params.append(job_type)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        sql = "DELETE FROM jobs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        cur = con.execute(sql, params)
        con.commit()
        return int(cur.rowcount or 0)
    finally:
        con.close()


def get_pipeline_health(db_path: Path) -> Dict[str, Any]:
    init_db(db_path)
    con = connect(db_path)
    try:
        job_rows = con.execute(
            """
            SELECT job_type, status, COALESCE(error_msg, '') AS error_msg
            FROM jobs
            ORDER BY created_at DESC
            """
        ).fetchall()
        status_counts = {"COMPLETED": 0, "RUNNING": 0, "RETRYABLE": 0, "FAILED": 0, "QUEUED": 0}
        job_types: Dict[str, Dict[str, Any]] = {}
        for row in job_rows:
            status = row["status"] or "QUEUED"
            status_counts[status] = status_counts.get(status, 0) + 1
            bucket = job_types.setdefault(row["job_type"], {"total": 0, "failures": 0, "latest_error": ""})
            bucket["total"] += 1
            if status in {"FAILED", "RETRYABLE"}:
                bucket["failures"] += 1
                if row["error_msg"] and not bucket["latest_error"]:
                    bucket["latest_error"] = row["error_msg"]
        missing_managed = con.execute(
            """
            SELECT COUNT(*)
            FROM assets a
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            WHERE m.asset_id IS NULL
               OR COALESCE(m.managed_path, '') = ''
               OR m.status IN ('PLANNED', 'FAILED', 'MISSING')
            """
        ).fetchone()[0]
        missing_previews = con.execute(
            """
            SELECT COUNT(*)
            FROM assets a
            LEFT JOIN thumbnails t ON t.asset_id = a.id AND t.kind IN ('primary', 'video_preview')
            WHERE a.media_type IN ('image', 'raw', 'video')
              AND (t.path IS NULL OR t.status <> 'READY')
            """
        ).fetchone()[0]
        errored_assets = con.execute("SELECT COUNT(*) FROM assets WHERE status='ERROR'").fetchone()[0]
        retryable_jobs = status_counts.get("RETRYABLE", 0)
        failed_jobs = status_counts.get("FAILED", 0)
        healthy = (errored_assets == 0 and retryable_jobs == 0 and failed_jobs == 0 and missing_managed == 0)
        return {
            "healthy": healthy,
            "job_status_counts": status_counts,
            "job_types": job_types,
            "missing_managed_assets": missing_managed,
            "missing_previews": missing_previews,
            "errored_assets": errored_assets,
        }
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
    sort: str = "recent",
    media_type: Optional[str] = None,
    source: Optional[str] = None,
    review_state: Optional[str] = None,
    favorite_only: bool = False,
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
        where_sql, params = _asset_filter_sql(
            query=query,
            include_hidden=include_hidden,
            media_type=media_type,
            source=source,
            review_state=review_state,
            favorite_only=favorite_only,
        )
        order_clause = _asset_order_clause(sort)
        if where_sql:
            sql += " WHERE " + where_sql
        sql += f" ORDER BY {order_clause} LIMIT ? OFFSET ?"
        params.extend([limit, max(0, offset)])
        rows = con.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def list_assets_on_this_day(
    db_path: Path,
    *,
    limit: Optional[int] = 12,
    month_day: Optional[str] = None,
) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        month_day = month_day or datetime.now().strftime("%m-%d")
        sql = """
            SELECT a.id, a.source, a.source_kind, a.media_type, a.orig_filename, a.dt_original,
                   a.status, a.target_relpath, a.target_filename, m.status AS managed_status,
                   m.managed_path
            FROM assets a
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            WHERE a.dt_original IS NOT NULL
              AND substr(a.dt_original, 6, 5) = ?
            ORDER BY a.dt_original DESC, a.id DESC
            """
        params: list[Any] = [month_day]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = con.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def list_on_this_day_groups(
    db_path: Path,
    *,
    preferred_month_day: Optional[str] = None,
    limit: Optional[int] = 12,
) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        preferred_month_day = preferred_month_day or datetime.now().strftime("%m-%d")
        rows = con.execute(
            """
            SELECT substr(a.dt_original, 6, 5) AS month_day,
                   COUNT(*) AS asset_count,
                   MAX(a.dt_original) AS latest_dt
            FROM assets a
            WHERE a.dt_original IS NOT NULL
              AND length(a.dt_original) >= 10
            GROUP BY substr(a.dt_original, 6, 5)
            """,
        ).fetchall()
        groups = [dict(row) for row in rows]

        def day_number(value: str) -> int:
            try:
                parsed = datetime.strptime(f"2000-{value}", "%Y-%m-%d").date()
            except ValueError:
                return 0
            return parsed.timetuple().tm_yday

        preferred_day = day_number(preferred_month_day)

        def days_back(group: Dict[str, Any]) -> tuple[int, str]:
            group_month_day = str(group.get("month_day") or "")
            group_day = day_number(group_month_day)
            if not group_day:
                return (367, group_month_day)
            return ((preferred_day - group_day) % 366, group_month_day)

        groups.sort(key=days_back)
        if limit is not None:
            groups = groups[:limit]
        return groups
    finally:
        con.close()


def _asset_filter_sql(
    *,
    query: Optional[str] = None,
    include_hidden: bool = False,
    media_type: Optional[str] = None,
    source: Optional[str] = None,
    review_state: Optional[str] = None,
    favorite_only: bool = False,
) -> tuple[str, List[Any]]:
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
    if media_type:
        clauses.append("a.media_type = ?")
        params.append(media_type)
    if source:
        clauses.append("a.source = ?")
        params.append(source)
    if review_state:
        clauses.append("COALESCE(a.review_state, 'NEW') = ?")
        params.append(review_state)
    if favorite_only:
        clauses.append("COALESCE(a.favorite, 0) = 1")
    return " AND ".join(clauses), params


def count_assets(
    db_path: Path,
    *,
    query: Optional[str] = None,
    include_hidden: bool = False,
    media_type: Optional[str] = None,
    source: Optional[str] = None,
    review_state: Optional[str] = None,
    favorite_only: bool = False,
) -> int:
    con = connect(db_path)
    try:
        where_sql, params = _asset_filter_sql(
            query=query,
            include_hidden=include_hidden,
            media_type=media_type,
            source=source,
            review_state=review_state,
            favorite_only=favorite_only,
        )
        sql = "SELECT COUNT(*) FROM assets a"
        if where_sql:
            sql += " WHERE " + where_sql
        row = con.execute(sql, params).fetchone()
        return int(row[0] if row else 0)
    finally:
        con.close()


def list_asset_sources(db_path: Path) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT COALESCE(NULLIF(TRIM(source), ''), 'unknown') AS source,
                   COUNT(*) AS asset_count
            FROM assets
            GROUP BY COALESCE(NULLIF(TRIM(source), ''), 'unknown')
            ORDER BY lower(source) ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def _asset_order_clause(sort: str) -> str:
    return {
        "recent": "CASE WHEN COALESCE(m.status, '') = 'BUILT' THEN 0 ELSE 1 END, a.dt_original DESC, a.id DESC",
        "rated": "COALESCE(a.user_rating, 0) DESC, COALESCE(a.review_score, 0) DESC, a.dt_original DESC, a.id DESC",
        "favorites": "COALESCE(a.favorite, 0) DESC, COALESCE(a.user_rating, 0) DESC, COALESCE(a.review_score, 0) DESC, a.dt_original DESC, a.id DESC",
        "review": "COALESCE(a.user_rating, 0) ASC, COALESCE(a.review_score, 0) DESC, a.dt_original DESC, a.id DESC",
    }.get(sort, "a.dt_original DESC, a.id DESC")


def save_saved_search(
    db_path: Path,
    *,
    label: str,
    params: Dict[str, Any],
) -> str:
    con = connect(db_path)
    try:
        search_id = hashlib.sha1(f"{label}|{json.dumps(params, sort_keys=True)}".encode("utf-8")).hexdigest()[:20]
        ts = utcnow_iso()
        con.execute(
            """
            INSERT INTO saved_searches (id, label, params_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              label=excluded.label,
              params_json=excluded.params_json,
              updated_at=excluded.updated_at
            """,
            (search_id, label.strip() or "Saved Search", json.dumps(params, sort_keys=True), ts, ts),
        )
        con.commit()
        return search_id
    finally:
        con.close()


def list_saved_searches(db_path: Path, *, limit: int = 20) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT id, label, params_json, created_at, updated_at
            FROM saved_searches
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
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


def get_import_progress(db_path: Path) -> Dict[str, Any]:
    con = connect(db_path)
    try:
        total_assets = con.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
        ready_assets = con.execute(
            """
            SELECT COUNT(DISTINCT a.id)
            FROM assets a
            JOIN managed_assets m ON m.asset_id = a.id
            LEFT JOIN thumbnails t
              ON t.asset_id = a.id
             AND (
                    (a.media_type IN ('image', 'raw') AND t.kind = 'primary' AND t.status = 'READY')
                 OR (a.media_type = 'video' AND t.kind = 'video_preview' AND t.status = 'READY')
             )
            WHERE a.status IN ('EMBEDDED', 'COPIED')
              AND m.status = 'BUILT'
              AND t.asset_id IS NOT NULL
            """
        ).fetchone()[0]
        managed_ready = con.execute(
            "SELECT COUNT(*) FROM managed_assets WHERE status='BUILT'"
        ).fetchone()[0]
        previews_ready = con.execute(
            "SELECT COUNT(*) FROM thumbnails WHERE status='READY'"
        ).fetchone()[0]
        completion_pct = round((ready_assets / total_assets) * 100, 1) if total_assets else 0.0
        return {
            "total_assets": total_assets,
            "ready_assets": ready_assets,
            "completion_pct": completion_pct,
            "pending_assets": max(total_assets - ready_assets, 0),
            "managed_ready": managed_ready,
            "previews_ready": previews_ready,
        }
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


def list_duplicate_groups(
    db_path: Path,
    limit: int = 50,
    *,
    min_score: Optional[float] = None,
    group_type: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        clauses: List[str] = []
        params: List[Any] = []
        if group_type:
            clauses.append("g.group_type = ?")
            params.append(group_type)
        if status:
            clauses.append("g.status = ?")
            params.append(status)
        visible_count_expr = "SUM(CASE WHEN COALESCE(i.keep_decision, '') <> 'HIDE' THEN 1 ELSE 0 END)"
        visible_score_expr = "AVG(CASE WHEN COALESCE(i.keep_decision, '') <> 'HIDE' THEN COALESCE(i.score, 0) END)"
        having_clauses: List[str] = []
        if status == "OPEN":
            having_clauses.append(f"{visible_count_expr} > 1")
        if min_score is not None:
            having_clauses.append(f"COALESCE({visible_score_expr}, 0) >= ?")
        query_params: List[Any] = list(params)
        if min_score is not None:
            query_params.append(min_score)
        query_params.append(limit)
        rows = con.execute(
            """
            SELECT g.*, COUNT(CASE WHEN COALESCE(i.keep_decision, '') <> 'HIDE' THEN 1 END) AS item_count,
                   COALESCE(SUM(CASE WHEN COALESCE(i.keep_decision, '') <> 'HIDE' THEN 1 ELSE 0 END), 0) AS visible_item_count,
                   COALESCE(AVG(CASE WHEN COALESCE(i.keep_decision, '') <> 'HIDE' THEN COALESCE(i.score, 0) END), 0) AS match_score,
                   COALESCE(MAX(CASE WHEN COALESCE(i.keep_decision, '') <> 'HIDE' THEN COALESCE(i.score, 0) END), 0) AS best_score,
                   a.orig_filename AS canonical_filename
            FROM duplicate_groups g
            LEFT JOIN duplicate_items i ON i.group_id = g.id
            LEFT JOIN assets a ON a.id = g.canonical_asset_id
            """
            + (f" WHERE {' AND '.join(clauses)}" if clauses else "")
            + """
            GROUP BY g.id
            """
            + (" HAVING " + " AND ".join(having_clauses) if having_clauses else "")
            + """
            ORDER BY COALESCE(AVG(CASE WHEN COALESCE(i.keep_decision, '') <> 'HIDE' THEN COALESCE(i.score, 0) END), 0) DESC, g.created_at DESC
            LIMIT ?
            """,
            tuple(query_params),
        ).fetchall()
        result = [dict(row) for row in rows]
        for row in result:
            try:
                row["match_score_pct"] = round(float(row.get("match_score") or 0) * 100, 1)
                row["best_score_pct"] = round(float(row.get("best_score") or 0) * 100, 1)
            except Exception:
                row["match_score_pct"] = 0.0
                row["best_score_pct"] = 0.0
        return result
    finally:
        con.close()


def get_duplicate_group(db_path: Path, group_id: str) -> Optional[Dict[str, Any]]:
    con = connect(db_path)
    try:
        row = con.execute(
            """
            SELECT *
            FROM duplicate_groups
            WHERE id = ?
            """,
            (group_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def list_duplicate_group_items(db_path: Path, group_id: str, *, include_hidden: bool = False) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        hidden_clause = "" if include_hidden else " AND COALESCE(i.keep_decision, '') <> 'HIDE'"
        rows = con.execute(
            """
            SELECT i.*, a.orig_filename, a.media_type, a.dt_original, a.orig_size,
                   m.managed_path, t.path AS thumbnail_path
            FROM duplicate_items i
            JOIN assets a ON a.id = i.asset_id
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            LEFT JOIN thumbnails t ON t.asset_id = a.id AND t.kind IN ('primary', 'video_preview')
            WHERE i.group_id = ?
            """
            + hidden_clause
            + """
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
                   m.managed_path, c.suggested_label, c.score AS clarification_score, c.status AS clarification_status,
                   c.rationale AS clarification_rationale, c.suggested_identity_id
            FROM faces f
            LEFT JOIN face_identities i ON i.id = f.identity_id
            LEFT JOIN assets a ON a.id = f.asset_id
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            LEFT JOIN thumbnails t ON t.asset_id = a.id AND t.kind IN ('primary', 'video_preview')
            LEFT JOIN face_clarifications c ON c.face_id = f.id
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


def list_face_identities(
    db_path: Path,
    limit: int = 200,
    *,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        sql = """
            SELECT i.*, COUNT(f.id) AS face_count
            FROM face_identities i
            LEFT JOIN faces f ON f.identity_id = i.id
        """
        params: List[Any] = []
        if status:
            sql += " WHERE i.status = ?"
            params.append(status)
        sql += """
            GROUP BY i.id
            ORDER BY COUNT(f.id) DESC, i.label ASC
            LIMIT ?
        """
        params.append(limit)
        rows = con.execute(
            sql,
            params,
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def get_face_identity(db_path: Path, identity_id: str) -> Optional[Dict[str, Any]]:
    con = connect(db_path)
    try:
        row = con.execute(
            """
            SELECT i.*, COUNT(DISTINCT f.id) AS face_count, COUNT(DISTINCT f.asset_id) AS asset_count,
                   MAX(f.created_at) AS last_seen_at
            FROM face_identities i
            LEFT JOIN faces f ON f.identity_id = i.id AND f.status <> 'REJECTED'
            WHERE i.id = ?
            GROUP BY i.id
            """,
            (identity_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def list_face_clarifications(
    db_path: Path,
    *,
    limit: int = 200,
    status: str = "OPEN",
    suggested_identity_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        clause = "WHERE c.status = ?"
        params: list[Any] = [status]
        if suggested_identity_id:
            clause += " AND c.suggested_identity_id = ?"
            params.append(suggested_identity_id)
        rows = con.execute(
            f"""
            SELECT c.*, f.asset_id, f.identity_id, f.status AS face_status, f.bbox_json,
                   a.orig_filename, a.media_type, m.managed_path, t.path AS thumbnail_path,
                   i.label AS suggested_identity_label
            FROM face_clarifications c
            JOIN faces f ON f.id = c.face_id
            LEFT JOIN face_identities i ON i.id = c.suggested_identity_id
            LEFT JOIN assets a ON a.id = f.asset_id
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            LEFT JOIN thumbnails t ON t.asset_id = a.id AND t.kind IN ('primary', 'video_preview')
            {clause}
            ORDER BY c.score DESC, c.created_at DESC
            LIMIT ?
            """,
            (*params, limit),
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


def save_face_clarification(
    db_path: Path,
    face_id: str,
    *,
    suggested_identity_id: Optional[str],
    suggested_label: Optional[str],
    score: float,
    rationale: str,
) -> None:
    con = connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO face_clarifications (
              face_id, suggested_identity_id, suggested_label, score, rationale, status, created_at, resolved_at
            ) VALUES (?, ?, ?, ?, ?, 'OPEN', ?, NULL)
            ON CONFLICT(face_id) DO UPDATE SET
              suggested_identity_id=excluded.suggested_identity_id,
              suggested_label=excluded.suggested_label,
              score=excluded.score,
              rationale=excluded.rationale,
              status='OPEN',
              created_at=excluded.created_at,
              resolved_at=NULL
            """,
            (face_id, suggested_identity_id, suggested_label, score, rationale, utcnow_iso()),
        )
        con.execute("UPDATE faces SET status='REVIEW' WHERE id=?", (face_id,))
        con.commit()
    finally:
        con.close()


def resolve_face_clarification(db_path: Path, face_id: str, *, status: str = "RESOLVED") -> None:
    con = connect(db_path)
    try:
        con.execute(
            "UPDATE face_clarifications SET status=?, resolved_at=? WHERE face_id=?",
            (status, utcnow_iso(), face_id),
        )
        con.commit()
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


def save_embedding(
    db_path: Path,
    *,
    asset_id: str,
    embedding_type: str,
    vector: List[float],
    model_name: str,
    vector_ref: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> str:
    embedding_id = uuid.uuid4().hex[:20]
    con = connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO embeddings (id, asset_id, embedding_type, model_name, vector_ref, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                embedding_id,
                asset_id,
                embedding_type,
                model_name,
                vector_ref,
                json.dumps({"vector": vector, **(payload or {})}),
                utcnow_iso(),
            ),
        )
        con.commit()
        return embedding_id
    finally:
        con.close()


def replace_embedding(
    db_path: Path,
    *,
    asset_id: str,
    embedding_type: str,
    model_name: str,
    vector: List[float],
    vector_ref: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> str:
    con = connect(db_path)
    try:
        con.execute(
            "DELETE FROM embeddings WHERE asset_id = ? AND embedding_type = ? AND model_name = ?",
            (asset_id, embedding_type, model_name),
        )
        embedding_id = uuid.uuid4().hex[:20]
        con.execute(
            """
            INSERT INTO embeddings (id, asset_id, embedding_type, model_name, vector_ref, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                embedding_id,
                asset_id,
                embedding_type,
                model_name,
                vector_ref,
                json.dumps({"vector": vector, **(payload or {})}),
                utcnow_iso(),
            ),
        )
        con.commit()
        return embedding_id
    finally:
        con.close()


def list_asset_embeddings(db_path: Path, asset_id: str, *, embedding_type: Optional[str] = None) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        sql = """
            SELECT id, asset_id, embedding_type, model_name, vector_ref, payload_json, created_at
            FROM embeddings
            WHERE asset_id = ?
        """
        params: list[Any] = [asset_id]
        if embedding_type:
            sql += " AND embedding_type = ?"
            params.append(embedding_type)
        sql += " ORDER BY created_at ASC"
        rows = con.execute(sql, params).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                payload = json.loads(item.get("payload_json") or "{}")
            except Exception:
                payload = {}
            item["vector"] = payload.get("vector") or []
            item["payload"] = payload
            out.append(item)
        return out
    finally:
        con.close()


def list_face_embeddings(db_path: Path, *, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        sql = """
            SELECT f.id AS face_id, f.asset_id, f.identity_id, f.status, e.id AS embedding_id,
                   e.model_name, e.payload_json
            FROM faces f
            JOIN embeddings e ON e.id = f.embedding_ref
            WHERE f.status <> 'REJECTED'
              AND e.embedding_type = 'face'
        """
        params: list[Any] = []
        if model_name:
            sql += " AND e.model_name = ?"
            params.append(model_name)
        sql += " ORDER BY f.created_at ASC"
        rows = con.execute(sql, params).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            payload = json.loads(item.get("payload_json") or "{}")
            item["vector"] = payload.get("vector") or []
            out.append(item)
        return out
    finally:
        con.close()


def record_extraction_result(
    db_path: Path,
    asset_id: str,
    *,
    result_type: str,
    text_content: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    segment_start_ms: Optional[int] = None,
    segment_end_ms: Optional[int] = None,
    status: str = "READY",
) -> str:
    extraction_id = uuid.uuid4().hex[:20]
    con = connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO extraction_results (
              id, asset_id, result_type, segment_start_ms, segment_end_ms, text_content, payload_json, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                extraction_id,
                asset_id,
                result_type,
                segment_start_ms,
                segment_end_ms,
                text_content,
                json.dumps(payload or {}, sort_keys=True),
                status,
                utcnow_iso(),
            ),
        )
        con.commit()
        return extraction_id
    finally:
        con.close()


def list_extraction_results(db_path: Path, asset_id: str, *, result_type: Optional[str] = None) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        sql = """
            SELECT id, asset_id, result_type, segment_start_ms, segment_end_ms, text_content,
                   payload_json, status, created_at
            FROM extraction_results
            WHERE asset_id = ?
        """
        params: list[Any] = [asset_id]
        if result_type:
            sql += " AND result_type = ?"
            params.append(result_type)
        sql += " ORDER BY created_at ASC"
        rows = con.execute(sql, params).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item.get("payload_json") or "{}")
            except Exception:
                item["payload"] = {}
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
    refresh_people_for_identity(db_path, identity_id)
    refresh_confirmed_identity_matches(db_path, identity_id)


def merge_face_identities(db_path: Path, source_identity_id: str, target_identity_id: str) -> None:
    if source_identity_id == target_identity_id:
        return
    con = connect(db_path)
    try:
        source = con.execute("SELECT label FROM face_identities WHERE id=?", (source_identity_id,)).fetchone()
        target = con.execute("SELECT label FROM face_identities WHERE id=?", (target_identity_id,)).fetchone()
        if not source or not target:
            raise ValueError("face identity not found")
        asset_rows = con.execute(
            "SELECT DISTINCT asset_id FROM faces WHERE identity_id=? AND status <> 'REJECTED'",
            (source_identity_id,),
        ).fetchall()
        con.execute(
            "UPDATE faces SET identity_id=? WHERE identity_id=?",
            (target_identity_id, source_identity_id),
        )
        con.execute(
            "UPDATE face_clarifications SET suggested_identity_id=? WHERE suggested_identity_id=?",
            (target_identity_id, source_identity_id),
        )
        con.execute("DELETE FROM face_identities WHERE id=?", (source_identity_id,))
        for asset_row in asset_rows:
            asset_id = asset_row["asset_id"]
            con.execute(
                """
                DELETE FROM metadata_fields
                WHERE asset_id = ?
                  AND field_name = 'people'
                  AND source_name = 'face_label'
                  AND COALESCE(source_field, '') = COALESCE(?, '')
                """,
                (asset_id, source_identity_id),
            )
            _sync_people_label_for_asset(con, asset_id, target_identity_id, target["label"])
            _sync_people_json_from_metadata(con, asset_id)
        con.commit()
    finally:
        con.close()
    refresh_people_for_identity(db_path, target_identity_id)
    refresh_confirmed_identity_matches(db_path, target_identity_id)


def apply_identity_to_assets(db_path: Path, identity_id: str) -> None:
    con = connect(db_path)
    try:
        identity = con.execute("SELECT label FROM face_identities WHERE id=?", (identity_id,)).fetchone()
        if not identity:
            raise ValueError("identity not found")
        asset_rows = con.execute(
            "SELECT DISTINCT asset_id FROM faces WHERE identity_id=? AND status <> 'REJECTED'",
            (identity_id,),
        ).fetchall()
        for asset_row in asset_rows:
            _sync_people_label_for_asset(con, asset_row["asset_id"], identity_id, identity["label"])
        con.execute(
            "UPDATE faces SET status='LABELED' WHERE identity_id=? AND status <> 'REJECTED'",
            (identity_id,),
        )
        con.commit()
    finally:
        con.close()
    refresh_confirmed_identity_matches(db_path, identity_id)


def refresh_people_for_identity(db_path: Path, identity_id: str) -> None:
    con = connect(db_path)
    try:
        identity = con.execute("SELECT label FROM face_identities WHERE id=?", (identity_id,)).fetchone()
        if not identity:
            return
        asset_rows = con.execute(
            "SELECT DISTINCT asset_id FROM faces WHERE identity_id=? AND status <> 'REJECTED'",
            (identity_id,),
        ).fetchall()
        for asset_row in asset_rows:
            _sync_people_label_for_asset(con, asset_row["asset_id"], identity_id, identity["label"])
        con.commit()
    finally:
        con.close()
    refresh_confirmed_identity_matches(db_path, identity_id)


def replace_faces_for_asset(
    db_path: Path,
    asset_id: str,
    detections: List[Dict[str, Any]],
    *,
    source_name: str = "insightface",
) -> int:
    con = connect(db_path)
    try:
        prior_identity_rows = con.execute(
            """
            SELECT DISTINCT identity_id
            FROM faces
            WHERE asset_id=?
              AND identity_id IS NOT NULL
            """,
            (asset_id,),
        ).fetchall()
        prior_identity_ids = [str(row["identity_id"]) for row in prior_identity_rows if row["identity_id"]]
        if prior_identity_ids:
            placeholders = ", ".join("?" for _ in prior_identity_ids)
            con.execute(
                f"""
                DELETE FROM metadata_fields
                WHERE asset_id = ?
                  AND field_name = 'people'
                  AND source_name = 'face_label'
                  AND source_field IN ({placeholders})
                """,
                (asset_id, *prior_identity_ids),
            )
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
        _sync_people_json_from_metadata(con, asset_id)
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
    resolve_face_clarification(db_path, face_id, status="RESOLVED")


def reject_face(db_path: Path, face_id: str) -> None:
    con = connect(db_path)
    try:
        con.execute(
            "UPDATE faces SET status='REJECTED', identity_id=NULL WHERE id=?",
            (face_id,),
        )
        con.execute(
            "UPDATE face_clarifications SET status='DISMISSED', resolved_at=? WHERE face_id=?",
            (utcnow_iso(), face_id),
        )
        con.commit()
    finally:
        con.close()


def dismiss_face_suggestion(db_path: Path, face_id: str) -> None:
    con = connect(db_path)
    try:
        con.execute(
            """
            UPDATE faces
            SET status = CASE
                WHEN identity_id IS NULL THEN 'DETECTED'
                ELSE status
            END
            WHERE id = ?
            """,
            (face_id,),
        )
        con.execute(
            "UPDATE face_clarifications SET status='DISMISSED', resolved_at=? WHERE face_id=?",
            (utcnow_iso(), face_id),
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
        review = con.execute("SELECT COUNT(*) FROM faces WHERE status='REVIEW'").fetchone()[0]
        rejected = con.execute("SELECT COUNT(*) FROM faces WHERE status='REJECTED'").fetchone()[0]
        assets = con.execute("SELECT COUNT(DISTINCT asset_id) FROM faces WHERE status <> 'REJECTED'").fetchone()[0]
        identities = con.execute("SELECT COUNT(*) FROM face_identities").fetchone()[0]
        confirmed_identities = con.execute("SELECT COUNT(*) FROM face_identities WHERE status='CONFIRMED'").fetchone()[0]
        clustered_identities = con.execute("SELECT COUNT(*) FROM face_identities WHERE status='CLUSTERED'").fetchone()[0]
        strong_clustered_identities = con.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT i.id
              FROM face_identities i
              LEFT JOIN faces f ON f.identity_id = i.id AND f.status <> 'REJECTED'
              WHERE i.status = 'CLUSTERED'
              GROUP BY i.id
              HAVING COUNT(f.id) >= 3
            )
            """
        ).fetchone()[0]
        clarifications = con.execute("SELECT COUNT(*) FROM face_clarifications WHERE status='OPEN'").fetchone()[0]
        return {
            "total_faces": total,
            "unlabeled_faces": unlabeled,
            "clustered_faces": clustered,
            "labeled_faces": labeled,
            "review_faces": review,
            "rejected_faces": rejected,
            "clarifications_open": clarifications,
            "assets_with_faces": assets,
            "identities": identities,
            "confirmed_identities": confirmed_identities,
            "clustered_identities": clustered_identities,
            "strong_clustered_identities": strong_clustered_identities,
        }
    finally:
        con.close()


def list_person_albums(
    db_path: Path,
    *,
    limit: int = 100,
    query: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        params: List[Any] = []
        clauses: List[str] = []
        if query:
            clauses.append("lower(i.label) LIKE ?")
            params.append(f"%{query.lower()}%")
        if status:
            clauses.append("i.status = ?")
            params.append(status)
        clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = con.execute(
            f"""
            SELECT i.id, i.label, i.status, COUNT(DISTINCT f.id) AS face_count,
                   COUNT(DISTINCT f.asset_id) AS asset_count,
                   MAX(f.created_at) AS last_seen_at,
                   (
                     SELECT f2.asset_id
                     FROM faces f2
                     WHERE f2.identity_id = i.id
                       AND f2.status <> 'REJECTED'
                     ORDER BY f2.created_at DESC
                     LIMIT 1
                   ) AS cover_asset_id,
                   (
                     SELECT f2.id
                     FROM faces f2
                     WHERE f2.identity_id = i.id
                       AND f2.status <> 'REJECTED'
                     ORDER BY f2.created_at DESC
                     LIMIT 1
                   ) AS cover_face_id
            FROM face_identities i
            LEFT JOIN faces f ON f.identity_id = i.id AND f.status <> 'REJECTED'
            {clause}
            GROUP BY i.id
            {'' if status == 'CONFIRMED' else 'HAVING COUNT(DISTINCT f.id) > 0'}
            ORDER BY COUNT(DISTINCT f.id) DESC, i.label ASC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def list_assets_for_person(db_path: Path, label: str, *, limit: int = 48) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT a.id, a.orig_filename, a.media_type, a.dt_original, a.status,
                   a.user_rating, a.review_state, a.favorite, m.managed_path,
                   MAX(f.created_at) AS last_seen_at,
                   COUNT(f.id) AS face_count
            FROM faces f
            JOIN face_identities i ON i.id = f.identity_id
            JOIN assets a ON a.id = f.asset_id
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            WHERE lower(i.label) = lower(?)
              AND f.status <> 'REJECTED'
            GROUP BY a.id
            ORDER BY MAX(f.created_at) DESC, a.dt_original DESC, a.id DESC
            LIMIT ?
            """,
            (label, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def list_assets_tagged_with_person(db_path: Path, label: str, *, limit: int = 48) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT a.id, a.orig_filename, a.media_type, a.dt_original, a.status,
                   a.user_rating, a.review_state, a.favorite, a.people_json, m.managed_path
            FROM assets a
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            WHERE lower(COALESCE(a.people_json, '')) LIKE ?
            ORDER BY a.dt_original DESC, a.id DESC
            LIMIT ?
            """,
            (f"%{label.lower()}%", limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def list_assets_for_identity(db_path: Path, identity_id: str, *, limit: int = 48) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT a.id, a.orig_filename, a.media_type, a.dt_original, a.status,
                   a.user_rating, a.review_state, a.favorite, m.managed_path,
                   MAX(f.created_at) AS last_seen_at,
                   COUNT(f.id) AS face_count
            FROM faces f
            JOIN face_identities i ON i.id = f.identity_id
            JOIN assets a ON a.id = f.asset_id
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            WHERE i.id = ?
              AND f.status <> 'REJECTED'
            GROUP BY a.id
            ORDER BY MAX(f.created_at) DESC, a.dt_original DESC, a.id DESC
            LIMIT ?
            """,
            (identity_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def get_identity_model_name(db_path: Path, identity_id: str) -> Optional[str]:
    con = connect(db_path)
    try:
        preferred_model_name = get_preferred_face_model_name(db_path)
        if preferred_model_name:
            preferred = con.execute(
                """
                SELECT 1
                FROM faces f
                JOIN embeddings e ON e.id = f.embedding_ref
                WHERE f.identity_id = ?
                  AND f.status <> 'REJECTED'
                  AND e.embedding_type = 'face'
                  AND e.model_name = ?
                LIMIT 1
                """,
                (identity_id, preferred_model_name),
            ).fetchone()
            if preferred:
                return preferred_model_name
        row = con.execute(
            """
            SELECT e.model_name, COUNT(*) AS count
            FROM faces f
            JOIN embeddings e ON e.id = f.embedding_ref
            WHERE f.identity_id = ?
              AND f.status <> 'REJECTED'
              AND e.embedding_type = 'face'
              AND COALESCE(e.model_name, '') <> ''
            GROUP BY e.model_name
            ORDER BY count DESC, e.model_name ASC
            LIMIT 1
            """,
            (identity_id,),
        ).fetchone()
        return str(row["model_name"]) if row else None
    finally:
        con.close()


def get_preferred_face_model_name(db_path: Path) -> Optional[str]:
    con = connect(db_path)
    try:
        row = con.execute(
            """
            SELECT e.model_name, COUNT(*) AS count, MAX(e.created_at) AS last_seen_at
            FROM embeddings e
            WHERE e.embedding_type = 'face'
              AND COALESCE(e.model_name, '') <> ''
            GROUP BY e.model_name
            ORDER BY count DESC, last_seen_at DESC, e.model_name ASC
            LIMIT 1
            """
        ).fetchone()
        return str(row["model_name"]) if row else None
    finally:
        con.close()


def refresh_confirmed_identity_matches(
    db_path: Path,
    identity_id: str,
    *,
    auto_assign_threshold: float = PERSON_AUTO_ASSIGN_THRESHOLD,
) -> Dict[str, int]:
    identity = get_face_identity(db_path, identity_id)
    if not identity:
        return {"assigned": 0, "clarified": 0}
    if str(identity.get("status") or "") != "CONFIRMED":
        return {"assigned": 0, "clarified": 0}
    model_name = get_identity_model_name(db_path, identity_id)
    if not model_name:
        return {"assigned": 0, "clarified": 0}

    con = connect(db_path)
    try:
        own_embeddings = con.execute(
            """
            SELECT e.payload_json
            FROM faces f
            JOIN embeddings e ON e.id = f.embedding_ref
            WHERE f.identity_id = ?
              AND f.status <> 'REJECTED'
              AND e.embedding_type = 'face'
              AND e.model_name = ?
            """,
            (identity_id, model_name),
        ).fetchall()
        vectors: List[List[float]] = []
        for row in own_embeddings:
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except Exception:
                payload = {}
            vector = payload.get("vector") or []
            if vector:
                vectors.append([float(value) for value in vector])
        if not vectors:
            return {"assigned": 0, "clarified": 0}

        centroid = _centroid_from_vectors(vectors)

        face_count = max(0, int(identity.get("face_count") or 0))
        adaptive_auto_assign_threshold = _confirmed_auto_assign_threshold(
            face_count,
            vectors,
            base_threshold=auto_assign_threshold,
        )

        face_rows = con.execute(
            """
            SELECT f.id AS face_id, f.asset_id, e.payload_json
            FROM faces f
            JOIN embeddings e ON e.id = f.embedding_ref
            WHERE f.status <> 'REJECTED'
              AND e.embedding_type = 'face'
              AND e.model_name = ?
              AND f.identity_id IS NULL
            ORDER BY f.created_at DESC
            """,
            (model_name,),
        ).fetchall()

        assigned = 0
        clarified = 0
        touched_assets: set[str] = set()
        label = str(identity.get("label") or "")
        resolved_at = utcnow_iso()
        for row in face_rows:
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except Exception:
                payload = {}
            vector = payload.get("vector") or []
            if not vector:
                continue
            score = _cosine_similarity(vector, centroid)
            if score < PERSON_CANDIDATE_CLARIFICATION_THRESHOLD:
                continue
            if score < adaptive_auto_assign_threshold:
                con.execute(
                    """
                    INSERT INTO face_clarifications (
                      face_id, suggested_identity_id, suggested_label, score, rationale, status, created_at, resolved_at
                    ) VALUES (?, ?, ?, ?, ?, 'OPEN', ?, NULL)
                    ON CONFLICT(face_id) DO UPDATE SET
                      suggested_identity_id=excluded.suggested_identity_id,
                      suggested_label=excluded.suggested_label,
                      score=excluded.score,
                      rationale=excluded.rationale,
                      status='OPEN',
                      created_at=excluded.created_at,
                      resolved_at=NULL
                    """,
                    (
                        row["face_id"],
                        identity_id,
                        label,
                        score,
                        "confirmed identity candidate",
                        utcnow_iso(),
                    ),
                )
                con.execute("UPDATE faces SET status='REVIEW' WHERE id=?", (row["face_id"],))
                clarified += 1
                continue
            con.execute(
                "UPDATE faces SET identity_id=?, status='LABELED' WHERE id=?",
                (identity_id, row["face_id"]),
            )
            con.execute(
                "UPDATE face_clarifications SET status='RESOLVED', resolved_at=? WHERE face_id=?",
                (resolved_at, row["face_id"]),
            )
            asset_id = str(row["asset_id"])
            touched_assets.add(asset_id)
            _sync_people_label_for_asset(con, asset_id, identity_id, label)
            assigned += 1
        for asset_id in touched_assets:
            _sync_people_json_from_metadata(con, asset_id)
        con.commit()
    finally:
        con.close()

    return {"assigned": assigned, "clarified": clarified}


def prepare_face_rerun(db_path: Path) -> Dict[str, Any]:
    init_db(db_path)
    con = connect(db_path)
    try:
        cluster_rows = con.execute(
            "SELECT id FROM face_identities WHERE status <> 'CONFIRMED'"
        ).fetchall()
        cluster_identity_ids = [str(row["id"]) for row in cluster_rows if row["id"]]
        affected_asset_rows = con.execute(
            """
            SELECT DISTINCT asset_id
            FROM faces
            WHERE (status = 'REVIEW' AND identity_id IS NULL)
               OR identity_id IN (
                    SELECT id FROM face_identities WHERE status <> 'CONFIRMED'
               )
            """
        ).fetchall()
        affected_asset_ids = [str(row["asset_id"]) for row in affected_asset_rows if row["asset_id"]]

        cleared_faces = con.execute(
            """
            UPDATE faces
            SET identity_id = NULL,
                status = 'DETECTED'
            WHERE (status = 'REVIEW' AND identity_id IS NULL)
               OR identity_id IN (
                    SELECT id FROM face_identities WHERE status <> 'CONFIRMED'
               )
            """
        ).rowcount or 0

        cleared_clarifications = con.execute("DELETE FROM face_clarifications").rowcount or 0

        if cluster_identity_ids:
            placeholders = ", ".join("?" for _ in cluster_identity_ids)
            con.execute(
                f"""
                DELETE FROM metadata_fields
                WHERE field_name = 'people'
                  AND source_name = 'face_label'
                  AND source_field IN ({placeholders})
                """,
                cluster_identity_ids,
            )

        removed_identities = con.execute(
            "DELETE FROM face_identities WHERE status <> 'CONFIRMED'"
        ).rowcount or 0

        for asset_id in affected_asset_ids:
            _sync_people_json_from_metadata(con, asset_id)

        con.commit()
    finally:
        con.close()

    cleared_jobs = clear_jobs(db_path, job_type="face_detection", statuses=["RETRYABLE", "FAILED"])
    return {
        "cleared_faces": int(cleared_faces),
        "cleared_clarifications": int(cleared_clarifications),
        "removed_identities": int(removed_identities),
        "affected_assets": len(affected_asset_ids),
        "cleared_jobs": int(cleared_jobs),
        "preferred_model_name": get_preferred_face_model_name(db_path),
    }


def list_person_candidate_faces(db_path: Path, identity_id: str, *, limit: int = PERSON_CANDIDATE_LIMIT) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        identity = con.execute("SELECT label FROM face_identities WHERE id=?", (identity_id,)).fetchone()
        if not identity:
            return []
        model_name = get_identity_model_name(db_path, identity_id)
        if not model_name:
            return []
        own_embeddings = con.execute(
            """
            SELECT e.payload_json
            FROM faces f
            JOIN embeddings e ON e.id = f.embedding_ref
            WHERE f.identity_id = ?
              AND f.status <> 'REJECTED'
              AND e.embedding_type = 'face'
              AND e.model_name = ?
            """,
            (identity_id, model_name),
        ).fetchall()
        vectors: List[List[float]] = []
        for row in own_embeddings:
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except Exception:
                payload = {}
            vector = payload.get("vector") or []
            if vector:
                vectors.append([float(value) for value in vector])
        if not vectors:
            return []
        dim = max(len(vector) for vector in vectors)
        centroid = [0.0] * dim
        for vector in vectors:
            for index, value in enumerate(vector[:dim]):
                centroid[index] += float(value)
        scale = float(len(vectors)) or 1.0
        centroid = [value / scale for value in centroid]
        face_rows = con.execute(
            """
            SELECT f.id AS face_id, f.asset_id, f.identity_id, f.status, f.created_at,
                   e.payload_json, e.model_name,
                   a.orig_filename, a.media_type, m.managed_path,
                   t.path AS thumbnail_path,
                   i.label AS identity_label, i.status AS identity_status
            FROM faces f
            JOIN embeddings e ON e.id = f.embedding_ref
            LEFT JOIN assets a ON a.id = f.asset_id
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            LEFT JOIN thumbnails t ON t.asset_id = a.id AND t.kind IN ('primary', 'video_preview')
            LEFT JOIN face_identities i ON i.id = f.identity_id
            WHERE f.status <> 'REJECTED'
              AND e.embedding_type = 'face'
              AND e.model_name = ?
              AND f.identity_id IS NULL
            ORDER BY f.created_at DESC
            """,
            (model_name,),
        ).fetchall()
        ranked: List[Dict[str, Any]] = []
        for row in face_rows:
            item = dict(row)
            try:
                payload = json.loads(item.get("payload_json") or "{}")
            except Exception:
                payload = {}
            vector = payload.get("vector") or []
            if not vector:
                continue
            score = _cosine_similarity(vector, centroid)
            if score < PERSON_CANDIDATE_SCORE_FLOOR:
                continue
            item["vector"] = vector
            item["score"] = score
            ranked.append(item)
        ranked.sort(key=lambda item: (float(item.get("score") or 0.0), str(item.get("created_at") or "")), reverse=True)
        return ranked[:limit]
    finally:
        con.close()


def refresh_person_candidate_clarifications(
    db_path: Path,
    identity_id: str,
    *,
    limit: int = PERSON_CANDIDATE_REFRESH_LIMIT,
    threshold: float = PERSON_CANDIDATE_CLARIFICATION_THRESHOLD,
) -> int:
    return 0


def list_related_identity_candidates(
    db_path: Path,
    identity_id: str,
    *,
    limit: int = 12,
    threshold: float = 0.72,
) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        model_name = get_identity_model_name(db_path, identity_id)
        if not model_name:
            return []
        own_embeddings = con.execute(
            """
            SELECT e.payload_json
            FROM faces f
            JOIN embeddings e ON e.id = f.embedding_ref
            WHERE f.identity_id = ?
              AND f.status <> 'REJECTED'
              AND e.embedding_type = 'face'
              AND e.model_name = ?
            """,
            (identity_id, model_name),
        ).fetchall()
        own_vectors: List[List[float]] = []
        for row in own_embeddings:
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except Exception:
                payload = {}
            vector = payload.get("vector") or []
            if vector:
                own_vectors.append([float(value) for value in vector])
        if not own_vectors:
            return []
        dim = max(len(vector) for vector in own_vectors)
        target_centroid = [0.0] * dim
        for vector in own_vectors:
            for index, value in enumerate(vector[:dim]):
                target_centroid[index] += float(value)
        scale = float(len(own_vectors)) or 1.0
        target_centroid = [value / scale for value in target_centroid]
        identity_rows = con.execute(
            """
            SELECT i.id, i.label, i.status,
                   COUNT(DISTINCT f.id) AS face_count,
                   COUNT(DISTINCT f.asset_id) AS asset_count,
                   MAX(f.created_at) AS last_seen_at,
                   (
                     SELECT f2.asset_id
                     FROM faces f2
                     WHERE f2.identity_id = i.id
                       AND f2.status <> 'REJECTED'
                     ORDER BY f2.created_at DESC
                     LIMIT 1
                   ) AS cover_asset_id,
                   (
                     SELECT f2.id
                     FROM faces f2
                     WHERE f2.identity_id = i.id
                       AND f2.status <> 'REJECTED'
                     ORDER BY f2.created_at DESC
                     LIMIT 1
                   ) AS cover_face_id
            FROM face_identities i
            JOIN faces f ON f.identity_id = i.id AND f.status <> 'REJECTED'
            WHERE i.id <> ?
              AND i.status = 'CLUSTERED'
            GROUP BY i.id
            ORDER BY face_count DESC, asset_count DESC, i.label ASC
            """,
            (identity_id,),
        ).fetchall()
        ranked: List[Dict[str, Any]] = []
        for row in identity_rows:
            embedding_rows = con.execute(
                """
                SELECT e.payload_json
                FROM faces f
                JOIN embeddings e ON e.id = f.embedding_ref
                WHERE f.identity_id = ?
                  AND f.status <> 'REJECTED'
                  AND e.embedding_type = 'face'
                  AND e.model_name = ?
                """,
                (row["id"], model_name),
            ).fetchall()
            vectors: List[List[float]] = []
            for embedding_row in embedding_rows:
                try:
                    payload = json.loads(embedding_row["payload_json"] or "{}")
                except Exception:
                    payload = {}
                vector = payload.get("vector") or []
                if vector:
                    vectors.append([float(value) for value in vector])
            if not vectors:
                continue
            other_dim = max(len(vector) for vector in vectors)
            centroid = [0.0] * other_dim
            for vector in vectors:
                for index, value in enumerate(vector[:other_dim]):
                    centroid[index] += float(value)
            other_scale = float(len(vectors)) or 1.0
            centroid = [value / other_scale for value in centroid]
            score = _cosine_similarity(target_centroid, centroid)
            if score < threshold:
                continue
            item = dict(row)
            item["score"] = score
            ranked.append(item)
        ranked.sort(key=lambda item: (float(item.get("score") or 0.0), int(item.get("face_count") or 0)), reverse=True)
        return ranked[:limit]
    finally:
        con.close()


def list_geotagged_assets(db_path: Path, *, limit: int = 32) -> List[Dict[str, Any]]:
    con = connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT a.id, a.orig_filename, a.media_type, a.dt_original, a.gps_lat, a.gps_lon,
                   a.gps_alt, a.status, m.managed_path
            FROM assets a
            LEFT JOIN managed_assets m ON m.asset_id = a.id
            WHERE a.gps_lat IS NOT NULL AND a.gps_lon IS NOT NULL
            ORDER BY a.dt_original DESC, a.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
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


def _parse_iso_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _sequence_candidate_score(left: Dict[str, Any], right: Dict[str, Any]) -> tuple[float, str]:
    left_dt = _parse_iso_dt(left.get("dt_original"))
    right_dt = _parse_iso_dt(right.get("dt_original"))
    gap_seconds = abs((right_dt - left_dt).total_seconds()) if left_dt and right_dt else None
    if gap_seconds is None or gap_seconds > 180:
        return 0.0, "time gap too wide"

    score = 0.55
    rationale_bits: list[str] = []
    if left.get("source_locator") and left.get("source_locator") == right.get("source_locator"):
        score += 0.12
        rationale_bits.append("same export")
    if left.get("source_root") and left.get("source_root") == right.get("source_root"):
        score += 0.05
        rationale_bits.append("same root")
    if left_dt and right_dt and left_dt.date() == right_dt.date():
        score += 0.05
        rationale_bits.append("same day")
    if gap_seconds <= 10:
        score += 0.15
    elif gap_seconds <= 30:
        score += 0.10
    elif gap_seconds <= 60:
        score += 0.06
    else:
        score += 0.03
    size_left = float(left.get("orig_size") or 0)
    size_right = float(right.get("orig_size") or 0)
    if size_left > 0 and size_right > 0:
        ratio = min(size_left, size_right) / max(size_left, size_right)
        if ratio >= 0.95:
            score += 0.05
            rationale_bits.append("similar size")
        elif ratio >= 0.80:
            score += 0.03
            rationale_bits.append("close size")
    score = min(score, 0.95)
    rationale = f"gap={int(gap_seconds)}s" + (f"; {'; '.join(rationale_bits)}" if rationale_bits else "")
    return round(score, 3), rationale


def build_snapchat_sequence_groups(db_path: Path, *, limit: Optional[int] = None) -> int:
    con = connect(db_path)
    try:
        exact_asset_rows = con.execute(
            """
            SELECT di.asset_id
            FROM duplicate_groups g
            JOIN duplicate_items di ON di.group_id = g.id
            WHERE g.group_type = 'EXACT_SHA256'
              AND g.status = 'OPEN'
            """
        ).fetchall()
        exact_asset_ids = {row["asset_id"] for row in exact_asset_rows}
        params: list[Any] = []
        sql = """
            SELECT a.id, a.source, a.source_locator, a.source_root, a.media_type,
                   a.orig_filename, a.orig_size, a.dt_original
            FROM assets a
            WHERE a.source = 'snapchat'
              AND a.media_type = 'video'
              AND a.dt_original IS NOT NULL
        """
        if exact_asset_ids:
            placeholders = ",".join("?" for _ in exact_asset_ids)
            sql += f" AND a.id NOT IN ({placeholders})"
            params.extend(sorted(exact_asset_ids))
        sql += " ORDER BY COALESCE(a.source_locator, ''), a.dt_original ASC, a.orig_size DESC, a.id ASC"
        rows = [dict(row) for row in con.execute(sql, tuple(params)).fetchall()]
        if limit is not None:
            rows = rows[: int(limit)]

        by_source: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = str(row.get("source_locator") or row.get("source_root") or "")
            by_source.setdefault(key, []).append(row)

        groups: list[dict[str, Any]] = []
        for items in by_source.values():
            if len(items) < 2:
                continue
            parent: dict[str, str] = {item["id"]: item["id"] for item in items}

            def find(node: str) -> str:
                while parent[node] != node:
                    parent[node] = parent[parent[node]]
                    node = parent[node]
                return node

            def union(left: str, right: str) -> None:
                root_left = find(left)
                root_right = find(right)
                if root_left != root_right:
                    parent[root_right] = root_left

            for index, left in enumerate(items):
                for right in items[index + 1 :]:
                    score, _rationale = _sequence_candidate_score(left, right)
                    if score >= 0.68:
                        union(left["id"], right["id"])
                    else:
                        left_dt = _parse_iso_dt(left.get("dt_original"))
                        right_dt = _parse_iso_dt(right.get("dt_original"))
                        if left_dt and right_dt and abs((right_dt - left_dt).total_seconds()) > 180:
                            break

            clusters: dict[str, list[dict[str, Any]]] = {}
            for item in items:
                clusters.setdefault(find(item["id"]), []).append(item)

            for cluster_items in clusters.values():
                if len(cluster_items) < 2:
                    continue
                ranked = sorted(
                    cluster_items,
                    key=lambda item: (
                        -(item.get("orig_size") or 0),
                        item.get("dt_original") or "",
                        item["id"],
                    ),
                )
                canonical_asset_id = ranked[0]["id"]
                group_items: list[dict[str, Any]] = []
                for item in ranked:
                    if item["id"] == canonical_asset_id:
                        group_items.append(
                            {
                                "asset_id": item["id"],
                                "score": 0.95,
                                "rationale": "canonical sequence candidate",
                            }
                        )
                        continue
                    score, rationale = _sequence_candidate_score(ranked[0], item)
                    if score <= 0:
                        continue
                    group_items.append(
                        {
                            "asset_id": item["id"],
                            "score": score,
                            "rationale": rationale,
                        }
                    )
                if len(group_items) < 2:
                    continue
                groups.append(
                    {
                        "canonical_asset_id": canonical_asset_id,
                        "items": group_items,
                    }
                )

        return replace_duplicate_groups(db_path, group_type="SNAPCHAT_SEQUENCE", groups=groups)
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
        before = _capture_duplicate_group_snapshot(con, group_id)
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
        after = _capture_duplicate_group_snapshot(con, group_id)
        _write_mutation_history(
            con,
            action_type="RESOLVE_DUPLICATE_GROUP",
            target_id=group_id,
            summary="Resolved duplicate group",
            before=before,
            after=after,
        )
        con.commit()
    finally:
        con.close()


def keep_all_duplicate_group(db_path: Path, group_id: str) -> None:
    con = connect(db_path)
    try:
        before = _capture_duplicate_group_snapshot(con, group_id)
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
        after = _capture_duplicate_group_snapshot(con, group_id)
        _write_mutation_history(
            con,
            action_type="KEEP_ALL_DUPLICATE_GROUP",
            target_id=group_id,
            summary="Kept all duplicate items",
            before=before,
            after=after,
        )
        con.commit()
    finally:
        con.close()


def skip_duplicate_group(db_path: Path, group_id: str) -> None:
    con = connect(db_path)
    try:
        before = _capture_duplicate_group_snapshot(con, group_id)
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
        after = _capture_duplicate_group_snapshot(con, group_id)
        _write_mutation_history(
            con,
            action_type="SKIP_DUPLICATE_GROUP",
            target_id=group_id,
            summary="Skipped duplicate group",
            before=before,
            after=after,
        )
        con.commit()
    finally:
        con.close()


def hide_duplicate_group_item(db_path: Path, group_id: str, asset_id: str) -> None:
    con = connect(db_path)
    try:
        before = _capture_duplicate_group_snapshot(con, group_id)
        row = con.execute(
            "SELECT 1 FROM duplicate_items WHERE group_id=? AND asset_id=?",
            (group_id, asset_id),
        ).fetchone()
        if not row:
            raise ValueError("duplicate item not found")
        con.execute(
            "UPDATE duplicate_items SET keep_decision='HIDE' WHERE group_id=? AND asset_id=?",
            (group_id, asset_id),
        )
        visible_rows = con.execute(
            """
            SELECT asset_id
            FROM duplicate_items
            WHERE group_id=? AND COALESCE(keep_decision, '') <> 'HIDE'
            ORDER BY asset_id
            """,
            (group_id,),
        ).fetchall()
        visible_ids = [row["asset_id"] for row in visible_rows]
        if len(visible_ids) == 1:
            canonical_asset_id = visible_ids[0]
            con.execute(
                """
                UPDATE duplicate_groups
                SET canonical_asset_id=?, status='RESOLVED', resolved_at=?
                WHERE id=?
                """,
                (canonical_asset_id, utcnow_iso(), group_id),
            )
            con.execute(
                "UPDATE duplicate_items SET keep_decision='CANONICAL' WHERE group_id=? AND asset_id=?",
                (group_id, canonical_asset_id),
            )
        elif not visible_ids:
            con.execute(
                "UPDATE duplicate_groups SET status='SKIPPED', resolved_at=? WHERE id=?",
                (utcnow_iso(), group_id),
            )
        con.execute(
            """
            INSERT INTO review_decisions (id, asset_id, decision_type, value_json, created_at)
            VALUES (?, ?, 'duplicate_item_hidden', ?, ?)
            """,
            (
                uuid.uuid4().hex[:20],
                asset_id,
                json.dumps({"group_id": group_id, "hidden_asset_id": asset_id}),
                utcnow_iso(),
            ),
        )
        after = _capture_duplicate_group_snapshot(con, group_id)
        _write_mutation_history(
            con,
            action_type="HIDE_DUPLICATE_ITEM",
            target_id=group_id,
            summary="Removed duplicate item",
            before=before,
            after=after,
        )
        con.commit()
    finally:
        con.close()
