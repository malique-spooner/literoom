from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import ExifTags, Image, ImageFilter, ImageStat

from .metadata import manifest
from .utils.location import parse_location_candidate
from .utils.exiftool import read_core_metadata_batch


WEAK_TIME_SOURCES = {"filename", "file_mtime", "zip_mtime", "unknown", None, ""}
APPLE_CSV_PREFIX = "Photo Details-"

GPS_INFO_TAG = next((tag for tag, name in ExifTags.TAGS.items() if name == "GPSInfo"), None)
DATETIME_ORIGINAL_TAG = next((tag for tag, name in ExifTags.TAGS.items() if name == "DateTimeOriginal"), None)
DATETIME_TAG = next((tag for tag, name in ExifTags.TAGS.items() if name == "DateTime"), None)
MAKE_TAG = next((tag for tag, name in ExifTags.TAGS.items() if name == "Make"), None)
MODEL_TAG = next((tag for tag, name in ExifTags.TAGS.items() if name == "Model"), None)
SOFTWARE_TAG = next((tag for tag, name in ExifTags.TAGS.items() if name == "Software"), None)


def _parse_apple_csv_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    raw = value.strip()
    try:
        if raw.endswith("Z"):
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return datetime.fromisoformat(raw)
    except Exception:
        pass
    for fmt in ("%A %B %d,%Y %I:%M %p GMT", "%A %B %d,%Y %H:%M GMT"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _first_nested_value(obj: Any, keys: Iterable[str], *, max_depth: int = 6) -> Any:
    wanted = {key.lower() for key in keys}

    def walk(node: Any, depth: int) -> Any:
        if depth < 0 or node in (None, "", False):
            return None
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in wanted and value not in (None, "", False):
                    return value
            if depth == 0:
                return None
            for value in node.values():
                found = walk(value, depth - 1)
                if found not in (None, "", False):
                    return found
        elif isinstance(node, (list, tuple)):
            if depth == 0:
                return None
            for item in node:
                found = walk(item, depth - 1)
                if found not in (None, "", False):
                    return found
        return None

    return walk(obj, max_depth)


def _normalize_people_value(value: Any) -> list[str]:
    if value in (None, "", False):
        return []
    people: list[str] = []
    if isinstance(value, dict):
        label = _first_nested_value(value, ("name", "label", "displayName", "fullName", "title", "text"))
        if label:
            people.append(str(label))
        else:
            for nested in value.values():
                people.extend(_normalize_people_value(nested))
        return [item for item in people if item]
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                people.extend(_normalize_people_value(item))
            elif item:
                people.append(str(item))
        return [item for item in people if item]
    text = str(value).strip()
    return [text] if text else []


def _normalize_text_value(value: Any) -> Optional[str]:
    if value in (None, "", False):
        return None
    if isinstance(value, dict):
        nested = _first_nested_value(value, ("title", "description", "caption", "text", "value", "name", "label", "filename"))
        if nested not in (None, "", False):
            return str(nested)
        return None
    if isinstance(value, list):
        for item in value:
            text = _normalize_text_value(item)
            if text:
                return text
        return None
    text = str(value).strip()
    return text or None


def _build_apple_csv_index(source_roots: Iterable[Path]) -> dict[str, datetime]:
    index: dict[str, datetime] = {}
    for root in source_roots:
        root = Path(root)
        if not root.exists():
            continue
        for csv_path in root.rglob(f"{APPLE_CSV_PREFIX}*.csv"):
            try:
                with csv_path.open("r", encoding="utf-8", errors="replace") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        name = _normalize_text_value(
                            _first_nested_value(
                                row,
                                ("imgName", "filename", "fileName", "originalFilename", "name", "photoName", "assetName"),
                            )
                        )
                        dt = _parse_apple_csv_datetime(
                            _normalize_text_value(
                                _first_nested_value(
                                    row,
                                    ("originalCreationDate", "creationDate", "takenDate", "photoTakenTime", "dateCreated", "captureDate"),
                                )
                            )
                            or ""
                        )
                        if not name or not dt:
                            continue
                        index.setdefault(name.lower(), dt)
            except OSError:
                continue
    return index


def _resolve_source_path(row: dict, source_roots: Iterable[Path]) -> Optional[Path]:
    current = Path(row.get("source_locator") or row.get("abs_zip") or "")
    if current.exists():
        return current
    source_path = row.get("source_path")
    if source_path:
        for root in source_roots:
            candidate = Path(root) / str(source_path)
            if candidate.exists():
                return candidate
    return None


def _asset_pair_key(filename: str) -> str:
    stem = Path(filename).stem.lower()
    for suffix in ("_original", "-original", "-edited", "_edited", "-main", "_main", "-overlay", "_overlay"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _location_payload(record: dict) -> Optional[dict]:
    lat = record.get("gps_lat")
    lon = record.get("gps_lon")
    if lat is None or lon is None:
        return None
    return {
        "gps_lat": lat,
        "gps_lon": lon,
        "gps_alt": record.get("gps_alt"),
        "gps_source": record.get("gps_source") or "paired_asset",
    }


def _text_payload(record: dict) -> dict:
    payload: dict = {}
    for key in ("title", "description"):
        if record.get(key):
            payload[key] = record[key]
    keywords = record.get("keywords")
    if keywords:
        payload["keywords"] = [str(item) for item in keywords if item]
    return payload


def _device_payload(record: dict) -> dict:
    keys = (
        "camera_make",
        "camera_model",
        "software",
        "lens_model",
        "pixel_width",
        "pixel_height",
        "format",
        "duration_seconds",
        "frame_rate",
        "bitrate",
        "audio_channels",
        "video_codec",
        "audio_codec",
        "rotation_degrees",
        "content_identifier",
        "container_format",
        "encoder",
        "mime_type",
    )
    return {key: record[key] for key in keys if record.get(key) not in (None, "", [], {})}


def _metadata_confidence(source_name: str, field_name: str, *, paired: bool = False) -> float:
    base = {
        "embedded_metadata": 0.97,
        "image_exif": 0.96,
        "source_read": 0.95,
        "apple_csv": 0.92,
        "sidecar_metadata": 0.9,
        "metadata_repair": 0.88,
        "paired_metadata": 0.72,
        "screenshot_detector": 0.8,
        "face_label": 0.98,
    }.get(source_name, 0.7)
    if field_name in {"title", "description", "keywords"} and source_name == "sidecar_metadata":
        base = 0.94
    if field_name in {"captured_at", "captured_at_utc", "timezone_name"} and source_name == "apple_csv":
        base = 0.95
    if paired:
        base = min(base, 0.72)
    return base


def _parse_timezone_offset(value: Any) -> Optional[timezone]:
    if value in (None, "", "Z", "UTC", "GMT"):
        return timezone.utc if value in {"Z", "UTC", "GMT"} else None
    text = str(value).strip()
    match = re.match(r"^([+-])(\d{2}):?(\d{2})$", text)
    if not match:
        return None
    sign = 1 if match.group(1) == "+" else -1
    hours = int(match.group(2))
    minutes = int(match.group(3))
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def _timezone_name_for_offset(tz_value: Any) -> Optional[str]:
    if tz_value in (None, ""):
        return None
    text = str(tz_value).strip()
    if text in {"Z", "UTC", "GMT"}:
        return "UTC"
    tz = _parse_timezone_offset(text)
    if not tz:
        return text
    offset = tz.utcoffset(None) or timedelta(0)
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    total_minutes = abs(total_minutes)
    hours, minutes = divmod(total_minutes, 60)
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


def _attach_timezone(dt_text: str, tz_value: Any) -> tuple[str, Optional[str], Optional[str]]:
    if not dt_text:
        return dt_text, None, None
    try:
        naive = datetime.strptime(dt_text[:19], "%Y-%m-%dT%H:%M:%S")
    except Exception:
        return dt_text, None, None
    tz = _parse_timezone_offset(tz_value)
    if not tz:
        return dt_text, None, _timezone_name_for_offset(tz_value)
    aware = naive.replace(tzinfo=tz)
    local_text = aware.isoformat(timespec="seconds")
    utc_text = aware.astimezone(timezone.utc).isoformat(timespec="seconds")
    return local_text, utc_text, _timezone_name_for_offset(tz_value)


def _gps_to_decimal(values, ref) -> Optional[float]:
    if not values or ref is None:
        return None
    try:
        d = float(values[0])
        m = float(values[1])
        s = float(values[2])
        value = d + (m / 60.0) + (s / 3600.0)
        if str(ref).upper() in {"S", "W"}:
            value *= -1
        return value
    except Exception:
        return None


def _read_image_metadata(path: Path) -> dict:
    result: dict = {}
    try:
        with Image.open(path) as img:
            result["pixel_width"], result["pixel_height"] = img.size
            result["format"] = img.format
            exif = img.getexif()
            if not exif:
                return result
            dt_value = None
            if DATETIME_ORIGINAL_TAG and exif.get(DATETIME_ORIGINAL_TAG):
                dt_value = exif.get(DATETIME_ORIGINAL_TAG)
                result["dt_source"] = "exif_datetimeoriginal"
            elif DATETIME_TAG and exif.get(DATETIME_TAG):
                dt_value = exif.get(DATETIME_TAG)
                result["dt_source"] = "exif_datetime"
            if dt_value:
                try:
                    parsed = datetime.strptime(str(dt_value), "%Y:%m:%d %H:%M:%S")
                    result["dt_original"] = parsed.strftime("%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    pass

            if MAKE_TAG and exif.get(MAKE_TAG):
                result["camera_make"] = str(exif.get(MAKE_TAG))
            if MODEL_TAG and exif.get(MODEL_TAG):
                result["camera_model"] = str(exif.get(MODEL_TAG))
            if SOFTWARE_TAG and exif.get(SOFTWARE_TAG):
                result["software"] = str(exif.get(SOFTWARE_TAG))

            gps = exif.get(GPS_INFO_TAG) if GPS_INFO_TAG else None
            if gps:
                lat = _gps_to_decimal(gps.get(2), gps.get(1))
                lon = _gps_to_decimal(gps.get(4), gps.get(3))
                alt = None
                if gps.get(6) is not None:
                    try:
                        alt = float(gps.get(6))
                    except Exception:
                        alt = None
                if lat is not None and lon is not None:
                    result["gps_lat"] = lat
                    result["gps_lon"] = lon
                    result["gps_alt"] = alt
                    result["gps_source"] = "exif_gps"
    except Exception:
        return result
    return result


def _candidate_sidecar_paths(source_path: Path) -> list[Path]:
    return [
        source_path.with_suffix(".json"),
        source_path.with_name(source_path.name + ".json"),
        source_path.with_suffix(".supplemental-metadata.json"),
        source_path.with_name(source_path.name + ".supplemental-metadata.json"),
    ]


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _coerce_sidecar_timestamp(value: Any) -> Optional[str]:
    seen: set[int] = set()

    def coerce(node: Any, depth: int = 6) -> Optional[str]:
        if depth < 0 or node in (None, "", False):
            return None
        marker = id(node)
        if marker in seen:
            return None
        seen.add(marker)

        if isinstance(node, dict):
            for key in ("timestamp", "seconds", "secondsSinceEpoch", "unixTime", "time"):
                if key in node:
                    coerced = coerce(node.get(key), depth - 1)
                    if coerced:
                        return coerced
            for key in ("formatted", "dateTime", "datetime"):
                if key in node:
                    cleaned = _normalize_text_value(node.get(key))
                    if cleaned:
                        cleaned = cleaned.replace("UTC", "").strip()
                        for fmt in ("%d %b %Y, %H:%M:%S", "%d %b %Y, %H:%M"):
                            try:
                                return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%dT%H:%M:%S")
                            except Exception:
                                continue
                    return None
            if depth == 0:
                return None
            for nested in node.values():
                coerced = coerce(nested, depth - 1)
                if coerced:
                    return coerced
            return None

        if isinstance(node, (list, tuple)):
            if depth == 0:
                return None
            for item in node:
                coerced = coerce(item, depth - 1)
                if coerced:
                    return coerced
            return None

        text = str(node).strip()
        if not text:
            return None
        if text.endswith("Z"):
            try:
                return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%Y-%m-%dT%H:%M:%S")
            except Exception:
                pass
        try:
            return datetime.fromisoformat(text).strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            pass
        try:
            ts = int(text)
            if abs(ts) >= 10_000_000_000:
                ts //= 1000
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None

    return coerce(value)


def _parse_sidecar_metadata(source_path: Path) -> dict:
    for candidate in _candidate_sidecar_paths(source_path):
        if not candidate.exists():
            continue
        obj = _read_json(candidate)
        if not isinstance(obj, dict):
            continue
        result: dict[str, Any] = {"sidecar_path": str(candidate)}
        loc = parse_location_candidate(
            obj.get("location")
            or obj.get("geoData")
            or obj.get("geoDataExif")
            or obj.get("geo")
            or obj.get("coordinates")
            or obj.get("position")
            or obj
        )
        if loc:
            result["gps_lat"] = loc["lat"]
            result["gps_lon"] = loc["lon"]
            result["gps_alt"] = loc.get("alt")
            result["gps_source"] = "sidecar_json"
        title = _normalize_text_value(_first_nested_value(obj, ("title", "filename", "name")))
        if title:
            result["title"] = title
        description = _normalize_text_value(_first_nested_value(obj, ("description", "caption", "Caption", "story")))
        if description:
            result["description"] = description
        keywords = _first_nested_value(obj, ("keywords", "tags", "subject", "labels"))
        if isinstance(keywords, list):
            result["keywords"] = [str(item) for item in keywords if item]
        elif keywords:
            result["keywords"] = [str(keywords)]
        people = _first_nested_value(obj, ("people", "faces", "faceNames", "recognizedFaces", "faceAnnotations"))
        normalized_people = _normalize_people_value(people)
        if normalized_people:
            result["people"] = normalized_people
        make = obj.get("deviceMake") or obj.get("make")
        model = obj.get("deviceModel") or obj.get("model")
        if not make and isinstance(obj.get("googlePhotosOrigin"), dict):
            mobile_upload = obj["googlePhotosOrigin"].get("mobileUpload") or {}
            if isinstance(mobile_upload, dict):
                make = mobile_upload.get("deviceType") or make
                model = mobile_upload.get("deviceModel") or model
        if make:
            result["camera_make"] = str(make)
        if model:
            result["camera_model"] = str(model)
        dt_original = _coerce_sidecar_timestamp(_first_nested_value(obj, ("photoTakenTime", "creationTime", "modificationTime")))
        dt_source = "sidecar_timestamp"
        if not dt_original:
            dt_original = _coerce_sidecar_timestamp(_first_nested_value(obj, ("creationTime", "modificationTime")))
        if dt_original:
            result["dt_original"] = dt_original
            result["dt_source"] = dt_source
        timezone_text = obj.get("timezone") or obj.get("timeZone") or obj.get("offsetTime")
        if timezone_text:
            result["timezone_offset"] = str(timezone_text)
        elif isinstance(obj.get("photoTakenTime"), dict) and obj["photoTakenTime"].get("formatted"):
            formatted = str(obj["photoTakenTime"]["formatted"])
            if "UTC" in formatted:
                result["timezone_offset"] = "UTC"
        return result
    return {}


def _blur_score(path: Path) -> float:
    try:
        with Image.open(path) as img:
            gray = np.asarray(img.convert("L"), dtype="float32")
            if gray.shape[0] < 3 or gray.shape[1] < 3:
                return 0.0
            center = gray[1:-1, 1:-1]
            laplacian = (
                4.0 * center
                - gray[1:-1, :-2]
                - gray[1:-1, 2:]
                - gray[:-2, 1:-1]
                - gray[2:, 1:-1]
            )
            return float(laplacian.var())
    except Exception:
        return 0.0


def _is_probable_blurry(path: Path, image_meta: dict) -> tuple[bool, float]:
    score = _blur_score(path)
    width = image_meta.get("pixel_width") or 0
    height = image_meta.get("pixel_height") or 0
    if max(width, height) < 800:
        return False, score
    return score < 18.0, score


def _is_probable_screenshot(row: dict, source_path: Path, image_meta: dict) -> bool:
    filename = (row.get("orig_filename") or "").lower()
    if "screenshot" in filename:
        return True
    if source_path.suffix.lower() == ".png" and "iphone take out" in str(source_path.parent).lower():
        return True
    width = image_meta.get("pixel_width")
    height = image_meta.get("pixel_height")
    if source_path.suffix.lower() == ".png" and width and height:
        if max(width, height) >= 2000 and min(width, height) >= 1000:
            return True
    return False


def repair_metadata(
    db_path: Path,
    managed_library_dir: Path,
    *,
    exiftool_path: Optional[Path] = None,
    source_roots: Optional[Iterable[Path]] = None,
    limit: int | None = None,
) -> dict:
    managed_library_dir = Path(managed_library_dir)
    roots = [Path(root) for root in (source_roots or [])]
    apple_csv_index = _build_apple_csv_index(roots)

    rows = list(manifest.iter_assets_for_metadata_repair(db_path=db_path, limit=limit))
    path_map: dict[str, dict] = {}
    managed_paths: list[Path] = []
    source_path_map: dict[str, dict] = {}
    source_paths: list[Path] = []
    paired_locations: dict[str, dict] = {}
    paired_text: dict[str, dict] = {}
    paired_device: dict[str, dict] = {}
    for row in rows:
        managed_path = row.get("managed_path")
        if not managed_path:
            continue
        full_path = managed_library_dir / managed_path
        if full_path.exists():
            normalized = str(full_path.resolve())
            path_map[normalized] = row
            managed_paths.append(full_path)
        source_path = _resolve_source_path(row, roots)
        if source_path and source_path.exists():
            normalized_source = os.path.normcase(os.path.abspath(str(source_path)))
            source_path_map[normalized_source] = row
            source_paths.append(source_path)

    readback = read_core_metadata_batch(managed_paths, exiftool_path=exiftool_path)
    source_readback = read_core_metadata_batch(source_paths, exiftool_path=exiftool_path)

    repaired = examined = screenshot_count = 0
    assets_with_location_after_primary: set[str] = set()
    for normalized, row in path_map.items():
        examined += 1
        probed = readback.get(normalized, {})
        source_path = _resolve_source_path(row, roots)
        image_meta: dict = {}
        sidecar_meta: dict = {}
        source_probed = source_readback.get(os.path.normcase(os.path.abspath(str(source_path))), {}) if source_path else {}
        if source_path and source_path.exists():
            if str(source_path) != str(row.get("source_locator") or row.get("abs_zip") or ""):
                manifest.update_source_locator(
                    db_path,
                    row["id"],
                    source_locator=str(source_path),
                    source_root=str(source_path.parent),
                )
            if row.get("media_type") in {"image", "raw"}:
                image_meta = _read_image_metadata(source_path)
            sidecar_meta = _parse_sidecar_metadata(source_path)

        dt_original = row.get("dt_original")
        source_dt = row.get("source_dt")
        gps_lat = row.get("gps_lat")
        gps_lon = row.get("gps_lon")
        updates: dict = {}

        dt_timezone = sidecar_meta.get("timezone_offset") or source_probed.get("timezone_offset") or probed.get("timezone_offset")
        if image_meta.get("dt_original") and (not dt_original or source_dt in WEAK_TIME_SOURCES):
            updates["dt_original"] = image_meta["dt_original"]
            updates["dt_source"] = image_meta.get("dt_source") or "image_exif"
        elif sidecar_meta.get("dt_original") and (not dt_original or source_dt in WEAK_TIME_SOURCES):
            updates["dt_original"] = sidecar_meta.get("dt_original")
            updates["dt_source"] = sidecar_meta.get("dt_source") or "sidecar_json"
        elif probed.get("dt_original") and (not dt_original or source_dt in WEAK_TIME_SOURCES):
            updates["dt_original"] = probed.get("dt_original")
            updates["dt_source"] = (probed.get("dt_source") or "managed_read").lower()
        elif row.get("source") == "local":
            apple_dt = apple_csv_index.get((row.get("orig_filename") or "").lower())
            if apple_dt and (not dt_original or source_dt in WEAK_TIME_SOURCES):
                updates["dt_original"] = apple_dt.strftime("%Y-%m-%dT%H:%M:%S")
                updates["dt_source"] = "apple_csv_gmt"
                manifest.set_metadata_field(
                    db_path,
                    row["id"],
                    field_name="captured_at_utc",
                    value=apple_dt.isoformat(),
                    source_name="apple_csv",
                    source_field="originalCreationDate",
                    is_canonical=True,
                    confidence=_metadata_confidence("apple_csv", "captured_at_utc"),
                )
                manifest.set_metadata_field(
                    db_path,
                    row["id"],
                    field_name="timezone_name",
                    value="UTC",
                    source_name="apple_csv",
                    source_field="originalCreationDate",
                    is_canonical=True,
                    confidence=_metadata_confidence("apple_csv", "timezone_name"),
                )

        final_dt = updates.get("dt_original") or row.get("dt_original")
        if final_dt and dt_timezone:
            localized_dt, utc_dt, timezone_name = _attach_timezone(final_dt, dt_timezone)
            if localized_dt and updates.get("dt_original"):
                updates["dt_original"] = localized_dt
            if utc_dt:
                manifest.set_metadata_field(
                    db_path,
                    row["id"],
                    field_name="captured_at_utc",
                    value=utc_dt,
                    source_name="embedded_metadata" if source_probed.get("timezone_offset") or probed.get("timezone_offset") else "sidecar_metadata",
                    source_field="timezone_offset",
                    is_canonical=True,
                    confidence=_metadata_confidence(
                        "embedded_metadata" if source_probed.get("timezone_offset") or probed.get("timezone_offset") else "sidecar_metadata",
                        "captured_at_utc",
                    ),
                )
            if timezone_name:
                manifest.set_metadata_field(
                    db_path,
                    row["id"],
                    field_name="timezone_name",
                    value=timezone_name,
                    source_name="embedded_metadata" if source_probed.get("timezone_offset") or probed.get("timezone_offset") else "sidecar_metadata",
                    source_field="timezone_offset",
                    is_canonical=True,
                    confidence=_metadata_confidence(
                        "embedded_metadata" if source_probed.get("timezone_offset") or probed.get("timezone_offset") else "sidecar_metadata",
                        "timezone_name",
                    ),
                )

        if image_meta.get("gps_lat") is not None and image_meta.get("gps_lon") is not None and (gps_lat is None or gps_lon is None):
            updates["gps_lat"] = image_meta.get("gps_lat")
            updates["gps_lon"] = image_meta.get("gps_lon")
            updates["gps_alt"] = image_meta.get("gps_alt")
            updates["gps_source"] = image_meta.get("gps_source") or "image_exif"
        elif sidecar_meta.get("gps_lat") is not None and sidecar_meta.get("gps_lon") is not None and (gps_lat is None or gps_lon is None):
            updates["gps_lat"] = sidecar_meta.get("gps_lat")
            updates["gps_lon"] = sidecar_meta.get("gps_lon")
            updates["gps_alt"] = sidecar_meta.get("gps_alt")
            updates["gps_source"] = sidecar_meta.get("gps_source") or "sidecar_json"
        elif source_probed.get("gps_lat") is not None and source_probed.get("gps_lon") is not None and (gps_lat is None or gps_lon is None):
            updates["gps_lat"] = source_probed.get("gps_lat")
            updates["gps_lon"] = source_probed.get("gps_lon")
            updates["gps_alt"] = source_probed.get("gps_alt")
            updates["gps_source"] = source_probed.get("gps_source") or "source_read"
        elif probed.get("gps_lat") is not None and probed.get("gps_lon") is not None and (gps_lat is None or gps_lon is None):
            updates["gps_lat"] = probed.get("gps_lat")
            updates["gps_lon"] = probed.get("gps_lon")
            updates["gps_alt"] = probed.get("gps_alt")
            updates["gps_source"] = probed.get("gps_source") or "managed_read"

        if sidecar_meta.get("title") and not row.get("title"):
            updates["title"] = sidecar_meta.get("title")
        elif source_probed.get("title") and not row.get("title"):
            updates["title"] = source_probed.get("title")
        elif probed.get("title") and not row.get("title"):
            updates["title"] = probed.get("title")
        if sidecar_meta.get("description") and not row.get("description"):
            updates["description"] = sidecar_meta.get("description")
        elif source_probed.get("description") and not row.get("description"):
            updates["description"] = source_probed.get("description")
        elif probed.get("description") and not row.get("description"):
            updates["description"] = probed.get("description")
        if sidecar_meta.get("people") and not row.get("people"):
            updates["people"] = sidecar_meta.get("people")
        elif source_probed.get("people") and not row.get("people"):
            updates["people"] = source_probed.get("people")
        elif probed.get("people") and not row.get("people"):
            updates["people"] = probed.get("people")
        if sidecar_meta.get("keywords"):
            existing_keywords = manifest.export_asset_metadata_payload(db_path, row["id"])["normalized_metadata"].get("keywords") or []
            merged = sorted({str(item) for item in existing_keywords + sidecar_meta.get("keywords", []) if item})
            if merged:
                updates["keywords"] = merged
        elif source_probed.get("keywords"):
            existing_keywords = manifest.export_asset_metadata_payload(db_path, row["id"])["normalized_metadata"].get("keywords") or []
            merged = sorted({str(item) for item in existing_keywords + source_probed.get("keywords", []) if item})
            if merged:
                updates["keywords"] = merged
        elif probed.get("keywords"):
            existing_keywords = manifest.export_asset_metadata_payload(db_path, row["id"])["normalized_metadata"].get("keywords") or []
            merged = sorted({str(item) for item in existing_keywords + probed.get("keywords", []) if item})
            if merged:
                updates["keywords"] = merged

        if updates:
            manifest.apply_metadata_updates(
                db_path,
                row["id"],
                source_name="metadata_repair",
                confidence=_metadata_confidence("metadata_repair", "captured_at"),
                **updates,
            )
            repaired += 1

        final_location = _location_payload(updates) or _location_payload(source_probed) or _location_payload(image_meta) or _location_payload(probed) or _location_payload(row)
        if final_location:
            assets_with_location_after_primary.add(row["id"])
            paired_locations.setdefault(_asset_pair_key(row.get("orig_filename") or row["id"]), final_location)
        final_text = _text_payload(updates) or _text_payload(sidecar_meta) or _text_payload(source_probed) or _text_payload(probed)
        if final_text:
            paired_text.setdefault(_asset_pair_key(row.get("orig_filename") or row["id"]), final_text)
        final_device = _device_payload(sidecar_meta) or _device_payload(source_probed) or _device_payload(probed) or _device_payload(image_meta)
        if final_device:
            paired_device.setdefault(_asset_pair_key(row.get("orig_filename") or row["id"]), final_device)

        for field_name, source_field in (
            ("camera_make", "Make"),
            ("camera_model", "Model"),
            ("software", "Software"),
            ("pixel_width", "PixelWidth"),
            ("pixel_height", "PixelHeight"),
            ("format", "Format"),
        ):
            if image_meta.get(field_name) is not None:
                manifest.set_metadata_field(
                    db_path,
                    row["id"],
                    field_name=field_name,
                    value=image_meta[field_name],
                    source_name="image_exif",
                    source_field=source_field,
                    is_canonical=True,
                    confidence=_metadata_confidence("image_exif", field_name),
                )
        for field_name, source_field in (
            ("camera_make", "Make"),
            ("camera_model", "Model"),
            ("software", "Software"),
            ("lens_model", "LensModel"),
            ("pixel_width", "PixelWidth"),
            ("pixel_height", "PixelHeight"),
            ("duration_seconds", "Duration"),
            ("frame_rate", "VideoFrameRate"),
            ("bitrate", "AvgBitrate"),
            ("audio_channels", "AudioChannels"),
            ("video_codec", "CompressorName"),
            ("audio_codec", "HandlerDescription"),
            ("rotation_degrees", "Rotation"),
            ("content_identifier", "ContentIdentifier"),
            ("container_format", "FileType"),
            ("encoder", "Encoder"),
            ("mime_type", "MIMEType"),
        ):
            value = source_probed.get(field_name)
            if value is None:
                value = probed.get(field_name)
            if value is not None:
                manifest.set_metadata_field(
                    db_path,
                    row["id"],
                    field_name=field_name,
                    value=value,
                    source_name="embedded_metadata",
                    source_field=source_field,
                    is_canonical=True,
                    confidence=_metadata_confidence("embedded_metadata", field_name),
                )
        for field_name in ("camera_make", "camera_model"):
            if sidecar_meta.get(field_name):
                manifest.set_metadata_field(
                    db_path,
                    row["id"],
                    field_name=field_name,
                    value=sidecar_meta[field_name],
                    source_name="sidecar_metadata",
                    source_field=field_name,
                    is_canonical=False,
                    confidence=_metadata_confidence("sidecar_metadata", field_name),
                )

        if source_path and _is_probable_screenshot(row, source_path, image_meta):
            manifest.set_metadata_field(
                db_path,
                row["id"],
                field_name="is_screenshot",
                value=True,
                source_name="screenshot_detector",
                source_field="png_iphone_heuristic",
                is_canonical=True,
                confidence=_metadata_confidence("screenshot_detector", "is_screenshot"),
            )
            screenshot_count += 1
        blurry, blur_score = _is_probable_blurry(source_path, image_meta) if source_path else (False, 0.0)
        if blurry:
            manifest.set_metadata_field(
                db_path,
                row["id"],
                field_name="is_blurry",
                value=True,
                source_name="quality_detector",
                source_field="edge_variance",
                is_canonical=True,
                confidence=0.74,
            )
            manifest.set_metadata_field(
                db_path,
                row["id"],
                field_name="blur_score",
                value=round(blur_score, 2),
                source_name="quality_detector",
                source_field="edge_variance",
                is_canonical=True,
                confidence=0.74,
            )

    paired_repairs = 0
    for normalized, row in path_map.items():
        if row["id"] in assets_with_location_after_primary:
            continue
        if row.get("gps_lat") is not None and row.get("gps_lon") is not None:
            continue
        pair_location = paired_locations.get(_asset_pair_key(row.get("orig_filename") or row["id"]))
        if not pair_location:
            continue
        manifest.apply_metadata_updates(
            db_path,
            row["id"],
            source_name="metadata_repair",
            confidence=_metadata_confidence("paired_metadata", "location", paired=True),
            **pair_location,
        )
        paired_repairs += 1

    paired_text_repairs = 0
    for normalized, row in path_map.items():
        pair_text = paired_text.get(_asset_pair_key(row.get("orig_filename") or row["id"]))
        if not pair_text:
            continue
        updates: dict[str, Any] = {}
        if pair_text.get("title") and not row.get("title"):
            updates["title"] = pair_text["title"]
        if pair_text.get("description") and not row.get("description"):
            updates["description"] = pair_text["description"]
        if pair_text.get("keywords"):
            current_keywords = manifest.export_asset_metadata_payload(db_path, row["id"])["normalized_metadata"].get("keywords") or []
            merged = sorted({str(item) for item in current_keywords + pair_text["keywords"] if item})
            if merged and merged != current_keywords:
                updates["keywords"] = merged
        if updates:
            manifest.apply_metadata_updates(
                db_path,
                row["id"],
                source_name="paired_metadata",
                confidence=_metadata_confidence("paired_metadata", "title", paired=True),
                **updates,
            )
            paired_text_repairs += 1

    paired_device_repairs = 0
    for normalized, row in path_map.items():
        pair_device = paired_device.get(_asset_pair_key(row.get("orig_filename") or row["id"]))
        if not pair_device:
            continue
        existing = manifest.export_asset_metadata_payload(db_path, row["id"])["normalized_metadata"]
        changed = False
        for field_name, value in pair_device.items():
            if existing.get(field_name) in (None, "", [], {}):
                manifest.set_metadata_field(
                    db_path,
                    row["id"],
                    field_name=field_name,
                    value=value,
                    source_name="paired_metadata",
                    source_field=field_name,
                    is_canonical=True,
                    confidence=_metadata_confidence("paired_metadata", field_name, paired=True),
                )
                changed = True
        if changed:
            paired_device_repairs += 1

    return {
        "examined": examined,
        "repaired": repaired + paired_repairs + paired_text_repairs + paired_device_repairs,
        "screenshots": screenshot_count,
        "paired_location_repairs": paired_repairs,
        "paired_text_repairs": paired_text_repairs,
        "paired_device_repairs": paired_device_repairs,
    }
