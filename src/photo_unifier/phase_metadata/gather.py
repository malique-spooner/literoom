from __future__ import annotations

import csv
import io
import itertools
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

try:
    from timezonefinder import TimezoneFinder
except ImportError:  # pragma: no cover - optional dependency fallback
    TimezoneFinder = None
from zoneinfo import ZoneInfo

from . import manifest
from ..utils.location import parse_location_candidate


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".heic", ".png", ".gif", ".tif", ".tiff", ".bmp", ".webp",
    ".dng", ".raw", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2",
}
RAW_EXTS = {".dng", ".raw", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2"}
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mts", ".m2ts", ".3gp", ".mkv", ".wmv"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS


def _classify_media_type(ext: str) -> str:
    ext = ext.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in RAW_EXTS:
        return "raw"
    if ext in IMAGE_EXTS:
        return "image"
    return "other"


def detect_source(path: Path, inner_names_sample: Iterable[str] = ()) -> str:
    name = path.name.lower()
    if "icloud" in name or "apple" in name:
        return "apple"
    if "takeout" in name or "google" in name:
        return "google"
    inner = " ".join(n.lower() for n in itertools.islice(inner_names_sample, 20))
    if "takeout" in inner or "metadata.json" in inner:
        return "google"
    if "icloud" in inner or "apple" in inner:
        return "apple"
    return "local"


def _list_zip_members(zp: zipfile.ZipFile) -> List[zipfile.ZipInfo]:
    return [zi for zi in zp.infolist() if not zi.is_dir()]


def _read_text(zp: zipfile.ZipFile, member: zipfile.ZipInfo) -> Optional[str]:
    try:
        with zp.open(member, "r") as fh:
            return fh.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _read_json_from_zip(zp: zipfile.ZipFile, member: zipfile.ZipInfo) -> Optional[Dict[str, Any]]:
    s = _read_text(zp, member)
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _zip_mtime_to_iso(zi: zipfile.ZipInfo) -> str:
    y, m, d, hh, mm, ss = zi.date_time
    return f"{y:04d}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}:{ss:02d}"


_TZ_TOKEN = re.compile(r"\s+(UTC|GMT|BST|CEST|CET|EDT|EST|PDT|PST|IST)$", re.IGNORECASE)


def _clean_apple_local_string(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    s = re.sub(r",(\d{4})", r", \1", s)
    s = _TZ_TOKEN.sub("", s)
    formats = [
        "%A %B %d, %Y %I:%M %p",
        "%A %B %d, %Y %H:%M",
        "%B %d, %Y %I:%M %p",
        "%Y-%m-%d %H:%M:%S",
        "%Y:%m:%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


_FN_PATTERNS = [
    re.compile(r"(?P<y>20\d{2})(?P<m>\d{2})(?P<d>\d{2})[_-]?(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})"),
    re.compile(r"IMG[_-](?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})[_-](?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})"),
    re.compile(r"PXL[_-](?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})[_-](?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})"),
]


def _parse_from_filename(name: str) -> Optional[datetime]:
    base = Path(name).stem
    for pattern in _FN_PATTERNS:
        match = pattern.search(base)
        if not match:
            continue
        try:
            return datetime(
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


def _looks_like_apple_csv(path: str) -> bool:
    p = path.lower()
    return ("photo details" in p) and p.endswith(".csv")


def _parse_apple_csv(zp: zipfile.ZipFile, member: zipfile.ZipInfo) -> Dict[str, datetime]:
    txt = _read_text(zp, member)
    if not txt:
        return {}
    out: Dict[str, datetime] = {}
    reader = csv.DictReader(io.StringIO(txt))
    for row in reader:
        name = _normalize_text_value(
            _first_nested_value(
                row,
                ("imgName", "filename", "fileName", "originalFilename", "name", "photoName", "assetName"),
            )
        )
        dt = _normalize_text_value(
            _first_nested_value(
                row,
                ("originalCreationDate", "creationDate", "takenDate", "photoTakenTime", "dateCreated", "captureDate"),
            )
        )
        if not name or not dt:
            continue
        parsed = _clean_apple_local_string(dt)
        if parsed:
            out[name.lower()] = parsed
    return out


def _parse_apple_csv_file(csv_path: Path) -> Dict[str, datetime]:
    try:
        txt = csv_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    out: Dict[str, datetime] = {}
    reader = csv.DictReader(io.StringIO(txt))
    for row in reader:
        name = _normalize_text_value(
            _first_nested_value(
                row,
                ("imgName", "filename", "fileName", "originalFilename", "name", "photoName", "assetName"),
            )
        )
        dt = _normalize_text_value(
            _first_nested_value(
                row,
                ("originalCreationDate", "creationDate", "takenDate", "photoTakenTime", "dateCreated", "captureDate"),
            )
        )
        if not name or not dt:
            continue
        parsed = _clean_apple_local_string(dt)
        if parsed:
            out[name.lower()] = parsed
    return out


def _build_apple_csv_index(files: List[zipfile.ZipInfo], zp: zipfile.ZipFile) -> Dict[str, datetime]:
    idx: Dict[str, datetime] = {}
    for zi in files:
        if _looks_like_apple_csv(zi.filename):
            idx.update(_parse_apple_csv(zp, zi))
    return idx


def _build_apple_csv_index_from_root(root: Path) -> Dict[str, datetime]:
    idx: Dict[str, datetime] = {}
    if not root.exists():
        return idx
    for csv_path in root.rglob(f"{APPLE_CSV_PREFIX}*.csv"):
        idx.update(_parse_apple_csv_file(csv_path))
    return idx


def _build_sidecar_index(files: List[zipfile.ZipInfo]) -> Dict[str, zipfile.ZipInfo]:
    return {zi.filename: zi for zi in files if zi.filename.lower().endswith(".json")}


def _candidate_sidecars(media_path: str) -> List[str]:
    p = Path(media_path)
    sibling = (p.parent / (p.stem + ".json")).as_posix()
    exact = f"{media_path}.json"
    supp1 = sibling.replace(".json", ".supplemental-metadata.json")
    supp2 = exact.replace(".json", ".supplemental-metadata.json")
    return [sibling, exact, supp1, supp2]


def _find_sidecar(media_path: str, sidecars: Dict[str, zipfile.ZipInfo]) -> Optional[zipfile.ZipInfo]:
    for candidate in _candidate_sidecars(media_path):
        if candidate in sidecars:
            return sidecars[candidate]
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


def _parse_timestamp_payload(value: Any) -> tuple[Optional[int], Optional[str]]:
    seen: set[int] = set()

    def coerce(node: Any, depth: int = 6) -> tuple[Optional[int], Optional[str]]:
        if depth < 0 or node in (None, "", False):
            return None, None
        marker = id(node)
        if marker in seen:
            return None, None
        seen.add(marker)

        if isinstance(node, dict):
            for key in ("timestamp", "seconds", "secondsSinceEpoch", "unixTime", "time"):
                if key in node:
                    ts, fmt = coerce(node.get(key), depth - 1)
                    if ts is not None or fmt is not None:
                        return ts, fmt
            for key in ("formatted", "dateTime", "datetime"):
                if key in node:
                    formatted = _normalize_text_value(node.get(key))
                    if formatted:
                        return None, formatted
            if depth == 0:
                return None, None
            for nested in node.values():
                ts, fmt = coerce(nested, depth - 1)
                if ts is not None or fmt is not None:
                    return ts, fmt
            return None, None

        if isinstance(node, (list, tuple)):
            if depth == 0:
                return None, None
            for item in node:
                ts, fmt = coerce(item, depth - 1)
                if ts is not None or fmt is not None:
                    return ts, fmt
            return None, None

        if isinstance(node, str):
            text = node.strip()
            if not text:
                return None, None
            if text.endswith("Z"):
                try:
                    return None, datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%Y-%m-%dT%H:%M:%S")
                except Exception:
                    pass
            try:
                try:
                    return None, datetime.fromisoformat(text).strftime("%Y-%m-%dT%H:%M:%S")
                except Exception:
                    pass
                if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
                    node = text
                else:
                    return None, text
            except Exception:
                return None, None

        try:
            ts = int(str(node))
            if abs(ts) >= 10_000_000_000:
                ts = ts // 1000
            return ts, None
        except Exception:
            return None, None

    return coerce(value)


@dataclass
class GoogleMeta:
    ts_utc: Optional[int] = None
    fmt_utc: Optional[str] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    gps_alt: Optional[float] = None
    title: Optional[str] = None
    description: Optional[str] = None
    people: Optional[list[str]] = None


def _parse_google(obj: Dict[str, Any]) -> GoogleMeta:
    meta = GoogleMeta()
    ptt = _first_nested_value(obj, ("photoTakenTime", "creationTime", "mediaCreationTime", "captureTime", "timestamp"))
    ts_utc, fmt_utc = _parse_timestamp_payload(ptt)
    if ts_utc is not None:
        meta.ts_utc = ts_utc
    if fmt_utc:
        meta.fmt_utc = fmt_utc
    geo = _first_nested_value(obj, ("geoData", "geoDataExif", "location", "geo", "coordinates", "position"))
    parsed_geo = parse_location_candidate(geo if geo is not None else obj)
    if parsed_geo:
        meta.gps_lat = parsed_geo["lat"]
        meta.gps_lon = parsed_geo["lon"]
        meta.gps_alt = parsed_geo.get("alt")
    title = _normalize_text_value(_first_nested_value(obj, ("title", "filename", "name")))
    description = _normalize_text_value(_first_nested_value(obj, ("description", "caption", "Caption", "story")))
    if title:
        meta.title = title
    if description:
        meta.description = description
    people = _first_nested_value(obj, ("people", "faces", "faceNames", "recognizedFaces", "faceAnnotations"))
    normalized_people = _normalize_people_value(people) if people is not None else []
    meta.people = normalized_people or None
    return meta


_tf = TimezoneFinder() if TimezoneFinder is not None else None


def _tz_at(lat: float, lon: float) -> Optional[str]:
    if _tf is None:
        return None
    try:
        return _tf.timezone_at(lat=lat, lng=lon)
    except Exception:
        return None


def _fmt_naive_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _prefer_filename_if_whole_hour_gap(fn: Optional[datetime], other: Optional[datetime]) -> Optional[datetime]:
    if fn and other:
        delta = abs((fn - other).total_seconds())
        if delta > 0 and (abs(delta % 3600) <= 90 or abs(3600 - (delta % 3600)) <= 90):
            return fn
    return other or fn


def _choose_canonical_time(
    filename_dt: Optional[datetime],
    apple_local_dt: Optional[datetime],
    google_meta: Optional[GoogleMeta],
    gps_lat: Optional[float],
    gps_lon: Optional[float],
) -> Optional[str]:
    g_local_from_epoch: Optional[datetime] = None
    g_fmt_local: Optional[datetime] = None
    if google_meta:
        if google_meta.fmt_utc:
            try:
                s = google_meta.fmt_utc.strip().replace("UTC", "").strip()
                for fmt in ("%d %b %Y, %H:%M:%S", "%d %b %Y, %H:%M"):
                    try:
                        g_fmt_local = datetime.strptime(s, fmt)
                        break
                    except Exception:
                        continue
            except Exception:
                pass
        if google_meta.ts_utc is not None:
            try:
                g_local_from_epoch = datetime.fromtimestamp(
                    google_meta.ts_utc, tz=timezone.utc
                ).replace(tzinfo=timezone.utc)
            except Exception:
                g_local_from_epoch = None

    if gps_lat is not None and gps_lon is not None:
        tzid = _tz_at(gps_lat, gps_lon)
        if tzid:
            tz = ZoneInfo(tzid)
            if g_local_from_epoch is not None:
                loc = g_local_from_epoch.astimezone(tz).replace(tzinfo=None)
                return _fmt_naive_iso(_prefer_filename_if_whole_hour_gap(filename_dt, loc))
            if apple_local_dt:
                return _fmt_naive_iso(_prefer_filename_if_whole_hour_gap(filename_dt, apple_local_dt))
            if filename_dt:
                return _fmt_naive_iso(filename_dt)
            if google_meta and google_meta.fmt_utc:
                for fmt in ("%d %b %Y, %H:%M:%S UTC", "%d %b %Y, %H:%M UTC"):
                    try:
                        aware = datetime.strptime(google_meta.fmt_utc.strip(), fmt).replace(tzinfo=timezone.utc)
                        loc = aware.astimezone(tz).replace(tzinfo=None)
                        return _fmt_naive_iso(_prefer_filename_if_whole_hour_gap(filename_dt, loc))
                    except Exception:
                        continue

    if filename_dt:
        return _fmt_naive_iso(filename_dt)
    if apple_local_dt:
        return _fmt_naive_iso(apple_local_dt)
    if g_fmt_local:
        return _fmt_naive_iso(g_fmt_local)
    if g_local_from_epoch is not None:
        return _fmt_naive_iso(g_local_from_epoch.replace(tzinfo=None))
    return None


def scan_zip(zip_path: Path, source_hint: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    with zipfile.ZipFile(zip_path, "r") as zp:
        files = _list_zip_members(zp)
        src = (source_hint or detect_source(zip_path, (zi.filename for zi in files))).lower()
        side_idx = _build_sidecar_index(files)
        apple_csv_idx = _build_apple_csv_index(files, zp)

        for zi in files:
            inner = zi.filename
            ext = Path(inner).suffix.lower()
            media_type = _classify_media_type(ext)
            if media_type not in {"image", "video", "raw"}:
                continue

            base_name = Path(inner).name
            filename_dt = _parse_from_filename(base_name)
            gps_lat = gps_lon = gps_alt = None
            title = description = None
            people = None
            source_dt = None
            source_gps = None

            apple_local = apple_csv_idx.get(base_name.lower())
            if src == "apple" or apple_local is not None:
                sc = _find_sidecar(inner, side_idx)
                if sc:
                    obj = _read_json_from_zip(zp, sc) or {}
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
                        gps_lat = loc["lat"]
                        gps_lon = loc["lon"]
                        gps_alt = loc.get("alt")
                        source_gps = "apple_json"
                    title = _normalize_text_value(_first_nested_value(obj, ("title", "filename", "name")))
                    description = _normalize_text_value(_first_nested_value(obj, ("description", "caption", "Caption", "story")))
                    people = _normalize_people_value(_first_nested_value(obj, ("people", "faces", "faceNames", "recognizedFaces", "faceAnnotations"))) or None
                dt_final = _choose_canonical_time(filename_dt, apple_local, None, gps_lat, gps_lon)
                source_dt = "apple_csv" if apple_local else ("filename" if filename_dt else "zip_mtime")
            elif src == "google":
                sc = _find_sidecar(inner, side_idx)
                gmeta = None
                if sc:
                    obj = _read_json_from_zip(zp, sc) or {}
                    gmeta = _parse_google(obj)
                    if gmeta.gps_lat is not None and gmeta.gps_lon is not None:
                        gps_lat, gps_lon, gps_alt = gmeta.gps_lat, gmeta.gps_lon, gmeta.gps_alt
                        source_gps = "google_json"
                    title = gmeta.title
                    description = gmeta.description
                    people = gmeta.people
                dt_final = _choose_canonical_time(filename_dt, None, gmeta, gps_lat, gps_lon)
                if gmeta and gmeta.ts_utc is not None:
                    source_dt = "google_epoch"
                elif gmeta and gmeta.fmt_utc:
                    source_dt = "google_formatted"
                else:
                    source_dt = "filename" if filename_dt else "zip_mtime"
            else:
                dt_final = _fmt_naive_iso(filename_dt) if filename_dt else None
                source_dt = "filename" if filename_dt else "zip_mtime"

            yield {
                "source": src,
                "abs_zip": str(zip_path),
                "zip_path": inner,
                "source_kind": "zip",
                "source_root": str(zip_path.parent),
                "source_locator": str(zip_path),
                "source_path": inner,
                "media_type": media_type,
                "orig_filename": base_name,
                "orig_ext": ext,
                "orig_size": zi.file_size,
                "dt_original": dt_final,
                "gps_lat": gps_lat,
                "gps_lon": gps_lon,
                "gps_alt": gps_alt,
                "title": title,
                "description": description,
                "keywords": None,
                "people": people,
                "live_group_id": None,
                "burst_id": None,
                "src_mtime": _zip_mtime_to_iso(zi),
                "source_dt": source_dt,
                "source_gps": source_gps,
            }


def scan_directory(root: Path, source_hint: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    src = (source_hint or detect_source(root)).lower()
    apple_csv_idx = _build_apple_csv_index_from_root(root)
    for path in root.rglob("*"):
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        ext = path.suffix.lower()
        media_type = _classify_media_type(ext)
        if media_type not in {"image", "video", "raw"}:
            continue
        try:
            stat = path.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%dT%H:%M:%S")
            size = stat.st_size
        except OSError:
            mtime = None
            size = None
        filename_dt = _parse_from_filename(path.name)
        apple_local = apple_csv_idx.get(path.name.lower())
        yield {
            "source": src,
            "abs_zip": str(path),
            "zip_path": str(path.relative_to(root).as_posix()),
            "source_kind": "file",
            "source_root": str(root),
            "source_locator": str(path),
            "source_path": str(path.relative_to(root).as_posix()),
            "media_type": media_type,
            "orig_filename": path.name,
            "orig_ext": ext,
            "orig_size": size,
            "dt_original": _fmt_naive_iso(apple_local) if apple_local else (_fmt_naive_iso(filename_dt) if filename_dt else mtime),
            "gps_lat": None,
            "gps_lon": None,
            "gps_alt": None,
            "title": None,
            "description": None,
            "keywords": None,
            "people": None,
            "live_group_id": None,
            "burst_id": None,
            "src_mtime": mtime,
            "source_dt": "apple_csv" if apple_local else ("filename" if filename_dt else "file_mtime"),
            "source_gps": None,
        }


def scan_file(file_path: Path, source_hint: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    if file_path.suffix.lower() == ".zip":
        yield from scan_zip(file_path, source_hint=source_hint)
        return
    ext = file_path.suffix.lower()
    media_type = _classify_media_type(ext)
    if media_type not in {"image", "video", "raw"}:
        return
    try:
        stat = file_path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%dT%H:%M:%S")
        size = stat.st_size
    except OSError:
        mtime = None
        size = None
    filename_dt = _parse_from_filename(file_path.name)
    yield {
        "source": (source_hint or detect_source(file_path)).lower(),
        "abs_zip": str(file_path),
        "zip_path": file_path.name,
        "source_kind": "file",
        "source_root": str(file_path.parent),
        "source_locator": str(file_path),
        "source_path": file_path.name,
        "media_type": media_type,
        "orig_filename": file_path.name,
        "orig_ext": ext,
        "orig_size": size,
        "dt_original": _fmt_naive_iso(filename_dt) if filename_dt else mtime,
        "gps_lat": None,
        "gps_lon": None,
        "gps_alt": None,
        "title": None,
        "description": None,
        "keywords": None,
        "people": None,
        "live_group_id": None,
        "burst_id": None,
        "src_mtime": mtime,
        "source_dt": "filename" if filename_dt else "file_mtime",
        "source_gps": None,
    }


def scan_sources(paths: Iterable[Path], source_hint: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    for item in paths:
        p = Path(item)
        if not p.exists():
            continue
        if p.is_dir():
            yield from scan_directory(p, source_hint=source_hint)
        elif p.is_file():
            yield from scan_file(p, source_hint=source_hint)


def run(
    paths: List[Path],
    db_path: Path,
    source_hint: Optional[str] = None,
    batch_size: int = 500,
    job_id: Optional[str] = None,
) -> int:
    candidates = [Path(p) for p in paths if Path(p).exists()]
    manifest.init_db(Path(db_path))

    total = 0
    batch: List[Dict[str, Any]] = []
    for row in scan_sources(candidates, source_hint=source_hint):
        batch.append(row)
        if len(batch) >= batch_size:
            total += manifest.upsert_raw(batch, db_path=Path(db_path), job_id=job_id)
            batch.clear()
    if batch:
        total += manifest.upsert_raw(batch, db_path=Path(db_path), job_id=job_id)
        batch.clear()
    return total


__all__ = ["detect_source", "run", "scan_directory", "scan_file", "scan_sources", "scan_zip"]
