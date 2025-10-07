"""Index assets (photos + videos) to their metadata **without extraction**.

Builds a manifest mapping each asset to Google JSON or Apple CSV metadata
*inside the same container* (ZIP or directory). Matching does not rely on
filenames being unique across different inputs; we use a stable virtual path.

Virtual paths:
    * ZIP member:  zip://<zip_path>!<inner/path>
    * File on disk: file://<absolute/path>
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

# ---------- asset types (photos + videos) ----------

IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif",
    ".tif", ".tiff", ".webp", ".bmp", ".gif",
    ".dng", ".arw", ".cr2", ".rw2",
}
VIDEO_EXTS = {
    ".mp4", ".mov", ".m4v", ".avi", ".mts", ".m2ts",
    ".3gp", ".3g2", ".mkv", ".wmv",
}
ASSET_EXTS = IMG_EXTS | VIDEO_EXTS


# ---------- data model ----------

@dataclass
class ManifestRow:
    """A single mapping of an asset to its sidecar/CSV metadata.

    Attributes:
        provider: "google"|"apple"|"unknown" (best guess from matched data).
        source_type: "zip"|"dir".
        source_root: Top-level container path (zip file or directory root).
        vpath: Stable virtual path ("zip://...!inner" or "file://...").
        container_chain: Friendly label (zip filename).
        inner_path: Path inside the container (or absolute path in dir mode).
        filename: Asset base filename.
        size_bytes: Member/file size when known.
        matched: Whether *any* metadata matched this asset.
        match_type: "json"|"csv"|"".
        taken_time, gps_*: Time/GPS (stringy; we don’t normalize here).
        description, album, people: Optional descriptive fields.
    """

    provider: str
    source_type: str
    source_root: str
    vpath: str
    container_chain: str
    inner_path: str
    filename: str
    size_bytes: int | None
    matched: bool
    match_type: str
    taken_time: str | None
    gps_lat: str | None
    gps_lon: str | None
    gps_alt: str | None
    description: str | None
    album: str | None
    people: str | None


# ---------- small helpers ----------

def _is_asset_name(name: str) -> bool:
    """True if the basename looks like a supported photo/video."""
    return Path(name).suffix.lower() in ASSET_EXTS


def _normalize_key(name: str) -> str:
    """Lowercased base filename for same-container matching."""
    return Path(name).name.lower()


def _build_vpath_zip(source_root: str, inner: str) -> str:
    """Virtual path for ZIP member."""
    return f"zip://{source_root}!{inner}"


def _build_vpath_file(path: str) -> str:
    """Virtual path for normal file."""
    return f"file://{path}"


def _read_text_from_zip(zf: zipfile.ZipFile, member: zipfile.ZipInfo) -> str:
    """UTF-8 text from a ZIP member (replace decode errors)."""
    with zf.open(member, "r") as fh:
        return io.TextIOWrapper(fh, encoding="utf-8", errors="replace").read()


# ---------- Google JSON parsing (defensive) ----------

def _parse_google_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        return {}


def _extract_google_fields(js: dict) -> Tuple[
    str | None, str | None, str | None, str | None, str | None, str | None
]:
    """Return (taken_time, lat, lon, alt, description, people_csv)."""
    # time
    taken = None
    pt = js.get("photoTakenTime") or js.get("creationTime")
    if isinstance(pt, dict) and "timestamp" in pt:
        taken = str(pt["timestamp"])
    elif isinstance(pt, (str, int, float)):
        taken = str(pt)

    # description/title
    desc = js.get("description") or js.get("title")

    # geo (several layouts exist)
    lat = lon = alt = None
    geo = js.get("geoDataExif") or js.get("geoData") or {}
    if isinstance(geo, dict):
        if geo.get("latitude") is not None:
            lat = str(geo.get("latitude"))
        if geo.get("longitude") is not None:
            lon = str(geo.get("longitude"))
        if geo.get("altitude") is not None:
            alt = str(geo.get("altitude"))

    # people
    people_csv = None
    if isinstance(js.get("people"), list):
        names = []
        for p in js["people"]:
            if isinstance(p, dict):
                n = p.get("name")
                if isinstance(n, str) and n.strip():
                    names.append(n.strip())
        if names:
            people_csv = ", ".join(names)

    return taken, lat, lon, alt, desc, people_csv


# ---------- Apple CSV detection (content-driven, no hardcoded names) ----------

# Regexes to heuristically pick metadata columns regardless of exact header names.
_RE_TIME = re.compile(r"(creation|taken|capture|date|time)", re.I)
_RE_DESC = re.compile(r"(description|caption|notes?|title)", re.I)
_RE_ALBUM = re.compile(r"(album|collection|folder|library)", re.I)
_RE_LAT = re.compile(r"lat(itude)?", re.I)
_RE_LON = re.compile(r"lon(gitude)?", re.I)
_RE_ALT = re.compile(r"alt(itude)?", re.I)

def _looks_like_filename(value: str) -> bool:
    v = value.strip().lower()
    return Path(v).suffix in ASSET_EXTS if v else False


def _pick_filename_column(headers: List[str], rows: List[dict]) -> Optional[str]:
    """Choose the header that behaves like a filename column."""
    if not headers:
        return None
    # Score headers by name and by values looking like filenames.
    scores: Dict[str, int] = {h: 0 for h in headers}
    for h in headers:
        hl = h.lower().replace("_", " ").replace("-", " ")
        if "file" in hl and "name" in hl:
            scores[h] += 2
        elif "name" in hl:
            scores[h] += 1
    for row in rows[: min(200, len(rows))]:
        for h in headers:
            if _looks_like_filename(str(row.get(h, ""))):
                scores[h] += 3
    # Best header with any non-empty values.
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    has_any = any(str(r.get(best) or "").strip() for r in rows)
    return best if has_any else None


def _apple_fields_from_row(row: dict) -> Tuple[
    str | None, str | None, str | None, str | None, str | None, str | None
]:
    """Return (taken_time, lat, lon, alt, description, album) via header regex."""
    def pick(rx: re.Pattern[str]) -> str | None:
        for k, v in row.items():
            if rx.search(str(k)) and str(v).strip():
                return str(v).strip()
        return None

    taken = pick(_RE_TIME)
    desc = pick(_RE_DESC)
    album = pick(_RE_ALBUM)
    lat = pick(_RE_LAT)
    lon = pick(_RE_LON)
    alt = pick(_RE_ALT)
    return taken, lat, lon, alt, desc, album


def _image_basenames_in_zip(zf: zipfile.ZipFile) -> set[str]:
    out: set[str] = set()
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        if _is_asset_name(name):
            out.add(_normalize_key(Path(name).name))
    return out


def _apple_map_from_csv_text(text: str, image_keys: set[str]) -> Dict[str, dict]:
    """Map filename->row using the most filename-like column; prefer overlap with assets."""
    rdr = csv.DictReader(io.StringIO(text))
    headers = rdr.fieldnames or []
    rows = list(rdr)
    if not headers or not rows:
        return {}
    fname_col = _pick_filename_column(headers, rows)
    if not fname_col:
        return {}
    mapping: Dict[str, dict] = {}
    for row in rows:
        raw = str(row.get(fname_col) or "").strip()
        if not raw:
            continue
        key = _normalize_key(raw)
        mapping[key] = row
        # also allow bare stem form (helps when JSON omits extension)
        mapping[_normalize_key(Path(raw).stem)] = row
    # Prefer rows that overlap assets in the same container (if any).
    if image_keys:
        stem_set = {Path(x).stem for x in image_keys}
        filtered = {
            k: v for k, v in mapping.items()
            if (k in image_keys) or (Path(k).stem in stem_set)
        }
        return filtered or mapping
    return mapping


def _load_best_apple_csv_map_from_zip(zf: zipfile.ZipFile) -> Dict[str, dict]:
    """Scan *all* CSVs; return the mapping that overlaps most with assets."""
    image_keys = _image_basenames_in_zip(zf)
    best: Dict[str, dict] = {}
    best_score = -1
    for info in zf.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".csv"):
            continue
        try:
            text = _read_text_from_zip(zf, info)
        except Exception:
            continue
        m = _apple_map_from_csv_text(text, image_keys)
        score = len(m)
        if score > best_score:
            best, best_score = m, score
    return best


def _collect_json_sidecars_in_zip(zf: zipfile.ZipFile) -> Dict[str, dict]:
    """Return keys -> JSON mapping; keys include with and without extension."""
    out: Dict[str, dict] = {}
    for info in zf.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".json"):
            continue
        base = Path(info.filename).name  # e.g., IMG.jpg.json or IMG.json
        if not base.lower().endswith(".json"):
            continue
        stem = base[:-5]  # drop ".json"
        try:
            js = json.loads(_read_text_from_zip(zf, info))
        except Exception:
            js = {}
        # two keys: with ext (IMG.jpg), and bare (IMG)
        out[stem.lower()] = js
        out[Path(stem).stem.lower()] = js
    return out


# ---------- public indexers ----------

def index_zip(zip_path: Union[str, Path]) -> Iterator[ManifestRow]:
    """Yield a ManifestRow per asset inside a ZIP (no extraction)."""
    zroot = str(zip_path)
    if not Path(zroot).is_file():
        raise FileNotFoundError(zroot)
    with zipfile.ZipFile(zroot, "r") as zf:
        json_map = _collect_json_sidecars_in_zip(zf)
        apple_map = _load_best_apple_csv_map_from_zip(zf)

        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if not _is_asset_name(name):
                continue

            base = Path(name).name
            key_ext = base.lower()
            key_bare = Path(base).stem.lower()

            js = json_map.get(key_ext) or json_map.get(key_bare)

            matched = False
            match_type = ""
            taken = lat = lon = alt = desc = album = people = None
            provider = "unknown"

            if js:
                matched = True
                match_type = "json"
                provider = "google"
                taken, lat, lon, alt, desc, people = _extract_google_fields(js)

            if not matched and apple_map:
                row = apple_map.get(key_ext) or apple_map.get(key_bare)
                if row:
                    matched = True
                    match_type = "csv"
                    provider = "apple"
                    taken, lat, lon, alt, desc, album = _apple_fields_from_row(row)

            yield ManifestRow(
                provider=provider,
                source_type="zip",
                source_root=zroot,
                vpath=_build_vpath_zip(zroot, name),
                container_chain=Path(zroot).name,
                inner_path=name,
                filename=base,
                size_bytes=int(info.file_size),
                matched=matched,
                match_type=match_type,
                taken_time=taken,
                gps_lat=lat,
                gps_lon=lon,
                gps_alt=alt,
                description=desc,
                album=album,
                people=people,
            )


def index_dir(root: Union[str, Path]) -> Iterator[ManifestRow]:
    """Yield a ManifestRow per asset in a directory tree (no extraction)."""
    r = Path(root).resolve()
    if not r.is_dir():
        raise FileNotFoundError(str(r))

    # Google JSON sidecars next to assets; map by both keys.
    json_map: Dict[str, dict] = {}
    for p in r.rglob("*.json"):
        try:
            js = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            js = {}
        json_map[_normalize_key(p.stem)] = js         # "img.jpg"
        json_map[_normalize_key(Path(p.stem).stem)] = js  # "img"

    # Build the best Apple CSV mapping across all CSVs on disk.
    # We prefer the mapping that overlaps the most with discovered assets.
    # First collect asset keys to judge overlap.
    asset_keys: set[str] = set()
    for p in r.rglob("*"):
        if p.is_file() and _is_asset_name(p.name):
            asset_keys.add(_normalize_key(p.name))

    apple_map: Dict[str, dict] = {}
    best_score = -1
    for p in r.rglob("*.csv"):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        m = _apple_map_from_csv_text(text, asset_keys)
        if len(m) > best_score:
            apple_map, best_score = m, len(m)

    # Walk assets and match to JSON / Apple CSV.
    for p in r.rglob("*"):
        if not p.is_file() or not _is_asset_name(p.name):
            continue

        base = p.name
        key_ext = base.lower()
        key_bare = Path(base).stem.lower()

        js = json_map.get(key_ext) or json_map.get(key_bare)

        matched = False
        match_type = ""
        taken = lat = lon = alt = desc = album = people = None
        provider = "unknown"

        if js:
            matched = True
            match_type = "json"
            provider = "google"
            taken, lat, lon, alt, desc, people = _extract_google_fields(js)

        if not matched and apple_map:
            row = apple_map.get(key_ext) or apple_map.get(key_bare)
            if row:
                matched = True
                match_type = "csv"
                provider = "apple"
                taken, lat, lon, alt, desc, album = _apple_fields_from_row(row)

        try:
            size = p.stat().st_size
        except Exception:
            size = None

        yield ManifestRow(
            provider=provider,
            source_type="dir",
            source_root=str(r),
            vpath=_build_vpath_file(str(p)),
            container_chain="",
            inner_path=str(p),
            filename=base,
            size_bytes=size,
            matched=matched,
            match_type=match_type,
            taken_time=taken,
            gps_lat=lat,
            gps_lon=lon,
            gps_alt=alt,
            description=desc,
            album=album,
            people=people,
        )
