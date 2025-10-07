r"""
Dry-run AND apply-mode embedder for portable metadata (photos + videos).

Dry-run (no writes):
  python -m photo_unifier.embed --manifest F:\_unifier_state\manifest.csv --limit 300
  python -m photo_unifier.embed --manifest F:\_unifier_state\manifest.csv --out F:\_unifier_state\embed_plan.csv

Apply (copies to clean library and writes .xmp sidecars; originals untouched):
  python -m photo_unifier.embed --manifest F:\_unifier_state\manifest.csv ^
    --dest F:\_library --apply --sidecar-only --limit 100

Notes:
- Date resolution priority:
    1) manifest taken_time  (from indexer)
    2) EXIF (images) / MediaInfo (videos)
    3) FILENAME FALLBACK (parse common date patterns from filename)
    4) container/file timestamp (zip member or filesystem mtime)
- Everything writes to XMP sidecars next to copied media (portable/safe).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import shutil
import zipfile
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# ---------- image EXIF ----------
from PIL import Image, ImageFile  # pillow
Image.MAX_IMAGE_PIXELS = None          # allow very large images
ImageFile.LOAD_TRUNCATED_IMAGES = True # tolerate slightly corrupt files
try:
    import pillow_heif  # optional: enables HEIC/HEIF EXIF
    pillow_heif.register_heif_opener()
except Exception:
    pass

# ---------- video metadata (MediaInfo) ----------
try:
    from pymediainfo import MediaInfo  # optional: video creation time
except Exception:
    MediaInfo = None  # graceful fallback if not installed

SAFE_INFILE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mts", ".m2ts", ".3gp", ".3g2", ".mkv", ".wmv"}

_ZIP_PREFIX = "zip://"
_FILE_PREFIX = "file://"
_DATE_RE = re.compile(r"(\d{4})[-:](\d{2})[-:](\d{2})")


# ---------- model ----------

@dataclass
class EmbedPlanRow:
    vpath: str
    filename: str
    ext: str
    provider: str
    match_type: str
    taken_time: str
    gps_lat: str
    gps_lon: str
    gps_alt: str
    description: str
    album: str
    people: str
    will_write_xmp_sidecar: bool
    will_write_infile_safe: bool
    is_video: bool


def _plan_headers() -> List[str]:
    return [f.name for f in fields(EmbedPlanRow)]


# ---------- manifest streaming ----------

def _iter_manifest(manifest_csv: str) -> Iterator[Dict[str, str]]:
    with open(manifest_csv, "r", encoding="utf-8", newline="") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            if isinstance(row, dict):
                yield row


def _row_to_plan(row: Dict[str, str]) -> EmbedPlanRow:
    fn = (row.get("filename") or "").strip()
    ext = Path(fn).suffix.lower()
    is_video = ext in VIDEO_EXTS

    def g(key: str) -> str:
        val = row.get(key)
        return val if isinstance(val, str) else ("" if val is None else str(val))

    return EmbedPlanRow(
        vpath=g("vpath"),
        filename=fn,
        ext=ext,
        provider=(row.get("provider") or "unknown"),
        match_type=(row.get("match_type") or ""),
        taken_time=g("taken_time"),
        gps_lat=g("gps_lat"),
        gps_lon=g("gps_lon"),
        gps_alt=g("gps_alt"),
        description=g("description"),
        album=g("album"),
        people=g("people"),
        will_write_xmp_sidecar=True,                  # portable for photos + videos
        will_write_infile_safe=(ext in SAFE_INFILE_EXTS),
        is_video=is_video,
    )


def iter_plan(manifest_csv: str, limit: Optional[int] = None) -> Iterator[EmbedPlanRow]:
    count = 0
    for row in _iter_manifest(manifest_csv):
        yield _row_to_plan(row)
        count += 1
        if limit is not None and limit > 0 and count >= limit:
            break


def save_plan_csv(plans: Iterable[EmbedPlanRow], out_csv: str) -> None:
    headers = _plan_headers()
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        for p in plans:
            w.writerow(asdict(p))


def _print_sample(plans: Iterable[EmbedPlanRow], sample: int = 10) -> None:
    shown = 0
    for p in plans:
        print(
            f"- {p.filename} | ext={p.ext} | sidecar={p.will_write_xmp_sidecar} "
            f"| infile_safe={p.will_write_infile_safe} | video={p.is_video} "
            f"| time={p.taken_time or '-'} | gps=({p.gps_lat or '-'}, {p.gps_lon or '-'}) "
            f"| people={p.people or '-'}"
        )
        shown += 1
        if shown >= sample:
            break


# ---------- XMP helpers ----------

_XMP_ESC = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;"}
def _xml_escape(s: str) -> str:
    return "".join(_XMP_ESC.get(ch, ch) for ch in s)


def _guess_xmp_datetime(raw: str) -> str:
    """Return ISO 8601 (UTC 'Z' when possible) for common raw formats, else raw."""
    if not raw:
        return ""
    raw = raw.strip()
    # Google timestamps (seconds or ms)
    if raw.isdigit() and len(raw) in (10, 13):
        try:
            sec = int(raw[:10])
            return dt.datetime.utcfromtimestamp(sec).replace(microsecond=0).isoformat() + "Z"
        except Exception:
            pass
    # Apple/EXIF patterns (note one variant has no space after comma)
    for pat in (
        "%A %B %d,%Y %I:%M %p %Z",
        "%A %B %d, %Y %I:%M %p %Z",
        "%Y-%m-%d %H:%M:%S",
        "%Y:%m:%d %H:%M:%S",
    ):
        try:
            t = dt.datetime.strptime(raw, pat)
            return t.replace(microsecond=0).isoformat() + "Z"
        except Exception:
            continue
    return raw  # fallback (leave as is)


def _xmp_packet(plan: EmbedPlanRow, effective_iso: str = "") -> str:
    """Minimal, broadly compatible XMP packet."""
    desc = _xml_escape(plan.description) if plan.description else ""
    album = _xml_escape(plan.album) if plan.album else ""
    xmp_date = effective_iso or (_guess_xmp_datetime(plan.taken_time) or "")
    people = [n.strip() for n in (plan.people or "").split(",") if n.strip()]

    people_rdf = ""
    if people:
        items = "\n".join(f'            <rdf:li>{_xml_escape(n)}</rdf:li>' for n in people)
        people_rdf = f"""
        <photoshop:PersonInImage>
          <rdf:Bag>
{items}
          </rdf:Bag>
        </photoshop:PersonInImage>"""

    album_rdf = ""
    if album:
        album_escaped = _xml_escape(f"Albums|{album}")
        album_rdf = f"""
        <lr:hierarchicalSubject>
          <rdf:Bag>
            <rdf:li>{album_escaped}</rdf:li>
          </rdf:Bag>
        </lr:hierarchicalSubject>"""

    gps_rdf = ""
    if plan.gps_lat and plan.gps_lon:
        gps_rdf = f"""
        <exif:GPSLatitude>{_xml_escape(plan.gps_lat)}</exif:GPSLatitude>
        <exif:GPSLongitude>{_xml_escape(plan.gps_lon)}</exif:GPSLongitude>"""
        if plan.gps_alt:
            gps_rdf += f"""
        <exif:GPSAltitude>{_xml_escape(plan.gps_alt)}</exif:GPSAltitude>"""

    return f"""<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
          xmlns:xmp="http://ns.adobe.com/xap/1.0/"
          xmlns:dc="http://purl.org/dc/elements/1.1/"
          xmlns:exif="http://ns.adobe.com/exif/1.0/"
          xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
          xmlns:lr="http://ns.adobe.com/lightroom/1.0/">
  <rdf:Description>
    <dc:description>
      <rdf:Alt>
        <rdf:li xml:lang="x-default">{desc}</rdf:li>
      </rdf:Alt>
    </dc:description>{album_rdf}{people_rdf}{gps_rdf}
    <xmp:CreateDate>{_xml_escape(xmp_date)}</xmp:CreateDate>
    <xmp:ModifyDate>{_xml_escape(xmp_date)}</xmp:ModifyDate>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


# ---------- vpath & file I/O ----------

def _parse_vpath(vpath: str) -> Optional[Tuple[str, str]]:
    if vpath.startswith(_ZIP_PREFIX):
        try:
            rest = vpath[len(_ZIP_PREFIX):]
            zip_path, inner = rest.split("!", 1)
            return zip_path, inner
        except Exception:
            return None
    if vpath.startswith(_FILE_PREFIX):
        return "", vpath[len(_FILE_PREFIX):]
    return None


def _copy_from_vpath_to(vpath: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(dest_path.suffix + ".part")
    parsed = _parse_vpath(vpath)
    if not parsed:
        raise ValueError(f"Unsupported vpath: {vpath}")
    zip_path, inner = parsed
    if zip_path:
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(inner, "r") as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
    else:
        shutil.copyfile(inner, tmp)
    os.replace(tmp, dest_path)


def _longpath(p: Path) -> str:
    s = str(p)
    if os.name == "nt":
        # use \\?\ long-path prefix (handles deep folders and odd edge-cases)
        if not s.startswith("\\\\?\\"):
            try:
                return "\\\\?\\" + str(p.resolve())
            except Exception:
                return "\\\\?\\" + s
    return s

def _write_sidecar_xmp(dest_asset: Path, xmp_text: str) -> Path:
    """
    Robust writer for Windows:
      - ensures parent exists
      - uses \\?\ long path
      - if Windows complains about the name, falls back to a short, hashed sidecar filename
    """
    sidecar = dest_asset.with_suffix(dest_asset.suffix + ".xmp")
    sidecar.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(_longpath(sidecar), "w", encoding="utf-8", newline="") as fh:
            fh.write(xmp_text)
        return sidecar
    except OSError:
        # Fallback: hashed name that is guaranteed to be simple and short
        import hashlib
        h = hashlib.sha1(str(dest_asset).encode("utf-8", "ignore")).hexdigest()[:12]
        fallback = sidecar.parent / f"{dest_asset.stem}_sidecar_{h}.xmp"
        with open(_longpath(fallback), "w", encoding="utf-8", newline="") as fh:
            fh.write(xmp_text)
        return fallback


# ---------- date helpers (EXIF, MediaInfo, filename, container/file fallback) ----------

def _exif_taken_datetime_from_file(path: Path) -> str | None:
    """ISO8601 from EXIF DateTimeOriginal/Digitized/DateTime for images."""
    try:
        with Image.open(path) as im:
            exif = getattr(im, "getexif", lambda: {})() or {}
            if not exif:
                return None
            for tag in (36867, 36868, 306):  # DateTimeOriginal, Digitized, DateTime
                v = exif.get(tag)
                if not v:
                    continue
                if isinstance(v, bytes):
                    v = v.decode(errors="ignore")
                v = str(v).strip()
                for pat in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
                    try:
                        t = dt.datetime.strptime(v, pat)
                        return t.replace(microsecond=0).isoformat()
                    except Exception:
                        continue
    except Exception:
        return None
    return None


def _video_creation_datetime_from_file(path: Path) -> str | None:
    """
    ISO8601 using MediaInfo (QuickTime/MP4 'Encoded date'/'Tagged date' etc.).
    Works for .mp4/.mov/.m4v/... if MediaInfo is installed.
    """
    if MediaInfo is None:
        return None
    try:
        mi = MediaInfo.parse(str(path))
        for track in mi.to_data().get("tracks", []):
            if track.get("track_type") != "General":
                continue
            for key in ("encoded_date", "tagged_date", "file_last_modification_date", "mastered_date"):
                s = (track.get(key) or "").strip()
                if not s:
                    continue
                if s.upper().startswith("UTC "):
                    s = s[4:]
                for pat in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
                    try:
                        t = dt.datetime.strptime(s, pat)
                        return t.replace(microsecond=0).isoformat() + "Z"
                    except Exception:
                        pass
                if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?$", s):
                    return s if s.endswith("Z") else (s + "Z")
    except Exception:
        pass
    return None


def _date_from_filename(name: str) -> str | None:
    """
    Try to derive an ISO date from common filename patterns.
    Returns ISO8601 with 'Z' when parsed, else None.

    Examples:
      20211024_153000.jpg
      IMG_20201231_235959.MOV
      2021-10-24 15.30.00.png
      WhatsApp Image 2021-10-24 at 15.30.00.jpeg
    """
    if not name:
        return None
    s = Path(name).stem

    # YYYYMMDD_HHMMSS or YYYYMMDD-HHMMSS or YYYYMMDDHHMMSS
    m = re.search(r'(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})[_\-]?(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})', s)
    if m:
        try:
            t = dt.datetime(int(m['y']), int(m['m']), int(m['d']), int(m['H']), int(m['M']), int(m['S']))
            return t.replace(microsecond=0).isoformat() + "Z"
        except Exception:
            pass

    # YYYY-MM-DD HH.MM.SS or YYYY-MM-DD HH:MM:SS (covers WhatsApp "at" format too)
    m = re.search(r'(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})[ _]?(?:at[ _])?(?P<H>\d{2})[\.:](?P<M>\d{2})[\.:](?P<S>\d{2})', s, re.I)
    if m:
        try:
            t = dt.datetime(int(m['y']), int(m['m']), int(m['d']), int(m['H']), int(m['M']), int(m['S']))
            return t.replace(microsecond=0).isoformat() + "Z"
        except Exception:
            pass

    # IMG_YYYYMMDD_HHMMSS / VID_YYYYMMDD_HHMMSS
    m = re.search(r'(?:IMG|VID)[_\-]?(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})[_\-]?(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})', s, re.I)
    if m:
        try:
            t = dt.datetime(int(m['y']), int(m['m']), int(m['d']), int(m['H']), int(m['M']), int(m['S']))
            return t.replace(microsecond=0).isoformat() + "Z"
        except Exception:
            pass

    # Bare YYYY-MM-DD (no time → noon)
    m = re.search(r'(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})', s)
    if m:
        try:
            t = dt.datetime(int(m['y']), int(m['m']), int(m['d']), 12, 0, 0)
            return t.replace(microsecond=0).isoformat() + "Z"
        except Exception:
            pass

    return None


def _fallback_datetime_from_vpath(vpath: str) -> str | None:
    """ISO8601 derived from ZIP member timestamp or filesystem mtime."""
    parsed = _parse_vpath(vpath)
    if not parsed:
        return None
    zip_path, inner = parsed
    try:
        if zip_path:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zi = zf.getinfo(inner)
                y, mo, d, h, mi, s = zi.date_time
                dt_local = dt.datetime(y, mo, d, h, mi, s)
                return dt_local.replace(microsecond=0).isoformat()
        else:
            ts = os.path.getmtime(inner)
            return dt.datetime.fromtimestamp(ts).replace(microsecond=0).isoformat()
    except Exception:
        return None


def _resolve_effective_datetime_iso(plan: EmbedPlanRow, src_local_path: Optional[Path]) -> str:
    """
    Best ISO datetime:
      manifest taken_time → EXIF (images) → MediaInfo (videos) → filename fallback → ZIP/file mtime → "".
    """
    # 1) manifest
    if plan.taken_time:
        iso = _guess_xmp_datetime(plan.taken_time)
        if iso:
            return iso

    # 2) local metadata (if we have a staged copy)
    if src_local_path and src_local_path.exists():
        ext = plan.ext.lower()
        if ext in {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".webp", ".heic", ".heif"}:
            exif_iso = _exif_taken_datetime_from_file(src_local_path)
            if exif_iso:
                return exif_iso
        if ext in VIDEO_EXTS:
            vid_iso = _video_creation_datetime_from_file(src_local_path)
            if vid_iso:
                return vid_iso

    # 3) filename-encoded date (last resort before timestamps)
    fname_iso = _date_from_filename(plan.filename)
    if fname_iso:
        return fname_iso

    # 4) container/file timestamp
    vpath_iso = _fallback_datetime_from_vpath(plan.vpath)
    if vpath_iso:
        return _guess_xmp_datetime(vpath_iso) or vpath_iso

    return ""


# ---------- destination layout ----------

def _safe_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    return cleaned[:240]


def _dest_for(plan: EmbedPlanRow, dest_root: Path, effective_iso: str) -> Path:
    """
    Layout: dest_root/YYYY/YYYY-MM/YYYY-MM-DD/<stamp>_<originalname>.<ext>
    """
    m = _DATE_RE.search(effective_iso)
    if m:
        y, mo, d = m.groups()
        y_str, ym, ymd = y, f"{y}-{mo}", f"{y}-{mo}-{d}"
        stamp = effective_iso.replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
    else:
        y_str = ym = ymd = "Unknown"
        stamp = "Unknown"

    folder = dest_root / y_str / ym / ymd
    base = Path(plan.filename).name or "asset"
    stem = f"{stamp}_{base}" if stamp != "Unknown" else base
    return folder / _safe_filename(stem)


# ---------- APPLY MODE (copy + sidecars) ----------

def apply_sidecars(manifest_csv: str, dest_root: str, limit: Optional[int]) -> Tuple[int, int, int]:
    """
    Copy assets into dest_root and write .xmp sidecars (no in-file edits).
    Returns (processed, copied, sidecars).
    """
    processed = copied = sidecars = 0
    dest_root_p = Path(dest_root)
    staging = dest_root_p / "_staging"
    staging.mkdir(parents=True, exist_ok=True)

    for plan in iter_plan(manifest_csv, limit=limit):
        processed += 1

        # 1) Copy to staging (we'll move to final path after date resolution)
        stage_name = _safe_filename(plan.filename or ("asset" + plan.ext))
        stage_path = staging / stage_name
        if not stage_path.exists():
            _copy_from_vpath_to(plan.vpath, stage_path)
            copied += 1

        # 2) Resolve effective ISO once; use for folder + XMP
        effective_iso = _resolve_effective_datetime_iso(plan, stage_path)

        # 3) Decide final destination
        final_dest = _dest_for(plan, dest_root_p, effective_iso)
        final_dest.parent.mkdir(parents=True, exist_ok=True)

        # 4) Move from staging to final (idempotent)
        if not final_dest.exists():
            os.replace(stage_path, final_dest)
        else:
            try:
                stage_path.unlink(missing_ok=True)
            except Exception:
                pass

        # 5) Write/refresh sidecar next to final file
        xmp = _xmp_packet(plan, effective_iso=effective_iso)
        _write_sidecar_xmp(final_dest, xmp)
        sidecars += 1

    # Cleanup empty staging dir (best-effort)
    try:
        next(staging.iterdir())
    except StopIteration:
        try:
            staging.rmdir()
        except Exception:
            pass
    except Exception:
        pass

    return processed, copied, sidecars


# ---------- CLI ----------

def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="DRY-RUN plan and APPLY sidecars for manifest assets.")
    ap.add_argument("--manifest", required=True, help="Path to manifest.csv from the index step.")
    ap.add_argument("--out", default=None, help="Optional path to write a dry-run plan CSV.")
    ap.add_argument("--limit", type=int, default=300, help="Limit number of rows (0 = all).")
    ap.add_argument("--print-sample", type=int, default=10, help="Sample rows to print (dry-run).")
    ap.add_argument("--dest", default=None, help="Destination clean library root (e.g., F:\\_library).")
    ap.add_argument("--apply", action="store_true", help="Enable apply mode: copy + .xmp sidecars into --dest.")
    ap.add_argument("--sidecar-only", action="store_true", help="(default) Only sidecars; no in-file edits.")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)
    limit = None if args.limit == 0 else args.limit

    # Small capability banner
    print(f"HEIC support: {'ON' if 'pillow_heif' in globals() else 'OFF'} | "
          f"Video dates (MediaInfo): {'ON' if MediaInfo else 'OFF'}")

    if args.apply:
        if not args.dest:
            print("ERROR: --dest is required in --apply mode (e.g., --dest F:\\_library).")
            return 2
        dest_root = Path(args.dest)
        dest_root.mkdir(parents=True, exist_ok=True)
        print(f"APPLY: copying to {dest_root} with sidecars (limit={limit or 'ALL'})")
        processed, copied, sidecars = apply_sidecars(args.manifest, args.dest, limit)
        print(f"APPLY DONE: processed={processed} copied={copied} sidecars={sidecars}")
        return 0

    # DRY-RUN path
    print(f"Building DRY-RUN plan from: {args.manifest}")
    print(f"Limit: {limit or 'ALL'} rows")
    plan_iter_for_print = iter_plan(args.manifest, limit=limit)
    plan_iter_for_save = iter_plan(args.manifest, limit=limit)
    _print_sample(plan_iter_for_print, sample=max(0, args.print_sample))
    if args.out:
        save_plan_csv(plan_iter_for_save, args.out)
        print(f"Wrote plan CSV: {args.out}")
    else:
        print("No --out provided; plan not written to disk (dry-run preview only).")
    print("Done (no files were modified).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
