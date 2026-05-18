from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import subprocess
import shutil
import os
import re
import json

from .location import parse_location_candidate

# Common video extensions for conditional tagging
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mts", ".m2ts", ".3gp", ".mkv"}

def _find_exiftool(exiftool_path: Optional[Path]) -> Optional[str]:
    if exiftool_path:
        p = str(exiftool_path)
        return p if os.path.exists(p) else None
    env = os.getenv("EXIFTOOL_PATH")
    if env and os.path.exists(env):
        return env
    found = shutil.which("exiftool")
    return found

# -----------------------------
# Write metadata with exiftool
# -----------------------------
def write_fields(
    file_path: Path,
    fields: Dict[str, Any],
    exiftool_path: Optional[Path] = None,
    backup_ext: str = ".bak",  # unused, we overwrite original
) -> Tuple[bool, str]:
    """
    Write common metadata into images/videos using exiftool.
    We assume fields['dt_original'] is *local Europe/London* (wall-clock).
    For images: EXIF + XMP (dates & GPS & text)
    For videos: QuickTime media dates + XMP (dates & GPS & text) + LocationISO6709
    """
    et = _find_exiftool(exiftool_path)
    if not et:
        raise RuntimeError("exiftool not found. Set EXIFTOOL_PATH or install it.")

    ext = file_path.suffix.lower()
    is_video = ext in VIDEO_EXTS

    args: List[str] = [et, "-overwrite_original", "-api", "largefilesupport=1", "-m", "-P", "-n"]

    # Title / Description
    title = fields.get("title")
    desc = fields.get("description")
    if title:
        args += [f"-XMP-dc:Title={title}"]
        if is_video:
            args += [f"-Keys:Title={title}"]
    if desc:
        args += [f"-XMP-dc:Description={desc}"]
        if is_video:
            args += [f"-Keys:Description={desc}"]

    # Date/Time (assumed local wall-clock — write as-is)
    dt = fields.get("dt_original")
    if dt:
        dt_norm = dt.replace("T", " ").split(".")[0]
        if is_video:
            args += [
                f"-QuickTime:CreateDate={dt_norm}",
                f"-QuickTime:ModifyDate={dt_norm}",
                f"-MediaCreateDate={dt_norm}",
                f"-TrackCreateDate={dt_norm}",
                f"-TrackModifyDate={dt_norm}",
                f"-XMP:CreateDate={dt_norm}",
            ]
        else:
            args += [
                f"-EXIF:DateTimeOriginal={dt_norm}",
                f"-EXIF:CreateDate={dt_norm}",
                f"-XMP:CreateDate={dt_norm}",
            ]

    # GPS (write both EXIF + XMP for stills; XMP + QT ISO6709 for videos)
    lat = fields.get("gps_lat")
    lon = fields.get("gps_lon")
    alt = fields.get("gps_alt")
    if isinstance(lat,(int,float)) and isinstance(lon,(int,float)):
        if is_video:
            # XMP
            args += [f"-XMP:GPSLatitude={lat}", f"-XMP:GPSLongitude={lon}"]
            if isinstance(alt,(int,float)): args += [f"-XMP:GPSAltitude={alt}"]
            # QuickTime ISO6709 (+lat -lon / altitude optional)
            # Format like +51.5074-0.1278/  or +51.5074-0.1278+45.00/
            iso6709 = f"{'+' if lat>=0 else ''}{lat}{'+' if lon>=0 else ''}{lon}"
            if isinstance(alt,(int,float)):
                iso6709 = f"{iso6709}{'+' if alt>=0 else ''}{alt:.2f}"
            iso6709 += "/"
            args += [f"-QuickTime:LocationISO6709={iso6709}"]
        else:
            # EXIF + XMP
            args += [f"-EXIF:GPSLatitude={lat}", f"-EXIF:GPSLongitude={lon}"]
            if isinstance(alt,(int,float)): args += [f"-EXIF:GPSAltitude={alt}"]
            args += [f"-XMP:GPSLatitude={lat}", f"-XMP:GPSLongitude={lon}"]
            if isinstance(alt,(int,float)): args += [f"-XMP:GPSAltitude={alt}"]

    # Keywords / People -> XMP Subject
    tags: List[str] = []
    if fields.get("keywords"): tags.extend(fields["keywords"])
    if fields.get("people"):   tags.extend(fields["people"])
    for t in sorted({str(x) for x in tags if x}):
        args += [f"-XMP-dc:Subject+={t}"]

    args += ["--", str(file_path)]

    proc = subprocess.run(args, capture_output=True, text=True)
    ok = (proc.returncode == 0)

    ver_proc = subprocess.run([et, "-ver"], capture_output=True, text=True)
    ver = (ver_proc.stdout or "").strip() or "unknown"

    return ok, ver

# -----------------------------
# Read “Date Taken” (batch) - optional fallback
# -----------------------------
def _normalize_exif_dt(s: str) -> str:
    s = s.strip()
    m = re.match(r"(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:{m.group(6)}"
    s = s.replace(" ", "T", 1)
    return s.split(".", 1)[0]

def _norm_abs(p: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def _parse_gps_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if not value:
        return None
    text = str(value).strip()
    try:
        return float(text)
    except Exception:
        pass
    match = re.match(
        r'^\s*(-?\d+(?:\.\d+)?)\s+deg\s+(\d+(?:\.\d+)?)\'\s+(\d+(?:\.\d+)?)"\s+([NSEW])\s*$',
        text,
    )
    if not match:
        return None
    deg = float(match.group(1))
    minutes = float(match.group(2))
    seconds = float(match.group(3))
    ref = match.group(4).upper()
    sign = -1.0 if ref in {"S", "W"} else 1.0
    return sign * (abs(deg) + minutes / 60.0 + seconds / 3600.0)


def _first_present(obj: Dict[str, Any], names: List[str]) -> Any:
    for name in names:
        value = obj.get(name)
        if value not in (None, ""):
            return value
    return None


def _record_location_from_candidate(record: Dict[str, Any], value: Any, *, source: str) -> bool:
    parsed = parse_location_candidate(value)
    if not parsed:
        return False
    record["gps_lat"] = parsed["lat"]
    record["gps_lon"] = parsed["lon"]
    if "alt" in parsed:
        record["gps_alt"] = parsed["alt"]
    record["gps_source"] = source
    return True

def read_best_datetime_batch(paths: List[Path], exiftool_path: Optional[Path] = None) -> Dict[str, str]:
    if not paths:
        return {}
    et = _find_exiftool(exiftool_path)
    if not et:
        return {}
    def chunks(items: List[Path], size: int = 256):
        for idx in range(0, len(items), size):
            yield items[idx:idx + size]
    tags = [
        "-json",
        "-n",
        "-DateTimeOriginal",
        "-CreateDate",
        "-MediaCreateDate",
        "-QuickTime:CreateDate",
        "-TrackCreateDate",
        "-FileModifyDate",
    ]
    out: Dict[str, str] = {}
    for batch in chunks(paths):
        args = [et, *tags, *[str(p) for p in batch]]
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0 or not proc.stdout.strip():
            continue
        try:
            arr = json.loads(proc.stdout)
        except Exception:
            continue
        order = ["DateTimeOriginal", "CreateDate", "MediaCreateDate", "QuickTime:CreateDate", "TrackCreateDate", "FileModifyDate"]
        for obj in arr:
            sf = obj.get("SourceFile")
            if not sf:
                continue
            key = _norm_abs(sf)
            for k in order:
                v = obj.get(k)
                if v:
                    try:
                        out[key] = _normalize_exif_dt(str(v))
                    except Exception:
                        pass
                    break
    return out


def read_core_metadata_batch(
    paths: List[Path],
    exiftool_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    if not paths:
        return {}
    et = _find_exiftool(exiftool_path)
    if not et:
        return {}
    def chunks(items: List[Path], size: int = 128):
        for idx in range(0, len(items), size):
            yield items[idx:idx + size]
    tags = [
        "-json",
        "-n",
        "-DateTimeOriginal",
        "-CreateDate",
        "-MediaCreateDate",
        "-QuickTime:CreateDate",
        "-TrackCreateDate",
        "-OffsetTimeOriginal",
        "-OffsetTime",
        "-TimeZone",
        "-GPSLatitude",
        "-GPSLongitude",
        "-GPSAltitude",
        "-GPSPosition",
        "-GPSCoordinates",
        "-GPSLatitudeRef",
        "-GPSLongitudeRef",
        "-GPSAltitudeRef",
        "-QuickTime:LocationISO6709",
        "-Make",
        "-Model",
        "-Software",
        "-LensModel",
        "-ImageWidth",
        "-ImageHeight",
        "-ExifImageWidth",
        "-ExifImageHeight",
        "-Duration",
        "-VideoFrameRate",
        "-CompressorName",
        "-HandlerDescription",
        "-AvgBitrate",
        "-AudioChannels",
        "-Rotation",
        "-ContentIdentifier",
        "-MajorBrand",
        "-Encoder",
        "-FileType",
        "-MIMEType",
        "-XMP-dc:Title",
        "-XMP-dc:Description",
        "-XMP-dc:Subject",
    ]
    out: Dict[str, Dict[str, Any]] = {}
    dt_order = [
        "DateTimeOriginal",
        "CreateDate",
        "MediaCreateDate",
        "QuickTime:CreateDate",
        "TrackCreateDate",
    ]
    for batch in chunks(paths):
        proc = subprocess.run(
            [et, *tags, *[str(p) for p in batch]],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            continue
        try:
            arr = json.loads(proc.stdout)
        except Exception:
            continue

        for obj in arr:
            sf = obj.get("SourceFile")
            if not sf:
                continue
            key = _norm_abs(sf)
            record: Dict[str, Any] = {}
            for tag in dt_order:
                value = obj.get(tag)
                if value:
                    try:
                        record["dt_original"] = _normalize_exif_dt(str(value))
                        record["dt_source"] = tag
                    except Exception:
                        pass
                    break
            tz_offset = _first_present(obj, ["OffsetTimeOriginal", "OffsetTime", "TimeZone"])
            if tz_offset:
                record["timezone_offset"] = str(tz_offset)
            lat = _parse_gps_number(obj.get("GPSLatitude"))
            lon = _parse_gps_number(obj.get("GPSLongitude"))
            alt = _parse_gps_number(obj.get("GPSAltitude"))
            if lat is not None and lon is not None:
                record["gps_lat"] = lat
                record["gps_lon"] = lon
                if alt is not None:
                    record["gps_alt"] = alt
                record["gps_source"] = "GPS"
            else:
                gps_position = obj.get("GPSPosition") or obj.get("GPSCoordinates")
                if gps_position:
                    _record_location_from_candidate(record, gps_position, source="GPSPosition")
                if "gps_lat" not in record and "QuickTime:LocationISO6709" in obj:
                    _record_location_from_candidate(record, obj.get("QuickTime:LocationISO6709"), source="QuickTime:LocationISO6709")
                if "gps_lat" not in record:
                    qt_location = obj.get("LocationISO6709") or obj.get("QuickTime:LocationISO6709")
                    if qt_location:
                        _record_location_from_candidate(record, qt_location, source="LocationISO6709")
            camera_make = _first_present(obj, ["Make"])
            if camera_make:
                record["camera_make"] = str(camera_make)
            camera_model = _first_present(obj, ["Model"])
            if camera_model:
                record["camera_model"] = str(camera_model)
            software = _first_present(obj, ["Software"])
            if software:
                record["software"] = str(software)
            lens_model = _first_present(obj, ["LensModel"])
            if lens_model:
                record["lens_model"] = str(lens_model)
            pixel_width = _first_present(obj, ["ImageWidth", "ExifImageWidth"])
            if isinstance(pixel_width, (int, float)):
                record["pixel_width"] = int(pixel_width)
            pixel_height = _first_present(obj, ["ImageHeight", "ExifImageHeight"])
            if isinstance(pixel_height, (int, float)):
                record["pixel_height"] = int(pixel_height)
            duration = _first_present(obj, ["Duration"])
            if isinstance(duration, (int, float)):
                record["duration_seconds"] = float(duration)
            frame_rate = _first_present(obj, ["VideoFrameRate"])
            if isinstance(frame_rate, (int, float)):
                record["frame_rate"] = float(frame_rate)
            avg_bitrate = _first_present(obj, ["AvgBitrate"])
            if isinstance(avg_bitrate, (int, float)):
                record["bitrate"] = int(avg_bitrate)
            video_codec = _first_present(obj, ["CompressorName"])
            if video_codec:
                record["video_codec"] = str(video_codec)
            audio_codec = _first_present(obj, ["HandlerDescription"])
            if audio_codec:
                record["audio_codec"] = str(audio_codec)
            audio_channels = _first_present(obj, ["AudioChannels"])
            if isinstance(audio_channels, (int, float)):
                record["audio_channels"] = int(audio_channels)
            rotation = _first_present(obj, ["Rotation"])
            if isinstance(rotation, (int, float)):
                record["rotation_degrees"] = int(rotation)
            content_identifier = _first_present(obj, ["ContentIdentifier"])
            if content_identifier:
                record["content_identifier"] = str(content_identifier)
            container = _first_present(obj, ["FileType", "MajorBrand"])
            if container:
                record["container_format"] = str(container)
            encoder = _first_present(obj, ["Encoder"])
            if encoder:
                record["encoder"] = str(encoder)
            mime_type = _first_present(obj, ["MIMEType"])
            if mime_type:
                record["mime_type"] = str(mime_type)
            title = obj.get("Title")
            if title:
                record["title"] = title
            desc = obj.get("Description")
            if desc:
                record["description"] = desc
            subject = obj.get("Subject")
            if isinstance(subject, list):
                record["keywords"] = [str(item) for item in subject if item]
            elif subject:
                record["keywords"] = [str(subject)]
            out[key] = record
    return out
