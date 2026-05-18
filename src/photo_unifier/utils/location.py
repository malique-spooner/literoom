from __future__ import annotations

import re
from typing import Any, Optional


_DMS_RE = re.compile(
    r"^\s*(?P<deg>-?\d+(?:\.\d+)?)\s+deg\s+"
    r"(?P<min>\d+(?:\.\d+)?)'\s+"
    r"(?P<sec>\d+(?:\.\d+)?)\"\s*"
    r"(?P<ref>[NSEW])?\s*$",
    re.IGNORECASE,
)
_DECIMAL_RE = re.compile(r"^\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<ref>[NSEW])?\s*$", re.IGNORECASE)
_ISO6709_RE = re.compile(
    r"^\s*(?P<lat>[+-]\d+(?:\.\d+)?)(?P<lon>[+-]\d+(?:\.\d+)?)(?P<alt>[+-]\d+(?:\.\d+)?)?/\s*$"
)

_SNAPCHAT_LOCATION_RE = re.compile(
    r"^\s*Latitude,\s*Longitude:\s*(?P<lat>-?\d+(?:\.\d+)?)\s*,\s*(?P<lon>-?\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, "", False):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _parse_coordinate_component(value: Any, *, expect_latitude: bool | None = None) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        num = float(value)
        if expect_latitude is True and not (-90.0 <= num <= 90.0):
            return None
        if expect_latitude is False and not (-180.0 <= num <= 180.0):
            return None
        return num

    if value in (None, "", False):
        return None
    text = str(value).strip()
    if not text:
        return None

    match = _DMS_RE.match(text)
    if match:
        deg = float(match.group("deg"))
        minutes = float(match.group("min"))
        seconds = float(match.group("sec"))
        ref = (match.group("ref") or "").upper()
        sign = -1.0 if ref in {"S", "W"} else 1.0
        num = sign * (abs(deg) + minutes / 60.0 + seconds / 3600.0)
        if expect_latitude is True and not (-90.0 <= num <= 90.0):
            return None
        if expect_latitude is False and not (-180.0 <= num <= 180.0):
            return None
        return num

    match = _DECIMAL_RE.match(text)
    if match:
        num = float(match.group("value"))
        ref = (match.group("ref") or "").upper()
        if ref in {"S", "W"}:
            num = -abs(num)
        elif ref in {"N", "E"}:
            num = abs(num)
        if expect_latitude is True and not (-90.0 <= num <= 90.0):
            return None
        if expect_latitude is False and not (-180.0 <= num <= 180.0):
            return None
        return num

    return None


def _parse_iso6709(text: str) -> Optional[dict[str, float]]:
    match = _SNAPCHAT_LOCATION_RE.match(text)
    if match:
        lat = _coerce_float(match.group("lat"))
        lon = _coerce_float(match.group("lon"))
        if lat is None or lon is None:
            return None
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None
        return {"lat": lat, "lon": lon}

    match = _ISO6709_RE.match(text)
    if not match:
        return None
    lat = _coerce_float(match.group("lat"))
    lon = _coerce_float(match.group("lon"))
    alt = _coerce_float(match.group("alt"))
    if lat is None or lon is None:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    payload = {"lat": lat, "lon": lon}
    if alt is not None:
        payload["alt"] = alt
    return payload


def _parse_pair_text(text: str) -> Optional[dict[str, float]]:
    text = text.strip()
    if not text:
        return None
    iso = _parse_iso6709(text)
    if iso:
        return iso

    parts = [part.strip() for part in re.split(r"\s*,\s*|\s*;\s*|\s+\|\s+", text) if part.strip()]
    if len(parts) >= 2:
        lat = _parse_coordinate_component(parts[0], expect_latitude=True)
        lon = _parse_coordinate_component(parts[1], expect_latitude=False)
        alt = _parse_coordinate_component(parts[2]) if len(parts) >= 3 else None
        if lat is not None and lon is not None:
            payload = {"lat": lat, "lon": lon}
            if alt is not None:
                payload["alt"] = alt
            return payload

    tokens = [token for token in re.split(r"\s+", text) if token]
    if len(tokens) >= 2:
        joined = " ".join(tokens)
        iso = _parse_iso6709(joined)
        if iso:
            return iso
        lat = _parse_coordinate_component(tokens[0], expect_latitude=True)
        lon = _parse_coordinate_component(tokens[1], expect_latitude=False)
        alt = _parse_coordinate_component(tokens[2]) if len(tokens) >= 3 else None
        if lat is not None and lon is not None:
            payload = {"lat": lat, "lon": lon}
            if alt is not None:
                payload["alt"] = alt
            return payload

    return None


def parse_location_candidate(value: Any) -> Optional[dict[str, float]]:
    if value in (None, "", False):
        return None
    if isinstance(value, dict):
        nested_keys = (
            "location",
            "geoData",
            "geoDataExif",
            "geo",
            "coordinates",
            "coordinate",
            "position",
            "mapPoint",
            "mapLocation",
            "latLng",
            "geoPoint",
            "coordinates",
        )
        for key in nested_keys:
            nested = value.get(key)
            if nested not in (None, "", False):
                parsed = parse_location_candidate(nested)
                if parsed:
                    return parsed

        lat = None
        lon = None
        alt = None

        lat_candidates = ("latitude", "lat", "gps_lat", "latitudeE7", "latE7", "latDegrees", "latitudeDegrees")
        lon_candidates = ("longitude", "lon", "lng", "gps_lon", "longitudeE7", "lonE7", "lonDegrees", "longitudeDegrees")
        alt_candidates = ("altitude", "alt", "gps_alt", "altitudeMeters", "elevation", "height")

        for key in lat_candidates:
            if key in value:
                candidate = value.get(key)
                if key.endswith("E7"):
                    num = _coerce_float(candidate)
                    if num is not None:
                        lat = num / 10_000_000.0
                else:
                    lat = _parse_coordinate_component(candidate, expect_latitude=True)
                if lat is not None:
                    break

        for key in lon_candidates:
            if key in value:
                candidate = value.get(key)
                if key.endswith("E7"):
                    num = _coerce_float(candidate)
                    if num is not None:
                        lon = num / 10_000_000.0
                else:
                    lon = _parse_coordinate_component(candidate, expect_latitude=False)
                if lon is not None:
                    break

        for key in alt_candidates:
            if key in value:
                alt = _coerce_float(value.get(key))
                if alt is not None:
                    break

        if lat is not None and lon is not None:
            payload = {"lat": lat, "lon": lon}
            if alt is not None:
                payload["alt"] = alt
            return payload

        for key in ("location", "geoData", "geoDataExif", "geo", "coordinates", "position"):
            nested = value.get(key)
            if nested not in (None, "", False):
                parsed = parse_location_candidate(nested)
                if parsed:
                    return parsed
        for nested in value.values():
            if nested is value:
                continue
            parsed = parse_location_candidate(nested)
            if parsed:
                return parsed
        return None

    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            lat = _parse_coordinate_component(value[0], expect_latitude=True)
            lon = _parse_coordinate_component(value[1], expect_latitude=False)
            alt = _parse_coordinate_component(value[2]) if len(value) >= 3 else None
            if lat is not None and lon is not None:
                payload = {"lat": lat, "lon": lon}
                if alt is not None:
                    payload["alt"] = alt
                return payload
        for item in value:
            parsed = parse_location_candidate(item)
            if parsed:
                return parsed
        return None

    if isinstance(value, str):
        return _parse_pair_text(value)

    return None
