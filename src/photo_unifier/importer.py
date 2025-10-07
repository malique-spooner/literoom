"""Importer utilities for scanning Apple/Google Takeout inputs.

Supports two safe workflows:

* :func:`peek_scan` — **no extraction**; quickly counts images/sidecars and
  estimates uncompressed size requirements by peeking inside ZIPs and folders.
* :func:`unzip_all` — extracts selected ZIPs into a chosen workspace so they
  can be scanned with :func:`scan`.
* :func:`scan` — walks directories to list images and sidecars.

All functions are pure and side-effect free **except** :func:`unzip_all`, which
creates an extraction workspace (ideally on a large external drive).

Typical usage:
    >>> report = peek_scan(["F:/Apple", "F:/Google"])  # no disk writes
    >>> roots, info = unzip_all(["F:/Apple"], workspace="F:/_unifier_tmp")
    >>> summary = scan(roots)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import tempfile
import zipfile

# Known image extensions to ingest.
IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif",
    ".tif", ".tiff", ".webp", ".bmp", ".gif",
    ".dng", ".arw", ".cr2", ".rw2",
}

# Apple Photos CSV filenames sometimes present in exports.
APPLE_CSV_NAMES = {"Photos.csv", "Photos Metadata.csv", "photos.csv", "metadata.csv"}


def is_image(path: Path) -> bool:
    """Return True if the path looks like an image file.

    Args:
        path: File path to inspect.

    Returns:
        True if the suffix matches a known image extension; False otherwise.
    """
    return path.suffix.lower() in IMG_EXTS


def _iter_all_zips_under(root: Path) -> List[Path]:
    """Recursively find all ``.zip`` files under a directory.

    Args:
        root: Directory to search.

    Returns:
        A list of ZIP file paths discovered under ``root``.
    """
    return [p for p in root.rglob("*.zip") if p.is_file()]


def _is_image_name(name: str) -> bool:
    """Check if an archive member name looks like an image.

    Args:
        name: Archive member name (e.g., ``"Photos/IMG_0001.HEIC"``).

    Returns:
        True if the suffix is a known image extension; False otherwise.
    """
    return Path(name).suffix.lower() in IMG_EXTS


def _is_sidecar_name(name: str) -> bool:
    """Check if an archive member name looks like a JSON sidecar.

    Args:
        name: Archive member name.

    Returns:
        True if ``.json``; False otherwise.
    """
    return Path(name).suffix.lower() == ".json"


def peek_scan(inputs: Iterable[str]) -> Dict[str, int | float | List[str]]:
    """Count what **would** be processed without extracting anything.

    For directories:
      * Count loose images/sidecars/Apple CSV files.
      * Open any nested ZIPs and count entries (no extraction).

    For ZIP inputs:
      * Open the archive and count entries (no extraction).

    Also accumulates **uncompressed byte sizes** to estimate workspace needs.

    Args:
        inputs: Iterable of filesystem paths (ZIPs or directories).

    Returns:
        A dictionary of counts, size sums (bytes), and small previews:
            {
              "total_zip_archives": int,
              "images_loose": int,
              "sidecars_loose": int,
              "apple_csv_loose": int,
              "images_in_zips": int,
              "sidecars_in_zips": int,
              "bytes_images_loose": int,
              "bytes_sidecars_loose": int,
              "bytes_csv_loose": int,
              "bytes_images_in_zips": int,
              "bytes_sidecars_in_zips": int,
              "total_uncompressed_bytes": int,
              "preview_images_loose": List[str],
              "preview_sidecars_loose": List[str],
              "preview_zip_entries": List[str],
            }
    """
    total_zips = 0
    images_loose = 0
    sidecars_loose = 0
    apple_csv_loose = 0
    images_in_zips = 0
    sidecars_in_zips = 0

    # Size accumulators (bytes)
    bytes_images_loose = 0
    bytes_sidecars_loose = 0
    bytes_csv_loose = 0
    bytes_images_in_zips = 0
    bytes_sidecars_in_zips = 0

    preview_images: List[str] = []
    preview_sidecars: List[str] = []
    preview_zip_entries: List[str] = []

    def _add_preview(bucket: List[str], value: str, maxn: int = 5) -> None:
        if len(bucket) < maxn:
            bucket.append(value)

    for raw in inputs:
        p = Path(raw)
        if p.is_file() and p.suffix.lower() == ".zip":
            total_zips += 1
            with zipfile.ZipFile(p, "r") as zf:
                for info in zf.infolist():
                    name = info.filename
                    if name.endswith("/"):
                        continue
                    if _is_image_name(name):
                        images_in_zips += 1
                        bytes_images_in_zips += int(info.file_size)
                        _add_preview(preview_zip_entries, f"{p.name}:{name}")
                    elif _is_sidecar_name(name):
                        sidecars_in_zips += 1
                        bytes_sidecars_in_zips += int(info.file_size)
                    elif Path(name).name in APPLE_CSV_NAMES:
                        bytes_csv_loose += int(info.file_size)
        elif p.is_dir():
            # Loose files
            for path in p.rglob("*"):
                if not path.is_file():
                    continue
                if is_image(path):
                    images_loose += 1
                    try:
                        bytes_images_loose += path.stat().st_size
                    except Exception:
                        pass
                    _add_preview(preview_images, str(path))
                elif path.name in APPLE_CSV_NAMES:
                    apple_csv_loose += 1
                    try:
                        bytes_csv_loose += path.stat().st_size
                    except Exception:
                        pass
                elif path.suffix.lower() == ".json":
                    sidecars_loose += 1
                    try:
                        bytes_sidecars_loose += path.stat().st_size
                    except Exception:
                        pass
                    _add_preview(preview_sidecars, str(path))
            # Peek inside nested zips
            for z in _iter_all_zips_under(p):
                total_zips += 1
                with zipfile.ZipFile(z, "r") as zf:
                    for info in zf.infolist():
                        name = info.filename
                        if name.endswith("/"):
                            continue
                        if _is_image_name(name):
                            images_in_zips += 1
                            bytes_images_in_zips += int(info.file_size)
                            _add_preview(preview_zip_entries, f"{z.name}:{name}")
                        elif _is_sidecar_name(name):
                            sidecars_in_zips += 1
                            bytes_sidecars_in_zips += int(info.file_size)
                        elif Path(name).name in APPLE_CSV_NAMES:
                            bytes_csv_loose += int(info.file_size)

    total_uncompressed_bytes = (
        bytes_images_loose
        + bytes_sidecars_loose
        + bytes_csv_loose
        + bytes_images_in_zips
        + bytes_sidecars_in_zips
    )

    return {
        "total_zip_archives": total_zips,
        "images_loose": images_loose,
        "sidecars_loose": sidecars_loose,
        "apple_csv_loose": apple_csv_loose,
        "images_in_zips": images_in_zips,
        "sidecars_in_zips": sidecars_in_zips,
        "bytes_images_loose": bytes_images_loose,
        "bytes_sidecars_loose": bytes_sidecars_loose,
        "bytes_csv_loose": bytes_csv_loose,
        "bytes_images_in_zips": bytes_images_in_zips,
        "bytes_sidecars_in_zips": bytes_sidecars_in_zips,
        "total_uncompressed_bytes": total_uncompressed_bytes,
        "preview_images_loose": preview_images,
        "preview_sidecars_loose": preview_sidecars,
        "preview_zip_entries": preview_zip_entries,
    }


def unzip_all(
    inputs: Iterable[str],
    workspace: str | None = None,
    zip_filter: str | None = None,
    limit_zips: int | None = None,
) -> Tuple[List[Path], Dict[str, object]]:
    """Extract selected ZIPs into a workspace and return roots to scan.

    This function is the **only** place that writes to disk. Prefer
    :func:`peek_scan` for preflight checks.

    Args:
        inputs: Filesystem paths (ZIPs or directories). Directories are scanned
            for nested ZIPs in addition to being included as scan roots.
        workspace: Extraction root directory. If ``None``, a temp directory on
            the system drive is created (not recommended for large sets—pass a
            path on your external drive).
        zip_filter: If set, only extract ZIPs whose **filename** contains this
            case-insensitive substring (applied to both top-level and nested ZIPs).
        limit_zips: If set, extract only the **first N** matching ZIPs (for
            testing). Directory roots are still returned for scanning.

    Returns:
        A tuple ``(roots, info)`` where:

        * ``roots``: List of directories to scan (original directories + extracted ZIP folders).
        * ``info``: Dict with extraction metadata:
            {
              "tmp_root": str,
              "workspace_explicit": bool,
              "extracted_dirs": List[str],
              "n_input_zips": int,    # number of top-level ZIPs extracted
              "n_nested_zips": int,   # number of nested ZIPs extracted
              "zip_filter": str,
              "limit_zips": int,
              "extracted_count": int,
            }

    Raises:
        FileNotFoundError: If any provided path does not exist.
        zipfile.BadZipFile: If a ZIP archive is malformed.
    """
    if workspace:
        tmp_root = Path(workspace).resolve()
        tmp_root.mkdir(parents=True, exist_ok=True)
        workspace_explicit = True
    else:
        tmp_root = Path(tempfile.mkdtemp(prefix="unifier_"))
        workspace_explicit = False

    # Gather roots and ZIPs to consider.
    roots: List[Path] = []
    top_level_zips: List[Path] = []
    nested_zips: List[Path] = []

    for raw in inputs:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(str(p))
        if p.is_file() and p.suffix.lower() == ".zip":
            top_level_zips.append(p)
        elif p.is_dir():
            roots.append(p)  # include original directory
            nested_zips.extend(_iter_all_zips_under(p))

    # Apply optional filename filter and limiting.
    all_candidates = top_level_zips + nested_zips
    if zip_filter:
        q = zip_filter.lower()
        all_candidates = [z for z in all_candidates if q in z.name.lower()]
    if limit_zips is not None:
        all_candidates = all_candidates[: int(limit_zips)]

    # Partition for reporting.
    tl_set = set(top_level_zips)
    selected_top = [z for z in all_candidates if z in tl_set]
    selected_nested = [z for z in all_candidates if z not in tl_set]

    # Extract selected ZIPs.
    extracted_dirs: List[Path] = []
    for z in all_candidates:
        out_dir = tmp_root / z.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(out_dir)
        extracted_dirs.append(out_dir)
        roots.append(out_dir)

    info = {
        "tmp_root": str(tmp_root),
        "workspace_explicit": workspace_explicit,
        "extracted_dirs": [str(d) for d in extracted_dirs],
        "n_input_zips": len(selected_top),
        "n_nested_zips": len(selected_nested),
        "zip_filter": zip_filter or "",
        "limit_zips": limit_zips or 0,
        "extracted_count": len(extracted_dirs),
    }
    return roots, info


def find_sidecar_for(img: Path) -> Optional[Path]:
    """Find a likely JSON sidecar for a given image on disk.

    Google Photos takeouts commonly use:
      * ``<file>.ext.json`` or
      * ``<file>.json``

    Args:
        img: The image file path.

    Returns:
        Path to a JSON sidecar if found; otherwise ``None``.
    """
    c1 = img.with_suffix(img.suffix + ".json")  # e.g., IMG.jpg.json
    if c1.exists():
        return c1
    c2 = Path(str(img) + ".json")  # e.g., IMG.jpg -> IMG.jpg.json
    if c2.exists():
        return c2
    return None


def _preview(paths: List[Path], n: int = 5) -> List[str]:
    """Return a string preview of the first ``n`` paths.

    Args:
        paths: Paths to preview.
        n: Max number of entries.

    Returns:
        List of up to ``n`` stringified paths.
    """
    return [str(p) for p in paths[:n]]


def scan(roots: Iterable[Path]) -> Dict[str, object]:
    """Scan roots for images, JSON sidecars, and Apple CSV files.

    Args:
        roots: Directories to recursively scan.

    Returns:
        Summary dictionary with counts and short previews:
            {
              "total_files": int,
              "images_count": int,
              "sidecars_count": int,
              "apple_csv_count": int,
              "preview_images": List[str],
              "preview_sidecars": List[str],
              "preview_apple_csv": List[str],
            }
    """
    images: List[Path] = []
    sidecars: List[Path] = []
    apple_csvs: List[Path] = []
    total_files = 0

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            total_files += 1
            if is_image(path):
                images.append(path)
                sc = find_sidecar_for(path)
                if sc:
                    sidecars.append(sc)
            else:
                if path.name in APPLE_CSV_NAMES:
                    apple_csvs.append(path)

    sidecars = sorted(set(sidecars))
    apple_csvs = sorted(set(apple_csvs))

    return {
        "total_files": total_files,
        "images_count": len(images),
        "sidecars_count": len(sidecars),
        "apple_csv_count": len(apple_csvs),
        "preview_images": _preview(images),
        "preview_sidecars": _preview(sidecars),
        "preview_apple_csv": _preview(apple_csvs),
    }
