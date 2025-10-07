#!/usr/bin/env python3
r"""
photo_unifier.cli
-----------------
Step 1 CLI: Build and *append* to a single master manifest (deduped by `vpath`).

Usage examples:

  # Build temp manifest from 3 sources, then merge into a master (dedupe by vpath)
  python -m photo_unifier.cli index ^
    "F:\MSp\Camera\Apple takeout" ^
    "F:\MSp\Camera\Google takeout - maliquespooner" ^
    "F:\MSp\Camera\Google takeout - mtspalace" ^
    --manifest "F:\MSp\Camera\_pipeline_state\manifest_tmp.csv" ^
    --to-master "F:\MSp\Camera\_pipeline_state\manifest_master.csv"

  # Verify master integrity
  python -m photo_unifier.cli manifest-verify --master "F:\MSp\Camera\_pipeline_state\manifest_master.csv"

  # Snapshot of external drive + project code (non-verbose)
  python -m photo_unifier.cli snapshot --ext-root "F:\" --project-root "C:\Users\maliq\Desktop\Projects\photo-unifier"
"""

from __future__ import annotations

import csv
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import click

# -----------------------------
# Media type filters (extensions)
# -----------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".heif", ".gif", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mts", ".m2ts", ".3gp", ".3g2", ".mkv", ".wmv"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS

ZIP_EXTS = {".zip"}

# vpath scheme (compatible with your embed pipeline)
ZIP_PREFIX = "zip://"
FILE_PREFIX = "file://"

# -----------------------------
# Helpers
# -----------------------------

def _is_media_name(name: str) -> bool:
    return Path(name).suffix.lower() in MEDIA_EXTS

def _guess_provider_from_path(p: Path) -> str:
    s = str(p).lower()
    if "apple" in s or "icloud" in s:
        return "apple"
    if "google" in s or "takeout" in s:
        return "google"
    return "unknown"

def _iter_zip_media(zip_path: Path) -> Iterable[Tuple[str, str]]:
    """
    Yield (vpath, filename) for media entries in a zip. vpath: 'zip://<zip>!<inner>'
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if not name or name.endswith("/"):
                    continue
                if _is_media_name(name):
                    vpath = f"{ZIP_PREFIX}{str(zip_path)}!{name}"
                    yield vpath, Path(name).name
    except zipfile.BadZipFile:
        # Skip corrupt zips gracefully
        return

def _iter_dir_media(root: Path) -> Iterable[Tuple[str, str]]:
    """
    Yield (vpath, filename) for media files under a directory. vpath: 'file://<path>'
    """
    for p in root.rglob("*"):
        try:
            if p.is_file() and _is_media_name(p.name):
                yield f"{FILE_PREFIX}{str(p)}", p.name
        except OSError:
            # Skip unreadable
            continue

@dataclass
class ManifestRow:
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

    def as_dict(self) -> Dict[str, str]:
        return {
            "vpath": self.vpath,
            "filename": self.filename,
            "ext": self.ext,
            "provider": self.provider,
            "match_type": self.match_type,
            "taken_time": self.taken_time,
            "gps_lat": self.gps_lat,
            "gps_lon": self.gps_lon,
            "gps_alt": self.gps_alt,
            "description": self.description,
            "album": self.album,
            "people": self.people,
        }

MANIFEST_HEADERS = list(ManifestRow("", "", "", "", "", "", "", "", "", "", "", "").as_dict().keys())

def _write_manifest(rows: Iterable[ManifestRow], out_csv: Path) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_HEADERS)
        w.writeheader()
        for r in rows:
            w.writerow(r.as_dict())
            count += 1
    return count

def _merge_csv_into_master(temp_csv: Path, master_csv: Path, key: str = "vpath") -> dict:
    """
    Merge temp_csv into master_csv, dedupe by `key`.
    New rows with the same key OVERWRITE old rows.
    Returns stats dict.
    """
    temp_csv = Path(temp_csv)
    master_csv = Path(master_csv)

    if not temp_csv.exists():
        raise FileNotFoundError(f"temp manifest not found: {temp_csv}")

    with temp_csv.open("r", encoding="utf-8", newline="") as fh:
        new_reader = csv.DictReader(fh)
        new_rows = list(new_reader)
        new_headers = list(new_reader.fieldnames or [])

    if key not in new_headers:
        raise KeyError(f"Column '{key}' not found in new manifest")

    # If master not present, bootstrap it
    if not master_csv.exists():
        master_csv.parent.mkdir(parents=True, exist_ok=True)
        with master_csv.open("w", encoding="utf-8", newline="") as out:
            w = csv.DictWriter(out, fieldnames=new_headers)
            w.writeheader()
            w.writerows(new_rows)
        return {"before": 0, "new": len(new_rows), "after": len(new_rows), "replaced": 0, "skipped": 0}

    # Load master
    with master_csv.open("r", encoding="utf-8", newline="") as fh:
        master_reader = csv.DictReader(fh)
        master_rows = list(master_reader)
        master_headers = list(master_reader.fieldnames or [])

    if key not in master_headers:
        # allow migrating old masters that lack the key, as long as new has it
        pass

    before = len(master_rows)

    # headers = union preserving master order
    headers = list(master_headers)
    for h in new_headers:
        if h not in headers:
            headers.append(h)

    master_map: Dict[str, Dict[str, str]] = {}
    for r in master_rows:
        kval = r.get(key)
        if kval:
            master_map[kval] = r

    replaced = 0
    added = 0
    skipped = 0

    for r in new_rows:
        kval = r.get(key)
        if not kval:
            skipped += 1
            continue
        if kval in master_map:
            replaced += 1
        else:
            added += 1
        merged = {h: (r.get(h, "") if r.get(h) is not None else "") for h in headers}
        master_map[kval] = merged

    with master_csv.open("w", encoding="utf-8", newline="") as out:
        w = csv.DictWriter(out, fieldnames=headers)
        w.writeheader()
        w.writerows(master_map.values())

    after = len(master_map)
    return {"before": before, "new": len(new_rows), "after": after, "replaced": replaced, "skipped": skipped}

# -----------------------------
# CLI
# -----------------------------

@click.group()
def app():
    """photo-unifier CLI (Step 1: Manifest build/append)."""
    pass

@app.command("index")
@click.argument("paths", nargs=-1, type=click.Path(path_type=Path, exists=True))
@click.option("--manifest", "manifest_out", required=True, type=click.Path(dir_okay=False, path_type=Path),
              help="Write the temporary manifest CSV here (will overwrite).")
@click.option("--to-master", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Merge the temp manifest into this master CSV (dedupe by --dedupe-key).")
@click.option("--dedupe-key", default="vpath", show_default=True,
              help="CSV column to use for deduping when merging into --to-master.")
@click.option("--source-tag", default=None,
              help="Optional source label to stamp in the 'provider' column (overrides auto-guess).")
def index_cmd(paths: Tuple[Path, ...], manifest_out: Path, to_master: Optional[Path],
              dedupe_key: str, source_tag: Optional[str]):
    """
    Scan PATHS (dirs or zips) for media and produce a temp manifest.
    Optionally merge into a persistent master.
    """
    if not paths:
        click.echo("ERROR: Provide one or more PATHS (folders or ZIPs).", err=True)
        sys.exit(2)

    # Collect media from all inputs
    rows: List[ManifestRow] = []
    zips = 0
    dirs = 0

    click.echo(f"Indexing from {len(paths)} input(s)...")

    for root in paths:
        if root.is_file() and root.suffix.lower() in ZIP_EXTS:
            zips += 1
            prov = source_tag or _guess_provider_from_path(root)
            for vpath, fname in _iter_zip_media(root):
                rows.append(ManifestRow(
                    vpath=vpath,
                    filename=fname,
                    ext=Path(fname).suffix.lower(),
                    provider=prov,
                    match_type="",     # available for future use
                    taken_time="",     # leave blank; embed stage will resolve from EXIF/MediaInfo/filename/mtime
                    gps_lat="",
                    gps_lon="",
                    gps_alt="",
                    description="",
                    album="",
                    people="",
                ))
        elif root.is_dir():
            dirs += 1
            prov = source_tag or _guess_provider_from_path(root)
            for vpath, fname in _iter_dir_media(root):
                rows.append(ManifestRow(
                    vpath=vpath,
                    filename=fname,
                    ext=Path(fname).suffix.lower(),
                    provider=prov,
                    match_type="",
                    taken_time="",
                    gps_lat="",
                    gps_lon="",
                    gps_alt="",
                    description="",
                    album="",
                    people="",
                ))
        else:
            click.echo(f"WARNING: Skipping unsupported path: {root}", err=True)

    click.echo(f"Found media rows: {len(rows)} | ZIPs scanned: {zips} | Dirs scanned: {dirs}")

    # Write temp manifest
    written = _write_manifest(rows, manifest_out)
    click.echo(f"Wrote manifest: {manifest_out} (rows={written})")

    # Optional merge into master
    if to_master:
        try:
            stats = _merge_csv_into_master(manifest_out, to_master, key=dedupe_key)
            click.echo(
                f"Merged into master: {to_master} | "
                f"before={stats['before']} new={stats['new']} after={stats['after']} "
                f"replaced={stats['replaced']} skipped={stats['skipped']}"
            )
        except Exception as e:
            click.echo(f"ERROR merging into master: {e}", err=True)
            raise

@app.command("manifest-verify")
@click.option("--master", required=True, type=click.Path(dir_okay=False, path_type=Path),
              help="Path to the master manifest CSV.")
@click.option("--key", default="vpath", show_default=True, help="Key column to check for duplicates.")
def manifest_verify(master: Path, key: str):
    """Check master manifest for duplicate keys and print basic stats."""
    if not master.exists():
        click.echo(f"ERROR: Master manifest not found: {master}", err=True)
        sys.exit(2)

    with master.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        headers = reader.fieldnames or []

    click.echo(f"Rows: {len(rows)}")
    if key not in headers:
        click.echo(f"ERROR: key '{key}' not found in headers {headers}", err=True)
        sys.exit(2)

    seen = {}
    dupes = 0
    for r in rows:
        kval = r.get(key)
        if not kval:
            continue
        if kval in seen:
            dupes += 1
        else:
            seen[kval] = 1
    if dupes:
        click.echo(f"⚠️ Duplicates by '{key}': {dupes}")
    else:
        click.echo("No duplicates by key. ✅")

@app.command("snapshot")
@click.option("--ext-root", required=True, type=click.Path(path_type=Path),
              help="External drive root (e.g., F:\\).")
@click.option("--project-root", required=True, type=click.Path(path_type=Path),
              help="Project root folder (this repo).")
def snapshot(ext_root: Path, project_root: Path):
    """
    Print a concise snapshot of external state (unifier_state, library, takeouts)
    and project code layout. Does not list every file.
    """
    def pexists(p: Path) -> bool:
        try:
            return p.exists()
        except OSError:
            return False

    def count_files(p: Path, pattern: str = "*") -> int:
        try:
            return sum(1 for _ in p.rglob(pattern))
        except OSError:
            return 0

    def approx_media_counts(year_dir: Path) -> int:
        cnt = 0
        try:
            for y in sorted([d for d in year_dir.iterdir() if d.is_dir()]):
                # Just sample per-year roughly by counting files under year (not recursive deep detail)
                cnt += sum(1 for _ in y.rglob("*") if _.is_file() and _.suffix.lower() in MEDIA_EXTS)
        except OSError:
            pass
        return cnt

    state = ext_root / "_unifier_state"
    library = ext_root / "_library"  # may be different in your final pipeline; this is informational
    camera_root = ext_root / "MSp" / "Camera"

    print(f"External project snapshot @ {ext_root}")
    # _unifier_state
    if pexists(state):
        manifest = state / "manifest.csv"
        shards = state / "shards"
        logs = state / "logs"
        print(f"\n{state}")
        if manifest.exists():
            try:
                rows = sum(1 for _ in csv.reader(manifest.open("r", encoding="utf-8", newline="")))
                rows = max(0, rows - 1)  # minus header
            except Exception:
                rows = "?"
            print(f"├── manifest.csv  (~{rows} rows)")
        else:
            print("├── manifest.csv  (not found)")
        if pexists(shards):
            shard_files = list(shards.glob("*.csv"))
            print(f"├── shards\\  ({len(shard_files)} shard files)")
        if pexists(logs):
            log_files = sorted(logs.glob("*.log"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
            print(f"├── logs\\  ({len(log_files)} logs)")
            if log_files:
                print(f"│   └── newest: {log_files[-1].name}")

    # library (quick stats)
    if pexists(library):
        media = 0
        xmps = 0
        try:
            for p in library.rglob("*"):
                if not p.is_file():
                    continue
                suf = p.suffix.lower()
                if suf == ".xmp":
                    xmps += 1
                elif suf in MEDIA_EXTS:
                    media += 1
        except OSError:
            pass
        print(f"\n{library}")
        print(f"├── summary: media={media:,}  sidecars={xmps:,}  files={(media + xmps):,}")

        # Print a handful of year folders with approximate counts
        years = []
        try:
            for d in library.iterdir():
                if d.is_dir() and re.match(r"^\d{4}$", d.name):
                    years.append(d)
            years = sorted(years, key=lambda p: p.name)[:12]
        except OSError:
            years = []

        for y in years:
            approx = sum(1 for _ in y.rglob("*") if _.is_file() and _.suffix.lower() in MEDIA_EXTS)
            print(f"├── {y.name}\\  (~{approx} media)")
        # Unknown
        unk = library / "Unknown"
        if pexists(unk):
            approx_unk = sum(1 for _ in unk.rglob("*") if _.is_file() and _.suffix.lower() in MEDIA_EXTS)
            print(f"├── Unknown\\  (~{approx_unk} media)")

    # takeouts
    if pexists(camera_root):
        apple = camera_root / "Apple takeout"
        g_ms = camera_root / "Google takeout - maliquespooner"
        g_mt = camera_root / "Google takeout - mtspalace"

        for folder in [apple, g_ms, g_mt]:
            if pexists(folder):
                zips = [p for p in folder.glob("*.zip")]
                print(f"\n{folder}")
                print(f"├── zip files: {len(zips)}")
                if zips:
                    sample = ", ".join(z.name for z in zips[:3])
                    more = len(zips) - 3
                    if more > 0:
                        print(f"│   ├── sample: {sample}")
                        print(f"│   └── … {more} more")
                    else:
                        print(f"│   └── sample: {sample}")

    # project code
    print(f"\nProject code @ {project_root}")
    src = project_root / "src" / "photo_unifier"
    tests = project_root / "tests"
    venv = project_root / ".venv"

    if pexists(src):
        py_files = list(src.glob("*.py"))
        subs = [d.name for d in src.iterdir() if d.is_dir()]
        print(f"├── src\\photo_unifier\\  (py files: {len(py_files)})")
        if subs:
            print(f"│   ├── subpackages: {', '.join(subs)}")
        tops = ", ".join(p.name for p in sorted(py_files, key=lambda p: p.name)[:8])
        print(f"│   └── top files: {tops}")

    if pexists(tests):
        tcount = len(list(tests.glob("**/*.py")))
        print(f"├── tests\\  (py files: {tcount})")

    # project files (common)
    project_files = []
    for name in ["print_tree.py", "pyproject.toml", "README.md", "requirements.txt"]:
        if pexists(project_root / name):
            project_files.append(name)
    if project_files:
        print(f"├── project files: {', '.join(project_files)}")

    if pexists(venv) and pexists(venv / "Scripts" / "python.exe"):
        print(f"└── .venv\\  (with python.exe)")

# Entry point
def main():
    app(prog_name="photo-unifier")

if __name__ == "__main__":
    main()
