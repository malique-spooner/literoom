#!/usr/bin/env python3
"""
Print a directory tree OR a compact "project snapshot" of the external-drive
folders used by the photo-unifier pipeline, PLUS a quick summary of code files
in your project root.

Examples
--------
# Traditional tree (unchanged behavior)
python print_tree.py --root . --max-depth 4 --file-limit 50

# Project snapshot: external drive + project code
python print_tree.py --project-snapshot --ext-root F:\ --project-root .

# Save snapshot to a file
python print_tree.py --project-snapshot --ext-root F:\ --project-root . --output F:\_unifier_state\project_snapshot.txt
"""

from __future__ import annotations
import argparse
import fnmatch
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------- Generic tree (existing behavior) ----------------

DEFAULT_IGNORES = [
    ".git", ".venv", "__pycache__", "build", "dist",
    "*.egg-info", "node_modules", ".mypy_cache", ".pytest_cache",
    ".DS_Store", "Thumbs.db"
]

def should_ignore(name: str, ignore_patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in ignore_patterns)

def iter_entries(path: Path, ignore_patterns: List[str]) -> tuple[list[Path], list[Path]]:
    dirs, files = [], []
    try:
        with os.scandir(path) as it:
            for entry in it:
                name = entry.name
                if should_ignore(name, ignore_patterns):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        dirs.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        files.append(Path(entry.path))
                except OSError:
                    continue
    except OSError:
        return [], []
    dirs.sort(key=lambda p: p.name.lower())
    files.sort(key=lambda p: p.name.lower())
    return dirs, files

def print_tree(root: Path, max_depth: Optional[int], ignore_patterns: List[str], file_limit: Optional[int]) -> str:
    lines: List[str] = [str(root.resolve())]

    def walk(dir_path: Path, prefix: str, depth: int):
        if max_depth is not None and depth > max_depth:
            return
        dirs, files = iter_entries(dir_path, ignore_patterns)
        entries: List[Path] = dirs + files

        if file_limit is not None and file_limit > 0 and len(entries) > file_limit:
            visible = entries[:file_limit]
            remainder = len(entries) - file_limit
        else:
            visible = entries
            remainder = 0

        for i, p in enumerate(visible):
            connector = "└── " if i == len(visible) - 1 and remainder == 0 else "├── "
            lines.append(prefix + connector + p.name)
            if p.is_dir():
                extension = "    " if (i == len(visible) - 1 and remainder == 0) else "│   "
                if max_depth is None or depth < max_depth:
                    walk(p, prefix + extension, depth + 1)

        if remainder > 0:
            lines.append(prefix + f"└── … {remainder} more")

    walk(root, "", 1)
    return "\n".join(lines)

# ---------------- Project snapshot (external drive) ----------------

IMAGE_EXTS = {".jpg", ".jpeg", ".heic", ".heif", ".png", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mts", ".m2ts", ".3gp", ".3g2", ".mkv", ".wmv"}
SIDE_EXT   = ".xmp"
YEAR_RE    = re.compile(r"^\d{4}$")

def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def _count_lines_fast(p: Path) -> int:
    try:
        with p.open("rb") as fh:
            return sum(1 for _ in fh)
    except Exception:
        return 0

def _list_files(p: Path, patterns: Tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    try:
        with os.scandir(p) as it:
            for e in it:
                if e.is_file():
                    name = e.name.lower()
                    if any(fnmatch.fnmatch(name, pat) for pat in patterns):
                        out.append(Path(e.path))
    except Exception:
        pass
    out.sort(key=lambda x: x.name.lower())
    return out

def _count_media_recursive(root: Path) -> tuple[int, int, int]:
    media = sidecars = total = 0
    for _dirpath, _dirs, files in os.walk(root):
        total += len(files)
        for f in files:
            fn = f.lower()
            if fn.endswith(SIDE_EXT):
                sidecars += 1
            elif any(fn.endswith(ext) for ext in IMAGE_EXTS | VIDEO_EXTS):
                media += 1
    return media, sidecars, total

def _count_media_in_dir(root: Path) -> int:
    c = 0
    for _dirpath, _dirs, files in os.walk(root):
        for f in files:
            fn = f.lower()
            if any(fn.endswith(ext) for ext in IMAGE_EXTS | VIDEO_EXTS):
                c += 1
    return c

def _summarize_years(library_root: Path) -> List[str]:
    lines: List[str] = []
    try:
        with os.scandir(library_root) as it:
            years = []
            unknowns = []
            for e in it:
                if not e.is_dir():
                    continue
                nm = e.name
                if YEAR_RE.match(nm):
                    years.append(Path(e.path))
                elif nm.lower() == "unknown":
                    unknowns.append(Path(e.path))
        years.sort(key=lambda p: p.name)
        for y in years[:20]:
            cnt = _count_media_in_dir(y)
            lines.append(f"├── {y.name}\\  (~{cnt} media)")
        if len(years) > 20:
            lines.append(f"├── … {len(years) - 20} more years")
        for u in unknowns:
            cnt = _count_media_in_dir(u)
            lines.append(f"├── {u.name}\\  (~{cnt} media)")
    except Exception:
        pass
    return lines

def print_project_snapshot(ext_root: Path, project_root: Path) -> str:
    lines: List[str] = []
    R = ext_root

    state_dir   = R / "_unifier_state"
    lib_a       = R / "_library_embedded"
    lib_b       = R / "_library"
    msp_root    = R / "MSp" / "Camera"
    apple_dir   = msp_root / "Apple takeout"
    g1_dir      = msp_root / "Google takeout - maliquespooner"
    g2_dir      = msp_root / "Google takeout - mtspalace"

    # Header
    lines.append(f"External project snapshot @ {R}")
    lines.append("")

    # _unifier_state
    if _exists(state_dir):
        lines.append(str(state_dir))
        manifest = state_dir / "manifest.csv"
        if _exists(manifest):
            rows = max(0, _count_lines_fast(manifest) - 1)
            lines.append(f"├── manifest.csv  (~{rows} rows)")
        shards = state_dir / "shards"
        logs   = state_dir / "logs"
        if _exists(shards):
            shard_files = _list_files(shards, ("manifest_part_*.csv",))
            lines.append(f"├── shards\\  ({len(shard_files)} shard files)")
            if shard_files:
                lines.append(f"│   ├── first: {shard_files[0].name}")
                lines.append(f"│   └── last : {shard_files[-1].name}")
        if _exists(logs):
            log_files = _list_files(logs, ("*.log",))
            lines.append(f"├── logs\\  ({len(log_files)} logs)")
            if log_files:
                lines.append(f"│   └── newest: {log_files[-1].name}")
        lines.append("")

    # Destination library (prefer embedded if present)
    lib = lib_a if _exists(lib_a) else (lib_b if _exists(lib_b) else None)
    if lib:
        lines.append(str(lib))
        media, side, total = _count_media_recursive(lib)
        lines.append(f"├── summary: media={media:,}  sidecars={side:,}  files={total:,}")
        lines.extend(_summarize_years(lib))
        lines.append("")

    # Source takeouts
    for label, p in (("Apple takeout", apple_dir),
                     ("Google takeout - maliquespooner", g1_dir),
                     ("Google takeout - mtspalace", g2_dir)):
        if _exists(p):
            lines.append(str(p))
            zips = _list_files(p, ("*.zip",))
            subdirs = []
            try:
                with os.scandir(p) as it:
                    for e in it:
                        if e.is_dir():
                            subdirs.append(e.name)
            except Exception:
                pass
            lines.append(f"├── zip files: {len(zips)}")
            if zips[:3]:
                lines.append("│   ├── sample: " + ", ".join(z.name for z in zips[:3]))
            if len(zips) > 3:
                lines.append(f"│   └── … {len(zips) - 3} more")
            if subdirs:
                lines.append(f"├── subfolders: {len(subdirs)} (top-level)")
            lines.append("")

    # ---------------- Project code summary (new) ----------------
    lines.append(f"Project code @ {project_root.resolve()}")
    src_pkg = project_root / "src" / "photo_unifier"
    tests   = project_root / "tests"

    def _count_py_recursive(p: Path) -> int:
        n = 0
        for _dp, _ds, files in os.walk(p):
            for f in files:
                if f.lower().endswith(".py"):
                    n += 1
        return n

    def _list_dir_names(p: Path, limit: int) -> list[str]:
        out = []
        try:
            with os.scandir(p) as it:
                for e in it:
                    if e.is_dir():
                        out.append(e.name)
        except Exception:
            pass
        out.sort(key=str.lower)
        return out[:limit]

    def _list_files_names(p: Path, patterns: Tuple[str, ...], limit: int) -> list[str]:
        out = []
        try:
            with os.scandir(p) as it:
                for e in it:
                    if e.is_file():
                        nm = e.name
                        if any(fnmatch.fnmatch(nm.lower(), pat) for pat in patterns):
                            out.append(nm)
        except Exception:
            pass
        out.sort(key=str.lower)
        return out[:limit]

    # src/photo_unifier
    if _exists(src_pkg):
        py_count = _count_py_recursive(src_pkg)
        subdirs  = _list_dir_names(src_pkg, limit=8)
        py_files = _list_files_names(src_pkg, ("*.py",), limit=12)
        lines.append(f"├── src\\photo_unifier\\  (py files: {py_count})")
        if subdirs:
            lines.append(f"│   ├── subpackages: {', '.join(subdirs)}")
        if py_files:
            lines.append(f"│   └── top files: {', '.join(py_files)}")
    else:
        lines.append("├── src\\photo_unifier\\  (missing)")

    # tests
    if _exists(tests):
        t_count = _count_py_recursive(tests)
        t_files = _list_files_names(tests, ("test_*.py","*_test.py","*.py"), limit=10)
        lines.append(f"├── tests\\  (py files: {t_count})")
        if t_files:
            lines.append(f"│   └── top files: {', '.join(t_files)}")
    else:
        lines.append("├── tests\\  (missing)")

    # Top-level config files
    top_files = _list_files_names(
        project_root,
        ("pyproject.toml","requirements*.txt","README*","Run-Overnight.ps1","print_tree.py",".env","*.md"),
        limit=20
    )
    if top_files:
        lines.append(f"├── project files: {', '.join(top_files)}")

    # venv presence
    venv = project_root / ".venv"
    if _exists(venv):
        py = venv / "Scripts" / "python.exe"
        lines.append(f"└── .venv\\  ({'with' if _exists(py) else 'no'} python.exe)")
    else:
        lines.append("└── .venv\\  (missing)")

    lines.append("")
    return "\n".join(lines)

# ---------------- CLI ----------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Print a directory tree or an external-drive project snapshot.")
    ap.add_argument("--root", default=".", help="Root folder to print (tree mode).")
    ap.add_argument("--max-depth", type=int, default=None, help="Limit recursion depth (tree mode).")
    ap.add_argument(
        "--ignore",
        default=",".join(DEFAULT_IGNORES),
        help=f"Comma-separated patterns to ignore (tree mode default: {','.join(DEFAULT_IGNORES)})",
    )
    ap.add_argument("--file-limit", type=int, default=None, help="Show at most N entries per directory (tree mode).")
    ap.add_argument("--output", type=str, default=None, help="Write output to this file.")
    ap.add_argument("--project-snapshot", action="store_true",
                    help="Print compact snapshot of external-drive folders used by the project + project code summary.")
    ap.add_argument("--ext-root", default="F:\\", help="External drive root for snapshot (default: F:\\)")
    ap.add_argument("--project-root", default=".", help="Project root for code summary (default: current directory)")
    args = ap.parse_args()

    if args.project_snapshot:
        text = print_project_snapshot(Path(args.ext_root), Path(args.project_root))
    else:
        root = Path(args.root).resolve()
        ignore_patterns = [p.strip() for p in args.ignore.split(",") if p.strip()]
        text = print_tree(root, args.max_depth, ignore_patterns, args.file_limit)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"✅ Wrote tree to: {args.output}")
    else:
        print(text)

if __name__ == "__main__":
    main()
