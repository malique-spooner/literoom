r"""De-duplicate photos & videos in the clean library.

Two passes:
1) EXACT duplicates (all media): group by (size, SHA-1). Keep one, mark others as duplicates.
2) NEAR duplicates (images only, optional): pHash within *each day folder* (YYYY/YYYY-MM/YYYY-MM-DD).
   Pairs within a Hamming distance threshold (default 6) are marked as near duplicates.

Dry-run (report only):
  python -m photo_unifier.dedupe --root F:\_library --out F:\_unifier_state\dedupe_report.csv --phash --threshold 6

Apply (moves duplicates to quarantine; nothing is deleted):
  python -m photo_unifier.dedupe --root F:\_library --apply --quarantine F:\_unifier_state\quarantine --phash --threshold 6
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# Images we can pHash for near-duplicate detection
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mts", ".m2ts", ".3gp", ".3g2", ".mkv", ".wmv"}

# Optional: HEIC/HEIF opener so Pillow can read those if present
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:
    pass

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # allow very large images you trust

try:
    import imagehash  # type: ignore
except Exception:
    imagehash = None  # near-dup disabled if not installed

DAY_PATH_RE = re.compile(r"[\\/](\d{4})[\\/](\d{4}-\d{2})[\\/](\d{4}-\d{2}-\d{2})[\\/]")

@dataclass
class DupeRow:
    """One duplicate detected relative to a chosen 'keeper'."""
    reason: str            # "exact" or "near"
    group_id: str          # stable id per group (e.g., sha1 for exact; YYYYMMDD+anchor hash for near)
    keep_path: str         # path we decided to keep
    dupe_path: str         # path to move to quarantine on --apply
    size_bytes: int
    sha1: str              # for exact; may be blank for near
    phash_keep: str        # hex pHash (images) for keep; blank for videos
    phash_dupe: str        # hex pHash (images) for dupe; blank for videos
    distance: int          # hamming distance for near; 0 for exact

def _is_media_file(p: Path) -> bool:
    if not p.is_file():
        return False
    if p.suffix.lower() == ".xmp":
        return False
    return p.suffix.lower() in IMAGE_EXTS.union(VIDEO_EXTS)

def _iter_media_files(root: Path) -> Iterator[Path]:
    for p in root.rglob("*"):
        if _is_media_file(p):
            yield p

def _sha1(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _phash(path: Path) -> Optional[imagehash.ImageHash]:
    if imagehash is None:
        return None
    if path.suffix.lower() not in IMAGE_EXTS:
        return None
    try:
        with Image.open(path) as im:
            return imagehash.phash(im)
    except Exception:
        return None

def _day_key(p: Path) -> Optional[str]:
    """Return YYYY-MM-DD key from path (clean-library layout) or None."""
    m = DAY_PATH_RE.search(str(p))
    if not m:
        return None
    return m.group(3)

def _group_by_size(files: Iterable[Path]) -> Dict[int, List[Path]]:
    g: Dict[int, List[Path]] = {}
    for p in files:
        try:
            sz = p.stat().st_size
        except Exception:
            continue
        g.setdefault(sz, []).append(p)
    return g

def _exact_dupes(root: Path) -> List[DupeRow]:
    """Find exact duplicates across the whole tree using (size, sha1)."""
    print("Scanning for exact duplicates...")
    files = list(_iter_media_files(root))
    by_size = _group_by_size(files)

    dupes: List[DupeRow] = []
    for size, group in by_size.items():
        if len(group) < 2:
            continue
        # compute sha1 only when needed
        by_hash: Dict[str, List[Path]] = {}
        for p in group:
            try:
                h = _sha1(p)
            except Exception:
                continue
            by_hash.setdefault(h, []).append(p)

        for sha, paths in by_hash.items():
            if len(paths) < 2:
                continue
            # choose keeper: shortest path (stable)
            keeper = min(paths, key=lambda x: (len(str(x)), str(x).lower()))
            for dupe in paths:
                if dupe == keeper:
                    continue
                phk = ""  # images only (not needed for exact)
                phd = ""
                if dupe.suffix.lower() in IMAGE_EXTS:
                    hk = _phash(keeper)
                    hd = _phash(dupe)
                    phk = hk.hash.flatten().tobytes().hex() if (hk and hasattr(hk, "hash")) else ""
                    phd = hd.hash.flatten().tobytes().hex() if (hd and hasattr(hd, "hash")) else ""
                dupes.append(DupeRow(
                    reason="exact",
                    group_id=sha,
                    keep_path=str(keeper),
                    dupe_path=str(dupe),
                    size_bytes=size,
                    sha1=sha,
                    phash_keep=phk,
                    phash_dupe=phd,
                    distance=0,
                ))
    print(f"Exact duplicates found: {len(dupes)}")
    return dupes

def _near_dupes_by_day(root: Path, threshold: int = 6) -> List[DupeRow]:
    """Find near-duplicate IMAGES within each day folder using pHash distance."""
    if imagehash is None:
        print("Near-duplicate check skipped (imagehash not installed).")
        return []
    print(f"Scanning for near duplicates (threshold={threshold})...")
    # bucket images by day key
    by_day: Dict[str, List[Path]] = {}
    for p in _iter_media_files(root):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        key = _day_key(p)
        if not key:
            continue
        by_day.setdefault(key, []).append(p)

    dupes: List[DupeRow] = []
    for day, paths in by_day.items():
        # compute pHash for each image in the day
        hashes: List[Tuple[Path, Optional[imagehash.ImageHash]]] = []
        for p in paths:
            hashes.append((p, _phash(p)))
        # naive pairwise within the day (days are typically small; fast enough)
        n = len(hashes)
        for i in range(n):
            pi, hi = hashes[i]
            if hi is None:
                continue
            # keeper pick: earliest lexicographically (stable)
            for j in range(i + 1, n):
                pj, hj = hashes[j]
                if hj is None:
                    continue
                dist = int(hi - hj)  # hamming distance
                if dist <= threshold:
                    keeper = min(pi, pj, key=lambda x: str(x).lower())
                    dupe   = pj if keeper == pi else pi
                    phk = hi.hash.flatten().tobytes().hex() if hasattr(hi, "hash") else ""
                    phd = hj.hash.flatten().tobytes().hex() if hasattr(hj, "hash") else ""
                    dupes.append(DupeRow(
                        reason="near",
                        group_id=f"{day}:{phk[:8]}",
                        keep_path=str(keeper),
                        dupe_path=str(dupe),
                        size_bytes=dupe.stat().st_size if dupe.exists() else 0,
                        sha1="",
                        phash_keep=phk,
                        phash_dupe=phd,
                        distance=dist,
                    ))
    print(f"Near duplicates found: {len(dupes)}")
    return dupes

def _write_report(rows: List[DupeRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()) if rows else [
            "reason","group_id","keep_path","dupe_path","size_bytes","sha1","phash_keep","phash_dupe","distance"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

def _move_with_xmp(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)
    xmp_src = src.with_suffix(src.suffix + ".xmp")
    xmp_dst = dst.with_suffix(dst.suffix + ".xmp")
    if xmp_src.exists():
        try:
            os.replace(xmp_src, xmp_dst)
        except Exception:
            pass

def _apply_moves(rows: List[DupeRow], quarantine: Path) -> int:
    """Move duplicate files (and their XMP) into quarantine. Returns count moved."""
    moved = 0
    for r in rows:
        src = Path(r.dupe_path)
        if not src.exists():
            continue
        reason_dir = "exact" if r.reason == "exact" else f"near_d{r.distance}"
        # preserve relative path after YYYY folder if present (nice for triage)
        rel = src
        try:
            parts = list(src.parts)
            # find index of the first 4-digit year folder
            idx = next(i for i, p in enumerate(parts) if re.match(r"^\d{4}$", p))
            rel = Path(*parts[idx:])  # keep YYYY/.. substructure
        except Exception:
            rel = src.name  # fallback
        dst = quarantine / reason_dir / str(rel)
        try:
            _move_with_xmp(src, dst)
            moved += 1
        except Exception:
            continue
    return moved

def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="De-duplicate exact and near-duplicate media.")
    ap.add_argument("--root", required=True, help="Clean library root (e.g., F:\\_library).")
    ap.add_argument("--out", default=None, help="CSV report path for dry-run/apply results.")
    ap.add_argument("--phash", action="store_true", help="Enable near-duplicate detection for images (pHash).")
    ap.add_argument("--threshold", type=int, default=6, help="Hamming distance threshold for near-duplicate (default 6).")
    ap.add_argument("--apply", action="store_true", help="Move duplicates to quarantine (no deletes).")
    ap.add_argument("--quarantine", default=None, help="Quarantine folder (required with --apply).")
    return ap

def main(argv: Optional[List[str]] = None) -> int:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: --root not found: {root}")
        return 2

    # 1) Exact dupes across the whole tree
    exact = _exact_dupes(root)

    # 2) Optional near dupes by day folders (images only)
    near: List[DupeRow] = []
    if args.phash:
        near = _near_dupes_by_day(root, threshold=args.threshold)

    rows = exact + near
    print(f"TOTAL duplicates flagged: {len(rows)}")

    # Write report if requested
    if args.out:
        _write_report(rows, Path(args.out))
        print(f"Wrote report: {args.out}")

    # Apply (move dupes) if requested
    if args.apply:
        if not args.quarantine:
            print("ERROR: --quarantine is required with --apply")
            return 2
        q = Path(args.quarantine)
        moved = _apply_moves(rows, q)
        print(f"Moved {moved} duplicates to quarantine: {q}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
