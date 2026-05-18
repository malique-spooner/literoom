from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image

from .phase_metadata import manifest


def _thumbnail_path(derivatives_dir: Path, asset_id: str, kind: str) -> Path:
    return derivatives_dir / kind / f"{asset_id}.jpg"


def _build_image_thumbnail(src: Path, dest: Path, size: int) -> tuple[int, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((size, size))
        img.save(dest, format="JPEG", quality=90)
        return img.width, img.height


def _build_video_preview(src: Path, dest: Path, ffmpeg_path: str, offset_seconds: int) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-ss",
            str(offset_seconds),
            "-i",
            str(src),
            "-frames:v",
            "1",
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def build_derivatives(
    db_path: Path,
    managed_library_dir: Path,
    derivatives_dir: Path,
    *,
    image_thumbnail_size: int = 512,
    video_preview_offset_seconds: int = 1,
    ffmpeg_path: Optional[str] = None,
    limit: int | None = None,
) -> dict:
    managed_library_dir = Path(managed_library_dir)
    derivatives_dir = Path(derivatives_dir)
    derivatives_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = ffmpeg_path or shutil.which("ffmpeg")

    built = failed = skipped = 0
    for row in manifest.iter_assets_for_derivatives(db_path=db_path, limit=limit):
        src = managed_library_dir / row["managed_path"]
        if not src.exists():
            manifest.record_thumbnail(
                db_path,
                row["id"],
                "primary",
                "",
                "FAILED",
                error="managed asset missing",
            )
            failed += 1
            continue

        if row["media_type"] in {"image", "raw"}:
            dest = _thumbnail_path(derivatives_dir, row["id"], "primary")
            try:
                width, height = _build_image_thumbnail(src, dest, image_thumbnail_size)
                manifest.record_thumbnail(
                    db_path, row["id"], "primary", str(dest), "READY", width=width, height=height
                )
                built += 1
            except Exception as exc:
                manifest.record_thumbnail(
                    db_path, row["id"], "primary", str(dest), "FAILED", error=str(exc)
                )
                failed += 1
            continue

        if row["media_type"] == "video":
            dest = _thumbnail_path(derivatives_dir, row["id"], "video_preview")
            if not ffmpeg_bin:
                manifest.record_thumbnail(
                    db_path,
                    row["id"],
                    "video_preview",
                    str(dest),
                    "SKIPPED",
                    error="ffmpeg not available",
                )
                skipped += 1
                continue
            ok = _build_video_preview(src, dest, ffmpeg_bin, video_preview_offset_seconds)
            if ok:
                manifest.record_thumbnail(
                    db_path, row["id"], "video_preview", str(dest), "READY"
                )
                built += 1
            else:
                manifest.record_thumbnail(
                    db_path,
                    row["id"],
                    "video_preview",
                    str(dest),
                    "FAILED",
                    error="ffmpeg preview extraction failed",
                )
                failed += 1

    return {"built": built, "failed": failed, "skipped": skipped}
