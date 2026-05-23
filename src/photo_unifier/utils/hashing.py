from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_from_path(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def average_hash(path: Path, *, hash_size: int = 8) -> str:
    img = _image_from_path(path)
    gray = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype="float32").reshape(-1)
    if pixels.size == 0:
        return "0" * ((hash_size * hash_size + 3) // 4)
    avg = sum(pixels) / len(pixels)
    bits = ["1" if px >= avg else "0" for px in pixels]
    value = int("".join(bits), 2)
    width = (hash_size * hash_size + 3) // 4
    return f"{value:0{width}x}"


def difference_hash(path: Path, *, hash_size: int = 8) -> str:
    img = _image_from_path(path).convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img, dtype="float32")
    bits = []
    for row in range(hash_size):
        for col in range(hash_size):
            bits.append("1" if pixels[row, col] > pixels[row, col + 1] else "0")
    value = int("".join(bits), 2)
    width = (hash_size * hash_size + 3) // 4
    return f"{value:0{width}x}"


def hamming_distance(left: str, right: str) -> int:
    try:
        a = int(left, 16)
        b = int(right, 16)
    except Exception:
        return max(len(left), len(right))
    return (a ^ b).bit_count()


def blur_score(path: Path) -> float:
    img = _image_from_path(path).convert("L")
    edges = img.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(edges)
    return float(stat.var[0] if stat.var else 0.0)


__all__ = ["average_hash", "blur_score", "difference_hash", "hamming_distance", "sha256_file"]
