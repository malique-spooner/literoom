from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from functools import lru_cache
import os
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image, ImageOps

from .metadata import manifest
from .utils.hashing import sha256_file

try:
    import imagehash
except Exception:  # pragma: no cover - optional dependency fallback
    class _FallbackHash:
        def __init__(self, bits: np.ndarray):
            self.hash = np.asarray(bits, dtype=bool)

        def __sub__(self, other: object) -> int:
            other_bits = getattr(other, "hash", None)
            if other_bits is None:
                return int(self.hash.size)
            left = self.hash.reshape(-1)
            right = np.asarray(other_bits, dtype=bool).reshape(-1)
            size = min(left.size, right.size)
            distance = int(np.count_nonzero(left[:size] != right[:size]))
            return distance + abs(left.size - right.size)

    class _FallbackCropHash(_FallbackHash):
        def matches(self, other: object, hamming_cutoff: int = 12) -> bool:
            return self - other <= hamming_cutoff

        def hash_diff(self, other: object) -> tuple[int, int]:
            other_bits = getattr(other, "hash", None)
            if other_bits is None:
                return self.hash.size, self.hash.size
            right = np.asarray(other_bits, dtype=bool).reshape(-1)
            left = self.hash.reshape(-1)
            size = min(left.size, right.size)
            distance = int(np.count_nonzero(left[:size] != right[:size]))
            return self.hash.size, distance + abs(left.size - right.size)

    def _resize_gray(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
        return np.asarray(ImageOps.exif_transpose(img).convert("L").resize(size, Image.Resampling.LANCZOS), dtype=np.float32)

    def _mean_bits(values: np.ndarray) -> np.ndarray:
        return values > float(values.mean())

    class _FallbackImageHashModule:
        @staticmethod
        def phash(img: Image.Image) -> _FallbackHash:
            gray = _resize_gray(img, (8, 8))
            return _FallbackHash(_mean_bits(gray))

        @staticmethod
        def dhash(img: Image.Image) -> _FallbackHash:
            gray = _resize_gray(img, (9, 8))
            diff = gray[:, 1:] > gray[:, :-1]
            return _FallbackHash(diff)

        @staticmethod
        def whash(img: Image.Image) -> _FallbackHash:
            gray = _resize_gray(img, (8, 8))
            softened = (gray * 0.75) + (gray.mean() * 0.25)
            return _FallbackHash(_mean_bits(softened))

        @staticmethod
        def colorhash(img: Image.Image) -> _FallbackHash:
            rgb = np.asarray(ImageOps.exif_transpose(img).convert("RGB").resize((8, 8), Image.Resampling.LANCZOS), dtype=np.float32)
            channel_means = rgb.mean(axis=(0, 1))
            bits = np.array(
                [
                    channel_means[0] > 96.0,
                    channel_means[0] > 160.0,
                    channel_means[1] > 96.0,
                    channel_means[1] > 160.0,
                    channel_means[2] > 96.0,
                    channel_means[2] > 160.0,
                    channel_means.mean() > 96.0,
                    channel_means.mean() > 160.0,
                ],
                dtype=bool,
            )
            return _FallbackHash(bits)

        @staticmethod
        def crop_resistant_hash(img: Image.Image) -> _FallbackCropHash:
            gray = _resize_gray(img, (8, 8))
            return _FallbackCropHash(_mean_bits(gray))

    imagehash = _FallbackImageHashModule()

open_clip = None
try_to_load_from_cache = None
torch = None
faiss = None


def _canonical_asset(items: List[Dict]) -> str:
    ranked = sorted(
        items,
        key=lambda item: (
            -(item.get("orig_size") or 0),
            item.get("dt_original") or "",
            item["id"],
        ),
    )
    return ranked[0]["id"]


def _load_image_for_hash(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def _hash_size(hash_obj: object) -> int:
    bits = getattr(hash_obj, "hash", None)
    size = getattr(bits, "size", 0)
    try:
        return max(1, int(size))
    except Exception:
        return 1


def _imagehash_similarity(left: object, right: object) -> float:
    if left is None or right is None:
        return 0.0
    try:
        distance = float(left - right)  # type: ignore[operator]
    except Exception:
        return 0.0
    scale = max(_hash_size(left), _hash_size(right))
    return max(0.0, 1.0 - min(1.0, distance / scale))


def _crop_resistant_similarity(left: object, right: object) -> float:
    if left is None or right is None:
        return 0.0
    try:
        if hasattr(left, "matches") and left.matches(right, hamming_cutoff=12):  # type: ignore[call-arg]
            return 1.0
        diff = left.hash_diff(right)  # type: ignore[attr-defined]
        if isinstance(diff, tuple) and len(diff) >= 2:
            distance = float(diff[1])
            scale = max(1.0, float(diff[0]) * 16.0)
            return max(0.0, 1.0 - min(1.0, distance / scale))
    except Exception:
        return 0.0
    return 0.0


def _preview_hash_bundle(path: Path) -> Dict[str, object]:
    img = _load_image_for_hash(path)
    bundle: Dict[str, object] = {
        "phash": imagehash.phash(img),
        "dhash": imagehash.dhash(img),
        "whash": imagehash.whash(img),
        "colorhash": imagehash.colorhash(img),
    }
    try:
        bundle["crop_hash"] = imagehash.crop_resistant_hash(img)
    except Exception:
        bundle["crop_hash"] = None
    return bundle


def _normalize_clip_spec(clip_model: Optional[str]) -> tuple[str, Optional[str]]:
    raw = (clip_model or "ViT-B-32").strip()
    if raw.startswith("open_clip:"):
        raw = raw.split(":", 1)[1].strip()
    if "@" in raw:
        model_name, pretrained = raw.split("@", 1)
        return model_name.strip() or "ViT-B-32", pretrained.strip() or None
    return raw or "ViT-B-32", None


def _clip_pretrained_candidates(pretrained: Optional[str]) -> List[Optional[str]]:
    if pretrained:
        return [pretrained]
    return ["laion2b_s34b_b79k", "laion400m_e32", "openai", None]


def _clip_weight_is_cached(model_name: str, pretrained_name: Optional[str]) -> bool:
    try:
        if open_clip is None or try_to_load_from_cache is None:
            from huggingface_hub import try_to_load_from_cache as _try_to_load_from_cache  # type: ignore
            import open_clip as _open_clip  # type: ignore

            globals()["open_clip"] = _open_clip
            globals()["try_to_load_from_cache"] = _try_to_load_from_cache
        clip_module = globals().get("open_clip")
        cache_lookup = globals().get("try_to_load_from_cache")
        if clip_module is None or cache_lookup is None:
            return False
        cfg = clip_module.get_pretrained_cfg(model_name, pretrained_name or "")
    except Exception:
        return False
    repo = str(cfg.get("hf_hub") or "").strip().rstrip("/")
    if not repo:
        return False
    filenames = []
    if cfg.get("url"):
        filenames.append(Path(urlparse(str(cfg["url"])).path).name)
    filenames.extend(["open_clip_model.safetensors", "open_clip_pytorch_model.bin"])
    for filename in dict.fromkeys(name for name in filenames if name):
        try:
            cached = cache_lookup(repo, filename)
        except Exception:
            cached = None
        if cached:
            return True
    return False


@lru_cache(maxsize=4)
def _clip_backend(clip_model: Optional[str]) -> Optional[Dict[str, object]]:
    global open_clip, torch
    if open_clip is None or torch is None:
        try:
            if open_clip is None:
                import open_clip as _open_clip  # type: ignore

                open_clip = _open_clip
            if torch is None:
                import torch as _torch  # type: ignore

                torch = _torch
        except Exception:
            return None
    if open_clip is None or torch is None:
        return None
    model_name, pretrained = _normalize_clip_spec(clip_model)
    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    last_error: Optional[Exception] = None
    for pretrained_name in _clip_pretrained_candidates(pretrained):
        if not _clip_weight_is_cached(model_name, pretrained_name):
            continue
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained_name,
                device=device,
            )
            model.eval()
            return {
                "model": model,
                "preprocess": preprocess,
                "device": device,
                "spec": f"{model_name}@{pretrained_name or 'default'}",
            }
        except Exception as exc:  # pragma: no cover - optional runtime path
            last_error = exc
            continue
    if last_error is not None:
        return None
    return None


def _clip_embedding(path: Path, clip_model: Optional[str]) -> Optional[List[float]]:
    backend = _clip_backend(clip_model)
    if not backend or torch is None:
        return None
    model = backend["model"]
    preprocess = backend["preprocess"]
    device = backend["device"]
    try:
        with Image.open(path) as img:
            image = ImageOps.exif_transpose(img).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            encoded = model.encode_image(tensor)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        vector = encoded[0].detach().float().cpu().numpy().astype("float32")
        return vector.tolist()
    except Exception:
        return None


def _build_faiss_neighbors(vectors: List[List[float]], *, top_k: int = 6):
    global faiss
    if faiss is None:
        try:
            import faiss as _faiss  # type: ignore

            faiss = _faiss
        except Exception:
            return None
    if faiss is None or not vectors:
        return None
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return None
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    scores, indices = index.search(matrix, min(top_k, matrix.shape[0]))
    return scores, indices


def run_exact(
    db_path: Path,
    managed_library_dir: Path,
    *,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    managed_library_dir = Path(managed_library_dir)
    by_hash: Dict[str, List[Dict]] = defaultdict(list)
    processed = hashed = missing = 0

    for row in manifest.iter_built_assets(db_path=db_path, limit=limit):
        processed += 1
        managed_rel = row.get("managed_path")
        if not managed_rel:
            missing += 1
            continue
        asset_path = managed_library_dir / managed_rel
        if not asset_path.exists():
            missing += 1
            continue

        digest = row.get("sha256")
        if not digest:
            digest = sha256_file(asset_path)
            manifest.record_hash(db_path, row["id"], "sha256", "managed", digest)
            hashed += 1
        row["sha256"] = digest
        by_hash[digest].append(row)

    groups: List[Dict] = []
    for digest, items in by_hash.items():
        if len(items) < 2:
            continue
        canonical_asset_id = _canonical_asset(items)
        groups.append(
            {
                "canonical_asset_id": canonical_asset_id,
                "items": [
                    {
                        "asset_id": item["id"],
                        "score": 1.0,
                        "rationale": f"exact sha256 match {digest}",
                    }
                    for item in items
                ],
            }
        )

    created = manifest.replace_duplicate_groups(
        db_path,
        group_type="EXACT_SHA256",
        groups=groups,
    )
    return {
        "processed": processed,
        "hashed": hashed,
        "missing": missing,
        "groups": created,
    }


def _preview_path(
    row: Dict,
    managed_library_dir: Path,
    derivatives_dir: Optional[Path] = None,
) -> Optional[Path]:
    if row.get("media_type") in {"image", "raw"} and row.get("managed_path"):
        candidate = managed_library_dir / row["managed_path"]
        if candidate.exists():
            return candidate
    if row.get("media_type") == "video":
        thumbnail = row.get("thumbnail_path")
        if thumbnail and Path(thumbnail).exists():
            return Path(thumbnail)
        if derivatives_dir:
            candidate = Path(derivatives_dir) / "video_preview" / f"{row['id']}.jpg"
            if candidate.exists():
                return candidate
    if row.get("managed_path"):
        candidate = managed_library_dir / row["managed_path"]
        if candidate.exists():
            return candidate
    return None


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _pair_score(left: Dict, right: Dict) -> tuple[float, str, float]:
    phash_score = _imagehash_similarity(left.get("phash"), right.get("phash"))
    dhash_score = _imagehash_similarity(left.get("dhash"), right.get("dhash"))
    whash_score = _imagehash_similarity(left.get("whash"), right.get("whash"))
    color_score = _imagehash_similarity(left.get("colorhash"), right.get("colorhash"))
    crop_score = _crop_resistant_similarity(left.get("crop_hash"), right.get("crop_hash"))
    hash_score = (
        phash_score * 0.42
        + dhash_score * 0.23
        + whash_score * 0.17
        + color_score * 0.10
        + crop_score * 0.08
    )
    average_distance = round((1.0 - hash_score) * 64.0, 2)

    dt_left = _parse_dt(left.get("dt_original"))
    dt_right = _parse_dt(right.get("dt_original"))
    time_score = 0.0
    time_note = "time=unknown"
    if dt_left and dt_right:
        delta_seconds = abs((dt_left - dt_right).total_seconds())
        if delta_seconds <= 30:
            time_score = 1.0
        elif delta_seconds <= 300:
            time_score = 0.85
        elif delta_seconds <= 1800:
            time_score = 0.65
        elif delta_seconds <= 43200:
            time_score = 0.35
        else:
            time_score = 0.1
        time_note = f"time_delta={int(delta_seconds)}s"

    source_score = 0.0
    source_bits: list[str] = []
    if left.get("source") and left.get("source") == right.get("source"):
        source_score += 0.12
        source_bits.append("same source")
    if left.get("source_kind") and left.get("source_kind") == right.get("source_kind"):
        source_score += 0.08
        source_bits.append("same source kind")
    if left.get("source_root") and left.get("source_root") == right.get("source_root"):
        source_score += 0.08
        source_bits.append("same source root")

    media_bias = 0.05 if left.get("media_type") == right.get("media_type") else -0.2
    score = (hash_score * 0.78) + (time_score * 0.14) + source_score + media_bias
    if crop_score >= 0.95:
        score += 0.04
    rationale = (
        (
            f"phash={round((1.0 - phash_score) * 64.0, 1)} "
            f"dhash={round((1.0 - dhash_score) * 64.0, 1)} "
            f"whash={round((1.0 - whash_score) * 64.0, 1)} "
            f"color={round((1.0 - color_score) * 64.0, 1)} "
            f"crop={round((1.0 - crop_score) * 64.0, 1)} "
        )
        + f"{time_note}"
        + (f" {'; '.join(source_bits)}" if source_bits else "")
    )
    return score, rationale, average_distance


def run_near(
    db_path: Path,
    managed_library_dir: Path,
    *,
    derivatives_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    max_distance: int = 6,
    clip_model: Optional[str] = None,
) -> Dict[str, int]:
    managed_library_dir = Path(managed_library_dir)
    derivatives_dir = Path(derivatives_dir) if derivatives_dir else None
    processed = hashed = missing = clip_hashed = 0
    buckets: Dict[tuple[str, str], List[Dict]] = defaultdict(list)
    clip_backend = _clip_backend(clip_model)

    for row in manifest.iter_built_assets(db_path=db_path, limit=limit):
        processed += 1
        preview = _preview_path(row, managed_library_dir, derivatives_dir)
        if not preview or not preview.exists():
            missing += 1
            continue
        try:
            bundle = _preview_hash_bundle(preview)
            clip_vector = _clip_embedding(preview, clip_model)
        except Exception:
            missing += 1
            continue
        hashed += 1
        row = {**row, **bundle, "preview_path": str(preview)}
        if clip_vector is not None:
            clip_hashed += 1
            row["clip_vector"] = clip_vector
            if clip_backend:
                manifest.replace_embedding(
                    db_path,
                    asset_id=row["id"],
                    embedding_type="duplicate_visual",
                    model_name=str(clip_backend["spec"]),
                    vector=clip_vector,
                    vector_ref=str(preview),
                    payload={"path": str(preview), "model": clip_backend["spec"]},
                )
        day_bucket = (row.get("dt_original") or "")[:10] or "unknown"
        buckets[(row.get("media_type") or "unknown", day_bucket)].append(row)

    groups: List[Dict] = []
    for (_media_type, _day_bucket), items in buckets.items():
        if len(items) < 2:
            continue
        parent = {item["id"]: item["id"] for item in items}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for i, left in enumerate(items):
            for right in items[i + 1 :]:
                score, rationale, avg_distance = _pair_score(left, right)
                if avg_distance <= max_distance or (score >= 0.82 and avg_distance <= max_distance + 4):
                    union(left["id"], right["id"])

        clip_items = [item for item in items if item.get("clip_vector")]
        neighbor_data = _build_faiss_neighbors([item["clip_vector"] for item in clip_items]) if clip_items and faiss is not None else None
        if neighbor_data:
            scores, indices = neighbor_data
            for i, neighbors in enumerate(indices):
                source_item = clip_items[i]
                for rank, j in enumerate(neighbors[1:], start=1):
                    if j < 0 or j >= len(clip_items):
                        continue
                    clip_score = float(scores[i, rank])
                    if clip_score < 0.95:
                        continue
                    candidate_item = clip_items[int(j)]
                    score, _rationale, _avg_distance = _pair_score(source_item, candidate_item)
                    if score >= 0.82 or clip_score >= 0.98:
                        union(source_item["id"], candidate_item["id"])

        clusters: Dict[str, List[Dict]] = defaultdict(list)
        for item in items:
            clusters[find(item["id"])].append(item)

        for cluster_items in clusters.values():
            if len(cluster_items) < 2:
                continue
            canonical_asset_id = _canonical_asset(cluster_items)
            groups.append(
                {
                    "canonical_asset_id": canonical_asset_id,
                    "items": [
                        {
                            "asset_id": item["id"],
                            "score": round(_pair_score(cluster_items[0], item)[0], 3),
                            "rationale": _pair_score(cluster_items[0], item)[1],
                        }
                        for item in cluster_items
                    ],
                }
            )

    created = manifest.replace_duplicate_groups(
        db_path,
        group_type="NEAR_VISUAL",
        groups=groups,
    )
    return {
        "processed": processed,
        "hashed": hashed,
        "clip_hashed": clip_hashed,
        "missing": missing,
        "groups": created,
    }
