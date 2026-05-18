from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..phase_metadata import manifest


def _cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency fallback
        raise RuntimeError("OpenCV is required for face detection") from exc
    return cv2


def _detectors() -> list[cv2.CascadeClassifier]:
    cv2 = _cv2()
    cascade_names = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
    ]
    candidate_dirs: list[Path] = []
    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if data_dir:
        candidate_dirs.append(Path(str(data_dir)))
    candidate_dirs.extend(
        [
            Path("/opt/homebrew/opt/opencv/share/opencv4/haarcascades"),
            Path("/opt/homebrew/Cellar/opencv/4.13.0_10/share/opencv4/haarcascades"),
            Path("/usr/local/opt/opencv/share/opencv4/haarcascades"),
        ]
    )
    module_path = Path(getattr(cv2, "__file__", "")).resolve()
    candidate_dirs.extend(
        [
            module_path.parent / "data",
            module_path.parent.parent / "share" / "opencv4" / "haarcascades",
            module_path.parent.parent.parent / "share" / "opencv4" / "haarcascades",
        ]
    )
    seen: set[str] = set()
    detectors: list[cv2.CascadeClassifier] = []
    for name in cascade_names:
        for base_dir in candidate_dirs:
            if not base_dir:
                continue
            candidate = (base_dir / name).resolve()
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if not candidate.exists():
                continue
            detector = cv2.CascadeClassifier(str(candidate))
            if not detector.empty():
                detectors.append(detector)
                break
    if not detectors:
        raise RuntimeError("OpenCV face cascades could not be loaded")
    return detectors


def _load_detection_image(asset: Dict, managed_library_dir: Path) -> tuple[Optional[str], Optional[int]]:
    if asset.get("media_type") == "video" and asset.get("thumbnail_path"):
        return asset["thumbnail_path"], 1000
    managed_path = asset.get("managed_path")
    if managed_path:
        return str((managed_library_dir / managed_path).resolve()), None
    return None, None


def _video_frame_offsets(duration_seconds: Optional[float]) -> List[float]:
    if not duration_seconds or duration_seconds <= 0:
        return [0.75, 1.5, 3.0]
    anchors = [0.15, 0.5, 0.85]
    offsets: List[float] = []
    for anchor in anchors:
        offset = max(0.5, min(duration_seconds * anchor, max(0.5, duration_seconds - 0.5)))
        if all(abs(offset - existing) > 0.35 for existing in offsets):
            offsets.append(offset)
    return offsets or [max(0.5, duration_seconds / 2.0)]


def _extract_video_frame(video_path: Path, offset_seconds: float) -> Optional[str]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin or not video_path.exists():
        return None
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        frame_path = tmp.name
    proc = subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{offset_seconds:.2f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            frame_path,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not Path(frame_path).exists():
        try:
            Path(frame_path).unlink(missing_ok=True)
        except Exception:
            pass
        return None
    return frame_path


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    a_area = aw * ah
    b_area = bw * bh
    union = a_area + b_area - inter_area
    return inter_area / union if union else 0.0


def _filter_boxes(boxes: List[Tuple[int, int, int, int]], width: int, height: int) -> List[Tuple[int, int, int, int]]:
    image_area = max(1, width * height)
    kept: List[Tuple[int, int, int, int]] = []
    for x, y, w, h in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        area_fraction = (w * h) / image_area
        aspect_ratio = (w / h) if h else 0
        if area_fraction < 0.012 or area_fraction > 0.7:
            continue
        if aspect_ratio < 0.55 or aspect_ratio > 1.6:
            continue
        if any(_iou((x, y, w, h), existing) > 0.3 for existing in kept):
            continue
        kept.append((x, y, w, h))
        if len(kept) >= 16:
            break
    return kept


def _face_quality(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    cv2 = _cv2()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    brightness = float(gray.mean())
    size_bonus = min(gray.shape[0], gray.shape[1]) / 160.0
    return sharpness * 0.45 + contrast * 1.2 + brightness * 0.1 + size_bonus * 100.0


def _compute_embedding(crop: np.ndarray) -> List[float]:
    cv2 = _cv2()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    normalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(normalized, (3, 3), 0)

    def _feature_map(img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(img, (12, 12), interpolation=cv2.INTER_AREA)
        pooled = resized.astype("float32").reshape(-1)
        pooled -= float(pooled.mean())
        pooled_std = float(pooled.std()) or 1.0
        pooled /= pooled_std

        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        angle = (cv2.phase(grad_x, grad_y, angleInDegrees=False) % (2.0 * math.pi)) / (2.0 * math.pi)
        edge_hist = np.zeros(8, dtype="float32")
        for idx in range(8):
            mask = (angle >= idx / 8.0) & (angle < (idx + 1) / 8.0)
            edge_hist[idx] = float(magnitude[mask].sum()) if np.any(mask) else 0.0
        edge_hist_sum = float(edge_hist.sum()) or 1.0
        edge_hist /= edge_hist_sum

        intensity_hist = cv2.calcHist([img], [0], None, [16], [0, 256]).astype("float32").reshape(-1)
        intensity_hist_sum = float(intensity_hist.sum()) or 1.0
        intensity_hist /= intensity_hist_sum

        combined = np.concatenate([pooled, edge_hist, intensity_hist], axis=0)
        return combined

    vector = (_feature_map(normalized) + _feature_map(cv2.flip(blurred, 1))) / 2.0
    norm = float(np.linalg.norm(vector)) or 1.0
    vector /= norm
    return vector.astype("float32").tolist()


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    aa = np.array(a, dtype="float32")
    bb = np.array(b, dtype="float32")
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb)) or 1.0
    return float(np.dot(aa, bb) / denom)


def _detect_boxes(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    cv2 = _cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    collected: List[Tuple[int, int, int, int]] = []
    for detector in _detectors():
        boxes = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=6, minSize=(56, 56))
        collected.extend(tuple(map(int, box)) for box in boxes)
    return _filter_boxes(collected, gray.shape[1], gray.shape[0])


def _cluster_label(index: int) -> str:
    return f"Cluster {index:03d}"


def _pairwise_similarity(items: List[Dict[str, object]]) -> float:
    if len(items) < 2:
        return 1.0
    scores: List[float] = []
    for i, left in enumerate(items):
        for right in items[i + 1 :]:
            scores.append(_cosine_similarity(left["vector"], right["vector"]))
    return float(sum(scores) / len(scores)) if scores else 0.0


def cluster_faces(db_path: Path, *, similarity_threshold: float = 0.9) -> Dict[str, int]:
    embeddings = manifest.list_face_embeddings(db_path)
    identities = manifest.list_face_identities(db_path, limit=5000)
    identity_index = {item["id"]: item for item in identities}

    centroids: dict[str, List[List[float]]] = {}
    for item in embeddings:
        identity_id = item.get("identity_id")
        vector = item.get("vector") or []
        if identity_id and vector:
            centroids.setdefault(identity_id, []).append(vector)

    def centroid_for(identity_id: str) -> List[float]:
        vectors = centroids.get(identity_id) or []
        if not vectors:
            return []
        arr = np.array(vectors, dtype="float32")
        mean = arr.mean(axis=0)
        norm = float(np.linalg.norm(mean)) or 1.0
        return (mean / norm).astype("float32").tolist()

    assigned = created = 0

    unmatched: List[Dict[str, object]] = []
    for item in embeddings:
        if not item.get("vector"):
            continue
        current_identity = item.get("identity_id")
        if current_identity and identity_index.get(current_identity, {}).get("status") == "CONFIRMED":
            continue
        best_identity: Optional[str] = None
        best_score = 0.0
        for identity_id, vectors in centroids.items():
            if identity_id == current_identity:
                continue
            score = _cosine_similarity(item["vector"], centroid_for(identity_id))
            if score > best_score:
                best_score = score
                best_identity = identity_id
        if best_identity and best_score >= similarity_threshold:
            manifest.update_face_cluster(
                db_path,
                item["face_id"],
                best_identity,
                labeled=identity_index.get(best_identity, {}).get("status") == "CONFIRMED",
            )
            centroids.setdefault(best_identity, []).append(item["vector"])
            assigned += 1
            continue
        unmatched.append(item)

    if unmatched:
        parent = {item["face_id"]: item["face_id"] for item in unmatched}

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

        for i, left in enumerate(unmatched):
            for right in unmatched[i + 1 :]:
                if left.get("asset_id") == right.get("asset_id"):
                    continue
                if _cosine_similarity(left["vector"], right["vector"]) >= max(similarity_threshold, 0.94):
                    union(left["face_id"], right["face_id"])

        clusters: dict[str, list[dict]] = {}
        for item in unmatched:
            clusters.setdefault(find(item["face_id"]), []).append(item)

        existing_cluster_numbers = [
            int(item["label"].split()[-1])
            for item in identities
            if str(item.get("label") or "").startswith("Cluster ") and str(item["label"]).split()[-1].isdigit()
        ]
        next_cluster = max(existing_cluster_numbers, default=0) + 1
        for cluster_items in clusters.values():
            unique_assets = {item.get("asset_id") for item in cluster_items if item.get("asset_id")}
            if len(cluster_items) < 2:
                continue
            if len(unique_assets) < 2 and len(cluster_items) < 3:
                continue
            if _pairwise_similarity(cluster_items) < 0.95:
                continue
            identity_id = manifest.create_face_identity(db_path, _cluster_label(next_cluster), status="CLUSTERED")
            next_cluster += 1
            identities = manifest.list_face_identities(db_path, limit=5000)
            identity_index = {identity["id"]: identity for identity in identities}
            for item in cluster_items:
                manifest.update_face_cluster(db_path, item["face_id"], identity_id, labeled=False)
                centroids.setdefault(identity_id, []).append(item["vector"])
            created += 1
    return {"assigned": assigned, "clusters_created": created}


def detect_faces(
    db_path: Path,
    managed_library_dir: Path,
    *,
    limit: int | None = None,
    force: bool = False,
) -> Dict[str, int]:
    try:
        _cv2()
    except RuntimeError:
        return {"processed": 0, "detected": 0, "failed": 0, "embedded": 0, "assigned": 0, "clusters_created": 0}
    processed = detected = failed = embedded = 0
    for asset in manifest.iter_assets_for_face_detection(db_path=db_path, limit=limit, include_existing=force):
        processed += 1
        image_path, frame_time_ms = _load_detection_image(asset, managed_library_dir)
        source_paths: List[tuple[str, Optional[int]]] = []
        if image_path and Path(image_path).exists():
            source_paths.append((image_path, frame_time_ms))
        if asset.get("media_type") == "video":
            managed_path = asset.get("managed_path")
            video_path = (managed_library_dir / managed_path).resolve() if managed_path else None
            duration_seconds = asset.get("duration_seconds")
            if video_path and video_path.exists():
                for offset in _video_frame_offsets(float(duration_seconds) if duration_seconds is not None else None):
                    extracted = _extract_video_frame(video_path, offset)
                    if extracted:
                        source_paths.append((extracted, int(offset * 1000)))
        if not source_paths:
            failed += 1
            continue
        cv2 = _cv2()
        detections = []
        seen_boxes: List[Tuple[int, int, int, int]] = []
        extracted_temp_paths = [Path(path) for path, _ in source_paths if path not in {image_path}]
        try:
            for candidate_path, candidate_frame_ms in source_paths:
                image = cv2.imread(candidate_path)
                if image is None:
                    continue
                for x, y, w, h in _detect_boxes(image):
                    if any(_iou((x, y, w, h), existing) > 0.7 for existing in seen_boxes):
                        continue
                    crop = image[max(0, y): max(0, y + h), max(0, x): max(0, x + w)]
                    quality = _face_quality(crop)
                    if quality < 35:
                        continue
                    seen_boxes.append((x, y, w, h))
                    face_id = uuid.uuid4().hex[:20]
                    detections.append(
                        {
                            "id": face_id,
                            "bbox": {
                                "x": int(x),
                                "y": int(y),
                                "w": int(w),
                                "h": int(h),
                                "image_width": int(image.shape[1]),
                                "image_height": int(image.shape[0]),
                                "quality": round(quality, 2),
                            },
                            "frame_time_ms": candidate_frame_ms,
                            "embedding_vector": _compute_embedding(crop),
                        }
                    )
        finally:
            for temp_path in extracted_temp_paths:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
        count = manifest.replace_faces_for_asset(db_path, asset["id"], detections, source_name="opencv_haar_clustered")
        detected += count
        for detection in detections:
            manifest.save_face_embedding(
                db_path,
                asset_id=asset["id"],
                face_id=detection["id"],
                vector=detection["embedding_vector"],
            )
            embedded += 1
    clustered = cluster_faces(db_path)
    return {
        "processed": processed,
        "detected": detected,
        "failed": failed,
        "embedded": embedded,
        **clustered,
    }
